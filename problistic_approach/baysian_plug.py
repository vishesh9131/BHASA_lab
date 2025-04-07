import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import math
from typing import Tuple, Optional, Dict

class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight and bias posterior distributions."""
    
    def __init__(self, in_features: int, out_features: int, prior_scale: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight posterior parameters (mean and log_var)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_log_var = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-9, 0.1))
        
        # Bias posterior parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_log_var = nn.Parameter(torch.Tensor(out_features).normal_(-9, 0.1))
        
        # Prior distributions
        self.weight_prior = dist.Normal(0, prior_scale)
        self.bias_prior = dist.Normal(0, prior_scale)
        
        self.kl_weight = 1.0
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            # Sample weights and biases from posterior
            weight_epsilon = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + torch.exp(0.5 * self.weight_log_var) * weight_epsilon
            
            bias_epsilon = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + torch.exp(0.5 * self.bias_log_var) * bias_epsilon
        else:
            # Use posterior means for prediction (MAP estimate)
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior distributions."""
        # KL for weights
        kl_weight = 0.5 * torch.sum(
            torch.exp(self.weight_log_var) + self.weight_mu**2 - 1 - self.weight_log_var
        )
        
        # KL for biases
        kl_bias = 0.5 * torch.sum(
            torch.exp(self.bias_log_var) + self.bias_mu**2 - 1 - self.bias_log_var
        )
        
        return (kl_weight + kl_bias) * self.kl_weight

class BayesianSSMKernel(nn.Module):
    """Bayesian implementation of the core SSM kernel in Mamba."""
    
    def __init__(self, d_model: int, d_state: int, prior_scale: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Parameter matrices for the SSM
        self.A_projection = BayesianLinear(d_model, d_state, prior_scale)
        self.B_projection = BayesianLinear(d_model, d_state, prior_scale)
        self.C_projection = BayesianLinear(d_state, d_model, prior_scale)
        
        # Input-dependent parameter projections
        self.dt_projection = BayesianLinear(d_model, d_state, prior_scale)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the SSM parameters based on the input.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            sample: Whether to sample from parameter posteriors
            
        Returns:
            The SSM output and updated hidden state
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute input-dependent SSM parameters
        A = -torch.exp(self.A_projection(x, sample)) # Make A stable with negative exponential
        dt = torch.exp(self.dt_projection(x, sample)) # Positive time delta
        B = self.B_projection(x, sample)
        C = self.C_projection(x, sample)
        
        # Initialize hidden state (h₀ = 0)
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        outputs = []
        
        # Discretized SSM recurrence
        for t in range(seq_len):
            # Update rule: h_t = (exp(A * dt) * h_{t-1}) + (B * x_t)
            A_dt = A[:, t] * dt[:, t]
            h = h * torch.exp(A_dt.unsqueeze(1)) + B[:, t].unsqueeze(1) * x[:, t].unsqueeze(1)
            y = (C[:, t].unsqueeze(1) * h).sum(dim=1)
            outputs.append(y)
        
        # Stack outputs along sequence dimension
        return torch.stack(outputs, dim=1)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute total KL divergence for all Bayesian layers."""
        kl_A = self.A_projection.kl_divergence()
        kl_B = self.B_projection.kl_divergence()
        kl_C = self.C_projection.kl_divergence()
        kl_dt = self.dt_projection.kl_divergence()
        
        return kl_A + kl_B + kl_C + kl_dt

class SelectiveScanMechanism(nn.Module):
    """Input-dependent parameter selection mechanism for Mamba."""
    
    def __init__(self, d_model: int, d_state: int, prior_scale: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Selective scan parameter projection
        self.S_proj = BayesianLinear(d_model, d_state * 2, prior_scale)  # For selective gating
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Compute the selective scan parameters.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            sample: Whether to sample from parameter posteriors
            
        Returns:
            Selective scan parameters
        """
        # Project input to get selection parameters
        S = self.S_proj(x, sample)
        
        # Split into selection parameters
        S_a, S_b = torch.chunk(S, 2, dim=-1)
        
        # Apply activation functions
        S_a = torch.sigmoid(S_a)  # Gate for A matrix
        S_b = torch.sigmoid(S_b)  # Gate for B matrix
        
        return S_a, S_b
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence."""
        return self.S_proj.kl_divergence()

class BayesianMambaBlock(nn.Module):
    """Full Bayesian Mamba block with selective state space model."""
    
    def __init__(
        self, 
        d_model: int, 
        d_state: int = 16, 
        d_conv: int = 4,
        expand_factor: int = 2,
        prior_scale: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor
        
        # Input projection (expands dimension)
        self.in_proj = BayesianLinear(d_model, self.d_inner, prior_scale)
        
        # 1D Convolution for local context
        self.conv_weight_mu = nn.Parameter(torch.randn(self.d_inner, 1, d_conv) * 0.02)
        self.conv_weight_log_var = nn.Parameter(torch.ones(self.d_inner, 1, d_conv) * -9)
        self.conv_bias_mu = nn.Parameter(torch.zeros(self.d_inner))
        self.conv_bias_log_var = nn.Parameter(torch.ones(self.d_inner) * -9)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # SSM components
        self.ssm_kernel = BayesianSSMKernel(self.d_inner, d_state, prior_scale)
        self.selection_mechanism = SelectiveScanMechanism(self.d_inner, d_state, prior_scale)
        
        # Output projection (reduced dimension)
        self.out_proj = BayesianLinear(self.d_inner, d_model, prior_scale)
        
        # Gating mechanism
        self.gate_proj = BayesianLinear(d_model, self.d_inner, prior_scale)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Process input sequence through the Bayesian Mamba block.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            sample: Whether to sample from parameter posteriors
            
        Returns:
            Processed sequence with same shape as input
        """
        residual = x
        x = self.norm(x)
        
        # Input projection
        x_proj = self.in_proj(x, sample)
        
        # Apply 1D convolution with sampled parameters
        if sample:
            conv_weight_epsilon = torch.randn_like(self.conv_weight_mu)
            conv_weight = self.conv_weight_mu + torch.exp(0.5 * self.conv_weight_log_var) * conv_weight_epsilon
            
            conv_bias_epsilon = torch.randn_like(self.conv_bias_mu)
            conv_bias = self.conv_bias_mu + torch.exp(0.5 * self.conv_bias_log_var) * conv_bias_epsilon
        else:
            conv_weight = self.conv_weight_mu
            conv_bias = self.conv_bias_mu
        
        # Apply 1D convolution for local context
        batch, seq_len, d_inner = x_proj.shape
        x_conv = x_proj.permute(0, 2, 1).contiguous()  # [batch, d_inner, seq_len]
        x_conv = F.pad(x_conv, (conv_weight.shape[-1] - 1, 0))  # Causal padding
        x_conv = F.conv1d(x_conv, conv_weight, conv_bias, groups=1)
        x_conv = x_conv.permute(0, 2, 1)  # [batch, seq_len, d_inner]
        
        # Apply activation function
        x_act = F.silu(x_conv)
        
        # Selection parameters
        S_a, S_b = self.selection_mechanism(x_act, sample)
        
        # Apply Bayesian SSM
        x_ssm = self.ssm_kernel(x_act * S_a, sample)
        
        # Gating mechanism
        gate = F.silu(self.gate_proj(x, sample))
        x_gated = gate * x_ssm
        
        # Output projection
        out = self.out_proj(x_gated, sample)
        
        return out + residual
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute total KL divergence for all Bayesian components."""
        # KL for linear layers
        kl_in = self.in_proj.kl_divergence()
        kl_gate = self.gate_proj.kl_divergence()
        kl_out = self.out_proj.kl_divergence()
        
        # KL for SSM components
        kl_ssm = self.ssm_kernel.kl_divergence()
        kl_select = self.selection_mechanism.kl_divergence()
        
        # KL for convolution parameters
        kl_conv_weight = 0.5 * torch.sum(
            torch.exp(self.conv_weight_log_var) + self.conv_weight_mu**2 - 1 - self.conv_weight_log_var
        )
        kl_conv_bias = 0.5 * torch.sum(
            torch.exp(self.conv_bias_log_var) + self.conv_bias_mu**2 - 1 - self.conv_bias_log_var
        )
        
        return kl_in + kl_gate + kl_out + kl_ssm + kl_select + kl_conv_weight + kl_conv_bias

class BayesianMambaModel(nn.Module):
    """Complete Bayesian Mamba model for sequence processing with uncertainty estimation."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        d_state: int = 16,
        n_layers: int = 12,
        dropout: float = 0.1,
        prior_scale: float = 0.1,
        kl_weight: float = 0.01,
        mc_samples: int = 5
    ):
        super().__init__()
        self.d_model = d_model
        self.mc_samples = mc_samples
        self.kl_weight = kl_weight
        
        # Token embedding (deterministic, could be made Bayesian if needed)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Positional encoding (deterministic)
        self.register_buffer("pos_encoding", self._create_pos_encoding(2048, d_model))
        
        # Mamba blocks
        self.layers = nn.ModuleList([
            BayesianMambaBlock(d_model, d_state, prior_scale=prior_scale)
            for _ in range(n_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = BayesianLinear(d_model, vocab_size, prior_scale)
    
    def _create_pos_encoding(self, max_seq_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        sample: bool = True,
        return_uncertainty: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Process input sequence through the Bayesian Mamba model.
        
        Args:
            x: Input tensor of shape [batch, seq_len] with token indices
            sample: Whether to sample from parameter posteriors
            return_uncertainty: Whether to compute prediction uncertainty
            
        Returns:
            Dictionary with model outputs and optionally uncertainty estimates
        """
        batch_size, seq_len = x.shape
        
        # Token embeddings
        x_emb = self.embedding(x)
        
        # Add positional encodings
        x_pos = x_emb + self.pos_encoding[:, :seq_len]
        x_pos = self.embedding_dropout(x_pos)
        
        # Process through Mamba blocks
        x_mamba = x_pos
        for layer in self.layers:
            x_mamba = layer(x_mamba, sample)
        
        # Output normalization
        x_norm = self.norm(x_mamba)
        
        if return_uncertainty and sample:
            # Monte Carlo sampling for uncertainty estimation
            logits_samples = []
            for _ in range(self.mc_samples):
                logits = self.output_proj(x_norm, sample=True)
                logits_samples.append(logits)
            
            # Stack samples
            logits_samples = torch.stack(logits_samples, dim=0)  # [samples, batch, seq, vocab]
            
            # Mean prediction
            mean_logits = logits_samples.mean(dim=0)
            
            # Uncertainty estimation (variance across samples)
            probs_samples = torch.softmax(logits_samples, dim=-1)
            mean_probs = probs_samples.mean(dim=0)
            variance = ((probs_samples - mean_probs.unsqueeze(0))**2).mean(dim=0).sum(dim=-1)
            
            return {
                "logits": mean_logits,
                "uncertainty": variance
            }
        else:
            # Standard forward pass
            logits = self.output_proj(x_norm, sample)
            return {"logits": logits}
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute total KL divergence across all Bayesian components."""
        kl_total = 0.0
        
        # KL for output projection
        kl_total += self.output_proj.kl_divergence()
        
        # KL for all Mamba blocks
        for layer in self.layers:
            kl_total += layer.kl_divergence()
        
        return kl_total * self.kl_weight
    
    def loss_function(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        sample: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute ELBO loss: reconstruction_loss + kl_divergence
        
        Args:
            x: Input sequence tensor [batch, seq_len]
            y: Target sequence tensor [batch, seq_len]
            sample: Whether to sample from parameter posteriors
            
        Returns:
            Total loss and dictionary with loss components
        """
        # Forward pass
        outputs = self.forward(x, sample)
        logits = outputs["logits"]
        
        # Reconstruction loss (cross-entropy)
        rec_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            y.reshape(-1),
            reduction='mean'
        )
        
        # KL divergence
        kl_loss = self.kl_divergence() if sample else 0.0
        
        # Total loss (ELBO)
        total_loss = rec_loss + kl_loss
        
        return total_loss, {
            "total_loss": total_loss,
            "rec_loss": rec_loss,
            "kl_loss": kl_loss
        }
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        num_samples: int = 30
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions with uncertainty estimation.
        
        Args:
            x: Input sequence tensor [batch, seq_len]
            num_samples: Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            Dictionary with mean predictions and uncertainty estimates
        """
        self.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            # Store original MC samples setting
            original_samples = self.mc_samples
            self.mc_samples = num_samples
            
            # Forward pass with uncertainty
            outputs = self.forward(x, sample=True, return_uncertainty=True)
            
            # Restore original setting
            self.mc_samples = original_samples
        
        return outputs

# Training loop with Bayesian principles
def train_bayesian_mamba(
    model: BayesianMambaModel,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
    kl_annealing: bool = True,
    device: str = "cuda"
):
    """Training loop for Bayesian Mamba model."""
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # KL annealing (optional)
        if kl_annealing:
            kl_weight = min(1.0, epoch / (epochs // 2)) * model.kl_weight
            model.kl_weight = kl_weight
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass with sampling
            loss, loss_components = model.loss_function(data, target, sample=True)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Rec: {loss_components['rec_loss']:.4f}, "
                      f"KL: {loss_components['kl_loss']:.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_epoch_loss:.4f}")
        
        # Evaluate periodically
        evaluate_bayesian_mamba(model, train_loader, device)

def evaluate_bayesian_mamba(
    model: BayesianMambaModel,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cuda"
):
    """Evaluate Bayesian Mamba model with uncertainty estimation."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Predict with uncertainty
            outputs = model.predict_with_uncertainty(data)
            logits = outputs["logits"]
            uncertainty = outputs["uncertainty"]
            
            # Compute loss (without KL since we're evaluating)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                target.reshape(-1)
            )
            total_loss += loss.item()
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == target).float().mean().item()
            
            # Average uncertainty
            avg_uncertainty = uncertainty.mean().item()
            
            print(f"Test Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, "
                  f"Avg Uncertainty: {avg_uncertainty:.4f}")
    
    avg_loss = total_loss / len(test_loader)
    print(f"Final Test Loss: {avg_loss:.4f}")