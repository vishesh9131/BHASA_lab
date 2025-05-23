import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import json
import os
import time
import requests
from pathlib import Path
import signal
import sys
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

# Global flag for graceful shutdown
training_interrupted = False

def signal_handler(signum, frame):
    global training_interrupted
    print("\n🛑 Training interrupted by user. Saving checkpoint...")
    training_interrupted = True

signal.signal(signal.SIGINT, signal_handler)

@dataclass
class MambaConfig:
    """Configuration for BHASA Mamba model"""
    vocab_size: int = 32000
    d_model: int = 768          # Model dimension
    n_layers: int = 12          # Number of Mamba blocks
    dt_rank: int = 128          # Rank of Δ (delta)
    d_state: int = 64           # State dimension (N in paper)
    expand_factor: int = 2      # Expansion factor in MLP
    d_conv: int = 4            # Local convolution width
    dt_min: float = 0.001      # Minimum delta
    dt_max: float = 0.1        # Maximum delta
    dt_init: str = "random"    # Delta initialization
    dt_scale: float = 1.0      # Delta scaling
    bias: bool = False         # Use bias in linear layers
    conv_bias: bool = True     # Use bias in convolution
    dropout: float = 0.1       # Dropout rate
    max_seq_len: int = 2048    # Maximum sequence length

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) - Core of Mamba architecture
    
    This implements the selective scan operation that makes Mamba unique.
    The key innovation is making A, B, C parameters input-dependent.
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.dt_rank = config.dt_rank
        self.d_conv = config.d_conv
        
        # S4D linear layer - creates A matrix (diagonal state matrix)
        self.A_log = nn.Parameter(torch.log(torch.rand(config.d_model, config.d_state)))
        
        # Linear layers for input-dependent B, C, and Δ
        self.x_proj = nn.Linear(config.d_model, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_model, bias=True)
        
        # Local convolution for better locality
        self.conv1d = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_model,
            padding=config.d_conv - 1
        )
        
        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        # Initialize dt_proj with special initialization
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        with torch.no_grad():
            self.dt_proj.weight.uniform_(-dt_init_std, dt_init_std)
            
        # Initialize A with S4D parameterization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_model, 1)
        self.A_log.data = torch.log(A)  # Ensures A is negative
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Apply local convolution (transpose for conv1d)
        x_conv = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim padding
        x_conv = x_conv.transpose(1, 2)  # Back to (batch, seq_len, d_model)
        
        # Apply SiLU activation
        x_conv = F.silu(x_conv)
        
        # Project to get dt, B, C (input-dependent parameters)
        x_dbl = self.x_proj(x_conv)  # (batch, seq_len, dt_rank + 2*d_state)
        
        # Split into dt, B, C
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Apply dt projection with softplus to ensure positivity
        dt = self.dt_proj(dt)  # (batch, seq_len, d_model)
        dt = F.softplus(dt)
        
        # Get A matrix (always negative for stability)
        A = -torch.exp(self.A_log.float())  # (d_model, d_state)
        
        # Perform selective scan
        y = self.selective_scan(x_conv, dt, A, B, C)
        
        # Output projection
        y = self.out_proj(y)
        
        return y
    
    def selective_scan(self, u, dt, A, B, C):
        """
        Selective scan operation - the heart of Mamba
        
        u: input (batch, seq_len, d_model)
        dt: delta (batch, seq_len, d_model)  
        A: state matrix (d_model, d_state)
        B: input matrix (batch, seq_len, d_state)
        C: output matrix (batch, seq_len, d_state)
        """
        batch_size, seq_len, d_model = u.shape
        d_state = A.shape[1]
        
        # Discretize continuous parameters
        # dt: (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        dt = dt.transpose(1, 2)  # (batch, d_model, seq_len)
        A = A.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch, d_model, d_state)
        
        # Discretization: A_discrete = exp(dt * A)
        # dt: (batch, d_model, seq_len) -> (batch, d_model, seq_len, 1)
        # A: (batch, d_model, d_state) -> (batch, d_model, 1, d_state)
        dtA = dt.unsqueeze(-1) * A.unsqueeze(2)  # (batch, d_model, seq_len, d_state)
        A_discrete = torch.exp(dtA)
        
        # Discretization: B_discrete = dt * u * B
        # u: (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        u_transposed = u.transpose(1, 2)  # (batch, d_model, seq_len)
        dt_u = dt * u_transposed  # (batch, d_model, seq_len)
        
        # B: (batch, seq_len, d_state) -> (batch, d_state, seq_len)
        B_transposed = B.transpose(1, 2)  # (batch, d_state, seq_len)
        
        # Broadcast for multiplication: (batch, d_model, seq_len, 1) * (batch, 1, seq_len, d_state)
        dt_u_expanded = dt_u.unsqueeze(-1)  # (batch, d_model, seq_len, 1)
        B_expanded = B_transposed.unsqueeze(1)  # (batch, 1, d_state, seq_len) -> (batch, 1, seq_len, d_state)
        B_expanded = B_expanded.transpose(2, 3)  # (batch, 1, seq_len, d_state)
        
        B_discrete = dt_u_expanded * B_expanded  # (batch, d_model, seq_len, d_state)
        
        # Initialize state
        state = torch.zeros(batch_size, d_model, d_state, device=u.device, dtype=u.dtype)
        
        outputs = []
        
        # Sequential scan (can be parallelized with associative scan)
        for i in range(seq_len):
            # Update state: x_{k+1} = A_k * x_k + B_k * u_k
            state = A_discrete[:, :, i] * state + B_discrete[:, :, i]
            
            # Compute output: y_k = C_k * x_k
            # C: (batch, seq_len, d_state) -> take step i -> (batch, d_state)
            # state: (batch, d_model, d_state)
            # We need to sum over d_state dimension
            C_i = C[:, i]  # (batch, d_state)
            # Expand C_i to match state dimensions for broadcasting
            C_i_expanded = C_i.unsqueeze(1)  # (batch, 1, d_state)
            output = torch.sum(C_i_expanded * state, dim=-1)  # (batch, d_model)
            outputs.append(output)
        
        # Stack outputs
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        
        return y

class MambaBlock(nn.Module):
    """
    A single Mamba block combining:
    1. Selective SSM
    2. Gated MLP (like SwiGLU)
    3. Residual connections
    4. Layer normalization
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        # Layer norm before SSM
        self.norm1 = nn.LayerNorm(config.d_model)
        
        # Selective SSM
        self.ssm = SelectiveSSM(config)
        
        # Layer norm before MLP
        self.norm2 = nn.LayerNorm(config.d_model)
        
        # Gated MLP
        self.mlp = GatedMLP(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        # SSM branch with residual connection
        residual = x
        x = self.norm1(x)
        x = self.ssm(x)
        x = self.dropout(x)
        x = x + residual
        
        # MLP branch with residual connection
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual
        
        return x

class GatedMLP(nn.Module):
    """
    Gated MLP similar to SwiGLU used in modern transformers
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        hidden_dim = config.d_model * config.expand_factor
        
        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class BhasaMamba(nn.Module):
    """
    BHASA (Bayesian Hyperdimensional Adaptive Sequential Architecture)
    
    Full Mamba language model with:
    - Token embedding
    - Stack of Mamba blocks  
    - Language modeling head
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.norm_f = nn.LayerNorm(config.d_model)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie embeddings (common practice)
        self.lm_head.weight = self.embedding.weight
        
    def _init_weights(self, module):
        """Initialize weights following best practices"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids, targets=None):
        """
        input_ids: (batch_size, seq_len)
        targets: (batch_size, seq_len) for training
        """
        # Token embeddings
        x = self.embedding(input_ids)  # (batch, seq_len, d_model)
        
        # Apply Mamba blocks
        for layer in self.layers:
            x = layer(x)
            
        # Final layer norm
        x = self.norm_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        loss = None
        if targets is not None:
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
            
        return logits, loss
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50):
        """Generate text from the model"""
        self.eval()
        device = input_ids.device
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits, _ = self(input_ids)
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if we hit max context
                if input_ids.shape[1] >= self.config.max_seq_len:
                    break
                    
        return input_ids

class TextDataset(Dataset):
    """Dataset for language modeling"""
    
    def __init__(self, text_data, tokenizer, seq_length=512):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Tokenize the entire text
        self.tokens = self.tokenizer.encode(text_data)
        
        # Create overlapping sequences
        self.examples = []
        for i in range(0, len(self.tokens) - seq_length, seq_length // 2):
            self.examples.append(self.tokens[i:i + seq_length + 1])
            
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, targets

class SimpleTokenizer:
    """Simple character-level tokenizer"""
    
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        
    def build_vocab(self, text):
        """Build vocabulary from text"""
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        
    def encode(self, text):
        """Encode text to token ids"""
        return [self.char_to_id.get(ch, 0) for ch in text]
    
    def decode(self, token_ids):
        """Decode token ids to text"""
        return ''.join([self.id_to_char.get(id, '<unk>') for id in token_ids])
    
    def save(self, path):
        """Save tokenizer"""
        data = {
            'char_to_id': self.char_to_id,
            'id_to_char': self.id_to_char,
            'vocab_size': self.vocab_size
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path):
        """Load tokenizer"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.char_to_id = data['char_to_id']
        self.id_to_char = {int(k): v for k, v in data['id_to_char'].items()}
        self.vocab_size = data['vocab_size']

def download_dataset(url, filename):
    """Download dataset if not exists"""
    if os.path.exists(filename):
        print(f"📁 Dataset {filename} already exists")
        return
    
    print(f"📥 Downloading dataset from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading"):
            f.write(chunk)
    
    print(f"✅ Downloaded {filename}")

def load_or_download_data():
    """Load training data"""
    # Try to use existing data first
    if os.path.exists('DUMP/data.txt'):
        print("📖 Loading existing data.txt")
        with open('DUMP/data.txt', 'r', encoding='utf-8') as f:
            return f.read()
    
    # Download a small dataset for training
    datasets = [
        ('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', 'shakespeare.txt'),
        ('https://www.gutenberg.org/files/11/11-0.txt', 'alice.txt'),  # Alice in Wonderland
    ]
    
    for url, filename in datasets:
        try:
            download_dataset(url, filename)
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
            # Clean the text a bit
            if 'gutenberg' in url.lower():
                # Remove Project Gutenberg header/footer
                lines = text.split('\n')
                start_idx = 0
                end_idx = len(lines)
                for i, line in enumerate(lines):
                    if 'START OF THE PROJECT GUTENBERG EBOOK' in line.upper():
                        start_idx = i + 1
                        break
                for i in range(len(lines) - 1, -1, -1):
                    if 'END OF THE PROJECT GUTENBERG EBOOK' in lines[i].upper():
                        end_idx = i
                        break
                text = '\n'.join(lines[start_idx:end_idx])
            
            return text[:500000]  # Use first 500k characters
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            continue
    
    # Fallback: create sample data
    print("🔄 Using fallback sample data")
    return """The quick brown fox jumps over the lazy dog. """ * 1000

def save_checkpoint(model, optimizer, epoch, loss, config, tokenizer, checkpoint_dir='checkpoints'):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config.__dict__,
        'tokenizer_vocab': {
            'char_to_id': tokenizer.char_to_id,
            'id_to_char': tokenizer.id_to_char,
            'vocab_size': tokenizer.vocab_size
        }
    }
    
    checkpoint_path = f"{checkpoint_dir}/bhasa_checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Also save as latest
    latest_path = f"{checkpoint_dir}/bhasa_latest.pt"
    torch.save(checkpoint, latest_path)
    print(f"💾 Latest checkpoint: {latest_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint.get('tokenizer_vocab')

def train_bhasa_mamba():
    """Main training function"""
    global training_interrupted
    
    print("🚀 Starting BHASA Mamba training...")
    
    # For MacBook Air M2, use CPU to avoid MPS issues
    device = torch.device('cpu')
    print("💻 Using CPU for stable training")
    
    # Load data
    print("📚 Loading training data...")
    text_data = load_or_download_data()
    print(f"📊 Loaded {len(text_data)} characters")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(text_data)
    print(f"🔤 Vocabulary size: {tokenizer.vocab_size}")
    
    # Save tokenizer
    tokenizer.save('bhasa_tokenizer.json')
    
    # Create smaller config for MacBook Air
    config = MambaConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256,       # Reduced from 512
        n_layers=4,        # Reduced from 8
        dt_rank=32,        # Reduced from 64
        d_state=16,        # Reduced from 32
        max_seq_len=256    # Reduced from 512
    )
    
    # Create model
    model = BhasaMamba(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model parameters: {total_params:,}")
    
    # Create dataset and dataloader
    dataset = TextDataset(text_data, tokenizer, seq_length=128)  # Reduced from 256
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)  # Reduced batch size
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)  # Slightly higher LR for smaller model
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader))
    
    # Check for existing checkpoint
    checkpoint_dir = 'checkpoints'
    latest_checkpoint = f"{checkpoint_dir}/bhasa_latest.pt"
    start_epoch = 0
    
    if os.path.exists(latest_checkpoint):
        print("🔄 Resuming from checkpoint...")
        try:
            start_epoch, last_loss, tokenizer_vocab = load_checkpoint(latest_checkpoint, model, optimizer)
            print(f"📍 Resumed from epoch {start_epoch}, loss: {last_loss:.4f}")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            start_epoch = 0
    
    # Training loop
    model.train()
    num_epochs = 20  # Reduced for faster testing
    
    print(f"🎯 Training for {num_epochs} epochs...")
    
    for epoch in range(start_epoch, num_epochs):
        if training_interrupted:
            break
            
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (input_ids, targets) in enumerate(progress_bar):
            if training_interrupted:
                break
                
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            try:
                # Forward pass
                logits, loss = model(input_ids, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{epoch_loss/num_batches:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.6f}"
                })
                
                # Save checkpoint every 50 batches (more frequent for smaller model)
                if batch_idx % 50 == 0 and batch_idx > 0:
                    save_checkpoint(model, optimizer, epoch, loss.item(), config, tokenizer)
                    
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                continue
        
        # Save checkpoint at end of epoch
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        save_checkpoint(model, optimizer, epoch + 1, avg_loss, config, tokenizer)
        
        print(f"📈 Epoch {epoch+1} completed - Average loss: {avg_loss:.4f}")
        
        # Generate sample text every 2 epochs
        if (epoch + 1) % 2 == 0:
            print("🎨 Generating sample text...")
            model.eval()
            with torch.no_grad():
                try:
                    prompt = "The meaning of life is"
                    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
                    generated = model.generate(input_ids, max_length=30, temperature=0.8)
                    generated_text = tokenizer.decode(generated[0].cpu().tolist())
                    print(f"📝 Generated: {generated_text}")
                except Exception as e:
                    print(f"⚠️ Generation error: {e}")
            model.train()
    
    # Final save
    print("🏁 Training completed!")
    save_checkpoint(model, optimizer, num_epochs, avg_loss, config, tokenizer)
    
    # Save final model
    final_model_path = "bhasa_mamba_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'tokenizer': {
            'char_to_id': tokenizer.char_to_id,
            'id_to_char': tokenizer.id_to_char,
            'vocab_size': tokenizer.vocab_size
        }
    }, final_model_path)
    print(f"🎉 Final model saved: {final_model_path}")

if __name__ == "__main__":
    train_bhasa_mamba() 