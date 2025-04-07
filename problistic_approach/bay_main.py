# author :  biswajit
import torch
from torch.utils.data import DataLoader, TensorDataset
from baysian_plug import BayesianMambaModel , train_bayesian_mamba , evaluate_bayesian_mamba

def create_synthetic_data(num_samples=1000, seq_len=32, vocab_size=10000):
    # Generate random sequences
    x = torch.randint(1, vocab_size, (num_samples, seq_len))
    # Target is next token prediction (shifted by 1)
    y = torch.cat([x[:, 1:], torch.randint(1, vocab_size, (num_samples, 1))], dim=1)
    return TensorDataset(x, y)
"""
assign a simple synthetic dataset
"""
# Create datasets
train_dataset = create_synthetic_data(num_samples=5000)
test_dataset = create_synthetic_data(num_samples=1000)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize model
vocab_size = 10000
model = BayesianMambaModel(
    vocab_size=vocab_size,
    d_model=128,
    d_state=16, 
    n_layers=2,
    prior_scale=0.1,
    kl_weight=0.001,
    mc_samples=5
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = "cuda" if torch.cuda.is_available() else "cpu"
train_bayesian_mamba(
    model, 
    train_loader,
    optimizer,
    epochs=5,
    kl_annealing=True,
    device=device
)

evaluate_bayesian_mamba(model, test_loader, device)

def generate_with_uncertainty(model, prompt, max_len=50, temperature=1.0):
    model.eval()
    with torch.no_grad():
        # Convert prompt to tensor
        if isinstance(prompt, str):

            prompt_tensor = torch.tensor([[1, 2, 3]]).to(next(model.parameters()).device)
        else:
            prompt_tensor = prompt.to(next(model.parameters()).device)
        
        generated = prompt_tensor.clone()
        uncertainties = []
        
        for _ in range(max_len):
            # Get predictions with uncertainty
            outputs = model.predict_with_uncertainty(generated[:, -32:] if generated.size(1) > 32 else generated)
            logits = outputs["logits"][:, -1, :] / temperature
            uncertainty = outputs["uncertainty"][:, -1]
            
            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            uncertainties.append(uncertainty.item())
        
        return generated, uncertainties

prompt = torch.tensor([[1, 2, 3]])
generated_sequence, uncertainties = generate_with_uncertainty(model, prompt, max_len=30)
print("Generated sequence:", generated_sequence)
print("Uncertainties:", uncertainties)