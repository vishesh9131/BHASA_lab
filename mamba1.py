import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import argparse
import pandas as pd
import json
import requests
import io
import gzip
import time

# Modified TextDataset class for HelpSteer2 data
class HelpSteer2Dataset(Dataset):
    def __init__(self, data, seq_length=50):
        # Combine all text into a single string
        self.text = ""
        for item in data:
            # Combine conversation into a single text using prompt and response
            conversation = f"User: {item['prompt']}\nAssistant: {item['response']}\n\n"
            self.text += conversation
            
        # Create character level tokenization
        self.seq_length = seq_length
        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data = [self.char_to_idx[ch] for ch in self.text]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx+self.seq_length]),
            torch.tensor(self.data[idx+1:idx+self.seq_length+1])
        )

# Define the Mamba model architecture
class MambaModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(MambaModel, self).__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

def download_helpsteer_data():
    print("Downloading HelpSteer dataset...")
    url = "https://huggingface.co/datasets/nvidia/HelpSteer/resolve/main/data/train.jsonl.gz"
    
    try:
        # Download the file
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Read the gzipped content
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
            data = []
            for line in f:
                data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        print(f"Dataset loaded successfully with {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        raise

# Load and prepare the HelpSteer2 dataset
def load_helpsteer2_data():
    print("Loading HelpSteer2 dataset...")
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("nvidia/HelpSteer2")
        # Combine train and validation splits
        data = list(dataset['train']) + list(dataset['validation'])
        print(f"Loaded dataset with {len(data)} samples")
        return data
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

# Modified training setup
def train_model(dataset, seq_length=50, hidden_size=128, num_layers=1, epochs=1, model_path='mamba_helpsteer555.pth', batch_size=32, resume=True):
    # Create dataset instance
    train_dataset = HelpSteer2Dataset(dataset, seq_length)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = MambaModel(len(train_dataset.chars), hidden_size, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_loss = float('inf')
    training_stats = {
        'epochs': [],
        'losses': [],
        'best_loss': float('inf'),
        'time_elapsed': 0
    }

    # Load checkpoint if resuming
    if resume:
        try:
            print(f"Loading checkpoint from {model_path}")
            checkpoint = torch.load(model_path)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # New checkpoint format
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint.get('epoch', 0)
                    best_loss = checkpoint.get('best_loss', float('inf'))
                    training_stats = checkpoint.get('training_stats', training_stats)
                else:
                    # Old format or different structure
                    model.load_state_dict(checkpoint)
            else:
                # Direct state dict
                model.load_state_dict(checkpoint)
                
            print(f"Resuming from epoch {start_epoch} with best loss: {best_loss:.4f}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
            start_epoch = 0
            best_loss = float('inf')

    print(f"Starting training with vocabulary size: {len(train_dataset.chars)}")
    
    start_time = time.time()
    try:
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            total_loss = 0
            batch_count = 0
            
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = nn.functional.one_hot(inputs, num_classes=len(train_dataset.chars)).float().to(device)
                
                batch_size = inputs.size(0)
                hidden = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                         torch.zeros(num_layers, batch_size, hidden_size).to(device))
                
                outputs, hidden = model(inputs, hidden)
                loss = criterion(outputs.view(-1, len(train_dataset.chars)), targets.view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                if batch_idx % 10 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    elapsed = time.time() - epoch_start_time
                    print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s')
            
            avg_epoch_loss = total_loss / batch_count
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time
            
            # Update training stats
            training_stats['epochs'].append(epoch + 1)
            training_stats['losses'].append(avg_epoch_loss)
            training_stats['time_elapsed'] = total_time
            
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                training_stats['best_loss'] = best_loss
            
            print(f'Epoch {epoch+1} completed in {epoch_time:.2f}s')
            print(f'Average Loss: {avg_epoch_loss:.4f}, Best Loss: {best_loss:.4f}')
            
            # Save checkpoint after each epoch
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'best_loss': best_loss,
                'vocab': train_dataset.chars,
                'training_stats': training_stats,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save both latest and best checkpoints
            torch.save(checkpoint, f'{model_path}.latest')
            print(f'Latest checkpoint saved: {model_path}.latest')
            
            if avg_epoch_loss == best_loss:
                torch.save(checkpoint, f'{model_path}.best')
                print(f'Best checkpoint saved: {model_path}.best')
            
            # Save periodic checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save(checkpoint, f'{model_path}.epoch{epoch+1}')
                print(f'Periodic checkpoint saved: {model_path}.epoch{epoch+1}')

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        
    finally:
        final_time = time.time() - start_time
        print(f"\nTraining completed in {final_time:.2f} seconds")
        print(f"Best loss achieved: {best_loss:.4f}")
        
        # Save final checkpoint
        final_checkpoint = {
            'model_state_dict': model.state_dict(),
            'vocab': train_dataset.chars,
            'training_stats': training_stats,
            'final_loss': best_loss,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        torch.save(final_checkpoint, model_path)
        print(f'Final model saved to {model_path}')
        
        return model, train_dataset

def load_model(vocab_size, hidden_size, num_layers, model_path='mamba_helpsteer2.pth'):
    # Force CPU usage
    device = torch.device('cpu')
    model = MambaModel(vocab_size, hidden_size, num_layers)
    # Load the model with explicit CPU mapping
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    model.eval()
    return model

def generate_text(model, start_text, length, dataset, hidden_size, num_layers, temperature=0.7):
    model.eval()  # Set the model to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize hidden state
    hidden = (torch.zeros(num_layers, 1, hidden_size).to(device),
              torch.zeros(num_layers, 1, hidden_size).to(device))
    
    # Convert start_text to indices and handle unknown characters
    input_indices = []
    for ch in start_text:
        if ch in dataset.char_to_idx:
            input_indices.append(dataset.char_to_idx[ch])
        else:
            # Skip or replace unknown characters
            continue
    
    input_tensor = torch.tensor(input_indices).unsqueeze(0).to(device)
    generated_text = start_text

    with torch.no_grad():
        for _ in range(length):
            # One-hot encode the input
            input_one_hot = nn.functional.one_hot(input_tensor, num_classes=len(dataset.chars)).float()
            
            # Generate prediction
            output, hidden = model(input_one_hot, hidden)
            
            # Apply temperature and sample
            output = output[:, -1, :] / temperature
            probs = nn.functional.softmax(output, dim=-1)
            next_char_idx = torch.multinomial(probs, 1).item()
            
            # Convert to character and append
            next_char = dataset.idx_to_char[next_char_idx]
            generated_text += next_char
            
            # Update input tensor for next iteration
            input_tensor = torch.tensor([[next_char_idx]]).to(device)
            
            # Optional: Add early stopping condition
            if next_char in ['.', '!', '?'] and len(generated_text) > length/2:
                break

    return generated_text

# Add a new method for chat-style responses
def get_chat_response(model, prompt, dataset, hidden_size=128, num_layers=1, max_length=100):
    """
    Generate a chat-style response to a given prompt
    """
    # Prepare the prompt with a conversation marker
    chat_prompt = f"\nUser: {prompt}\nAssistant: "
    
    # Generate response
    response = generate_text(
        model, 
        chat_prompt,
        max_length,
        dataset,
        hidden_size,
        num_layers,
        temperature=0.7
    )
    
    # Extract just the assistant's response
    try:
        assistant_response = response.split("Assistant: ")[1].strip()
    except IndexError:
        assistant_response = response.strip()
    
    return assistant_response

# Example usage
if __name__ == '__main__':
    # Load the HelpSteer2 dataset
    dataset = load_helpsteer2_data()
    
    # Training parameters
    params = {
        'seq_length': 100,
        'hidden_size': 256,
        'num_layers': 2,
        'epochs': 50,
        'batch_size': 32,
        'model_path': 'mamba_helpsteer555.pth'
    }
    
    print("Starting training with parameters:", params)
    
    # Train the model
    model, train_dataset = train_model(dataset, **params)
    
    # Test the model
    test_prompt = "User: tell me about death \nAssistant:"
    generated = generate_text(
        model,
        test_prompt,
        length=200,
        dataset=train_dataset,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers']
    )
    print("\nTest Generation:")
    print(generated)

