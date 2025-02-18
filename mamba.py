import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# Define a simple dataset class
class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data = [self.char_to_idx[ch] for ch in text]

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

# Training setup
def train_model(data, seq_length=50, hidden_size=128, num_layers=1, epochs=5, model_path='mamba_model.pth'):
    dataset = TextDataset(data, seq_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    
    # Check if GPU is available and use it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MambaModel(len(dataset.chars), hidden_size, num_layers).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    try:
        for epoch in range(epochs):
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = nn.functional.one_hot(inputs, num_classes=len(dataset.chars)).float().to(device)
                
                # Initialize hidden state with the correct batch size
                batch_size = inputs.size(0)
                hidden = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                          torch.zeros(num_layers, batch_size, hidden_size).to(device))
                
                outputs, hidden = model(inputs, hidden)
                loss = criterion(outputs.view(-1, len(dataset.chars)), targets.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if batch_idx % 10 == 0:  # Log every 10 batches
                    print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}')
            
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    except KeyboardInterrupt:
        print("Training interrupted. Saving model weights...")
    
    finally:
        # Save the model weights
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

def load_model(vocab_size, hidden_size, num_layers, model_path='mamba_model.pth'):
    model = MambaModel(vocab_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(model_path))
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
    with open('data.txt', 'r') as file:
        text_data = file.read()

    # Load the model for inference
    loaded_model = load_model(len(set(text_data)), 128, 1)  # Ensure vocab size is correct

    # Create a dataset instance for character mapping
    dataset = TextDataset(text_data, 50)

    # Generate text
    start_text = "Once upon a time"
    generated_length = 100  # Specify the length of text to generate
    generated = generate_text(loaded_model, start_text, generated_length, dataset, hidden_size=128, num_layers=1)
    print(generated)