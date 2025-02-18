import os
import torch
from chatty.mamba import load_model, TextDataset, generate_text, train_model

# Update model parameters to match the checkpoint
hidden_size = 256  # Updated to match the trained model's hidden size
num_layers = 2     # Updated to match the number of layers used in training
model_path = 'mamba_helpsteer2.pth'

def load_helpsteer2_data():
    with open('data.txt', 'r', encoding='utf-8') as file:
        data = file.read()
    return data

# Check if the model file exists
if os.path.exists(model_path):
    print("Loading existing model...")
    loaded_model, saved_vocab = load_model(hidden_size, num_layers, model_path)
else:
    print("Model not found. Training a new model...")
    dataset = load_helpsteer2_data()
    loaded_model, train_dataset = train_model(dataset, hidden_size=hidden_size, num_layers=num_layers, model_path=model_path)
    saved_vocab = train_dataset.chars

# Ensure vocab_size is an integer
vocab_size = len(saved_vocab) if saved_vocab else 0
if vocab_size == 0:
    raise ValueError("No saved vocabulary found in the model checkpoint.")

# Create dataset with the loaded vocabulary
dataset = TextDataset(''.join(saved_vocab), seq_length=50)
dataset.chars = saved_vocab
dataset.char_to_idx = {ch: i for i, ch in enumerate(saved_vocab)}
dataset.idx_to_char = {i: ch for i, ch in enumerate(saved_vocab)}

print("Model loaded successfully!")
print(f"Vocabulary size: {vocab_size}")

# Generate text based on user input
user_prompt = input("Enter your prompt: ")
length = 100  # Length of the generated text

generated = generate_text(
    model=loaded_model,
    start_text=user_prompt,
    length=length,
    dataset=dataset,
    hidden_size=hidden_size,
    num_layers=num_layers
)

print("Generated text:")
print(generated)