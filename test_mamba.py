from mamba import load_model, TextDataset, generate_text

# Load data and model
with open('data.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()

# Create dataset
dataset = TextDataset(text_data, 50)
print("Dataset created with vocab size:", len(dataset.chars))

# Load model
model = load_model(
    vocab_size=len(dataset.chars),
    hidden_size=128,
    num_layers=1,
    model_path='mamba_model.pth'
)

# Test generation
test_text = "Hello, how are you?"
generated = generate_text(
    model=model,
    start_text=test_text,
    length=100,
    dataset=dataset,
    hidden_size=128,
    num_layers=1
)

print("Input:", test_text)
print("Generated:", generated) 