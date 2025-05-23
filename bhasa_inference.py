import torch
import json
import argparse
from pathlib import Path
from bhasa_mamba import BhasaMamba, MambaConfig, SimpleTokenizer

def load_bhasa_model(model_path, device='cpu'):
    """Load the trained BHASA Mamba model"""
    print(f"🔄 Loading BHASA model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct config
    config_dict = checkpoint['config']
    config = MambaConfig(**config_dict)
    
    # Create and load model
    model = BhasaMamba(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Reconstruct tokenizer
    tokenizer = SimpleTokenizer()
    
    # Handle different checkpoint formats
    if 'tokenizer' in checkpoint:
        tokenizer_data = checkpoint['tokenizer']
    elif 'tokenizer_vocab' in checkpoint:
        tokenizer_data = checkpoint['tokenizer_vocab']
    else:
        # Try to load from separate tokenizer file
        try:
            with open('bhasa_tokenizer.json', 'r') as f:
                tokenizer_data = json.load(f)
        except:
            raise ValueError("No tokenizer data found in checkpoint or separate file")
    
    tokenizer.char_to_id = tokenizer_data['char_to_id']
    tokenizer.id_to_char = {int(k): v for k, v in tokenizer_data['id_to_char'].items()}
    tokenizer.vocab_size = tokenizer_data['vocab_size']
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🔤 Vocabulary size: {tokenizer.vocab_size}")
    
    return model, tokenizer, config

def generate_text(model, tokenizer, prompt, max_length=200, temperature=0.8, top_k=50, device='cpu'):
    """Generate text using the BHASA model"""
    model.eval()
    
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    print(f"🎯 Generating text...")
    print(f"📝 Prompt: '{prompt}'")
    print(f"🌡️ Temperature: {temperature}")
    print(f"🔝 Top-k: {top_k}")
    print(f"📏 Max length: {max_length}")
    print("-" * 50)
    
    with torch.no_grad():
        generated = model.generate(
            input_ids, 
            max_length=max_length, 
            temperature=temperature, 
            top_k=top_k
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(generated[0].cpu().tolist())
    
    return generated_text

def interactive_chat():
    """Interactive chat with BHASA"""
    # Check for available models
    model_paths = [
        "bhasa_mamba_final.pt",
        "checkpoints/bhasa_latest.pt"
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if model_path is None:
        print("❌ No trained model found!")
        print("🏃 Please run `python bhasa_mamba.py` to train the model first.")
        return
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🍎 Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("🔥 Using CUDA")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    # Load model
    model, tokenizer, config = load_bhasa_model(model_path, device)
    
    print("\n" + "="*60)
    print("🤖 BHASA (Bayesian Hyperdimensional Adaptive Sequential Architecture)")
    print("🚀 Ready for conversation!")
    print("💡 Type 'quit' to exit, 'help' for options")
    print("="*60)
    
    # Default settings
    temperature = 0.8
    top_k = 50
    max_length = 150
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\n📚 BHASA Help:")
                print("• Type any text to generate a response")
                print("• 'settings' - Adjust generation parameters")
                print("• 'info' - Show model information")
                print("• 'quit' - Exit the chat")
                continue
            
            elif user_input.lower() == 'settings':
                print(f"\n⚙️ Current settings:")
                print(f"🌡️ Temperature: {temperature}")
                print(f"🔝 Top-k: {top_k}")
                print(f"📏 Max length: {max_length}")
                
                try:
                    new_temp = input(f"New temperature (current: {temperature}): ").strip()
                    if new_temp:
                        temperature = float(new_temp)
                    
                    new_top_k = input(f"New top-k (current: {top_k}): ").strip()
                    if new_top_k:
                        top_k = int(new_top_k)
                    
                    new_max_len = input(f"New max length (current: {max_length}): ").strip()
                    if new_max_len:
                        max_length = int(new_max_len)
                    
                    print("✅ Settings updated!")
                except ValueError:
                    print("❌ Invalid input, settings unchanged.")
                continue
            
            elif user_input.lower() == 'info':
                print(f"\n🧠 Model Information:")
                print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
                print(f"🔤 Vocabulary size: {tokenizer.vocab_size}")
                print(f"🏗️ Architecture: {config.n_layers} layers, {config.d_model} dimensions")
                print(f"🎯 State size: {config.d_state}")
                print(f"💾 Model path: {model_path}")
                continue
            
            elif not user_input:
                continue
            
            # Generate response
            print("🤖 BHASA:", end=" ")
            generated_text = generate_text(
                model, tokenizer, user_input,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                device=device
            )
            
            # Extract just the generated part (after prompt)
            response = generated_text[len(user_input):].strip()
            if not response:
                response = "I need more context to respond meaningfully."
            
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("🔄 Continuing...")

def main():
    parser = argparse.ArgumentParser(description="BHASA Mamba Inference")
    parser.add_argument("--prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--model", type=str, default="bhasa_mamba_final.pt", help="Model checkpoint path")
    parser.add_argument("--max-length", type=int, default=200, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat mode")
    
    args = parser.parse_args()
    
    if args.interactive or not args.prompt:
        interactive_chat()
        return
    
    # Single generation mode
    if not Path(args.model).exists():
        print(f"❌ Model file {args.model} not found!")
        return
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Load model and generate
    model, tokenizer, config = load_bhasa_model(args.model, device)
    generated_text = generate_text(
        model, tokenizer, args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )
    
    print(f"\n🎨 Generated text:\n{generated_text}")

if __name__ == "__main__":
    main() 