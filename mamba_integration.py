import torch
from mamba import MambaModel, TextDataset, load_model, get_chat_response

class MambaChat:
    def __init__(self, model_path='mamba_model.pth', data_path='data.txt'):
        # Load training data for character mappings
        with open(data_path, 'r', encoding='utf-8') as f:
            self.text_data = f.read()
        
        # Create dataset for character mappings
        self.dataset = TextDataset(self.text_data, seq_length=50)
        
        # Initialize model
        self.model = load_model(
            vocab_size=len(self.dataset.chars),
            hidden_size=128,
            num_layers=1,
            model_path=model_path
        )
        
        self.model.eval()

    def get_response(self, prompt):
        try:
            response = get_chat_response(
                self.model,
                prompt,
                self.dataset,
                hidden_size=128,
                num_layers=1
            )
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    @staticmethod
    def is_model_ready():
        return True  # You can add more sophisticated checks here 