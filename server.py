from flask import Flask, request, jsonify
from flask_cors import CORS
from mamba import load_model, generate_text, TextDataset
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables for model and dataset
model = None
dataset = None

# Initialize model and dataset
try:
    print("Loading data and model...")
    with open('data.txt', 'r', encoding='utf-8') as file:
        text_data = file.read()
    print(f"Data loaded, length: {len(text_data)}")

    dataset = TextDataset(text_data, 50)
    print(f"Dataset created, vocab size: {len(dataset.chars)}")

    model = load_model(
        vocab_size=len(dataset.chars),
        hidden_size=128,
        num_layers=1,
        model_path='mamba_model.pth'
    )
    print("Model loaded successfully!")

except Exception as e:
    print(f"Error during initialization: {str(e)}")
    print(traceback.format_exc())

@app.route('/generate', methods=['POST', 'OPTIONS'])
def generate():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
        
    try:
        print("Received request data:", request.json)
        data = request.json
        
        # Try both possible keys
        message = data.get('message') or data.get('start_text', '')
        print(f"Processing message: {message}")
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
            
        if model is None or dataset is None:
            return jsonify({'error': 'Model or dataset not initialized'}), 500
            
        response = generate_text(
            model=model,
            start_text=message,
            length=100,
            dataset=dataset,
            hidden_size=128,
            num_layers=1
        )
        print(f"Generated response: {response}")
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'model_loaded': model is not None,
        'dataset_loaded': dataset is not None,
        'vocab_size': len(dataset.chars) if dataset else 0
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(port=5001, debug=True)