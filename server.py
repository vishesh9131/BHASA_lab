from flask import Flask, request, jsonify
from flask_cors import CORS
from mamba import load_model, generate_text, TextDataset

app = Flask(__name__)
CORS(app)

# Load model and dataset once at startup
with open('data.txt', 'r') as file:
    text_data = file.read()

dataset = TextDataset(text_data, 50)
model = load_model(len(dataset.chars), 128, 1, 'mamba_model.pth')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        text = data.get('start_text', '')
        
        response = generate_text(
            model=model,
            start_text=text,
            length=100,
            dataset=dataset,
            hidden_size=128,
            num_layers=1
        )
        return jsonify({'generated_text': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001) 