from fastapi import FastAPI
from pydantic import BaseModel
import torch

# Assuming mamba.py is in the same directory
from mamba import load_model, generate_text, TextDataset

app = FastAPI()

# Load the model
text_data = open('data.txt', 'r').read()
dataset = TextDataset(text_data, 50)
model = load_model(len(dataset.chars), 128, 1)

class TextGenerationRequest(BaseModel):
    start_text: str
    length: int

@app.post("/generate")
def generate(request: TextGenerationRequest):
    start_text = request.start_text
    generated_text = generate_text(
        model, 
        start_text,
        request.length, 
        dataset, 
        hidden_size=128, 
        num_layers=1
    )
    return {"generated_text": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 