import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
from tqdm import tqdm
import numpy as np
from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments

def main():
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-370m-hf")
    model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-370m-hf")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("Abirate/english_quotes", split="train")
    
    # Inspect the dataset to find the correct text field
    print("Sample data:", dataset[0])  # Print a sample to check field names
    
    # Preprocess the dataset
    def preprocess_function(examples):
        return tokenizer(examples["quote"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    # Define training arguments with reduced batch size and no fp16
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduced batch size
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=2e-3,
        gradient_accumulation_steps=4  # Use gradient accumulation
    )
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=8,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )
    
    # Initialize the trainer
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=lora_config,
        train_dataset=tokenized_dataset,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()

def evaluate_perplexity_wikitext(model, tokenizer):
    """Evaluate perplexity on WikiText-2 test set (small sample)"""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:20]")
    
    # Concatenate all texts
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    
    # Move to the same device as model
    input_ids = encodings.input_ids.to(model.device)
    
    # Calculate perplexity in chunks to avoid OOM
    max_length = 128  # Smaller for macOS
    stride = 64
    seq_len = input_ids.size(1)
    
    nlls = []
    for i in tqdm(range(0, seq_len, stride)):
        begin_loc = max(0, i)
        end_loc = min(i + max_length, seq_len)
        trg_len = end_loc - begin_loc
        
        if trg_len <= 0:
            continue
            
        with torch.no_grad():
            inputs = input_ids[:, begin_loc:end_loc]
            targets = inputs.clone()
            
            outputs = model(inputs, labels=targets)
            neg_log_likelihood = outputs.loss * trg_len
            
        nlls.append(neg_log_likelihood)
    
    if not nlls:
        return float('inf')
        
    ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
    return ppl.item()

def evaluate_perplexity_ptb(model, tokenizer):
    """Evaluate perplexity on Penn Treebank test set (small sample)"""
    dataset = load_dataset("ptb_text_only", "penn_treebank", split="test[:20]")
    
    # Concatenate all texts
    text = " ".join(dataset["sentence"])
    encodings = tokenizer(text, return_tensors="pt")
    
    # Move to the same device as model
    input_ids = encodings.input_ids.to(model.device)
    
    # Calculate perplexity in chunks to avoid OOM
    max_length = 128
    stride = 64
    seq_len = input_ids.size(1)
    
    nlls = []
    for i in tqdm(range(0, seq_len, stride)):
        begin_loc = max(0, i)
        end_loc = min(i + max_length, seq_len)
        trg_len = end_loc - begin_loc
        
        if trg_len <= 0:
            continue
            
        with torch.no_grad():
            inputs = input_ids[:, begin_loc:end_loc]
            targets = inputs.clone()
            
            outputs = model(inputs, labels=targets)
            neg_log_likelihood = outputs.loss * trg_len
            
        nlls.append(neg_log_likelihood)
    
    if not nlls:
        return float('inf')
        
    ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
    return ppl.item()

def evaluate_completion(model, tokenizer):
    """Evaluate text completion capability"""
    prompts = [
        "The capital of France is",
        "Machine learning is a subset of artificial intelligence that",
        "The Eiffel Tower was built in",
        "The theory of relativity was developed by",
        "Python is a programming language that"
    ]
    
    correct_completions = [
        "Paris",
        "uses statistical methods",
        "1889",
        "Einstein",
        "is interpreted"
    ]
    
    scores = []
    for prompt, correct in zip(prompts, correct_completions):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated_text[len(prompt):].strip()
        
        # Simple check if the correct completion is in the generated text
        score = 1.0 if correct.lower() in completion.lower() else 0.0
        scores.append(score)
        
        print(f"Prompt: {prompt}")
        print(f"Completion: {completion}")
        print(f"Score: {score}\n")
    
    return sum(scores) / len(scores)

def evaluate_qa(model, tokenizer):
    """Evaluate question answering capability"""
    qa_pairs = [
        {"question": "What is the capital of Japan?", "answer": "Tokyo"},
        {"question": "Who wrote 'Romeo and Juliet'?", "answer": "Shakespeare"},
        {"question": "What is the chemical symbol for gold?", "answer": "Au"},
        {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
        {"question": "What year did World War II end?", "answer": "1945"}
    ]
    
    scores = []
    for pair in qa_pairs:
        prompt = f"Question: {pair['question']}\nAnswer:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text[len(prompt):].strip()
        
        # Simple check if the correct answer is in the generated text
        score = 1.0 if pair['answer'].lower() in answer.lower() else 0.0
        scores.append(score)
        
        print(f"Question: {pair['question']}")
        print(f"Generated: {answer}")
        print(f"Expected: {pair['answer']}")
        print(f"Score: {score}\n")
    
    return sum(scores) / len(scores)

def evaluate_sentiment(model, tokenizer):
    """Evaluate sentiment analysis capability"""
    texts = [
        "I absolutely loved this movie, it was fantastic!",
        "The service at this restaurant was terrible and the food was cold.",
        "The product works as expected, nothing special but gets the job done.",
        "I'm extremely disappointed with the quality of this item.",
        "This book changed my life, I couldn't put it down!"
    ]
    
    sentiments = ["positive", "negative", "neutral", "negative", "positive"]
    
    scores = []
    for text, expected in zip(texts, sentiments):
        prompt = f"Text: {text}\nSentiment (positive, negative, or neutral):"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=5,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted = generated_text[len(prompt):].strip().lower()
        
        # Check if the predicted sentiment matches the expected one
        score = 1.0 if expected in predicted else 0.0
        scores.append(score)
        
        print(f"Text: {text}")
        print(f"Predicted: {predicted}")
        print(f"Expected: {expected}")
        print(f"Score: {score}\n")
    
    return sum(scores) / len(scores)

if __name__ == "__main__":
    main()