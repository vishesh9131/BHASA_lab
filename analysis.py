import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from tqdm import tqdm
import pandas as pd
from chatty.mamba import generate_text  # Import the generate_text function

# Download required NLTK data at startup
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')

def calculate_bleu_score(reference, candidate):
    """Calculate BLEU score between reference and candidate texts"""
    try:
        # Simple word tokenization by splitting on spaces
        reference_tokens = reference.lower().split()
        candidate_tokens = candidate.lower().split()
        
        # Calculate BLEU score with smoothing
        smoothing = SmoothingFunction().method1
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return 0.0

def calculate_coherence_score(text):
    """Calculate a simple coherence score based on text structure"""
    try:
        # Simple word-based tokenization
        words = text.split()
        if not words:
            return 0.0
            
        # Simple coherence metrics
        avg_word_length = np.mean([len(word) for word in words])
        has_punctuation = any(p in text for p in '.!?')
        unique_words = len(set(words)) / len(words) if words else 0
        
        # Combine metrics into a score between 0 and 1
        coherence = (
            (min(avg_word_length, 10) / 10) * 0.3 +  # Reward reasonable word lengths
            (has_punctuation * 0.3) +                 # Reward proper punctuation
            (unique_words * 0.4)                      # Reward vocabulary diversity
        )
        
        return coherence
    except Exception as e:
        print(f"Error calculating coherence: {e}")
        return 0.0

def evaluate_model(model, test_dataset, num_samples=1000):
    """Evaluate model performance on test dataset"""
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    metrics = {
        'bleu_scores': [],
        'response_lengths': [],
        'generation_times': [],
        'coherence_scores': []
    }
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    with torch.no_grad():
        for i, (input_seq, target_seq) in enumerate(tqdm(test_loader, desc="Evaluating")):
            if i >= num_samples:
                break
                
            try:
                # Generate response
                start_time = time.time()
                
                # Convert input sequence to text
                input_text = ''.join([test_dataset.idx_to_char[idx.item()] for idx in input_seq[0]])
                
                # Generate response
                generated_text = generate_text(
                    model,
                    input_text,
                    max_length=200,
                    dataset=test_dataset,
                    hidden_size=256,
                    num_layers=2,
                    temperature=0.7
                )
                
                generation_time = time.time() - start_time
                
                # Get reference text
                reference_text = ''.join([test_dataset.idx_to_char[idx.item()] for idx in target_seq[0]])
                
                # Calculate metrics
                bleu = calculate_bleu_score(reference_text, generated_text)
                coherence = calculate_coherence_score(generated_text)
                
                # Store metrics
                metrics['bleu_scores'].append(bleu)
                metrics['response_lengths'].append(len(generated_text))
                metrics['generation_times'].append(generation_time)
                metrics['coherence_scores'].append(coherence)
                
            except Exception as e:
                print(f"Error evaluating sample {i}: {e}")
                continue
    
    return metrics

def plot_metrics(metrics):
    """Create visualizations of model performance metrics"""
    # Use default style instead of seaborn
    plt.style.use('default')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # BLEU Score Distribution
    axes[0,0].hist(metrics['bleu_scores'], bins=20)
    axes[0,0].set_title('Distribution of BLEU Scores')
    axes[0,0].set_xlabel('BLEU Score')
    
    # Response Length Distribution
    axes[0,1].hist(metrics['response_lengths'], bins=20)
    axes[0,1].set_title('Distribution of Response Lengths')
    axes[0,1].set_xlabel('Length (characters)')
    
    # Generation Time Distribution
    axes[1,0].hist(metrics['generation_times'], bins=20)
    axes[1,0].set_title('Distribution of Generation Times')
    axes[1,0].set_xlabel('Time (seconds)')
    
    # Coherence Score Distribution
    axes[1,1].hist(metrics['coherence_scores'], bins=20)
    axes[1,1].set_title('Distribution of Coherence Scores')
    axes[1,1].set_xlabel('Coherence Score')
    
    plt.tight_layout()
    plt.savefig('model_performance_metrics.png')
    plt.close()
    
    # Create summary statistics
    summary = pd.DataFrame({
        'Metric': ['BLEU Score', 'Response Length', 'Generation Time', 'Coherence'],
        'Mean': [
            np.mean(metrics['bleu_scores']),
            np.mean(metrics['response_lengths']),
            np.mean(metrics['generation_times']),
            np.mean(metrics['coherence_scores'])
        ],
        'Median': [
            np.median(metrics['bleu_scores']),
            np.median(metrics['response_lengths']),
            np.median(metrics['generation_times']),
            np.median(metrics['coherence_scores'])
        ],
        'Std Dev': [
            np.std(metrics['bleu_scores']),
            np.std(metrics['response_lengths']),
            np.std(metrics['generation_times']),
            np.std(metrics['coherence_scores'])
        ]
    })
    
    return summary

if __name__ == "__main__":
    import time
    from chatty.mamba import load_model, TextDataset
    
    # Load your model and test dataset
    model, saved_vocab, vocab_size = load_model(256, 2, 'mamba_helpsteer2.pth')
    
    # Load test data
    with open('data.txt', 'r') as file:
        test_data = file.read()
    
    test_dataset = TextDataset(test_data, 50, vocab_size)
    test_dataset.chars = saved_vocab
    test_dataset.char_to_idx = {ch: i for i, ch in enumerate(saved_vocab)}
    test_dataset.idx_to_char = {i: ch for i, ch in enumerate(saved_vocab)}
    
    # Evaluate model
    print("Starting model evaluation...")
    metrics = evaluate_model(model, test_dataset)
    
    # Plot results
    summary = plot_metrics(metrics)
    print("\nPerformance Summary:")
    print(summary.to_string(index=False))
