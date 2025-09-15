import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse

from utils import download_shakespeare_data
from ShakespeareDataset import ShakespeareDataset
from RNNModel import RNNModel
from LSTMModel import LSTMModel


torch.manual_seed(42)
np.random.seed(42)
 

def save_model(model, optimizer, epoch, train_losses, filepath):
    """Save model, optimizer state, and training info"""
    checkpoint = {
        'model_config': model.get_config(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_losses': train_losses
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Reconstruct model based on saved config
    config = checkpoint['model_config']
    if config['model_type'] == 'RNNModel':
        model = RNNModel(
            vocab_size=config['vocab_size'],
            embed_size=config['embed_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        )
    elif config['model_type'] == 'LSTMModel':
        model = LSTMModel(
            vocab_size=config['vocab_size'],
            embed_size=config['embed_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Return model and training info
    return model, checkpoint

def train_model(model, dataloader, criterion, optimizer, device, epochs=5, save_path=None):
    """Train the model"""
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(data)
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}')
        
        # Save checkpoint after each epoch
        if save_path:
            save_model(model, optimizer, epoch + 1, train_losses, save_path)
    
    return train_losses

def generate_text(model, dataset, seed_text, length=200, temperature=0.8):
    """Generate text using the trained model"""
    model.eval()
    device = next(model.parameters()).device
    
    # Convert seed text to indices
    if len(seed_text) == 0:
        seed_text = " "
    
    current_seq = [dataset.char_to_idx.get(ch, 0) for ch in seed_text[-100:]]  # Use last 100 chars
    generated = seed_text
    
    with torch.no_grad():
        for _ in range(length):
            # Prepare input
            x = torch.tensor([current_seq], dtype=torch.long).to(device)
            
            # Get prediction
            output, _ = model(x)
            
            # Apply temperature and sample
            logits = output[0, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            next_char_idx = torch.multinomial(probs, 1).item()
            
            # Convert back to character and add to generated text
            next_char = dataset.idx_to_char[next_char_idx]
            generated += next_char
            
            # Update current sequence
            current_seq = current_seq[1:] + [next_char_idx]
    
    return generated

def calculate_perplexity(model, dataloader, criterion, device):
    """Calculate perplexity on validation set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    batch_count = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            output, _ = model(data)
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            
            # Add total loss for this batch (loss is already averaged over tokens)
            batch_tokens = target.numel()  # Total number of tokens in this batch
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            batch_count += 1
            
            # Debug: Print first few batch losses to check for anomalies
            if batch_count <= 3:
                print(f"  Batch {batch_count}: loss={loss.item():.4f}, tokens={batch_tokens}")
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    print(f"  Total batches: {batch_count}, Total tokens: {total_tokens}")
    print(f"  Average loss: {avg_loss:.4f}")
    
    return perplexity, avg_loss

def prepare_data():
    """Download and prepare the Shakespeare dataset"""
    print("Preparing Shakespeare dataset...")
    
    # Download text
    text = download_shakespeare_data()
    
    # Split data (80% train, 20% validation)
    split_idx = int(0.8 * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Create datasets
    seq_length = 100
    train_dataset = ShakespeareDataset(train_text, seq_length)
    val_dataset = ShakespeareDataset(val_text, seq_length)
    
    # Save vocabulary
    os.makedirs('models', exist_ok=True)
    train_dataset.save_vocab('models/vocab.json')
    
    print(f"Vocabulary size: {train_dataset.vocab_size}")
    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    print("Vocabulary saved to models/vocab.json")
    
    return train_dataset, val_dataset

def train_models():
    """Train both RNN and LSTM models"""
    print("=" * 60)
    print("Training RNN and LSTM Models on Shakespeare Dataset")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    train_dataset, val_dataset = prepare_data()
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model parameters
    vocab_size = train_dataset.vocab_size
    embed_size = 128
    hidden_size = 256
    num_layers = 2
    dropout = 0.3
    learning_rate = 0.001
    epochs = 5
    
    print(f"\nModel Configuration:")
    print(f"   Embedding size: {embed_size}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Number of layers: {num_layers}")
    print(f"   Dropout: {dropout}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {epochs}")
    
    # Initialize models
    rnn_model = RNNModel(vocab_size, embed_size, hidden_size, num_layers, dropout).to(device)
    lstm_model = LSTMModel(vocab_size, embed_size, hidden_size, num_layers, dropout).to(device)
    
    print(f"\nRNN Parameters: {sum(p.numel() for p in rnn_model.parameters()):,}")
    print(f"LSTM Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    # Loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
    
    # Train RNN
    print("\n" + "="*40)
    print("Training RNN")
    print("="*40)
    rnn_start_time = time.time()
    rnn_losses = train_model(rnn_model, train_loader, criterion, rnn_optimizer, device, epochs, 'models/rnn_model.pth')
    rnn_train_time = time.time() - rnn_start_time
    
    # Train LSTM
    print("\n" + "="*40)
    print("Training LSTM")
    print("="*40)
    lstm_start_time = time.time()
    lstm_losses = train_model(lstm_model, train_loader, criterion, lstm_optimizer, device, epochs, 'models/lstm_model.pth')
    lstm_train_time = time.time() - lstm_start_time
    
    print(f"\nTraining completed!")
    print(f"RNN training time: {rnn_train_time:.2f} seconds")
    print(f"LSTM training time: {lstm_train_time:.2f} seconds")
    print(f"Models saved in 'models/' directory")

def evaluate_models():
    """Evaluate pre-trained models"""
    print("=" * 60)
    print("Evaluating Pre-trained RNN and LSTM Models")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if models exist
    if not os.path.exists('models/rnn_model.pth') or not os.path.exists('models/lstm_model.pth'):
        print("Error: Pre-trained models not found!")
        print("Please run with --mode train first to train the models.")
        return
    
    # Load vocabulary
    if not os.path.exists('models/vocab.json'):
        print("Error: Vocabulary file not found!")
        print("Please run with --mode train first to create the vocabulary.")
        return
    
    # Load models
    print("\nLoading pre-trained models...")
    rnn_model, rnn_checkpoint = load_model('models/rnn_model.pth', device)
    lstm_model, lstm_checkpoint = load_model('models/lstm_model.pth', device)
    
    print(f"RNN trained for {rnn_checkpoint['epoch']} epochs")
    print(f"LSTM trained for {lstm_checkpoint['epoch']} epochs")
    
    # Prepare validation data using the SAME preprocessing as training
    print("\nPreparing validation data...")
    text = download_shakespeare_data()
    split_idx = int(0.8 * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Create dataset with training text to get the exact same vocabulary
    full_dataset = ShakespeareDataset(train_text, 100)
    
    # Now create validation dataset using the SAME character mappings
    val_dataset = ShakespeareDataset.__new__(ShakespeareDataset)
    val_dataset.seq_length = 100
    val_dataset.char_to_idx = full_dataset.char_to_idx
    val_dataset.idx_to_char = full_dataset.idx_to_char
    val_dataset.vocab_size = full_dataset.vocab_size
    val_dataset.data = [full_dataset.char_to_idx[ch] for ch in val_text]
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Loaded vocabulary with {val_dataset.vocab_size} characters")
    print(f"Validation sequences: {len(val_dataset)}")
    
    # Calculate perplexity
    print("\nCalculating perplexity...")
    criterion = nn.CrossEntropyLoss()
    rnn_perplexity, rnn_val_loss = calculate_perplexity(rnn_model, val_loader, criterion, device)
    lstm_perplexity, lstm_val_loss = calculate_perplexity(lstm_model, val_loader, criterion, device)
    
    print(f"\nValidation Results:")
    print(f"RNN  - Loss: {rnn_val_loss:.4f}, Perplexity: {rnn_perplexity:.2f}")
    print(f"LSTM - Loss: {lstm_val_loss:.4f}, Perplexity: {lstm_perplexity:.2f}")
    
    # Generate sample text
    print("\n" + "="*60)
    print("Text Generation Examples")
    print("="*60)
    
    seed_texts = ["ROMEO:", "To be or not to be", "JULIET:"]
    
    for seed in seed_texts:
        print(f"\nSeed: '{seed}'")
        print("-" * 50)
        
        print("RNN Output:")
        rnn_output = generate_text(rnn_model, full_dataset, seed, length=150, temperature=0.7)
        print(rnn_output)
        
        print(f"\nLSTM Output:")
        lstm_output = generate_text(lstm_model, full_dataset, seed, length=150, temperature=0.7)
        print(lstm_output)
        print()
    
    # Plot training curves
    print("Plotting training curves...")
    plt.figure(figsize=(15, 5))
    
    # Training loss curves
    plt.subplot(1, 3, 1)
    rnn_losses = rnn_checkpoint['train_losses']
    lstm_losses = lstm_checkpoint['train_losses']
    epochs_range = range(1, len(rnn_losses) + 1)
    plt.plot(epochs_range, rnn_losses, 'b-', label='RNN', linewidth=2)
    plt.plot(epochs_range, lstm_losses, 'r-', label='LSTM', linewidth=2)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Perplexity comparison
    plt.subplot(1, 3, 2)
    models = ['RNN', 'LSTM']
    perplexities = [rnn_perplexity, lstm_perplexity]
    colors = ['blue', 'red']
    bars = plt.bar(models, perplexities, color=colors, alpha=0.7)
    plt.title('Validation Perplexity Comparison')
    plt.ylabel('Perplexity (lower is better)')
    plt.grid(True, axis='y')
    
    # Add value labels on bars
    for bar, perp in zip(bars, perplexities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(perplexities)*0.01,
                f'{perp:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Parameter count comparison
    plt.subplot(1, 3, 3)
    rnn_params = sum(p.numel() for p in rnn_model.parameters())
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    param_counts = [rnn_params, lstm_params]
    bars = plt.bar(models, param_counts, color=colors, alpha=0.7)
    plt.title('Parameter Count Comparison')
    plt.ylabel('Number of Parameters')
    plt.grid(True, axis='y')
    
    # Add value labels on bars
    for bar, params in zip(bars, param_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(param_counts)*0.01,
                f'{params:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"{'Metric':<20} {'RNN':<15} {'LSTM':<15} {'Winner'}")
    print("-" * 60)
    print(f"{'Final Train Loss':<20} {rnn_losses[-1]:<15.4f} {lstm_losses[-1]:<15.4f} {'LSTM' if lstm_losses[-1] < rnn_losses[-1] else 'RNN'}")
    print(f"{'Validation Loss':<20} {rnn_val_loss:<15.4f} {lstm_val_loss:<15.4f} {'LSTM' if lstm_val_loss < rnn_val_loss else 'RNN'}")
    print(f"{'Perplexity':<20} {rnn_perplexity:<15.2f} {lstm_perplexity:<15.2f} {'LSTM' if lstm_perplexity < rnn_perplexity else 'RNN'}")
    print(f"{'Parameters':<20} {rnn_params:<15,} {lstm_params:<15,} {'RNN' if rnn_params < lstm_params else 'LSTM'}")
    
    print("\nKey Observations:")
    print("• LSTMs typically achieve better perplexity due to their ability to model long-term dependencies")
    print("• RNNs are simpler and faster to train but may struggle with vanishing gradients")
    print("• LSTMs have more parameters due to their gating mechanisms")
    print("• Text quality is subjective but LSTMs often produce more coherent sequences")

def interactive_generation():
    """Interactive text generation with pre-trained models"""
    print("=" * 60)
    print("Interactive Text Generation")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if models exist
    if not os.path.exists('models/rnn_model.pth') or not os.path.exists('models/lstm_model.pth'):
        print("Error: Pre-trained models not found!")
        print("Please run with --mode train first to train the models.")
        return
    
    # Load vocabulary and models
    dataset = ShakespeareDataset.load_vocab('models/vocab.json')
    rnn_model, _ = load_model('models/rnn_model.pth', device)
    lstm_model, _ = load_model('models/lstm_model.pth', device)
    
    print("Models loaded successfully!")
    print("Enter text prompts to generate Shakespeare-style text.")
    print("Commands: 'quit' to exit, 'rnn' or 'lstm' to switch models, 'both' for comparison")
    print("=" * 60)
    
    current_model = "both"
    
    while True:
        try:
            user_input = input(f"\n[{current_model.upper()}] Enter seed text (or command): ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'rnn':
                current_model = "rnn"
                print("Switched to RNN model")
                continue
            elif user_input.lower() == 'lstm':
                current_model = "lstm"
                print("Switched to LSTM model")
                continue
            elif user_input.lower() == 'both':
                current_model = "both"
                print("Switched to comparison mode")
                continue
            
            if not user_input:
                user_input = " "
            
            print("-" * 50)
            
            if current_model in ["rnn", "both"]:
                print("RNN Output:")
                rnn_output = generate_text(rnn_model, dataset, user_input, length=200, temperature=0.8)
                print(rnn_output)
                print()
            
            if current_model in ["lstm", "both"]:
                print("LSTM Output:")
                lstm_output = generate_text(lstm_model, dataset, user_input, length=200, temperature=0.8)
                print(lstm_output)
        
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='RNN vs LSTM Shakespeare Text Generation')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'interactive'], default='evaluate',
                        help='Mode: train models, evaluate pre-trained models, or interactive generation')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_models()
    elif args.mode == 'evaluate':
        evaluate_models()
    elif args.mode == 'interactive':
        interactive_generation()

if __name__ == "__main__":
    # If no command line args, default to evaluation mode
    import sys
    if len(sys.argv) == 1:
        evaluate_models()
    else:
        main()