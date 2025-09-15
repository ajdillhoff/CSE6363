import torch
from torch.utils.data import Dataset
import json


class ShakespeareDataset(Dataset):
    """Dataset class for Shakespeare text"""
    
    def __init__(self, text, seq_length=100):
        self.seq_length = seq_length
        
        # Create character mappings
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        
        # Convert text to indices
        self.data = [self.char_to_idx[ch] for ch in text]
    
    def save_vocab(self, filepath):
        """Save vocabulary mappings"""
        vocab_data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f)
    
    @classmethod
    def load_vocab(cls, filepath):
        """Load vocabulary mappings"""
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        # Create a dummy dataset instance
        dataset = cls.__new__(cls)
        dataset.char_to_idx = vocab_data['char_to_idx']
        dataset.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
        dataset.vocab_size = vocab_data['vocab_size']
        return dataset
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        # Return sequence and target (next character)
        seq = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        target = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return seq, target
