import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """Simple RNN model for text generation"""
    
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.2):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.dropout(output)
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    def get_config(self):
        """Return model configuration for saving"""
        return {
            'model_type': 'RNNModel',
            'vocab_size': self.vocab_size,
            'embed_size': self.embed_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }