import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryLSTM(nn.Module):
    """
    LSTM model for trajectory modeling, designed for next token prediction.
    Similar to language models but with LSTM architecture instead of transformer.
    """
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, n_layers=2, dropout=0.2):
        super(TrajectoryLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout
        
        # Embedding layer to convert token indices to vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        """
        Forward pass for next token prediction
        x: input tensor of shape [batch_size, seq_len]
        Returns: logits of shape [batch_size, seq_len, vocab_size]
        """
        # Embed input tokens
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # Process through LSTM
        output, (hidden, cell) = self.lstm(embedded)
        
        # Apply dropout to LSTM output
        output = self.dropout(output)
        
        # Project to vocabulary size
        logits = self.fc(output)
        
        return logits
    
    def get_perplexity(self, x):
        """
        Calculate perplexity for input sequences
        x: input tensor of shape [batch_size, seq_len]
        Returns: perplexity scores of shape [batch_size]
        """
        with torch.no_grad():
            # Forward pass excluding the last token
            logits = self.forward(x[:, :-1])
            
            # Calculate log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Get target tokens (shifted by 1 position)
            targets = x[:, 1:]
            
            # Extract log probability of each actual next token
            next_token_log_probs = torch.gather(
                log_probs, 
                dim=2, 
                index=targets.unsqueeze(-1)
            ).squeeze(-1)
            
            # Create mask for valid tokens (non-padding)
            mask = (targets != 0).float()  # Assuming 0 is padding token
            
            # Calculate average log probability per sequence
            seq_lengths = mask.sum(dim=1)
            seq_log_probs = (next_token_log_probs * mask).sum(dim=1)
            avg_log_probs = seq_log_probs / seq_lengths.clamp(min=1)  # Avoid division by zero
            
            # Convert to perplexity
            perplexity = torch.exp(-avg_log_probs)
            
        return perplexity
    
    def configure_optimizers(self, weight_decay, lr, betas, device):
        """
        Configure optimizer similar to LMTAD approach
        """
        # Separate weight decay and no-weight decay parameters
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if 'bias' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer
    
    def get_num_params(self):
        """
        Get number of parameters in the model
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params / 1e6  # Return in millions