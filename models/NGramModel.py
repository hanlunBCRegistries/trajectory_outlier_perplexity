import numpy as np
from collections import defaultdict, Counter
import pickle
import json
import os

class NGramModel:
    def __init__(self, n=3, smoothing=0.1):
        """
        N-gram model for trajectory anomaly detection
        
        Args:
            n: The 'n' in n-gram (context length)
            smoothing: Laplace smoothing parameter
        """
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = defaultdict(int)
        self.vocab = set()
        
    def fit(self, trajectories):
        """Train the model on a list of trajectories"""
        # Build vocabulary
        for traj in trajectories:
            self.vocab.update(traj)
        
        # Count n-grams
        for traj in trajectories:
            padded_traj = ['<START>'] * (self.n - 1) + list(traj)
            for i in range(len(padded_traj) - self.n + 1):
                context = tuple(padded_traj[i:i+self.n-1])
                next_token = padded_traj[i+self.n-1]
                self.ngram_counts[context][next_token] += 1
                self.context_counts[context] += 1
    
    def get_log_probability(self, traj):
        """Calculate the log probability of a trajectory"""
        log_prob = 0
        padded_traj = ['<START>'] * (self.n - 1) + list(traj)
        
        for i in range(len(padded_traj) - self.n + 1):
            context = tuple(padded_traj[i:i+self.n-1])
            next_token = padded_traj[i+self.n-1]
            
            # Calculate probability with Laplace smoothing
            count = self.ngram_counts[context][next_token]
            context_total = self.context_counts[context]
            
            if context_total > 0:
                prob = (count + self.smoothing) / (context_total + self.smoothing * len(self.vocab))
            else:
                prob = 1.0 / len(self.vocab)  # Uniform if context never seen
                
            log_prob += np.log(prob)
            
        return log_prob
    
    def get_perplexity(self, traj):
        """
        Calculate perplexity of a trajectory
        Perplexity = exp(-log_prob / N)
        Higher perplexity indicates more anomalous trajectories
        """
        log_prob = self.get_log_probability(traj)
        # Add 1 to account for the effective sequence length after padding
        N = max(1, len(traj) - self.n + 1)
        return np.exp(-log_prob / N)
    
    def detect_outliers(self, trajectories, threshold_percentile=95):
        """
        Detect outliers based on trajectory perplexity
        Returns indices of outlier trajectories
        Note: For perplexity, higher values (above threshold_percentile) are outliers
        """
        perplexities = [self.get_perplexity(traj) for traj in trajectories]
        threshold = np.percentile(perplexities, threshold_percentile)
        outliers = [i for i, perp in enumerate(perplexities) if perp >= threshold]
        return outliers, perplexities
    
    def score_trajectory(self, traj):
        """
        Returns the perplexity score for a trajectory
        Higher perplexity = more anomalous
        """
        return self.get_perplexity(traj)
    
    def save(self, filepath):
        """Save model to file"""
        model_data = {
            'n': self.n,
            'smoothing': self.smoothing,
            'ngram_counts': dict(self.ngram_counts),
            'context_counts': dict(self.context_counts),
            'vocab': list(self.vocab)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(n=model_data['n'], smoothing=model_data['smoothing'])
        model.ngram_counts = defaultdict(Counter, model_data['ngram_counts'])
        model.context_counts = defaultdict(int, model_data['context_counts'])
        model.vocab = set(model_data['vocab'])
        return model