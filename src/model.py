"""
Plaintext MLP model for voice command recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceCommandMLP(nn.Module):
    """
    Multi-Layer Perceptron for voice command classification.
    
    Architecture:
        Input → Dense(128) → ReLU → Dense(64) → ReLU → Dense(num_classes) → Softmax
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list = [128, 64],
        dropout: float = 0.2
    ):
        """
        Initialize MLP model.
        
        Args:
            input_dim: Input feature dimension (e.g., 13 * 32 = 416 for MFCC)
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions (default: [128, 64])
            dropout: Dropout probability (default: 0.2)
        """
        super(VoiceCommandMLP, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        return self.layers(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (with softmax).
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Probability distribution over classes
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

