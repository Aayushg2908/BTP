"""
HE-compatible model with polynomial activations for homomorphic encryption
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolynomialActivation(nn.Module):
    """
    Polynomial activation function for HE compatibility.
    
    Approximates ReLU using a polynomial: f(x) = a*x² + b*x + c
    Common choices:
    - Quadratic: f(x) = x² (for x >= 0 approximation)
    - Scaled quadratic: f(x) = 0.5*x² + 0.5*x (smooth ReLU approximation)
    - Chebyshev: T₂(x) = 2x² - 1 (scaled appropriately)
    """
    
    def __init__(self, poly_type='quadratic', scale=1.0):
        """
        Initialize polynomial activation.
        
        Args:
            poly_type: Type of polynomial ('quadratic', 'scaled_quad', 'chebyshev')
            scale: Scaling factor for the polynomial
        """
        super(PolynomialActivation, self).__init__()
        self.poly_type = poly_type
        self.scale = scale
    
    def forward(self, x):
        """
        Apply polynomial activation.
        
        Args:
            x: Input tensor
        
        Returns:
            Activated tensor
        """
        if self.poly_type == 'quadratic':
            # f(x) = x²
            return x * x
        elif self.poly_type == 'scaled_quad':
            # f(x) = 0.5*x² + 0.5*x (smooth approximation of ReLU)
            return 0.5 * x * x + 0.5 * x
        elif self.poly_type == 'chebyshev':
            # T₂(x) = 2x² - 1, scaled
            return self.scale * (2 * x * x - 1)
        else:
            raise ValueError(f"Unknown polynomial type: {self.poly_type}")


class HEMLP(nn.Module):
    """
    HE-compatible MLP for voice command classification.
    
    Uses polynomial activations instead of ReLU to enable homomorphic encryption.
    All operations are HE-friendly (only addition, subtraction, multiplication).
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list = [128, 64],
        poly_type: str = 'scaled_quad',
        scale: float = 1.0
    ):
        """
        Initialize HE-compatible MLP model.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            poly_type: Type of polynomial activation ('quadratic', 'scaled_quad', 'chebyshev')
            scale: Scaling factor for polynomial activation
        """
        super(HEMLP, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.poly_type = poly_type
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(PolynomialActivation(poly_type=poly_type, scale=scale))
            prev_dim = hidden_dim
        
        # Output layer (no activation, softmax applied externally)
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
    
    def get_weights(self) -> list:
        """
        Get model weights as numpy arrays for HE inference.
        
        Returns:
            List of weight matrices and bias vectors
        """
        weights = []
        for module in self.layers:
            if isinstance(module, nn.Linear):
                weights.append({
                    'weight': module.weight.detach().cpu().numpy(),
                    'bias': module.bias.detach().cpu().numpy()
                })
        return weights
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def convert_plaintext_to_he_model(plaintext_model: nn.Module, poly_type='scaled_quad', scale=1.0) -> HEMLP:
    """
    Convert a plaintext MLP model to HE-compatible model.
    
    Args:
        plaintext_model: Trained VoiceCommandMLP model
        poly_type: Type of polynomial activation
        scale: Scaling factor for polynomial
    
    Returns:
        HE-compatible model with same weights (except activations)
    """
    # Extract model configuration
    input_dim = plaintext_model.input_dim
    num_classes = plaintext_model.num_classes
    hidden_dims = plaintext_model.hidden_dims
    
    # Create HE model
    he_model = HEMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        poly_type=poly_type,
        scale=scale
    )
    
    # Copy weights from plaintext model (skip ReLU layers)
    plaintext_modules = [m for m in plaintext_model.layers if isinstance(m, nn.Linear)]
    he_modules = [m for m in he_model.layers if isinstance(m, nn.Linear)]
    
    for pt_module, he_module in zip(plaintext_modules, he_modules):
        he_module.weight.data = pt_module.weight.data.clone()
        he_module.bias.data = pt_module.bias.data.clone()
    
    return he_model

