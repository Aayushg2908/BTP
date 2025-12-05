"""
Utility functions for the voice command recognition project.
"""

import os
import numpy as np
from pathlib import Path


def ensure_dir(path):
    """Ensure a directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def normalize_features(features, mean=None, std=None):
    """
    Normalize features to zero mean and unit variance.
    
    Args:
        features: Feature array to normalize
        mean: Pre-computed mean (if None, computed from features)
        std: Pre-computed std (if None, computed from features)
    
    Returns:
        Normalized features, mean, std
    """
    if mean is None:
        mean = np.mean(features, axis=0, keepdims=True)
    if std is None:
        std = np.std(features, axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)  # Avoid division by zero
    
    normalized = (features - mean) / std
    return normalized, mean, std

