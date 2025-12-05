"""
Data loader for Google Speech Commands Dataset
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import soundfile as sf
import numpy as np
from typing import Optional, Tuple, List


class SpeechCommandsDataset(Dataset):
    """
    Dataset class for Google Speech Commands Dataset.
    
    Args:
        data_dir: Root directory containing train/validation/test folders
        split: One of 'train', 'validation', 'test'
        label_map: Optional dictionary mapping label names to indices
        transform: Optional transform to apply to audio
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        label_map: Optional[dict] = None,
        transform: Optional[callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load audio files
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        # Get all audio files, including those in label subdirectories
        self.audio_files = sorted(list(split_dir.rglob("*.wav")))
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {split_dir}")
        
        # Extract labels from subdirectory names (e.g., train/backward/file.wav -> "backward")
        # If files are directly in split_dir, extract from filename
        self.labels = []
        unique_labels = set()
        
        for audio_file in self.audio_files:
            # Get relative path from split_dir
            rel_path = audio_file.relative_to(split_dir)
            
            # If file is in a subdirectory, use subdirectory name as label
            # Otherwise, extract from filename (format: <label>_<speaker_id>_<nohash>.wav)
            if len(rel_path.parts) > 1:
                # File is in a subdirectory (label)
                label = rel_path.parts[0]
            else:
                # File is directly in split_dir, extract label from filename
                label = audio_file.stem.split('_')[0]
            
            self.labels.append(label)
            unique_labels.add(label)
        
        # Create or use provided label map
        if label_map is None:
            self.label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        else:
            self.label_map = label_map
        
        self.num_classes = len(self.label_map)
        
        # Convert labels to indices
        self.label_indices = [self.label_map[label] for label in self.labels]
        
        print(f"Loaded {len(self.audio_files)} samples from {split} split")
        print(f"Found {self.num_classes} classes: {sorted(self.label_map.keys())}")
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Returns:
            features: MFCC feature tensor (fixed size)
            label: Label index
        """
        audio_file = self.audio_files[idx]
        label_idx = self.label_indices[idx]
        
        # Load audio file
        audio, sr = sf.read(str(audio_file))
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Extract MFCC features (fixed size: 416 dimensions)
        # Import here to avoid circular import
        from src.feature_extraction import extract_mfcc
        # Don't normalize here - normalization should be done at dataset level
        # to preserve feature distributions across samples
        mfcc_features, _, _ = extract_mfcc(audio, sr=sr, normalize=False)
        
        # Convert to tensor
        features = torch.FloatTensor(mfcc_features)
        
        # Apply transform if provided (for additional augmentation)
        if self.transform:
            features = self.transform(features)
        
        return features, label_idx
    
    def get_label_map(self) -> dict:
        """Get the label mapping dictionary."""
        return self.label_map.copy()
    
    def get_label_name(self, idx: int) -> str:
        """Get label name from index."""
        reverse_map = {v: k for k, v in self.label_map.items()}
        return reverse_map.get(idx, "unknown")


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Optional[callable] = None,
    val_transform: Optional[callable] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root directory containing train/validation/test folders
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        train_transform: Optional transform for training data
        val_transform: Optional transform for validation/test data
    
    Returns:
        train_loader, val_loader, test_loader, label_map
    """
    # Create datasets
    train_dataset = SpeechCommandsDataset(
        data_dir=data_dir,
        split="train",
        transform=train_transform
    )
    
    # Use same label map for all splits
    label_map = train_dataset.get_label_map()
    
    val_dataset = SpeechCommandsDataset(
        data_dir=data_dir,
        split="validation",
        label_map=label_map,
        transform=val_transform
    )
    
    test_dataset = SpeechCommandsDataset(
        data_dir=data_dir,
        split="test",
        label_map=label_map,
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, label_map

