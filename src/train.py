"""
Training script for plaintext MLP model
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import VoiceCommandMLP
from src.data_loader import create_dataloaders
from src.feature_extraction import get_feature_dim
from src.utils import get_project_root


class NormalizeTransform:
    """Picklable normalization transform for multiprocessing."""
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std
    
    def __call__(self, features):
        features_np = features.numpy() if isinstance(features, torch.Tensor) else features
        features_norm = (features_np - self.mean) / self.std
        return torch.FloatTensor(features_norm)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for features, labels in pbar:
        # Features are already MFCC features (extracted in dataset)
        # Move to device (non-blocking for faster transfer)
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for features, labels in pbar:
            # Features are already MFCC features (extracted in dataset)
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='Train voice command MLP model')
    parser.add_argument('--data_dir', type=str, default='data/speech_commands_v0.02',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training (larger = faster)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64],
                       help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout probability')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save model checkpoints')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/mps/cpu). Auto-detect if not specified')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--fast_mode', action='store_true',
                       help='Fast mode: use subset of data and sample for normalization stats')
    parser.add_argument('--subset_size', type=int, default=None,
                       help='Use subset of training data (for faster iteration)')
    
    args = parser.parse_args()
    
    # Setup device (auto-detect if not specified)
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    project_root = get_project_root()
    data_dir = project_root / args.data_dir
    save_dir = project_root / args.save_dir
    save_dir.mkdir(exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Save directory: {save_dir}")
    print(f"Batch size: {args.batch_size}, Workers: {args.num_workers}")
    
    # Create dataloaders (without normalization first to compute stats)
    print("Loading dataset...")
    train_loader_temp, _, _, label_map = create_dataloaders(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Compute normalization statistics (use subset for speed if fast_mode)
    print("Computing normalization statistics from training set...")
    all_features = []
    max_samples = 10000 if args.fast_mode else None  # Use 10k samples for fast mode
    samples_collected = 0
    
    for features, _ in train_loader_temp:
        all_features.append(features.numpy())
        samples_collected += features.shape[0]
        if max_samples and samples_collected >= max_samples:
            print(f"Using {samples_collected} samples for normalization stats (fast mode)")
            break
    
    all_features = np.concatenate(all_features, axis=0)
    feature_mean = np.mean(all_features, axis=0)
    feature_std = np.std(all_features, axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)  # Avoid division by zero
    
    print(f"Feature mean range: [{np.min(feature_mean):.4f}, {np.max(feature_mean):.4f}]")
    print(f"Feature std range: [{np.min(feature_std):.4f}, {np.max(feature_std):.4f}]")
    
    # Create normalization transform (picklable class for multiprocessing)
    normalize_transform = NormalizeTransform(feature_mean, feature_std)
    
    # Recreate dataloaders with normalization transform
    from src.data_loader import SpeechCommandsDataset
    train_dataset = SpeechCommandsDataset(str(data_dir), "train", transform=normalize_transform)
    
    # Use subset if specified
    if args.subset_size:
        from torch.utils.data import Subset
        indices = torch.randperm(len(train_dataset))[:args.subset_size]
        train_dataset = Subset(train_dataset, indices)
        print(f"Using subset of {len(train_dataset)} samples for training")
    
    val_dataset = SpeechCommandsDataset(str(data_dir), "validation", label_map=label_map, transform=normalize_transform)
    test_dataset = SpeechCommandsDataset(str(data_dir), "test", label_map=label_map, transform=normalize_transform)
    
    # Use pin_memory only for CUDA
    pin_memory = device.type == 'cuda'
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0  # Keep workers alive between epochs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=pin_memory
    )
    
    # Get feature dimension
    feature_dim = get_feature_dim()
    num_classes = len(label_map)
    
    print(f"Feature dimension: {feature_dim}")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    model = VoiceCommandMLP(
        input_dim=feature_dim,
        num_classes=num_classes,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout
    ).to(device)
    
    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'label_map': label_map,
                'model_config': {
                    'input_dim': feature_dim,
                    'num_classes': num_classes,
                    'hidden_dims': args.hidden_dims,
                    'dropout': args.dropout
                },
                'feature_mean': feature_mean,
                'feature_std': feature_std
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f"Saved best model with val_acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(save_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

