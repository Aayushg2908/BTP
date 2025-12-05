"""
Quick script to check model predictions and debug accuracy issues
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import VoiceCommandMLP
from src.data_loader import SpeechCommandsDataset
from src.feature_extraction import extract_mfcc

def main():
    # Load model
    model_path = project_root / "models/best_model.pth"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = VoiceCommandMLP(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load a few samples
    dataset = SpeechCommandsDataset("data/speech_commands_v0.02", split="test")
    
    print("Checking model predictions...")
    print(f"Model config: {checkpoint['model_config']}")
    print(f"Number of classes: {len(checkpoint['label_map'])}")
    print()
    
    correct = 0
    total = 0
    
    for i in range(min(20, len(dataset))):
        features, label = dataset[i]
        label_name = dataset.get_label_name(label)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(features.unsqueeze(0))
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1).item()
            pred_name = dataset.get_label_name(pred)
            confidence = probs[0, pred].item()
        
        is_correct = pred == label
        if is_correct:
            correct += 1
        total += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Sample {i}: True={label_name:10s} Pred={pred_name:10s} Conf={confidence:.3f} Features: mean={features.mean():.3f} std={features.std():.3f}")
    
    print(f"\nAccuracy on {total} samples: {100*correct/total:.1f}%")
    
    # Check feature statistics
    print("\nFeature statistics:")
    all_features = []
    for i in range(min(100, len(dataset))):
        features, _ = dataset[i]
        all_features.append(features.numpy())
    all_features = np.array(all_features)
    print(f"  Mean across samples: {all_features.mean(axis=0).mean():.6f}")
    print(f"  Std across samples: {all_features.std(axis=0).mean():.6f}")
    print(f"  Per-sample mean (should be ~0): {all_features.mean(axis=1).mean():.6f}")
    print(f"  Per-sample std (should be ~1): {all_features.std(axis=1).mean():.6f}")

if __name__ == "__main__":
    main()

