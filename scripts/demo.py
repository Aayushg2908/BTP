"""
Interactive demo script for privacy-preserving voice command recognition.
This script demonstrates encrypted inference on a single audio sample.
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import soundfile as sf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference import EncryptedInference
from src.encryption import create_encryption_system
from src.model import VoiceCommandMLP
from src.he_model import HEMLP
from src.feature_extraction import extract_mfcc
from src.data_loader import SpeechCommandsDataset


def load_models(he_model_path, plaintext_model_path, device='cpu'):
    """Load HE and plaintext models."""
    print("Loading models...")
    
    # Load HE model
    he_checkpoint = torch.load(he_model_path, map_location=device, weights_only=False)
    model_config = he_checkpoint['model_config'].copy()
    if 'poly_scale' in model_config:
        model_config['scale'] = model_config.pop('poly_scale')
    he_model = HEMLP(**model_config)
    he_model.load_state_dict(he_checkpoint['model_state_dict'])
    he_model.eval()
    
    # Load plaintext model
    plaintext_checkpoint = torch.load(plaintext_model_path, map_location=device, weights_only=False)
    plaintext_model = VoiceCommandMLP(**plaintext_checkpoint['model_config'])
    plaintext_model.load_state_dict(plaintext_checkpoint['model_state_dict'])
    plaintext_model.to(device)
    plaintext_model.eval()
    
    # Get normalization stats
    feature_mean = plaintext_checkpoint.get('feature_mean', None)
    feature_std = plaintext_checkpoint.get('feature_std', None)
    if feature_mean is not None and feature_std is not None:
        feature_mean = np.array(feature_mean)
        feature_std = np.array(feature_std)
    
    return he_model, plaintext_model, feature_mean, feature_std


def predict_from_audio_file(
    audio_path: str,
    he_model: HEMLP,
    plaintext_model: VoiceCommandMLP,
    encryption_system,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    label_map: dict,
    device: str = 'cpu'
):
    """Predict voice command from audio file."""
    print(f"\nProcessing audio file: {audio_path}")
    
    # Load audio
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    print(f"Audio loaded: {len(audio)} samples at {sr}Hz")
    
    # Extract MFCC features
    print("Extracting MFCC features...")
    mfcc_features, _, _ = extract_mfcc(audio, sr=sr, normalize=False)
    print(f"MFCC features: {mfcc_features.shape}")
    
    # Normalize
    if feature_mean is not None and feature_std is not None:
        mfcc_features = (mfcc_features - feature_mean) / feature_std
        print("Features normalized using dataset statistics")
    
    # Plaintext inference
    print("\n--- Plaintext Inference ---")
    mfcc_tensor = torch.FloatTensor(mfcc_features).unsqueeze(0).to(device)
    with torch.no_grad():
        plaintext_outputs = plaintext_model(mfcc_tensor)
        plaintext_probs = torch.softmax(plaintext_outputs, dim=1)
        plaintext_pred = torch.argmax(plaintext_outputs, dim=1).item()
    
    plaintext_prob = plaintext_probs[0][plaintext_pred].item()
    plaintext_label = list(label_map.keys())[plaintext_pred]
    
    print(f"Predicted: {plaintext_label} (confidence: {plaintext_prob:.2%})")
    print(f"Top 3 predictions:")
    top3_probs, top3_indices = torch.topk(plaintext_probs[0], 3)
    for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
        label = list(label_map.keys())[idx.item()]
        print(f"  {i+1}. {label}: {prob.item():.2%}")
    
    # Encrypted inference
    print("\n--- Encrypted Inference ---")
    encrypted_inference = EncryptedInference(
        model=he_model,
        encryption_system=encryption_system,
        poly_type=he_model.poly_type
    )
    
    he_logits, timing = encrypted_inference.predict_encrypted(mfcc_features, return_timing=True)
    he_probs = np.exp(he_logits - np.max(he_logits))  # Softmax
    he_probs = he_probs / np.sum(he_probs)
    he_pred = np.argmax(he_logits)
    
    he_prob = he_probs[he_pred]
    he_label = list(label_map.keys())[he_pred]
    
    print(f"Predicted: {he_label} (confidence: {he_prob:.2%})")
    print(f"Top 3 predictions:")
    top3_indices = np.argsort(he_probs)[::-1][:3]
    for i, idx in enumerate(top3_indices):
        label = list(label_map.keys())[idx]
        print(f"  {i+1}. {label}: {he_probs[idx]:.2%}")
    
    print(f"\n--- Timing ---")
    print(f"Encryption: {timing['encrypt']*1000:.2f}ms")
    print(f"Inference: {timing['inference']*1000:.2f}ms")
    print(f"Decryption: {timing['decrypt']*1000:.2f}ms")
    print(f"Total: {timing['total']*1000:.2f}ms")
    
    print(f"\n--- Comparison ---")
    print(f"Agreement: {'✓ YES' if plaintext_pred == he_pred else '✗ NO'}")
    if plaintext_pred == he_pred:
        print(f"Both models predicted: {plaintext_label}")
    else:
        print(f"Plaintext: {plaintext_label}, HE: {he_label}")
    
    return plaintext_pred, he_pred, timing


def demo_random_sample(data_dir: str, he_model_path: str, plaintext_model_path: str, device='cpu'):
    """Demo with a random sample from test set."""
    print("="*60)
    print("Privacy-Preserving Voice Command Recognition Demo")
    print("="*60)
    
    # Load models
    he_model, plaintext_model, feature_mean, feature_std = load_models(
        he_model_path, plaintext_model_path, device
    )
    
    # Create encryption system
    print("\nCreating encryption system...")
    encryption_system = create_encryption_system(
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    print("Encryption system ready!")
    
    # Load dataset to get label map and random sample
    print("\nLoading test dataset...")
    test_dataset = SpeechCommandsDataset(data_dir, split="test")
    label_map = test_dataset.get_label_map()
    print(f"Found {len(label_map)} classes: {sorted(label_map.keys())}")
    
    # Get random sample
    import random
    idx = random.randint(0, len(test_dataset) - 1)
    audio_file = test_dataset.audio_files[idx]
    true_label = test_dataset.labels[idx]
    true_label_idx = label_map[true_label]
    
    print(f"\nRandom sample selected:")
    print(f"  File: {audio_file.name}")
    print(f"  True label: {true_label}")
    
    # Predict
    plaintext_pred, he_pred, timing = predict_from_audio_file(
        str(audio_file),
        he_model,
        plaintext_model,
        encryption_system,
        feature_mean,
        feature_std,
        label_map,
        device
    )
    
    print(f"\n--- Ground Truth ---")
    print(f"True label: {true_label}")
    print(f"Plaintext correct: {'✓' if plaintext_pred == true_label_idx else '✗'}")
    print(f"HE correct: {'✓' if he_pred == true_label_idx else '✗'}")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Interactive demo for encrypted voice recognition')
    parser.add_argument('--data_dir', type=str, default='data/speech_commands_v0.02',
                       help='Path to dataset directory')
    parser.add_argument('--he_model_path', type=str, default='models/best_he_model.pth',
                       help='Path to HE model checkpoint')
    parser.add_argument('--plaintext_model_path', type=str, default='models/best_model.pth',
                       help='Path to plaintext model checkpoint')
    parser.add_argument('--audio_file', type=str, default=None,
                       help='Path to specific audio file (if None, uses random sample)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda/mps)')
    
    args = parser.parse_args()
    
    if args.audio_file:
        # Predict from specific file
        he_model, plaintext_model, feature_mean, feature_std = load_models(
            args.he_model_path, args.plaintext_model_path, args.device
        )
        encryption_system = create_encryption_system()
        test_dataset = SpeechCommandsDataset(args.data_dir, split="test")
        label_map = test_dataset.get_label_map()
        
        predict_from_audio_file(
            args.audio_file,
            he_model,
            plaintext_model,
            encryption_system,
            feature_mean,
            feature_std,
            label_map,
            args.device
        )
    else:
        # Demo with random sample
        demo_random_sample(
            args.data_dir,
            args.he_model_path,
            args.plaintext_model_path,
            args.device
        )


if __name__ == "__main__":
    main()

