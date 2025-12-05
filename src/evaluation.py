"""
Evaluation metrics for encrypted inference
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
import time
from tqdm import tqdm

from src.inference import EncryptedInference
from src.model import VoiceCommandMLP
from src.he_model import HEMLP
from src.feature_extraction import extract_mfcc
from src.data_loader import SpeechCommandsDataset


def evaluate_accuracy(
    encrypted_inference: EncryptedInference,
    plaintext_model: VoiceCommandMLP,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    num_samples: int = None,
    feature_mean: np.ndarray = None,
    feature_std: np.ndarray = None
) -> Dict[str, float]:
    """
    Evaluate accuracy of encrypted inference vs plaintext inference.
    
    Args:
        encrypted_inference: Encrypted inference system
        plaintext_model: Plaintext model for comparison
        test_loader: Test data loader
        device: Device for plaintext model
        num_samples: Number of samples to evaluate (None for all)
    
    Returns:
        Dictionary with accuracy metrics
    """
    plaintext_model.eval()
    
    he_correct = 0
    plaintext_correct = 0
    total = 0
    agreement = 0
    
    samples_evaluated = 0
    target_samples = num_samples if num_samples else len(test_loader.dataset)
    
    # Create progress bar for samples, not batches
    pbar = tqdm(total=target_samples, desc="Evaluating accuracy", initial=0, miniters=1)
    
    with torch.no_grad():
        for features, labels in test_loader:
            if num_samples and samples_evaluated >= num_samples:
                break
            
            batch_size = features.size(0)
            features_np = features.numpy()
            
            for i in range(batch_size):
                if num_samples and samples_evaluated >= num_samples:
                    break
                
                # Features are already MFCC features from the dataloader
                mfcc_features = features_np[i]
                
                # Normalize using dataset statistics if provided
                if feature_mean is not None and feature_std is not None:
                    mfcc_features = (mfcc_features - feature_mean) / feature_std
                else:
                    # Fallback: normalize per-sample (not ideal but works)
                    mfcc_features = (mfcc_features - mfcc_features.mean()) / (mfcc_features.std() + 1e-8)
                
                # Encrypted inference
                he_logits, _ = encrypted_inference.predict_encrypted(mfcc_features, return_timing=False)
                he_pred = np.argmax(he_logits)
                
                # Plaintext inference
                mfcc_tensor = torch.FloatTensor(mfcc_features).unsqueeze(0).to(device)
                plaintext_outputs = plaintext_model(mfcc_tensor)
                plaintext_pred = torch.argmax(plaintext_outputs, dim=1).item()
                
                # Compare
                label = labels[i].item()
                if he_pred == label:
                    he_correct += 1
                if plaintext_pred == label:
                    plaintext_correct += 1
                if he_pred == plaintext_pred:
                    agreement += 1
                
                total += 1
                samples_evaluated += 1
                pbar.update(1)
    
    pbar.close()
    
    results = {
        'he_accuracy': 100 * he_correct / total if total > 0 else 0,
        'plaintext_accuracy': 100 * plaintext_correct / total if total > 0 else 0,
        'agreement': 100 * agreement / total if total > 0 else 0,
        'accuracy_difference': (100 * plaintext_correct / total - 100 * he_correct / total) if total > 0 else 0,
        'total_samples': total
    }
    
    return results


def benchmark_latency(
    encrypted_inference: EncryptedInference,
    test_loader: torch.utils.data.DataLoader,
    num_samples: int = 100,
    feature_mean: np.ndarray = None,
    feature_std: np.ndarray = None
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark latency of encrypted inference operations.
    
    Args:
        encrypted_inference: Encrypted inference system
        test_loader: Test data loader
        num_samples: Number of samples to benchmark
    
    Returns:
        Dictionary with latency statistics
    """
    encrypt_times = []
    inference_times = []
    decrypt_times = []
    total_times = []
    
    samples_processed = 0
    pbar = tqdm(total=num_samples, desc="Benchmarking latency")
    
    for features, _ in test_loader:
        if samples_processed >= num_samples:
            break
        
        batch_size = features.size(0)
        features_np = features.numpy()
        
        for i in range(batch_size):
            if samples_processed >= num_samples:
                break
            
            # Features are already MFCC features from the dataloader
            mfcc_features = features_np[i]
            
            # Normalize using dataset statistics if provided
            if feature_mean is not None and feature_std is not None:
                mfcc_features = (mfcc_features - feature_mean) / feature_std
            else:
                # Fallback: normalize per-sample
                mfcc_features = (mfcc_features - mfcc_features.mean()) / (mfcc_features.std() + 1e-8)
            
            # Encrypted inference with timing
            _, timing = encrypted_inference.predict_encrypted(mfcc_features, return_timing=True)
            
            encrypt_times.append(timing['encrypt'])
            inference_times.append(timing['inference'])
            decrypt_times.append(timing['decrypt'])
            total_times.append(timing['total'])
            
            samples_processed += 1
            pbar.update(1)
    
    pbar.close()
    
    def compute_stats(times):
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        }
    
    results = {
        'encrypt': compute_stats(encrypt_times),
        'inference': compute_stats(inference_times),
        'decrypt': compute_stats(decrypt_times),
        'total': compute_stats(total_times),
        'samples': samples_processed
    }
    
    return results


def add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Gaussian noise to audio signal to achieve target SNR.
    
    Args:
        audio: Input audio signal
        snr_db: Signal-to-noise ratio in dB
    
    Returns:
        Noisy audio signal
    """
    # Calculate signal power
    signal_power = np.mean(audio ** 2)
    
    # Calculate noise power for target SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate noise
    noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
    
    # Add noise
    noisy_audio = audio + noise
    
    return noisy_audio


def test_robustness(
    encrypted_inference: EncryptedInference,
    plaintext_model: VoiceCommandMLP,
    test_loader: torch.utils.data.DataLoader,
    snr_levels: List[float] = [20.0, 10.0, 5.0],
    num_samples_per_snr: int = 50,
    device: str = 'cpu',
    feature_mean: np.ndarray = None,
    feature_std: np.ndarray = None
) -> Dict[str, Dict[str, float]]:
    """
    Test model robustness across different noise levels.
    
    Args:
        encrypted_inference: Encrypted inference system
        plaintext_model: Plaintext model for comparison
        test_loader: Test data loader
        snr_levels: List of SNR levels in dB to test
        num_samples_per_snr: Number of samples per SNR level
        device: Device for plaintext model
    
    Returns:
        Dictionary with robustness results for each SNR level
    """
    plaintext_model.eval()
    results = {}
    
    for snr_db in snr_levels:
        print(f"\nTesting robustness at SNR = {snr_db} dB")
        
        he_correct = 0
        plaintext_correct = 0
        total = 0
        
        samples_processed = 0
        pbar = tqdm(total=num_samples_per_snr, desc=f"SNR={snr_db}dB")
        
        # Access dataset directly to get raw audio files
        dataset = test_loader.dataset
        if hasattr(dataset, 'dataset'):  # Handle Subset wrapper
            dataset = dataset.dataset
        
        with torch.no_grad():
            # Iterate through dataset indices to get raw audio
            indices = list(range(len(dataset)))[:num_samples_per_snr]
            
            for idx in indices:
                if samples_processed >= num_samples_per_snr:
                    break
                
                # Load raw audio from file
                audio_file = dataset.audio_files[idx]
                label_idx = dataset.label_indices[idx]
                
                import soundfile as sf
                audio, sr = sf.read(str(audio_file))
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                # Add noise to raw audio
                noisy_audio = add_noise(audio, snr_db)
                
                # Extract MFCC features from noisy audio (without normalization)
                mfcc_features, _, _ = extract_mfcc(noisy_audio, sr=sr, normalize=False)
                
                # Normalize using dataset statistics if provided
                if feature_mean is not None and feature_std is not None:
                    mfcc_features = (mfcc_features - feature_mean) / feature_std
                else:
                    # Fallback: normalize per-sample
                    mfcc_features = (mfcc_features - mfcc_features.mean()) / (mfcc_features.std() + 1e-8)
                
                # Encrypted inference
                he_logits, _ = encrypted_inference.predict_encrypted(mfcc_features, return_timing=False)
                he_pred = np.argmax(he_logits)
                
                # Plaintext inference
                mfcc_tensor = torch.FloatTensor(mfcc_features).unsqueeze(0).to(device)
                plaintext_outputs = plaintext_model(mfcc_tensor)
                plaintext_pred = torch.argmax(plaintext_outputs, dim=1).item()
                
                # Compare
                if he_pred == label_idx:
                    he_correct += 1
                if plaintext_pred == label_idx:
                    plaintext_correct += 1
                
                total += 1
                samples_processed += 1
                pbar.update(1)
        
        pbar.close()
        
        results[f'snr_{snr_db}db'] = {
            'he_accuracy': 100 * he_correct / total if total > 0 else 0,
            'plaintext_accuracy': 100 * plaintext_correct / total if total > 0 else 0,
            'samples': total
        }
    
    return results

