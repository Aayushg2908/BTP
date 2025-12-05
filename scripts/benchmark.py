"""
Comprehensive benchmarking script for encrypted inference
"""

import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import evaluate_accuracy, benchmark_latency, test_robustness
from src.inference import EncryptedInference
from src.encryption import create_encryption_system
from src.model import VoiceCommandMLP
from src.he_model import HEMLP
from src.data_loader import create_dataloaders
from src.utils import get_project_root


def format_latency_stats(stats: dict) -> str:
    """Format latency statistics for printing."""
    return (
        f"  Mean: {stats['mean']:.4f}s\n"
        f"  Std: {stats['std']:.4f}s\n"
        f"  Min: {stats['min']:.4f}s\n"
        f"  Max: {stats['max']:.4f}s\n"
        f"  Median: {stats['median']:.4f}s\n"
        f"  P95: {stats['p95']:.4f}s\n"
        f"  P99: {stats['p99']:.4f}s"
    )


def main():
    parser = argparse.ArgumentParser(description='Benchmark encrypted inference')
    parser.add_argument('--data_dir', type=str, default='data/speech_commands_v0.02',
                       help='Path to dataset directory')
    parser.add_argument('--he_model_path', type=str, default='models/best_he_model.pth',
                       help='Path to HE model checkpoint')
    parser.add_argument('--plaintext_model_path', type=str, default='models/best_model.pth',
                       help='Path to plaintext model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_accuracy_samples', type=int, default=200,
                       help='Number of samples for accuracy evaluation')
    parser.add_argument('--num_latency_samples', type=int, default=100,
                       help='Number of samples for latency benchmarking')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    project_root = get_project_root()
    data_dir = project_root / args.data_dir
    he_model_path = project_root / args.he_model_path
    plaintext_model_path = project_root / args.plaintext_model_path
    output_dir = project_root / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device(args.device)
    
    print("=" * 60)
    print("Encrypted Inference Benchmarking")
    print("=" * 60)
    
    # Load models
    print("\n1. Loading models...")
    
    # Load HE model
    print(f"Loading HE model from {he_model_path}...")
    he_checkpoint = torch.load(he_model_path, map_location=device, weights_only=False)
    model_config = he_checkpoint['model_config'].copy()
    # Fix parameter name: poly_scale -> scale
    if 'poly_scale' in model_config:
        model_config['scale'] = model_config.pop('poly_scale')
    he_model = HEMLP(**model_config)
    he_model.load_state_dict(he_checkpoint['model_state_dict'])
    he_model.eval()
    
    # Load plaintext model
    print(f"Loading plaintext model from {plaintext_model_path}...")
    plaintext_checkpoint = torch.load(plaintext_model_path, map_location=device, weights_only=False)
    plaintext_model = VoiceCommandMLP(**plaintext_checkpoint['model_config'])
    plaintext_model.load_state_dict(plaintext_checkpoint['model_state_dict'])
    plaintext_model.to(device)
    plaintext_model.eval()
    
    # Get normalization stats if available
    feature_mean = plaintext_checkpoint.get('feature_mean', None)
    feature_std = plaintext_checkpoint.get('feature_std', None)
    if feature_mean is not None and feature_std is not None:
        print("Found dataset normalization statistics in checkpoint")
        feature_mean = np.array(feature_mean)
        feature_std = np.array(feature_std)
    else:
        print("Warning: No normalization stats found. Will compute from test set.")
        feature_mean = None
        feature_std = None
    
    # Create encryption system
    print("Creating encryption system...")
    encryption_system = create_encryption_system(
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    
    # Create encrypted inference system
    poly_type = model_config.get('poly_type', 'scaled_quad')
    encrypted_inference = EncryptedInference(
        model=he_model,
        encryption_system=encryption_system,
        poly_type=poly_type
    )
    
    # Load test data
    print("\n2. Loading test data...")
    _, _, test_loader, label_map = create_dataloaders(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Accuracy evaluation
    print("\n3. Evaluating accuracy...")
    accuracy_results = evaluate_accuracy(
        encrypted_inference=encrypted_inference,
        plaintext_model=plaintext_model,
        test_loader=test_loader,
        device=device,
        num_samples=args.num_accuracy_samples,
        feature_mean=feature_mean,
        feature_std=feature_std
    )
    
    print("\nAccuracy Results:")
    print(f"  HE Inference Accuracy: {accuracy_results['he_accuracy']:.2f}%")
    print(f"  Plaintext Accuracy: {accuracy_results['plaintext_accuracy']:.2f}%")
    print(f"  Agreement: {accuracy_results['agreement']:.2f}%")
    print(f"  Accuracy Difference: {accuracy_results['accuracy_difference']:.2f}%")
    print(f"  Samples Evaluated: {accuracy_results['total_samples']}")
    
    # Latency benchmarking
    print("\n4. Benchmarking latency...")
    latency_results = benchmark_latency(
        encrypted_inference=encrypted_inference,
        test_loader=test_loader,
        num_samples=args.num_latency_samples,
        feature_mean=feature_mean,
        feature_std=feature_std
    )
    
    print("\nLatency Results:")
    print(f"\nEncryption Time:")
    print(format_latency_stats(latency_results['encrypt']))
    print(f"\nInference Time:")
    print(format_latency_stats(latency_results['inference']))
    print(f"\nDecryption Time:")
    print(format_latency_stats(latency_results['decrypt']))
    print(f"\nTotal Time:")
    print(format_latency_stats(latency_results['total']))
    print(f"\nSamples Processed: {latency_results['samples']}")
    
    # Robustness testing
    print("\n5. Testing robustness...")
    robustness_results = test_robustness(
        encrypted_inference=encrypted_inference,
        plaintext_model=plaintext_model,
        test_loader=test_loader,
        snr_levels=[20.0, 10.0, 5.0],
        num_samples_per_snr=50,
        device=device,
        feature_mean=feature_mean,
        feature_std=feature_std
    )
    
    print("\nRobustness Results:")
    for snr_key, results in robustness_results.items():
        snr = snr_key.replace('snr_', '').replace('db', '')
        print(f"\nSNR = {snr} dB:")
        print(f"  HE Accuracy: {results['he_accuracy']:.2f}%")
        print(f"  Plaintext Accuracy: {results['plaintext_accuracy']:.2f}%")
        print(f"  Samples: {results['samples']}")
    
    # Save results
    print("\n6. Saving results...")
    
    # Combine all results
    all_results = {
        'accuracy': accuracy_results,
        'latency': latency_results,
        'robustness': robustness_results,
        'config': {
            'num_accuracy_samples': args.num_accuracy_samples,
            'num_latency_samples': args.num_latency_samples,
            'poly_type': poly_type,
            'model_config': he_checkpoint['model_config']
        }
    }
    
    # Save as JSON
    results_json_path = output_dir / 'benchmark_results.json'
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {results_json_path}")
    
    # Save latency as CSV
    latency_df = pd.DataFrame({
        'operation': ['encrypt', 'inference', 'decrypt', 'total'],
        'mean': [
            latency_results['encrypt']['mean'],
            latency_results['inference']['mean'],
            latency_results['decrypt']['mean'],
            latency_results['total']['mean']
        ],
        'std': [
            latency_results['encrypt']['std'],
            latency_results['inference']['std'],
            latency_results['decrypt']['std'],
            latency_results['total']['std']
        ],
        'median': [
            latency_results['encrypt']['median'],
            latency_results['inference']['median'],
            latency_results['decrypt']['median'],
            latency_results['total']['median']
        ]
    })
    latency_csv_path = output_dir / 'latency_stats.csv'
    latency_df.to_csv(latency_csv_path, index=False)
    print(f"Latency stats saved to {latency_csv_path}")
    
    print("\n" + "=" * 60)
    print("Benchmarking complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

