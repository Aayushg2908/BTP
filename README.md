# Privacy-Preserving Voice Command Recognition Using Approximate Homomorphic Encryption

A research project exploring the feasibility of performing voice command recognition on encrypted audio features using CKKS (Cheon-Kim-Kim-Song) homomorphic encryption scheme.

## Overview

This project implements a privacy-preserving voice assistant that performs inference on encrypted MFCC (Mel-Frequency Cepstral Coefficients) features using Approximate Homomorphic Encryption. The system ensures that user voice data remains confidential even during processing, addressing privacy concerns with cloud-based voice assistants.

## Features

- **MFCC Feature Extraction**: Converts raw audio to low-dimensional numerical features
- **Homomorphic Encryption**: Uses TenSEAL (CKKS scheme) for encrypted inference
- **HE-Compatible Neural Network**: MLP with polynomial activations instead of ReLU
- **Fully Encrypted Inference**: Performs inference entirely in the encrypted domain
- **Comprehensive Evaluation**: Accuracy, latency, and robustness benchmarking

## Project Structure

```
BTP/
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── data/                    # Dataset storage
├── models/                  # Trained model checkpoints
├── src/
│   ├── data_loader.py      # Dataset loading utilities
│   ├── feature_extraction.py  # MFCC extraction
│   ├── model.py            # Plaintext MLP model
│   ├── he_model.py         # HE-compatible model
│   ├── train.py            # Training script for plaintext model
│   ├── train_he.py         # Training script for HE model
│   ├── encryption.py       # TenSEAL encryption utilities
│   ├── inference.py        # Encrypted inference pipeline
│   ├── evaluation.py       # Evaluation metrics
│   └── utils.py            # Helper functions
├── scripts/
│   ├── download_dataset.py # Download Google Speech Commands dataset
│   └── benchmark.py        # Comprehensive benchmarking
├── tests/
│   └── test_inference.py   # Unit tests
└── notebooks/
    └── exploration.ipynb   # Data exploration
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
cd /path/to/BTP
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: TenSEAL installation may require additional system dependencies. Refer to the [TenSEAL documentation](https://github.com/OpenMined/TenSEAL) for details.

## Usage

### 1. Download Dataset

Download and organize the Google Speech Commands Dataset:

```bash
python scripts/download_dataset.py
```

This will download the dataset to `data/speech_commands_v0.02/` and organize it into train/validation/test splits.

### 2. Train Plaintext Model

Train the baseline plaintext MLP model:

```bash
python src/train.py \
    --data_dir data/speech_commands_v0.02 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --save_dir models
```

The trained model will be saved to `models/best_model.pth`.

### 3. Train HE-Compatible Model

Convert and fine-tune the model for homomorphic encryption:

```bash
python src/train_he.py \
    --data_dir data/speech_commands_v0.02 \
    --plaintext_model models/best_model.pth \
    --batch_size 32 \
    --epochs 20 \
    --lr 0.0001 \
    --poly_type scaled_quad \
    --save_dir models
```

The HE-compatible model will be saved to `models/best_he_model.pth`.

### 4. Run Benchmarking

Evaluate the encrypted inference system:

```bash
python scripts/benchmark.py \
    --data_dir data/speech_commands_v0.02 \
    --he_model_path models/best_he_model.pth \
    --plaintext_model_path models/best_model.pth \
    --num_accuracy_samples 200 \
    --num_latency_samples 100 \
    --output_dir results
```

Results will be saved to `results/benchmark_results.json` and `results/latency_stats.csv`.

### 5. Run Tests

Run unit tests to verify functionality:

```bash
pytest tests/test_inference.py -v
```

## Methodology

### Feature Extraction

- **MFCC Parameters**: 13 coefficients, 32 time frames → 416-dimensional feature vector
- **Preprocessing**: Zero-padding/truncation to fixed length, zero-mean unit-variance normalization
- **Extraction**: Done locally on user's device (plaintext stage)

### Encryption

- **Scheme**: CKKS (Cheon-Kim-Kim-Song) approximate homomorphic encryption
- **Library**: TenSEAL (Python wrapper for Microsoft SEAL)
- **Parameters**: 
  - Polynomial modulus degree: 8192 (or 16384 for larger models)
  - Coefficient modulus bit sizes: [60, 40, 40, 60]
  - Scale: 2^40

### Model Architecture

- **Type**: Multi-Layer Perceptron (MLP)
- **Architecture**: Input(416) → Dense(128) → Poly → Dense(64) → Poly → Dense(num_classes)
- **Activations**: Polynomial approximations (scaled quadratic: 0.5*x² + 0.5*x) instead of ReLU
- **Operations**: Only addition, subtraction, and multiplication (HE-friendly)

### Encrypted Inference

The inference pipeline performs:
1. **Encryption**: Encrypt MFCC feature vector using CKKS
2. **Encrypted Forward Pass**: 
   - Matrix-vector multiplications in encrypted domain
   - Polynomial activations in encrypted domain
3. **Decryption**: Decrypt final logits
4. **Classification**: Apply softmax and predict class (plaintext)

## Evaluation Metrics

### Accuracy
- Compare HE inference accuracy vs plaintext baseline
- Measure prediction agreement between HE and plaintext models

### Latency
- Encryption time
- Inference time (encrypted operations)
- Decryption time
- Total end-to-end latency

### Robustness
- Test performance across different noise levels (SNR: 20dB, 10dB, 5dB)
- Compare HE vs plaintext model degradation

## Expected Results

Based on the implementation:

- **Accuracy**: HE inference should achieve within 5% of plaintext baseline
- **Latency**: Total inference time (encryption + inference + decryption) typically ranges from 5-15 seconds per sample
- **Model Performance**: Plaintext model accuracy >85% on test set

## Limitations and Future Work

### Current Limitations

1. **Matrix Multiplication**: The current implementation uses a simplified approach for encrypted matrix multiplication. A fully homomorphic version would use rotations more efficiently.

2. **Latency**: Encrypted inference is significantly slower than plaintext (100-1000x slower), making real-time applications challenging.

3. **Model Size**: HE operations are computationally expensive, limiting model complexity.

4. **Approximation Errors**: CKKS introduces small approximation errors that can accumulate through multiple layers.

### Future Improvements

- Optimize matrix multiplication using efficient rotation-based algorithms
- Implement batch processing for encrypted inference
- Explore quantization techniques to reduce computational overhead
- Investigate hybrid approaches (partial encryption)
- Optimize CKKS parameters for better performance/security trade-off

## Dataset

This project uses the [Google Speech Commands Dataset v0.02](https://www.tensorflow.org/datasets/catalog/speech_commands), which contains:
- 35 voice commands (e.g., "yes", "no", "up", "down", "left", "right")
- ~105,000 audio samples
- 16kHz sample rate, 1-second duration

## References

- **CKKS Scheme**: Cheon, J. H., Kim, A., Kim, M., & Song, Y. (2017). Homomorphic encryption for arithmetic of approximate numbers. ASIACRYPT 2017.
- **TenSEAL**: [https://github.com/OpenMined/TenSEAL](https://github.com/OpenMined/TenSEAL)
- **Google Speech Commands**: [https://www.tensorflow.org/datasets/catalog/speech_commands](https://www.tensorflow.org/datasets/catalog/speech_commands)

## License

This project is for research purposes. Please refer to the licenses of dependencies:
- TenSEAL: Apache License 2.0
- PyTorch: BSD-style license
- Google Speech Commands Dataset: CC BY 4.0

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{privacy-preserving-voice-recognition,
  title={Privacy-Preserving Voice Command Recognition Using Approximate Homomorphic Encryption},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/BTP}
}
```

## Contact

For questions or issues, please open an issue on the repository.

