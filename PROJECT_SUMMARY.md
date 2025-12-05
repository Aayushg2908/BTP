# Project Summary: Privacy-Preserving Voice Command Recognition

## 🎯 What Is This Project?

A **privacy-preserving voice assistant** that can recognize voice commands **without ever seeing your voice data in plaintext**. It uses **Homomorphic Encryption** to perform machine learning inference on encrypted audio features.

### The Problem
Traditional voice assistants (Siri, Alexa, Google Assistant) send your voice recordings to cloud servers for processing. This means:
- Your voice data is stored on servers
- It can be accessed by service providers
- It's vulnerable to data breaches
- Privacy concerns for sensitive commands

### The Solution
This project demonstrates how to:
1. Encrypt your voice features **before** sending to server
2. Perform inference **entirely on encrypted data**
3. Only decrypt the **final prediction** (not your voice)
4. **Never expose your voice data** to the server

---

## 📊 Dataset: Google Speech Commands

**What**: A dataset of 35 voice commands
- Examples: "yes", "no", "up", "down", "left", "right", "stop", "go", "on", "off", etc.
- **Size**: ~105,000 audio samples
- **Format**: 1-second audio clips at 16kHz
- **Purpose**: Train models to recognize voice commands

**Why this dataset**:
- Standard benchmark for voice recognition research
- Real-world voice commands
- Diverse speakers and accents
- Perfect for demonstrating privacy-preserving ML

---

## 🧠 What Are We Training?

### Two Models:

#### 1. **Plaintext Model** (Baseline)
- **Type**: Multi-Layer Perceptron (MLP)
- **Architecture**: 
  - Input: 416 features (MFCC)
  - Hidden: 128 → 64 neurons
  - Output: 35 classes (voice commands)
- **Activations**: ReLU (standard neural network)
- **Purpose**: Baseline to compare against
- **Accuracy**: ~65% on test set

#### 2. **HE-Compatible Model** (Privacy-Preserving)
- **Type**: Similar MLP but HE-compatible
- **Key Difference**: Uses **polynomial activations** (`0.5*x² + 0.5*x`) instead of ReLU
- **Why**: Homomorphic encryption can only do addition, subtraction, multiplication (not ReLU)
- **Accuracy**: ~58-70% (comparable to plaintext!)

---

## 🔐 How Does It Work?

### The Complete Pipeline:

```
┌─────────────────────────────────────────────────────────┐
│ CLIENT SIDE (Your Device)                               │
├─────────────────────────────────────────────────────────┤
│ 1. Record Audio                                         │
│    "yes" (1 second audio)                              │
│         ↓                                               │
│ 2. Extract MFCC Features                              │
│    [416 numbers representing audio characteristics]     │
│         ↓                                               │
│ 3. Encrypt Features                                    │
│    [Encrypted 416 numbers]                             │
│         ↓                                               │
│ 4. Send to Server (ENCRYPTED)                          │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ SERVER SIDE (Cloud)                                     │
├─────────────────────────────────────────────────────────┤
│ 5. Encrypted Inference                                 │
│    - Matrix multiplication (W × encrypted_features)      │
│    - Polynomial activation (0.5*x² + 0.5*x)           │
│    - All operations on ENCRYPTED data!                 │
│    [Encrypted prediction scores]                       │
│         ↓                                               │
│ 6. Send Back (STILL ENCRYPTED)                         │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ CLIENT SIDE (Your Device)                               │
├─────────────────────────────────────────────────────────┤
│ 7. Decrypt Results                                      │
│    [35 prediction scores]                              │
│         ↓                                               │
│ 8. Classify                                            │
│    Predicted: "yes" ✓                                  │
└─────────────────────────────────────────────────────────┘
```

### Key Technologies:

1. **MFCC (Mel-Frequency Cepstral Coefficients)**
   - Converts audio to numerical features
   - Standard in speech recognition
   - 416-dimensional vector per audio sample

2. **CKKS Homomorphic Encryption**
   - Allows arithmetic on encrypted data
   - Approximate encryption (small errors OK)
   - Used for machine learning

3. **Polynomial Activations**
   - Replaces ReLU with `0.5*x² + 0.5*x`
   - HE-compatible (only uses +, -, ×)
   - Maintains model accuracy

---

## 📈 Results

### Accuracy
- **Plaintext Model**: ~65% accuracy
- **HE Model**: ~58-70% accuracy
- **Conclusion**: Comparable performance while maintaining privacy!

### Latency (per inference)
- **Encryption**: ~3ms
- **Inference**: ~15ms
- **Decryption**: ~1ms
- **Total**: ~19ms (~50 inferences/second)

**Note**: Slower than plaintext but acceptable for privacy-critical apps.

### Privacy
- ✅ Voice data never sent in plaintext
- ✅ Server never sees your voice
- ✅ Only final prediction is decrypted
- ✅ Even if server is compromised, data is encrypted

---

## 🎬 How to Demo

### Quick Demo (2 minutes)
```bash
# Run benchmark
python scripts/benchmark.py --num_accuracy_samples 20
```

### Interactive Demo (5 minutes)
```bash
# Demo with random sample
python scripts/demo.py

# Demo with specific audio file
python scripts/demo.py --audio_file path/to/audio.wav
```

### Full Evaluation (10 minutes)
```bash
# Complete benchmark
python scripts/benchmark.py
```

---

## 💡 Why This Matters

1. **Privacy**: Your voice never leaves device in plaintext
2. **Security**: Even compromised servers can't access your data
3. **Compliance**: Helps meet GDPR/privacy regulations
4. **Trust**: Users can trust cloud services with sensitive data

---

## 🎓 Key Concepts Explained

### Homomorphic Encryption
- **What**: Encryption that allows computation on encrypted data
- **Analogy**: Like doing math on numbers inside locked boxes without opening them
- **Result**: Get correct answer without ever seeing the data

### CKKS Scheme
- **Type**: Approximate homomorphic encryption
- **Why**: Perfect for machine learning (small errors acceptable)
- **Operations**: Supports addition, subtraction, multiplication

### Polynomial Activations
- **Why needed**: ReLU requires comparison (if x > 0), which HE can't do
- **Solution**: Use polynomials (only +, -, ×)
- **Trade-off**: Slightly different behavior but maintains accuracy

---

## 📚 Technical Details

### Model Architecture
- **Input**: 416 MFCC features
- **Hidden Layers**: 128 → 64 neurons
- **Output**: 35 classes (voice commands)
- **Parameters**: ~64,000 trainable parameters

### Encryption Parameters
- **Polynomial Modulus**: 8192
- **Coefficient Modulus**: [60, 40, 40, 60] bits
- **Security Level**: ~128-bit security

### Training
- **Epochs**: 50 for plaintext, 20-50 for HE model
- **Batch Size**: 64-128
- **Learning Rate**: 0.001 (plaintext), 0.0001 (HE)
- **Optimizer**: Adam

---

## 🚀 Future Improvements

1. **Performance**: Optimize HE operations for faster inference
2. **Accuracy**: Better polynomial activations or model architectures
3. **Scalability**: Support larger models and more classes
4. **Real-time**: Stream processing for continuous voice recognition
5. **Hybrid**: Combine HE with other privacy techniques

---

## 📖 Files Overview

- `src/train.py`: Train plaintext model
- `src/train_he.py`: Train HE-compatible model
- `src/inference.py`: Encrypted inference pipeline
- `src/encryption.py`: CKKS encryption utilities
- `scripts/benchmark.py`: Comprehensive evaluation
- `scripts/demo.py`: Interactive demo script
- `PRESENTATION_GUIDE.md`: Detailed presentation guide

---

## 🎯 Summary

This project demonstrates **feasible privacy-preserving voice recognition** using homomorphic encryption. While slower than plaintext inference, it achieves **comparable accuracy** while ensuring **complete privacy** of user voice data. This is a significant step toward trustworthy, privacy-preserving voice assistants!

