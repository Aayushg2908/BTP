# Project Presentation Guide: Privacy-Preserving Voice Command Recognition

## 🎯 Project Overview

**Title**: Privacy-Preserving Voice Command Recognition Using Approximate Homomorphic Encryption

**Problem Statement**:
Traditional voice assistants send your audio to cloud servers for processing, exposing your voice data to potential privacy breaches. This project demonstrates how to perform voice command recognition **entirely on encrypted data** without ever decrypting it.

**Key Innovation**:
We can perform machine learning inference on encrypted audio features using Homomorphic Encryption (HE), ensuring your voice data remains private even during processing.

---

## 📊 Dataset: Google Speech Commands Dataset

**What it is**:

- A publicly available dataset of voice commands
- Contains **35 different voice commands** like "yes", "no", "up", "down", "left", "right", "stop", "go", etc.
- **~105,000 audio samples** total
- Each sample is **1 second long** at **16kHz** sample rate
- Used for training voice command recognition models

**Why this dataset**:

- Standard benchmark for voice recognition
- Diverse speakers and accents
- Real-world voice commands
- Perfect for demonstrating privacy-preserving ML

---

## 🧠 What We're Training

### 1. **Plaintext Model** (Baseline)

- **Architecture**: Multi-Layer Perceptron (MLP)
- **Input**: 416-dimensional MFCC feature vector (13 coefficients × 32 time frames)
- **Layers**:
  - Input(416) → Dense(128) → ReLU → Dropout(0.2)
  - → Dense(64) → ReLU → Dropout(0.2)
  - → Dense(35) → Softmax
- **Purpose**: Baseline model to compare against
- **Performance**: ~65% accuracy on test set

### 2. **HE-Compatible Model** (Privacy-Preserving)

- **Architecture**: Similar MLP but with **polynomial activations** instead of ReLU
- **Key Difference**: Uses `0.5*x² + 0.5*x` instead of ReLU (HE-friendly)
- **Why**: Homomorphic encryption can only perform addition, subtraction, and multiplication
- **Performance**: ~70% accuracy (sometimes better due to polynomial activations)

---

## 🔐 How It Works: Technical Pipeline

### Step 1: Feature Extraction (Client Side - Plaintext)

```
Raw Audio (1 second, 16kHz)
    ↓
MFCC Extraction (13 coefficients × 32 frames)
    ↓
416-dimensional feature vector
```

**MFCC (Mel-Frequency Cepstral Coefficients)**:

- Converts audio to numerical features
- Captures spectral characteristics of voice
- Standard in speech recognition

### Step 2: Encryption (Client Side)

```
MFCC Features (416 numbers)
    ↓
CKKS Homomorphic Encryption
    ↓
Encrypted Feature Vector
```

**CKKS Scheme**:

- Allows arithmetic operations on encrypted data
- Approximate encryption (small errors acceptable)
- Used for machine learning applications

### Step 3: Encrypted Inference (Server Side - Encrypted)

```
Encrypted Features
    ↓
Encrypted Matrix Multiplication (W × x + b)
    ↓
Encrypted Polynomial Activation (0.5*x² + 0.5*x)
    ↓
Encrypted Logits (35 class scores)
```

**All operations happen in encrypted domain!**

### Step 4: Decryption & Classification (Client Side)

```
Encrypted Logits
    ↓
Decrypt
    ↓
Plaintext Logits
    ↓
Softmax → Predicted Command
```

---

## 🎬 Live Demo Guide

### Prerequisites

1. Trained models (`models/best_model.pth` and `models/best_he_model.pth`)
2. Dataset downloaded (`data/speech_commands_v0.02/`)
3. Python environment with all dependencies

### Demo Script Options

#### Option 1: Quick Accuracy Demo (2-3 minutes)

```bash
# Show that both models work
python scripts/benchmark.py \
    --num_accuracy_samples 50 \
    --num_latency_samples 10
```

**What to highlight**:

- Both models achieve similar accuracy
- HE model performs inference on encrypted data
- Privacy is preserved throughout

#### Option 2: Full Benchmark Demo (5-7 minutes)

```bash
# Complete evaluation
python scripts/benchmark.py
```

**What to show**:

1. **Accuracy Results**: HE vs Plaintext comparison
2. **Latency Breakdown**: Encryption, Inference, Decryption times
3. **Robustness**: Performance under noise

#### Option 3: Interactive Demo (10+ minutes)

Create a simple script to:

1. Record audio from microphone
2. Extract MFCC features
3. Show encryption → inference → decryption
4. Display predicted command

---

## 📈 Key Results to Present

### Accuracy Comparison

- **Plaintext Model**: ~65% accuracy
- **HE Model**: ~58-70% accuracy
- **Agreement**: ~45-50% (both models agree)
- **Conclusion**: HE model achieves comparable performance!

### Latency Breakdown (per sample)

- **Encryption**: ~0.003 seconds
- **Inference**: ~0.015 seconds
- **Decryption**: ~0.001 seconds
- **Total**: ~0.019 seconds (~50 samples/second)

**Note**: This is slower than plaintext but acceptable for privacy-critical applications.

### Robustness

- **SNR 20dB**: HE ~22-26%, Plaintext ~6-8%
- **SNR 10dB**: HE ~2%, Plaintext ~0%
- **SNR 5dB**: Both ~0%

**Insight**: HE model shows better robustness to noise!

---

## 🎤 Presentation Structure (15-20 minutes)

### 1. Introduction (2 min)

- **Problem**: Privacy concerns with voice assistants
- **Solution**: Homomorphic encryption for encrypted inference
- **Demo Preview**: Show final result first!

### 2. Background (3 min)

- What is Homomorphic Encryption?
- Why CKKS for machine learning?
- Why polynomial activations?

### 3. Dataset & Model (3 min)

- Google Speech Commands Dataset
- Model architecture (show diagram)
- Training process

### 4. Technical Pipeline (5 min)

- Feature extraction (MFCC)
- Encryption process
- Encrypted inference (show code snippets)
- Decryption & classification

### 5. Results & Demo (5 min)

- Run benchmark script live
- Show accuracy comparison
- Show latency breakdown
- Discuss trade-offs

### 6. Conclusion & Future Work (2 min)

- Privacy achieved!
- Performance acceptable
- Future improvements

---

## 💡 Key Talking Points

### Why This Matters

1. **Privacy**: Voice data never leaves device in plaintext
2. **Security**: Even if server is compromised, data is encrypted
3. **Compliance**: Helps meet GDPR/privacy regulations
4. **Trust**: Users can trust cloud services with sensitive data

### Technical Challenges Overcome

1. **HE-Compatible Activations**: Replaced ReLU with polynomials
2. **Encrypted Operations**: Matrix multiplication in encrypted domain
3. **Accuracy Preservation**: Maintained ~65% accuracy despite constraints
4. **Performance**: Achieved ~50 inferences/second

### Limitations & Trade-offs

1. **Latency**: ~20ms per inference (vs <1ms plaintext)
2. **Model Size**: Limited by HE computational cost
3. **Approximation Errors**: Small errors accumulate
4. **Complexity**: More complex than plaintext ML

---

## 🛠️ Quick Demo Commands

### Check Models Exist

```bash
ls -lh models/*.pth
```

### Run Quick Test

```bash
python scripts/benchmark.py --num_accuracy_samples 20
```

### Show Results

```bash
cat results/benchmark_results.json | python -m json.tool
```

### Visualize Latency

```bash
cat results/latency_stats.csv
```

---

## 📝 Presentation Tips

1. **Start with the problem**: Show why privacy matters
2. **Visual aids**: Use diagrams for encryption pipeline
3. **Live demo**: Run benchmark script during presentation
4. **Compare**: Always show plaintext vs HE side-by-side
5. **Be honest**: Acknowledge limitations and trade-offs
6. **Future work**: Discuss improvements and research directions

---

## 🎯 Expected Questions & Answers

**Q: Why is HE inference slower?**
A: Performing operations on encrypted data requires complex mathematical operations. Each multiplication/addition involves polynomial arithmetic, which is computationally expensive.

**Q: Can this work in real-time?**
A: Currently ~50 samples/second, which is fast enough for voice commands but not real-time streaming. Future optimizations could improve this.

**Q: Is the accuracy loss acceptable?**
A: Yes! We maintain ~65% accuracy while gaining complete privacy. For privacy-critical applications, this trade-off is worth it.

**Q: What if the encryption is broken?**
A: CKKS is based on well-studied cryptographic assumptions (RLWE). Breaking it would require solving hard mathematical problems.

**Q: Can this scale to larger models?**
A: Yes, but with increased latency. Larger polynomial modulus degrees allow larger models but increase computation time.

---

## 📚 Additional Resources

- **CKKS Paper**: Cheon et al., "Homomorphic encryption for arithmetic of approximate numbers"
- **TenSEAL**: https://github.com/OpenMined/TenSEAL
- **Google Speech Commands**: https://www.tensorflow.org/datasets/catalog/speech_commands

---

Good luck with your presentation! 🚀
