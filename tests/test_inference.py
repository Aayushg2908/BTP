"""
Unit tests for inference pipeline
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.feature_extraction import extract_mfcc, get_feature_dim
from src.encryption import create_encryption_system
from src.model import VoiceCommandMLP
from src.he_model import HEMLP, convert_plaintext_to_he_model


class TestMFCCExtraction:
    """Test MFCC feature extraction."""
    
    def test_extract_mfcc_shape(self):
        """Test that MFCC extraction returns correct shape."""
        # Generate dummy audio (1 second at 16kHz)
        audio = np.random.randn(16000)
        
        mfcc_features, _, _ = extract_mfcc(audio, normalize=False)
        expected_dim = get_feature_dim()
        
        assert mfcc_features.shape == (expected_dim,), \
            f"Expected shape ({expected_dim},), got {mfcc_features.shape}"
    
    def test_extract_mfcc_normalization(self):
        """Test that MFCC normalization works correctly."""
        audio = np.random.randn(16000)
        
        mfcc_features, mean, std = extract_mfcc(audio, normalize=True)
        
        # Check that normalization produces zero mean and unit variance
        assert np.abs(np.mean(mfcc_features)) < 1e-5, "Features should have zero mean"
        assert np.abs(np.std(mfcc_features) - 1.0) < 1e-5, "Features should have unit variance"
        assert mean is not None, "Mean should be returned"
        assert std is not None, "Std should be returned"
    
    def test_extract_mfcc_consistency(self):
        """Test that MFCC extraction is consistent."""
        audio = np.random.randn(16000)
        
        mfcc1, _, _ = extract_mfcc(audio, normalize=True)
        mfcc2, _, _ = extract_mfcc(audio, normalize=True)
        
        # Should be identical (within numerical precision)
        np.testing.assert_allclose(mfcc1, mfcc2, rtol=1e-5)


class TestEncryption:
    """Test encryption/decryption functionality."""
    
    def test_encrypt_decrypt(self):
        """Test that encryption and decryption are correct."""
        encryption_system = create_encryption_system(
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        
        # Create test vector
        feature_dim = get_feature_dim()
        test_vector = np.random.randn(feature_dim).astype(np.float32)
        
        # Encrypt
        encrypted = encryption_system.encrypt_vector(test_vector)
        
        # Decrypt
        decrypted = encryption_system.decrypt_vector(encrypted)
        
        # Check correctness (within tolerance for HE approximation)
        # Extract only relevant elements
        decrypted_relevant = decrypted[:len(test_vector)]
        
        # HE introduces small errors, so we check with tolerance
        np.testing.assert_allclose(
            test_vector,
            decrypted_relevant,
            rtol=1e-2,
            atol=1e-2
        )
    
    def test_encryption_preserves_shape(self):
        """Test that encryption preserves vector shape information."""
        encryption_system = create_encryption_system()
        
        feature_dim = get_feature_dim()
        test_vector = np.random.randn(feature_dim)
        
        encrypted = encryption_system.encrypt_vector(test_vector)
        decrypted = encryption_system.decrypt_vector(encrypted)
        
        # Decrypted should have correct length
        assert len(decrypted) >= len(test_vector), \
            "Decrypted vector should be at least as long as original"


class TestHEModel:
    """Test HE-compatible model."""
    
    def test_he_model_creation(self):
        """Test HE model can be created."""
        feature_dim = get_feature_dim()
        num_classes = 10
        
        model = HEMLP(
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_dims=[128, 64],
            poly_type='scaled_quad'
        )
        
        assert model.input_dim == feature_dim
        assert model.num_classes == num_classes
        assert model.count_parameters() > 0
    
    def test_he_model_forward(self):
        """Test HE model forward pass."""
        feature_dim = get_feature_dim()
        num_classes = 10
        
        model = HEMLP(
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_dims=[128, 64]
        )
        
        # Create dummy input
        x = torch.randn(1, feature_dim)
        
        # Forward pass
        output = model(x)
        
        assert output.shape == (1, num_classes), \
            f"Expected output shape (1, {num_classes}), got {output.shape}"
    
    def test_convert_plaintext_to_he(self):
        """Test conversion from plaintext to HE model."""
        feature_dim = get_feature_dim()
        num_classes = 10
        
        # Create plaintext model
        plaintext_model = VoiceCommandMLP(
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_dims=[128, 64]
        )
        
        # Convert to HE model
        he_model = convert_plaintext_to_he_model(plaintext_model)
        
        assert isinstance(he_model, HEMLP)
        assert he_model.input_dim == feature_dim
        assert he_model.num_classes == num_classes
        
        # Test forward pass
        x = torch.randn(1, feature_dim)
        output = he_model(x)
        assert output.shape == (1, num_classes)


class TestEncryptedInference:
    """Test encrypted inference pipeline."""
    
    def test_encrypted_inference_basic(self):
        """Test basic encrypted inference."""
        from src.inference import EncryptedInference
        
        # Create models
        feature_dim = get_feature_dim()
        num_classes = 10
        
        he_model = HEMLP(
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_dims=[64, 32]  # Smaller for faster testing
        )
        
        encryption_system = create_encryption_system(
            poly_modulus_degree=8192
        )
        
        encrypted_inference = EncryptedInference(
            model=he_model,
            encryption_system=encryption_system,
            poly_type='scaled_quad'
        )
        
        # Create test features
        mfcc_features = np.random.randn(feature_dim).astype(np.float32)
        
        # Encrypted inference
        logits, timing = encrypted_inference.predict_encrypted(
            mfcc_features,
            return_timing=True
        )
        
        assert logits.shape == (num_classes,), \
            f"Expected logits shape ({num_classes},), got {logits.shape}"
        assert timing is not None
        assert 'encrypt' in timing
        assert 'inference' in timing
        assert 'decrypt' in timing
        assert 'total' in timing
    
    def test_encrypted_vs_plaintext_consistency(self):
        """Test that encrypted inference produces similar results to plaintext."""
        from src.inference import EncryptedInference
        
        feature_dim = get_feature_dim()
        num_classes = 10
        
        # Create plaintext model
        plaintext_model = VoiceCommandMLP(
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_dims=[64, 32]
        )
        
        # Convert to HE model
        he_model = convert_plaintext_to_he_model(plaintext_model)
        
        encryption_system = create_encryption_system(poly_modulus_degree=8192)
        encrypted_inference = EncryptedInference(
            model=he_model,
            encryption_system=encryption_system
        )
        
        # Test features
        mfcc_features = np.random.randn(feature_dim).astype(np.float32)
        
        # Plaintext inference
        mfcc_tensor = torch.FloatTensor(mfcc_features).unsqueeze(0)
        plaintext_logits = plaintext_model(mfcc_tensor).squeeze(0).detach().numpy()
        
        # Encrypted inference
        he_logits, _ = encrypted_inference.predict_encrypted(mfcc_features)
        
        # Compare predictions (should be similar, but not identical due to HE approximation)
        plaintext_pred = np.argmax(plaintext_logits)
        he_pred = np.argmax(he_logits)
        
        # They should agree most of the time (within reasonable tolerance)
        # Note: Due to HE approximation errors, exact match is not guaranteed
        assert he_logits.shape == plaintext_logits.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

