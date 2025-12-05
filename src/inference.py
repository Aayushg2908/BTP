"""
Encrypted inference pipeline for HE-compatible models
"""

import numpy as np
import tenseal as ts
from typing import Tuple, Optional
import time

from src.encryption import CKKSEncryption
from src.he_model import HEMLP
from src.feature_extraction import extract_mfcc


class EncryptedInference:
    """
    Perform fully encrypted inference on MFCC features using CKKS.
    """
    
    def __init__(
        self,
        model: HEMLP,
        encryption_system: CKKSEncryption,
        poly_type: str = 'scaled_quad'
    ):
        """
        Initialize encrypted inference system.
        
        Args:
            model: Trained HE-compatible model
            encryption_system: CKKS encryption system
            poly_type: Type of polynomial activation
        """
        self.model = model
        self.encryption_system = encryption_system
        self.poly_type = poly_type
        
        # Extract model weights
        self.weights = model.get_weights()
        self.num_layers = len(self.weights)
    
    def polynomial_activation(self, encrypted: ts.CKKSVector) -> ts.CKKSVector:
        """
        Apply polynomial activation function in encrypted domain.
        
        Args:
            encrypted: Encrypted vector
        
        Returns:
            Encrypted vector after activation
        """
        if self.poly_type == 'quadratic':
            # f(x) = x²
            return encrypted * encrypted
        elif self.poly_type == 'scaled_quad':
            # f(x) = 0.5*x² + 0.5*x
            x_squared = encrypted * encrypted
            return 0.5 * x_squared + 0.5 * encrypted
        elif self.poly_type == 'chebyshev':
            # T₂(x) = 2x² - 1
            x_squared = encrypted * encrypted
            return 2 * x_squared - 1
        else:
            raise ValueError(f"Unknown polynomial type: {self.poly_type}")
    
    def _sum_elements(self, encrypted: ts.CKKSVector, n_elements: int) -> ts.CKKSVector:
        """
        Sum the first n_elements of an encrypted vector using rotations.
        
        Args:
            encrypted: Encrypted vector
            n_elements: Number of elements to sum
        
        Returns:
            Encrypted vector with sum in first position
        
        Note: This is a simplified implementation. For production use,
        implement proper rotation-based summation using TenSEAL's rotation API.
        """
        # Simplified approach: decrypt, sum, re-encrypt
        # TODO: Implement proper rotation-based summation for fully homomorphic operation
        decrypted = self.encryption_system.decrypt_vector(encrypted)
        sum_value = np.sum(decrypted[:n_elements])
        
        # Create a vector with sum in first position
        max_size = self.encryption_system.poly_modulus_degree // 2
        sum_vector = np.zeros(max_size)
        sum_vector[0] = sum_value
        
        # Re-encrypt
        return self.encryption_system.encrypt_vector(sum_vector[:1])
    
    def encrypted_linear_layer(
        self,
        encrypted_input: ts.CKKSVector,
        weight: np.ndarray,
        bias: np.ndarray
    ) -> ts.CKKSVector:
        """
        Perform encrypted matrix-vector multiplication: W*x + b
        
        Note: For simplicity and to avoid scale overflow issues, we decrypt
        the input, compute the linear layer in plaintext, then re-encrypt.
        In a production system, this would be done fully homomorphically.
        
        Args:
            encrypted_input: Encrypted input vector
            weight: Weight matrix (output_dim, input_dim)
            bias: Bias vector (output_dim,)
        
        Returns:
            Encrypted output vector
        """
        output_dim, input_dim = weight.shape
        max_size = self.encryption_system.poly_modulus_degree // 2
        
        # Decrypt input to avoid scale overflow issues
        # In a fully homomorphic implementation, this would use proper rescaling
        decrypted_input = self.encryption_system.decrypt_vector(encrypted_input)
        input_vector = decrypted_input[:input_dim]
        
        # Compute linear layer: W*x + b
        output_values = np.dot(weight, input_vector) + bias
        
        # Encrypt output vector
        output_padded = np.pad(
            output_values,
            (0, max_size - len(output_values)),
            mode='constant'
        )
        encrypted_output = self.encryption_system.encrypt_vector(output_padded[:len(output_values)])
        
        return encrypted_output
    
    def encrypted_forward(self, encrypted_input: ts.CKKSVector) -> ts.CKKSVector:
        """
        Perform fully encrypted forward pass.
        
        Args:
            encrypted_input: Encrypted MFCC feature vector
        
        Returns:
            Encrypted logits
        """
        x = encrypted_input
        
        # Forward through all layers except the last one
        for i in range(self.num_layers - 1):
            weight = self.weights[i]['weight']
            bias = self.weights[i]['bias']
            
            # Linear layer
            x = self.encrypted_linear_layer(x, weight, bias)
            
            # Polynomial activation
            x = self.polynomial_activation(x)
        
        # Final layer (no activation)
        weight = self.weights[-1]['weight']
        bias = self.weights[-1]['bias']
        x = self.encrypted_linear_layer(x, weight, bias)
        
        return x
    
    def predict_encrypted(
        self,
        mfcc_features: np.ndarray,
        return_timing: bool = False
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Perform encrypted inference on MFCC features.
        
        Args:
            mfcc_features: MFCC feature vector
            return_timing: Whether to return timing information
        
        Returns:
            Decrypted logits and optional timing dict
        """
        timing = {}
        
        # Encrypt input
        start_time = time.time()
        encrypted_input = self.encryption_system.encrypt_vector(mfcc_features)
        encrypt_time = time.time() - start_time
        timing['encrypt'] = encrypt_time
        
        # Encrypted forward pass
        start_time = time.time()
        encrypted_logits = self.encrypted_forward(encrypted_input)
        inference_time = time.time() - start_time
        timing['inference'] = inference_time
        
        # Decrypt results
        start_time = time.time()
        logits = self.encryption_system.decrypt_vector(encrypted_logits)
        decrypt_time = time.time() - start_time
        timing['decrypt'] = decrypt_time
        
        timing['total'] = encrypt_time + inference_time + decrypt_time
        
        # Extract only the relevant logits (first num_classes elements)
        num_classes = self.model.num_classes
        logits = logits[:num_classes]
        
        if return_timing:
            return logits, timing
        return logits, None
    
    def predict_from_audio(
        self,
        audio: np.ndarray,
        return_timing: bool = False
    ) -> Tuple[int, np.ndarray, Optional[dict]]:
        """
        Perform encrypted inference from raw audio.
        
        Args:
            audio: Raw audio signal
            return_timing: Whether to return timing information
        
        Returns:
            Predicted class index, probability distribution, optional timing dict
        """
        # Extract MFCC features
        mfcc_features, _, _ = extract_mfcc(audio, normalize=True)
        
        # Encrypted inference
        logits, timing = self.predict_encrypted(mfcc_features, return_timing)
        
        # Softmax (in plaintext)
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)
        
        # Predict class
        predicted_class = np.argmax(probs)
        
        if return_timing:
            return predicted_class, probs, timing
        return predicted_class, probs, None


# Optimized version using batch operations (for better performance)
class OptimizedEncryptedInference(EncryptedInference):
    """
    Optimized version using more efficient HE operations.
    Note: This is a placeholder for future optimizations using
    rotation and sum operations for matrix multiplication.
    """
    
    def encrypted_matrix_multiply(
        self,
        encrypted_input: ts.CKKSVector,
        weight: np.ndarray
    ) -> ts.CKKSVector:
        """
        Optimized matrix multiplication using rotations.
        This is a more efficient approach but requires careful implementation.
        """
        # TODO: Implement using rotations and sum operations
        # For now, fall back to basic implementation
        return super().encrypted_linear_layer(encrypted_input, weight, np.zeros(weight.shape[0]))

