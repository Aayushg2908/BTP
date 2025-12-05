"""
TenSEAL CKKS encryption/decryption utilities for MFCC vectors
"""

import numpy as np
import tenseal as ts
from typing import Tuple, Optional
import pickle
from pathlib import Path


class CKKSEncryption:
    """
    CKKS encryption wrapper for homomorphic encryption operations.
    """
    
    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: list = [60, 40, 40, 60],
        scale: float = 2**40,
        context_path: Optional[str] = None
    ):
        """
        Initialize CKKS context.
        
        Args:
            poly_modulus_degree: Polynomial modulus degree (8192 or 16384)
            coeff_mod_bit_sizes: Coefficient modulus bit sizes
            scale: Scale for encoding (default: 2^40)
            context_path: Optional path to load context from file
        """
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.scale = scale
        
        if context_path and Path(context_path).exists():
            self.load_context(context_path)
        else:
            self._create_context()
    
    def _create_context(self):
        """Create CKKS context and keys."""
        # Create context
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes
        )
        
        # Set global scale
        self.context.global_scale = self.scale
        
        # Generate keys
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()
        
        # Create secret key (for decryption)
        self.secret_key = self.context.secret_key()
    
    def save_context(self, context_path: str):
        """Save context to file."""
        with open(context_path, 'wb') as f:
            f.write(self.context.serialize(save_secret_key=True))
    
    def load_context(self, context_path: str):
        """Load context from file."""
        with open(context_path, 'rb') as f:
            self.context = ts.context_from(f.read())
        self.secret_key = self.context.secret_key()
    
    def encrypt_vector(self, vector: np.ndarray) -> ts.CKKSVector:
        """
        Encrypt a vector using CKKS.
        
        Args:
            vector: Input vector as numpy array
        
        Returns:
            Encrypted CKKS vector
        """
        # Ensure vector is 1D
        if len(vector.shape) > 1:
            vector = vector.flatten()
        
        # Pad vector to match ciphertext size if needed
        max_size = self.poly_modulus_degree // 2
        if len(vector) > max_size:
            raise ValueError(f"Vector size {len(vector)} exceeds maximum {max_size}")
        
        # Pad with zeros if necessary
        if len(vector) < max_size:
            vector = np.pad(vector, (0, max_size - len(vector)), mode='constant')
        
        # Encrypt
        encrypted = ts.ckks_vector(self.context, vector.tolist())
        return encrypted
    
    def decrypt_vector(self, encrypted: ts.CKKSVector) -> np.ndarray:
        """
        Decrypt a CKKS vector.
        
        Args:
            encrypted: Encrypted CKKS vector
        
        Returns:
            Decrypted vector as numpy array
        """
        decrypted = encrypted.decrypt(self.secret_key)
        return np.array(decrypted)
    
    def get_public_context(self) -> bytes:
        """
        Get public context (without secret key) for client-side encryption.
        
        Returns:
            Serialized public context
        """
        return self.context.serialize(save_secret_key=False)
    
    @staticmethod
    def create_public_context(public_context_bytes: bytes) -> ts.Context:
        """
        Create a public context from serialized bytes (for client-side encryption).
        
        Args:
            public_context_bytes: Serialized public context
        
        Returns:
            Public context (without secret key)
        """
        return ts.context_from(public_context_bytes)


class ClientEncryption:
    """
    Client-side encryption wrapper (for scenarios where client encrypts data).
    Uses public context without secret key.
    """
    
    def __init__(self, public_context_bytes: bytes, scale: float = 2**40):
        """
        Initialize client encryption with public context.
        
        Args:
            public_context_bytes: Serialized public context
            scale: Scale for encoding
        """
        self.context = CKKSEncryption.create_public_context(public_context_bytes)
        self.context.global_scale = scale
        self.scale = scale
    
    def encrypt_vector(self, vector: np.ndarray) -> ts.CKKSVector:
        """
        Encrypt a vector using public context.
        
        Args:
            vector: Input vector as numpy array
        
        Returns:
            Encrypted CKKS vector
        """
        # Ensure vector is 1D
        if len(vector.shape) > 1:
            vector = vector.flatten()
        
        # Pad vector to match ciphertext size if needed
        max_size = self.context.poly_modulus_degree() // 2
        if len(vector) > max_size:
            raise ValueError(f"Vector size {len(vector)} exceeds maximum {max_size}")
        
        # Pad with zeros if necessary
        if len(vector) < max_size:
            vector = np.pad(vector, (0, max_size - len(vector)), mode='constant')
        
        # Encrypt
        encrypted = ts.ckks_vector(self.context, vector.tolist())
        return encrypted


def create_encryption_system(
    poly_modulus_degree: int = 8192,
    coeff_mod_bit_sizes: list = [60, 40, 40, 60],
    scale: float = 2**40,
    context_path: Optional[str] = None
) -> CKKSEncryption:
    """
    Create and return a CKKS encryption system.
    
    Args:
        poly_modulus_degree: Polynomial modulus degree
        coeff_mod_bit_sizes: Coefficient modulus bit sizes
        scale: Scale for encoding
        context_path: Optional path to load context from
    
    Returns:
        CKKSEncryption instance
    """
    return CKKSEncryption(
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        scale=scale,
        context_path=context_path
    )

