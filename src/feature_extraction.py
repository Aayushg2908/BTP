"""
MFCC (Mel-Frequency Cepstral Coefficients) feature extraction
"""

import numpy as np
import librosa
import torch
from typing import Tuple, Optional
from src.utils import normalize_features


# Default MFCC parameters
DEFAULT_N_MFCC = 13
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_MELS = 40
DEFAULT_SR = 16000  # Google Speech Commands uses 16kHz
DEFAULT_N_FRAMES = 32  # Target number of time frames


def extract_mfcc(
    audio: np.ndarray,
    sr: int = DEFAULT_SR,
    n_mfcc: int = DEFAULT_N_MFCC,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    n_frames: int = DEFAULT_N_FRAMES,
    normalize: bool = True,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract MFCC features from audio signal.
    
    Args:
        audio: Audio signal as numpy array
        sr: Sample rate (default: 16000)
        n_mfcc: Number of MFCC coefficients (default: 13)
        n_fft: FFT window size (default: 2048)
        hop_length: Hop length for STFT (default: 512)
        n_mels: Number of mel filterbanks (default: 40)
        n_frames: Target number of time frames (default: 32)
        normalize: Whether to normalize features (default: True)
        mean: Pre-computed mean for normalization (optional)
        std: Pre-computed std for normalization (optional)
    
    Returns:
        mfcc_features: MFCC feature array of shape (n_frames, n_mfcc) flattened to (n_frames * n_mfcc,)
        mean: Mean used for normalization (if normalize=True)
        std: Std used for normalization (if normalize=True)
    """
    # Ensure audio is 1D
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=0)
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # mfcc shape: (n_mfcc, n_time_frames)
    # Handle variable length: pad or truncate to n_frames
    n_time_frames = mfcc.shape[1]
    
    if n_time_frames < n_frames:
        # Pad with zeros
        pad_width = n_frames - n_time_frames
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    elif n_time_frames > n_frames:
        # Truncate (take middle portion)
        start_idx = (n_time_frames - n_frames) // 2
        mfcc = mfcc[:, start_idx:start_idx + n_frames]
    
    # Transpose to (n_frames, n_mfcc) and flatten to (n_frames * n_mfcc,)
    mfcc = mfcc.T  # (n_frames, n_mfcc)
    mfcc_flat = mfcc.flatten()  # (n_frames * n_mfcc,)
    
    # Normalize if requested
    if normalize:
        mfcc_flat, mean, std = normalize_features(
            mfcc_flat.reshape(1, -1),
            mean=mean.reshape(1, -1) if mean is not None else None,
            std=std.reshape(1, -1) if std is not None else None
        )
        mfcc_flat = mfcc_flat.flatten()
        mean = mean.flatten() if mean is not None else None
        std = std.flatten() if std is not None else None
    else:
        mean = None
        std = None
    
    return mfcc_flat, mean, std


def extract_mfcc_batch(
    audio_batch: np.ndarray,
    sr: int = DEFAULT_SR,
    n_mfcc: int = DEFAULT_N_MFCC,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    n_frames: int = DEFAULT_N_FRAMES,
    normalize: bool = True,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract MFCC features from a batch of audio signals.
    
    Args:
        audio_batch: Batch of audio signals, shape (batch_size, audio_length)
        sr: Sample rate (default: 16000)
        n_mfcc: Number of MFCC coefficients (default: 13)
        n_fft: FFT window size (default: 2048)
        hop_length: Hop length for STFT (default: 512)
        n_mels: Number of mel filterbanks (default: 40)
        n_frames: Target number of time frames (default: 32)
        normalize: Whether to normalize features (default: True)
        mean: Pre-computed mean for normalization (optional)
        std: Pre-computed std for normalization (optional)
    
    Returns:
        mfcc_batch: MFCC features, shape (batch_size, n_frames * n_mfcc)
        mean: Mean used for normalization
        std: Std used for normalization
    """
    batch_size = audio_batch.shape[0]
    feature_dim = n_frames * n_mfcc
    
    mfcc_batch = np.zeros((batch_size, feature_dim))
    
    for i in range(batch_size):
        mfcc_batch[i], _, _ = extract_mfcc(
            audio_batch[i],
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            n_frames=n_frames,
            normalize=False  # Normalize after batch extraction
        )
    
    # Normalize the entire batch
    if normalize:
        mfcc_batch, mean, std = normalize_features(mfcc_batch, mean=mean, std=std)
    else:
        mean = np.zeros(feature_dim)
        std = np.ones(feature_dim)
    
    return mfcc_batch, mean, std


def get_feature_dim(
    n_mfcc: int = DEFAULT_N_MFCC,
    n_frames: int = DEFAULT_N_FRAMES
) -> int:
    """Get the feature dimension."""
    return n_mfcc * n_frames

