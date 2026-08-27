"""
Image Normalization & Tensor Conversion Module
Implements Min-Max scaling [0, 1] and Channel-wise Z-Score standardization.
"""

import numpy as np
from PIL import Image


def images_to_numpy(images: list) -> np.ndarray:
    """Converts a list of PIL Images into a 4D NumPy array of shape (N, H, W, C)."""
    return np.stack([np.array(img, dtype=np.float32) for img in images], axis=0)


def normalize_minmax(tensor: np.ndarray) -> np.ndarray:
    """Scales pixel values from [0, 255] to [0.0, 1.0]."""
    return tensor / 255.0


def normalize_standard(tensor: np.ndarray, mean=None, std=None) -> tuple:
    """
    Applies channel-wise Z-score standardization: (X - mean) / std.
    If mean and std are not provided, computes them across the batch.
    """
    if mean is None:
        mean = np.mean(tensor, axis=(0, 1, 2), keepdims=True)
    if std is None:
        std = np.std(tensor, axis=(0, 1, 2), keepdims=True)
    # Avoid zero division
    std = np.where(std == 0, 1e-7, std)

    normalized = (tensor - mean) / std
    return normalized, mean, std


def flatten_images(tensor: np.ndarray) -> np.ndarray:
    """Flattens 4D image tensor (N, H, W, C) into 2D feature matrix (N, H*W*C)."""
    n_samples = tensor.shape[0]
    return tensor.reshape(n_samples, -1)
