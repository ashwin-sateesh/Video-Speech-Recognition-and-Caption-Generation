"""CNN feature extraction using pre-trained VGG16 and ResNet50.

Extracts spatial feature vectors from lip images by passing them through
pre-trained CNNs with the classification head removed.

    - VGG16: 512-dimensional feature vectors (via Global Average Pooling)
    - ResNet50: 2048-dimensional feature vectors (via Global Average Pooling)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_vgg16_extractor(input_shape: Tuple[int, int, int] = (25, 58, 3)) -> Model:
    """Build a VGG16 feature extractor with Global Average Pooling.

    Removes the classification head and applies GAP to produce
    512-dimensional feature vectors.

    Args:
        input_shape: (H, W, C) of the lip images.

    Returns:
        Keras Model that maps images to 512-D feature vectors.
    """
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=base_model.input, outputs=x, name="vgg16_extractor")


def build_resnet50_extractor(input_shape: Tuple[int, int, int] = (25, 58, 3)) -> Model:
    """Build a ResNet50 feature extractor with Global Average Pooling.

    Removes the classification head and applies GAP to produce
    2048-dimensional feature vectors.

    Args:
        input_shape: (H, W, C) of the lip images.

    Returns:
        Keras Model that maps images to 2048-D feature vectors.
    """
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=base_model.input, outputs=x, name="resnet50_extractor")


def extract_features(
    images: np.ndarray,
    extractor: Model,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract features from a batch of images using a CNN extractor.

    Args:
        images: Normalized image array of shape (N, H, W, C) with values in [0, 1].
        extractor: Keras feature extraction model (VGG16 or ResNet50).
        batch_size: Batch size for prediction.

    Returns:
        Feature array of shape (N, feature_dim).
    """
    return extractor.predict(images, batch_size=batch_size, verbose=1)


def extract_sequence_features(
    image_sequences: np.ndarray,
    extractor: Model,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract features from image sequences and reshape to (instances, seq_len, features).

    Args:
        image_sequences: Array of shape (num_instances, seq_len, H, W, C).
        extractor: Keras feature extraction model.
        batch_size: Batch size for prediction.

    Returns:
        Feature array of shape (num_instances, seq_len, feature_dim).
    """
    num_instances, seq_len = image_sequences.shape[:2]
    H, W, C = image_sequences.shape[2:]

    # Flatten instances and frames for batch extraction
    flat_images = image_sequences.reshape(-1, H, W, C)
    flat_features = extract_features(flat_images, extractor, batch_size)

    feature_dim = flat_features.shape[-1]
    return flat_features.reshape(num_instances, seq_len, feature_dim)


# ---------------------------------------------------------------------------
# Pickle I/O for pre-extracted features
# ---------------------------------------------------------------------------

def save_features(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save extracted features and labels to a pickle file.

    Args:
        features: Feature array of shape (N, seq_len, feature_dim).
        labels: Label array of shape (N,).
        output_path: Destination pickle file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"x": features, "y": labels}, f)


def load_features(pickle_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load pre-extracted features and labels from a pickle file.

    Args:
        pickle_path: Path to the pickle file containing {'x': ..., 'y': ...}.

    Returns:
        Tuple of (features, labels) numpy arrays.
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data["x"], data["y"]
