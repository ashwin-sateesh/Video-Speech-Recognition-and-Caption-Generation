"""Training and evaluation utilities.

Provides functions for compiling models, running training, and
computing evaluation metrics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 10,
    max_seq_len: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pad features, one-hot encode labels, and split into train/test.

    Args:
        features: Feature array. If a list of variable-length sequences,
            they will be zero-padded to ``max_seq_len``.
        labels: Integer label array.
        num_classes: Total number of classes.
        max_seq_len: Pad sequences to this length. If None, uses the
            maximum length found in the data.
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) where labels are one-hot.
    """
    # Pad if features is a list of variable-length arrays
    if isinstance(features, list) or (isinstance(features, np.ndarray) and features.dtype == object):
        if max_seq_len is None:
            max_seq_len = max(len(seq) for seq in features)
        features = pad_sequences(features, maxlen=max_seq_len, padding="pre", dtype="float32")

    # Zero-index labels if needed
    if labels.min() > 0:
        labels = labels - labels.min()

    y_onehot = to_categorical(labels, num_classes=num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        features, y_onehot, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model compilation
# ---------------------------------------------------------------------------

def compile_model(
    model: Model,
    optimizer: str = "adam",
    learning_rate: float = 0.001,
    momentum: float = 0.9,
    clip_value: Optional[float] = None,
) -> Model:
    """Compile a Keras model with the specified optimizer and loss.

    Args:
        model: Keras model to compile.
        optimizer: ``"adam"`` or ``"sgd"``.
        learning_rate: Optimizer learning rate.
        momentum: Momentum for SGD (ignored for Adam).
        clip_value: Gradient clipping value (None to disable).

    Returns:
        The compiled model.
    """
    if optimizer == "adam":
        opt = Adam(learning_rate=learning_rate, clipvalue=clip_value)
    elif optimizer == "sgd":
        opt = SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            clipvalue=clip_value or 0.0,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 50,
    batch_size: int = 128,
    early_stopping_patience: Optional[int] = None,
) -> tf.keras.callbacks.History:
    """Train a compiled Keras model.

    Args:
        model: Compiled Keras model.
        X_train: Training features.
        y_train: Training labels (one-hot).
        X_test: Validation features.
        y_test: Validation labels (one-hot).
        epochs: Maximum training epochs.
        batch_size: Training batch size.
        early_stopping_patience: Patience for early stopping. None to disable.

    Returns:
        Keras History object.
    """
    callbacks = []
    if early_stopping_patience:
        callbacks.append(
            EarlyStopping(
                monitor="val_accuracy",
                patience=early_stopping_patience,
                restore_best_weights=True,
            )
        )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Evaluate a trained model and return classification metrics.

    Args:
        model: Trained Keras model.
        X_test: Test features.
        y_test: Test labels (one-hot encoded).
        class_names: Optional list of class names for the report.

    Returns:
        Dict with 'accuracy', 'report', and 'confusion_matrix'.
    """
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
    }
