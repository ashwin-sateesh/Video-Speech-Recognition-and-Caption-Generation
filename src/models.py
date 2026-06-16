"""Sequence classification models for lip-reading.

Provides factory functions for building:
    - LSTM classifier
    - LSTM with Attention classifier
    - Transformer classifier (with traditional or learned positional encoding)
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Attention,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    Lambda,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model, Sequential


# ---------------------------------------------------------------------------
# Positional Encoding Layers
# ---------------------------------------------------------------------------

class TraditionalPositionalEncoding(Layer):
    """Sine-cosine positional encoding as described in 'Attention Is All You Need'.

    Generates fixed positional encodings based on sinusoidal functions,
    broadcast across the batch dimension.
    """

    def __init__(self, seq_len: int = 22, max_wavelength: float = 25.0, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.max_wavelength = max_wavelength

    def call(self, inputs, **kwargs):
        input_dim = inputs.shape[-1]
        k = np.arange(0, input_dim, dtype=np.float32)[None, :]
        pos = np.arange(0, self.seq_len, dtype=np.float32)[:, None]

        i = k // 2
        pe = pos / (self.max_wavelength ** (2 * i / input_dim))
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        pe = tf.constant(pe, dtype=tf.float32)
        pe = tf.expand_dims(pe, axis=0)  # (1, seq_len, input_dim)
        return pe

    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len, "max_wavelength": self.max_wavelength})
        return config


class LearnedPositionalEncoding(Layer):
    """Learned positional encoding using 1D convolutions.

    Applies two Conv1D layers followed by a Dense projection to learn
    position-dependent representations from the input features.
    """

    def __init__(self, filters: int = 128, kernel_size: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv1 = Conv1D(filters=filters, kernel_size=kernel_size, padding="same", activation="relu")
        self.conv2 = Conv1D(filters=filters, kernel_size=1, activation="linear")
        self.dense = Dense(units=1, activation=None)

    def call(self, inputs, **kwargs):
        input_dim = inputs.shape[-1]
        pe = inputs * tf.math.sqrt(tf.cast(input_dim, tf.float32))
        pe = Lambda(lambda x: tf.expand_dims(x, axis=-1))(pe)
        pe = self.conv1(pe)
        pe = self.conv2(pe)
        pe = self.dense(pe)
        pe = tf.reshape(pe, tf.shape(inputs))
        return pe

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size})
        return config


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

def _transformer_block(
    inputs: tf.Tensor,
    key_dim: int,
    num_heads: int,
    ff_dim: int,
    dropout: float,
    name: str,
) -> tf.Tensor:
    """Single Transformer encoder block with multi-head attention and FFN.

    Args:
        inputs: Input tensor of shape (batch, seq_len, d_model).
        key_dim: Dimensionality of attention keys.
        num_heads: Number of attention heads.
        ff_dim: Hidden dimension of the feed-forward network.
        dropout: Dropout rate.
        name: Name prefix for layers.

    Returns:
        Output tensor of the same shape as inputs.
    """
    attn_out = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"{name}_mha"
    )(inputs, inputs)
    attn_out = Dropout(dropout)(attn_out)
    out1 = LayerNormalization()(inputs + attn_out)

    ffn = Sequential(
        [Dense(ff_dim, activation="relu"), Dense(inputs.shape[-1])],
        name=f"{name}_ffn",
    )
    ffn_out = ffn(out1)
    ffn_out = Dropout(dropout)(ffn_out)
    return LayerNormalization()(out1 + ffn_out)


# ---------------------------------------------------------------------------
# Model factory functions
# ---------------------------------------------------------------------------

def build_lstm_model(
    input_shape: tuple,
    hidden_units: int = 128,
    num_classes: int = 10,
) -> Model:
    """Build a simple LSTM classifier.

    Args:
        input_shape: (seq_len, feature_dim).
        hidden_units: Number of LSTM hidden units.
        num_classes: Number of output classes.

    Returns:
        Compiled-ready Keras Model.
    """
    model = Sequential([
        LSTM(units=hidden_units, input_shape=input_shape),
        Dense(units=num_classes, activation="softmax"),
    ], name="lstm_classifier")
    return model


def build_lstm_attention_model(
    input_shape: tuple,
    hidden_units: int = 128,
    num_classes: int = 10,
) -> Model:
    """Build an LSTM with Attention classifier.

    Uses Keras built-in Attention layer over LSTM hidden states.

    Args:
        input_shape: (seq_len, feature_dim).
        hidden_units: Number of LSTM hidden units.
        num_classes: Number of output classes.

    Returns:
        Keras Model.
    """
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(hidden_units, return_sequences=True)(inputs)
    attn_out = Attention()([lstm_out, lstm_out])
    flat = Flatten()(attn_out)
    outputs = Dense(num_classes, activation="softmax")(flat)
    return Model(inputs=inputs, outputs=outputs, name="lstm_attention_classifier")


def build_transformer_model(
    input_shape: tuple,
    num_transformer_blocks: int = 2,
    key_dim: int = 64,
    num_heads: int = 16,
    ff_dim: int = 768,
    dropout: float = 0.1,
    num_classes: int = 10,
    positional_encoding: Literal["traditional", "learned", "none"] = "traditional",
) -> Model:
    """Build a Transformer encoder classifier for sequence classification.

    Args:
        input_shape: (seq_len, feature_dim).
        num_transformer_blocks: Number of stacked Transformer encoder blocks.
        key_dim: Attention key dimensionality.
        num_heads: Number of attention heads.
        ff_dim: Feed-forward hidden dimension.
        dropout: Dropout rate.
        num_classes: Number of output classes.
        positional_encoding: Type of positional encoding to use.

    Returns:
        Keras Model.
    """
    inputs = Input(shape=input_shape)

    if positional_encoding == "traditional":
        pe = TraditionalPositionalEncoding(seq_len=input_shape[0])(inputs)
    elif positional_encoding == "learned":
        pe = LearnedPositionalEncoding()(inputs)
    else:
        pe = tf.zeros_like(inputs)

    pe = tf.cast(pe, dtype=tf.float32)
    x = inputs + pe

    for i in range(num_transformer_blocks):
        x = x + _transformer_block(
            x,
            key_dim=key_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name=f"transformer_{i + 1}",
        )

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs, name="transformer_classifier")
