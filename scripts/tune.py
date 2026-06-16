#!/usr/bin/env python3
"""Hyperparameter tuning for the Transformer classifier using Keras Tuner.

Usage:
    python scripts/tune.py \
        --features ./data/miracl_feature_maps_resnet50_words.pkl \
        --max-epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import TrainingConfig
from src.feature_extraction import load_features
from src.models import build_transformer_model
from src.training import prepare_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Transformer")
    parser.add_argument("--features", type=str, required=True, help="Path to feature pickle file")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = TrainingConfig()

    print(f"Loading features from: {args.features}")
    features, labels = load_features(args.features)

    X_train, X_test, y_train, y_test = prepare_data(
        features, labels, num_classes=10,
        test_size=train_cfg.test_size,
        random_state=train_cfg.random_state,
    )
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"  Input shape: {input_shape}")

    def create_model(hp):
        num_transformers = hp.Int("num_transformers", 2, 4, step=1)
        num_heads = hp.Int("num_heads", 4, 16, step=4)
        ff_dim = hp.Int("ff_dim", 256, 1024, step=256)
        key_dim = hp.Int("key_dim", 32, 128, step=32)
        learning_rate = hp.Choice("learning_rate", [1e-2, 5e-3, 1e-3])
        optimizer_name = hp.Choice("optimizer", ["adam", "sgd"])

        model = build_transformer_model(
            input_shape,
            num_transformer_blocks=num_transformers,
            key_dim=key_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=0.1,
            num_classes=10,
            positional_encoding="traditional",
        )

        if optimizer_name == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    tuner = kt.Hyperband(
        create_model,
        objective="val_accuracy",
        max_epochs=args.max_epochs,
        factor=args.factor,
    )

    print("\nSearch space:")
    tuner.search_space_summary()

    print("\nRunning hyperparameter search...")
    tuner.search(
        X_train, y_train,
        epochs=args.max_epochs,
        validation_data=(X_test, y_test),
        batch_size=args.batch_size,
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest hyperparameters:")
    for key, val in best_hps.values.items():
        print(f"  {key}: {val}")

    best_model = tuner.get_best_models(num_models=1)[0]
    loss, acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nBest model accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
