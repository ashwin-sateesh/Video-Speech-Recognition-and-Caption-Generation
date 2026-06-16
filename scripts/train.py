#!/usr/bin/env python3
"""Train sequence models on pre-extracted lip-reading features.

Usage:
    python scripts/train.py \
        --features ./data/miracl_feature_maps_vgg16_words.pkl \
        --model lstm_attention \
        --output-dir ./artifacts

    python scripts/train.py \
        --features ./data/miracl_feature_maps_resnet50_words.pkl \
        --model transformer \
        --pos-encoding traditional \
        --output-dir ./artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import LSTMConfig, TrainingConfig, TransformerConfig
from src.feature_extraction import load_features
from src.models import (
    build_lstm_attention_model,
    build_lstm_model,
    build_transformer_model,
)
from src.training import compile_model, evaluate_model, prepare_data, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lip-reading sequence models")
    parser.add_argument("--features", type=str, required=True, help="Path to feature pickle file")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["lstm", "lstm_attention", "transformer"],
        help="Model architecture",
    )
    parser.add_argument("--pos-encoding", type=str, default="traditional", choices=["traditional", "learned", "none"])
    parser.add_argument("--output-dir", type=str, default="./artifacts")
    parser.add_argument("--epochs", type=int, default=None, help="Override default epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override default batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = TrainingConfig()

    print(f"Loading features from: {args.features}")
    features, labels = load_features(args.features)
    print(f"  Shape: {features.shape if isinstance(features, np.ndarray) else 'variable-length'}")
    print(f"  Classes: {len(np.unique(labels))}")

    # Determine input shape
    if isinstance(features, np.ndarray) and features.ndim == 3:
        seq_len, feat_dim = features.shape[1], features.shape[2]
    else:
        seq_len = max(len(seq) for seq in features)
        feat_dim = features[0].shape[-1] if hasattr(features[0], 'shape') else len(features[0][0])

    input_shape = (seq_len, feat_dim)
    print(f"  Input shape: {input_shape}")

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(
        features, labels,
        num_classes=10,
        max_seq_len=seq_len,
        test_size=train_cfg.test_size,
        random_state=train_cfg.random_state,
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Build model
    print(f"\nBuilding {args.model} model...")
    if args.model == "lstm":
        cfg = LSTMConfig()
        model = build_lstm_model(input_shape, cfg.hidden_units, cfg.num_classes)
        model = compile_model(model, optimizer=cfg.optimizer, learning_rate=0.001)
        epochs = args.epochs or cfg.epochs
        batch_size = args.batch_size or cfg.batch_size

    elif args.model == "lstm_attention":
        cfg = LSTMConfig()
        model = build_lstm_attention_model(input_shape, cfg.hidden_units, cfg.num_classes)
        model = compile_model(model, optimizer=cfg.optimizer, learning_rate=0.001)
        epochs = args.epochs or cfg.epochs
        batch_size = args.batch_size or cfg.batch_size

    elif args.model == "transformer":
        cfg = TransformerConfig()
        model = build_transformer_model(
            input_shape,
            num_transformer_blocks=cfg.num_transformer_blocks,
            key_dim=cfg.key_dim,
            num_heads=cfg.num_heads,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
            num_classes=cfg.num_classes,
            positional_encoding=args.pos_encoding,
        )
        model = compile_model(
            model,
            optimizer=cfg.optimizer,
            learning_rate=cfg.learning_rate,
            momentum=cfg.momentum,
            clip_value=cfg.clip_value,
        )
        epochs = args.epochs or cfg.epochs
        batch_size = args.batch_size or cfg.batch_size

    model.summary()

    # Train
    print(f"\nTraining for {epochs} epochs (batch_size={batch_size})...")
    history = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=epochs, batch_size=batch_size,
        early_stopping_patience=train_cfg.early_stopping_patience,
    )

    # Evaluate
    print("\nEvaluation:")
    from configs.config import WORD_LABELS
    class_names = list(WORD_LABELS.values())
    metrics = evaluate_model(model, X_test, y_test, class_names=class_names)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"\n{metrics['report']}")

    # Save
    feature_name = Path(args.features).stem.split("_")
    extractor = "resnet50" if "resnet" in args.features else "vgg16"
    category = "phrases" if "phrases" in args.features else "words"
    pe_suffix = f"_{args.pos_encoding}" if args.model == "transformer" else ""
    model_name = f"{extractor}_{args.model}{pe_suffix}_{category}"

    save_dir = Path(args.output_dir) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(save_dir / "model.keras")

    results = {
        "model": args.model,
        "extractor": extractor,
        "category": category,
        "positional_encoding": args.pos_encoding if args.model == "transformer" else None,
        "accuracy": float(metrics["accuracy"]),
        "epochs": epochs,
        "batch_size": batch_size,
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nModel saved to {save_dir}")


if __name__ == "__main__":
    main()
