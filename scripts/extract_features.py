#!/usr/bin/env python3
"""Extract CNN features from cropped lip image sequences.

Usage:
    python scripts/extract_features.py \
        --images-dir ./data/cropped_lips \
        --output-dir ./data \
        --extractor vgg16 \
        --category words
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import DatasetConfig
from src.feature_extraction import (
    build_resnet50_extractor,
    build_vgg16_extractor,
    extract_sequence_features,
    save_features,
)
from src.preprocessing import (
    load_image_sequences,
    normalize_images,
    parse_cropped_filenames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CNN features from lip images")
    parser.add_argument("--images-dir", type=str, required=True, help="Directory with cropped lip images")
    parser.add_argument("--output-dir", type=str, default="./data", help="Output directory for pickle files")
    parser.add_argument("--extractor", type=str, choices=["vgg16", "resnet50"], default="vgg16")
    parser.add_argument("--category", type=str, choices=["words", "phrases"], default="words")
    parser.add_argument("--max-seq-len", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DatasetConfig()

    print(f"Parsing cropped images from: {args.images_dir}")
    images_list = parse_cropped_filenames(args.images_dir)

    print(f"Loading image sequences (max_seq_len={args.max_seq_len})...")
    sequences, labels = load_image_sequences(
        args.images_dir, images_list,
        max_seq_len=args.max_seq_len,
        target_size=cfg.lip_image_size,
    )
    sequences = normalize_images(sequences)
    print(f"  Loaded {len(labels)} instances, shape: {sequences.shape}")

    print(f"Building {args.extractor} feature extractor...")
    if args.extractor == "vgg16":
        extractor = build_vgg16_extractor(cfg.lip_image_size)
    else:
        extractor = build_resnet50_extractor(cfg.lip_image_size)

    print("Extracting features...")
    features = extract_sequence_features(sequences, extractor, batch_size=args.batch_size)
    print(f"  Feature shape: {features.shape}")

    output_name = f"miracl_feature_maps_{args.extractor}_{args.category}.pkl"
    output_path = Path(args.output_dir) / output_name
    save_features(features, labels, output_path)
    print(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()
