#!/usr/bin/env python3
"""Run inference on a video: detect lips, classify speech, and add captions.

Usage:
    python scripts/predict.py \
        --video ./data/Web.mp4 \
        --model-dir ./artifacts/vgg16_lstm_attention_words \
        --extractor vgg16 \
        --output ./outputs/captioned_video.mp4
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import WORD_LABELS, PHRASE_LABELS
from src.feature_extraction import build_resnet50_extractor, build_vgg16_extractor
from src.inference import (
    add_caption_to_video,
    extract_lips_from_frames,
    extract_video_frames,
    predict_from_lip_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict spoken word from video and add caption")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory with saved model")
    parser.add_argument("--extractor", type=str, choices=["vgg16", "resnet50"], default="vgg16")
    parser.add_argument("--category", type=str, choices=["words", "phrases"], default="words")
    parser.add_argument("--output", type=str, default="./outputs/captioned_video.mp4")
    parser.add_argument("--max-seq-len", type=int, default=22)
    parser.add_argument("--sample-rate", type=int, default=3, help="Extract every Nth frame")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import tensorflow as tf
    classifier = tf.keras.models.load_model(Path(args.model_dir) / "model.keras")

    print(f"Building {args.extractor} feature extractor...")
    if args.extractor == "vgg16":
        extractor = build_vgg16_extractor()
    else:
        extractor = build_resnet50_extractor()

    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = Path(tmpdir) / "frames"
        lips_dir = Path(tmpdir) / "lips"

        print(f"Extracting frames from: {args.video}")
        frame_paths = extract_video_frames(args.video, frames_dir, sample_rate=args.sample_rate)
        print(f"  Extracted {len(frame_paths)} frames")

        print("Detecting and cropping lip regions...")
        lip_paths = extract_lips_from_frames(frame_paths, lips_dir)
        print(f"  Cropped {len(lip_paths)} lip images")

        if not lip_paths:
            print("Error: No lips detected in the video. Check video quality or face visibility.")
            sys.exit(1)

        print("Running sequence classification...")
        pred_idx = predict_from_lip_sequence(
            lip_paths, extractor, classifier, max_seq_len=args.max_seq_len,
        )

    labels = WORD_LABELS if args.category == "words" else PHRASE_LABELS
    predicted_text = labels.get(pred_idx, f"Unknown ({pred_idx})")
    print(f"\n  Predicted: '{predicted_text}' (class {pred_idx})")

    print(f"\nAdding caption to video...")
    output_path = add_caption_to_video(args.video, args.output, predicted_text)
    print(f"  Captioned video saved to: {output_path}")


if __name__ == "__main__":
    main()
