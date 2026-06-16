"""Inference pipeline: predict spoken words from video and add captions.

Handles the full inference flow from raw video → frame extraction →
lip cropping → feature extraction → sequence classification → captioned output.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model

from .preprocessing import create_lip_detector, crop_lip_region, normalize_images


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------

def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    sample_rate: int = 1,
) -> List[str]:
    """Extract frames from a video file.

    Args:
        video_path: Path to the input video.
        output_dir: Directory to save extracted frames.
        sample_rate: Save every Nth frame (1 = all frames).

    Returns:
        List of saved frame file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_paths = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            path = output_dir / f"frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(path), frame)
            frame_paths.append(str(path))
        frame_idx += 1

    cap.release()
    return frame_paths


# ---------------------------------------------------------------------------
# Lip extraction from video frames
# ---------------------------------------------------------------------------

def extract_lips_from_frames(
    frame_paths: List[str],
    output_dir: str | Path,
    target_size: Tuple[int, int] = (58, 25),
    min_face_size: int = 75,
) -> List[str]:
    """Detect and crop lip regions from video frames.

    Args:
        frame_paths: List of video frame image paths.
        output_dir: Directory to save cropped lip images.
        target_size: (width, height) to resize cropped lips.
        min_face_size: Minimum face size for MTCNN.

    Returns:
        Sorted list of cropped lip image paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = create_lip_detector(min_face_size)

    lip_paths = []
    for frame_path in frame_paths:
        img = cv2.imread(frame_path)
        if img is None:
            continue

        cropped = crop_lip_region(img, detector)
        if cropped is not None:
            resized = cv2.resize(cropped, target_size)
            stem = Path(frame_path).stem
            out_path = output_dir / f"lip_{stem}.jpg"
            cv2.imwrite(str(out_path), resized)
            lip_paths.append(str(out_path))

    return sorted(lip_paths)


# ---------------------------------------------------------------------------
# Feature extraction and prediction
# ---------------------------------------------------------------------------

def predict_from_lip_sequence(
    lip_paths: List[str],
    feature_extractor: Model,
    classifier: Model,
    max_seq_len: int = 22,
    target_size: Tuple[int, int, int] = (25, 58, 3),
) -> int:
    """Run the full inference pipeline on a sequence of lip images.

    Args:
        lip_paths: Sorted list of cropped lip image paths.
        feature_extractor: CNN feature extractor (VGG16 or ResNet50).
        classifier: Trained sequence model (LSTM, Attention, or Transformer).
        max_seq_len: Maximum sequence length for padding.
        target_size: Expected (H, W, C) for lip images.

    Returns:
        Predicted class index (integer).
    """
    H, W, C = target_size
    frames = []

    for path in lip_paths[:max_seq_len]:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (W, H))
            frames.append(img.astype(np.float32) / 255.0)

    # Pad to max_seq_len
    while len(frames) < max_seq_len:
        frames.append(np.zeros(target_size, dtype=np.float32))

    frames_array = np.stack(frames)  # (seq_len, H, W, C)

    # Extract features
    features = feature_extractor.predict(frames_array, verbose=0)  # (seq_len, feat_dim)
    features = features.reshape(1, max_seq_len, -1)  # (1, seq_len, feat_dim)

    # Classify
    prediction = classifier.predict(features, verbose=0)
    return int(np.argmax(prediction[0]))


# ---------------------------------------------------------------------------
# Video captioning
# ---------------------------------------------------------------------------

def add_caption_to_video(
    input_video: str | Path,
    output_video: str | Path,
    caption: str,
    display_from_fraction: float = 0.7,
    font_scale: float = 1.0,
    font_thickness: int = 2,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> Path:
    """Add a text caption to a video file.

    The caption appears in the lower portion of the video during the
    final fraction of playback.

    Args:
        input_video: Path to the source video.
        output_video: Path for the captioned output video.
        caption: Text string to display.
        display_from_fraction: Show caption starting at this fraction of
            total duration (e.g., 0.7 = last 30%).
        font_scale: OpenCV font scale.
        font_thickness: OpenCV font thickness.
        color: BGR text color.

    Returns:
        Path to the output video.
    """
    input_video = Path(input_video)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_time = total_frames / fps
    text_start_time = total_time * display_from_fraction

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if current_time >= text_start_time:
            text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - text_size[1] - 10
            cv2.putText(
                frame, caption, (text_x, text_y),
                font, font_scale, color, font_thickness,
            )

        out.write(frame)

    cap.release()
    out.release()
    return output_video
