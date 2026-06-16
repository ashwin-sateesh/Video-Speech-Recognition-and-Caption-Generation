"""Preprocessing pipeline: face detection, lip cropping, and frame sequence construction.

Handles the full path from raw MIRACL-VC1 video frames to padded, normalized
image arrays ready for feature extraction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from mtcnn import MTCNN


# ---------------------------------------------------------------------------
# Face and lip detection
# ---------------------------------------------------------------------------

def create_lip_detector(min_face_size: int = 75) -> MTCNN:
    """Initialize an MTCNN face detector.

    Args:
        min_face_size: Minimum face size in pixels for detection.

    Returns:
        Configured MTCNN detector instance.
    """
    return MTCNN(min_face_size=min_face_size)


def crop_lip_region(
    image: np.ndarray,
    detector: MTCNN,
    y_offset: int = 12,
    x_offset: int = 7,
    box_size: int = 65,
) -> Optional[np.ndarray]:
    """Detect a face and crop the lip region from an image.

    Uses MTCNN to locate the face bounding box, then extracts the lower
    portion corresponding to the lip area.

    Args:
        image: BGR image array (H, W, 3).
        detector: MTCNN detector instance.
        y_offset: Vertical offset from face center to lip region.
        x_offset: Horizontal shrink from bounding box edges.
        box_size: Base bounding box width for cropping.

    Returns:
        Cropped lip image (25, 58, 3) or None if no face detected.
    """
    faces = detector.detect_faces(image)
    if not faces:
        return None

    face = faces[0]
    x, y, w, h = face["box"]

    x1 = x + w // 2 - box_size // 2
    y1 = y + h // 2 + y_offset
    x2 = x1 + box_size - x_offset
    y2 = y1 + 25

    # Bounds checking
    y1, y2 = max(0, y1), min(image.shape[0], y2)
    x1, x2 = max(0, x1), min(image.shape[1], x2)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return None

    return cropped


def crop_lips_from_dataset(
    image_paths: List[str],
    output_dir: str | Path,
    detector: Optional[MTCNN] = None,
    min_face_size: int = 75,
) -> int:
    """Crop lip regions from all images and save to output directory.

    Args:
        image_paths: List of source image file paths.
        output_dir: Directory to save cropped lip images.
        detector: Optional pre-initialized MTCNN detector.
        min_face_size: Min face size if creating a new detector.

    Returns:
        Number of successfully cropped images.
    """
    if detector is None:
        detector = create_lip_detector(min_face_size)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            continue

        cropped = crop_lip_region(image, detector)
        if cropped is not None:
            filename = Path(img_path).stem + "_lip.jpg"
            cv2.imwrite(str(output_dir / filename), cropped)
            count += 1

    return count


# ---------------------------------------------------------------------------
# Dataset path traversal (MIRACL-VC1 structure)
# ---------------------------------------------------------------------------

def collect_frame_paths(
    dataset_dir: str | Path,
    category: str = "words",
) -> List[str]:
    """Traverse the MIRACL-VC1 directory structure and collect frame paths.

    Expected structure: dataset_dir/F01/words/01/01/color_001.jpg

    Args:
        dataset_dir: Root directory of the MIRACL-VC1 dataset.
        category: ``"words"`` or ``"phrases"``.

    Returns:
        Sorted list of image file paths.
    """
    dataset_dir = Path(dataset_dir)
    paths = []

    for person_dir in sorted(dataset_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        cat_dir = person_dir / category
        if not cat_dir.exists():
            continue
        for word_dir in sorted(cat_dir.iterdir()):
            for instance_dir in sorted(word_dir.iterdir()):
                for frame in sorted(instance_dir.glob("color_*.jpg")):
                    paths.append(str(frame))

    return paths


# ---------------------------------------------------------------------------
# Image sequence construction and padding
# ---------------------------------------------------------------------------

def parse_cropped_filenames(
    folder_path: str | Path,
) -> List[List[List[List[str]]]]:
    """Parse cropped lip image filenames into a structured nested list.

    Returns a nested structure: [persons][words][instances][frames]
    where each leaf is a filename string.

    Args:
        folder_path: Directory containing cropped lip images with naming
            convention ``cropped_{person}_{category}_{word}_instance_{inst}_frame_{f}.jpg``

    Returns:
        Nested list indexed by [person][word][instance][frame].
    """
    folder_path = Path(folder_path)
    images: Dict = {}

    for filename in sorted(folder_path.iterdir()):
        if not filename.suffix == ".jpg":
            continue
        parts = filename.stem.split("_")
        # Expected: cropped_{person}_{cat}_{word}_instance_{inst}_frame_{frame}
        if len(parts) < 8:
            continue

        person = parts[1]
        word = parts[3]
        instance = parts[5]
        frame = parts[7]

        images.setdefault(person, {})
        images[person].setdefault(word, {})
        images[person][word].setdefault(instance, {})
        images[person][word][instance][frame] = filename.name

    # Convert to nested list
    result = []
    for person in sorted(images.keys()):
        person_list = []
        for word in sorted(images[person].keys()):
            word_list = []
            for instance in sorted(images[person][word].keys()):
                inst_dict = images[person][word][instance]
                frame_list = [inst_dict[f] for f in sorted(inst_dict.keys())]
                word_list.append(frame_list)
            person_list.append(word_list)
        result.append(person_list)

    return result


def load_image_sequences(
    folder_path: str | Path,
    images_list: List[List[List[List[str]]]],
    max_seq_len: int = 22,
    target_size: Tuple[int, int, int] = (25, 58, 3),
) -> Tuple[np.ndarray, np.ndarray]:
    """Load cropped lip images into padded numpy arrays and generate labels.

    Args:
        folder_path: Directory containing the cropped images.
        images_list: Nested structure from ``parse_cropped_filenames()``.
        max_seq_len: Maximum sequence length (zero-padded if shorter).
        target_size: Expected (H, W, C) for each lip image.

    Returns:
        Tuple of (feature_sequences, labels):
            - feature_sequences: shape (num_instances, max_seq_len, H, W, C)
            - labels: shape (num_instances,) with integer class indices
    """
    folder_path = Path(folder_path)
    H, W, C = target_size

    sequences = []
    labels = []

    for person_idx, person in enumerate(images_list):
        for word_idx, word in enumerate(person):
            for instance in word:
                frames = []
                for fname in instance:
                    img = cv2.imread(str(folder_path / fname))
                    if img is not None:
                        frames.append(img.astype(np.float32))
                    else:
                        frames.append(np.zeros(target_size, dtype=np.float32))

                # Pad to max_seq_len
                while len(frames) < max_seq_len:
                    frames.append(np.zeros(target_size, dtype=np.float32))

                # Truncate if longer
                frames = frames[:max_seq_len]

                sequences.append(np.stack(frames))
                labels.append(word_idx)

    return np.array(sequences), np.array(labels)


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1].

    Args:
        images: Image array with uint8 or float values in [0, 255].

    Returns:
        Normalized array with values in [0, 1].
    """
    return images.astype(np.float32) / 255.0
