"""Centralized configuration for the Video Speech Recognition project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal


# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

WORD_LABELS: Dict[int, str] = {
    0: "Begin", 1: "Choose", 2: "Connection", 3: "Navigation", 4: "Next",
    5: "Previous", 6: "Start", 7: "Stop", 8: "Hello", 9: "Web",
}

PHRASE_LABELS: Dict[int, str] = {
    0: "Stop Navigation", 1: "Excuse me", 2: "I am sorry", 3: "Thank you",
    4: "Good bye", 5: "I love this game", 6: "Nice to meet you",
    7: "You are welcome", 8: "How are you", 9: "Have a good time",
}


@dataclass
class PathConfig:
    """File and directory paths."""

    data_dir: Path = Path("./data")
    artifacts_dir: Path = Path("./artifacts")

    # Raw dataset (MIRACL-VC1)
    raw_dataset_dir: str = "miracl_vc1"
    cropped_images_dir: str = "cropped_lips"

    # Pre-extracted feature pickles
    resnet_words_pkl: str = "miracl_feature_maps_resnet_words.pkl"
    resnet_phrases_pkl: str = "miracl_feature_maps_resnet_phrases.pkl"
    vgg_words_pkl: str = "miracl_feature_maps_vgg_words.pkl"
    vgg_phrases_pkl: str = "miracl_feature_maps_vgg_phrases.pkl"

    @property
    def raw_dataset_path(self) -> Path:
        return self.data_dir / self.raw_dataset_dir

    @property
    def cropped_path(self) -> Path:
        return self.data_dir / self.cropped_images_dir


@dataclass
class DatasetConfig:
    """Dataset properties."""

    num_words: int = 10
    num_phrases: int = 10
    num_speakers: int = 15
    instances_per_speaker: int = 10
    lip_image_size: tuple = (25, 58, 3)
    max_seq_len_words: int = 22
    max_seq_len_phrases: int = 27
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class FaceDetectionConfig:
    """MTCNN and lip cropping parameters."""

    min_face_size: int = 75
    lip_box_width: int = 58
    lip_box_height: int = 25
    y_offset: int = 12
    x_offset: int = 7


@dataclass
class LSTMConfig:
    """Hyperparameters for LSTM and LSTM-Attention models."""

    hidden_units: int = 128
    num_classes: int = 10
    batch_size: int = 128
    epochs: int = 50
    optimizer: str = "adam"
    loss: str = "categorical_crossentropy"


@dataclass
class TransformerConfig:
    """Hyperparameters for the Transformer sequence classifier."""

    key_dim: int = 64
    num_heads: int = 16
    num_transformer_blocks: int = 2
    ff_dim: int = 768
    dropout: float = 0.1
    num_classes: int = 10
    positional_encoding: Literal["traditional", "learned", "none"] = "traditional"
    # Training
    optimizer: str = "sgd"
    learning_rate: float = 0.001
    momentum: float = 0.9
    clip_value: float = 0.5
    batch_size: int = 64
    epochs: int = 50


@dataclass
class TrainingConfig:
    """General training configuration."""

    test_size: float = 0.2
    random_state: int = 42
    early_stopping_patience: int = 5
