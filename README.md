# Automatic Video Speech Recognition and Caption Generation

A lip-reading system that detects and generates spoken words from short videos using only visual cues. Combines pre-trained CNN feature extractors (VGG16, ResNet50) with sequence models (LSTM, LSTM-Attention, Transformer) to classify lip movements into words and phrases, then adds captions to the source video.

![Architecture](/assets/system_architecture.png)

## Architecture Overview

The pipeline operates in four stages:

**Stage 1 — Face and Lip Detection:** Each video frame is processed through an MTCNN face detector, which localizes the face bounding box. The lip region is cropped from the lower portion of the face and resized to a uniform 25×58×3 dimension.

**Stage 2 — Feature Extraction:** Cropped lip images are passed through pre-trained CNNs with the classification head removed. VGG16 produces 512-dimensional feature vectors and ResNet50 produces 2048-dimensional vectors via Global Average Pooling. Each video instance becomes a sequence of feature vectors preserving temporal order.

**Stage 3 — Sequence Classification:** The feature sequences are fed into sequence models that learn the temporal dynamics of lip movements. Three architectures are compared: vanilla LSTM, LSTM with Attention (self-attention over hidden states), and Transformer encoder (with traditional sine-cosine or learned 1D-CNN positional encoding). The models output a probability distribution over 10 word or 10 phrase classes.

**Stage 4 — Caption Generation:** The predicted class is mapped to its text label and overlaid onto the original video as a caption.

## Project Structure

```
video-speech-recognition-and-caption-generation/
├── configs/
│   ├── __init__.py
│   └── config.py              # PathConfig, DatasetConfig, LSTMConfig, TransformerConfig, etc.
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # MTCNN lip detection, frame parsing, sequence padding
│   ├── feature_extraction.py  # VGG16/ResNet50 extractors, pickle I/O
│   ├── models.py              # LSTM, LSTM-Attention, Transformer (with positional encodings)
│   ├── training.py            # Data prep, compilation, training loops, evaluation metrics
│   └── inference.py           # Video frame extraction, lip cropping, prediction, captioning
├── scripts/
│   ├── extract_features.py    # CNN feature extraction pipeline
│   ├── train.py               # Train any model/extractor combination
│   ├── tune.py                # Keras Tuner hyperparameter search
│   └── predict.py             # End-to-end video → caption inference
├── data/                      # Dataset and feature pickles (not tracked)
├── artifacts/                 # Trained model checkpoints (not tracked)
├── assets/                    # Architecture diagrams
├── requirements.txt
├── .gitignore
└── README.md
```

## Results

Trained on the MIRACL-VC1 dataset (15 speakers, 10 words, 10 phrases, ~37,000 lip images):

| Model | Accuracy (Words) | Accuracy (Phrases) |
|---|---|---|
| ResNet50 + LSTM | 71.3% | 72.3% |
| ResNet50 + LSTM-Attention | 86.1% | 75.4% |
| ResNet50 + Transformer (Traditional PE) | 82.4% | 88.3% |
| ResNet50 + Transformer (Learned PE) | 78.7% | 82.3% |
| VGG16 + LSTM | 89.1% | 86.6% |
| **VGG16 + LSTM-Attention** | **91.3%** | 77.6% |
| VGG16 + Transformer (Traditional PE) | 86.6% | 87.6% |
| VGG16 + Transformer (Learned PE) | 82.3% | 80.6% |

## Setup

```bash
git clone https://github.com/ashwin-sateesh/Video-Speech-Recognition-and-Caption-Generation.git
cd Video-Speech-Recognition-and-Caption-Generation

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Dataset

The project uses the [MIRACL-VC1 dataset](http://www.oulu.fi/cmvs/node/41316), which contains lip-reading video frames of 15 individuals speaking 10 words and 10 phrases (10 instances each).

Place the raw dataset in `data/miracl_vc1/` with the standard directory structure:
```
data/miracl_vc1/F01/words/01/01/color_001.jpg
```

## Usage

**1. Extract CNN features from cropped lip images:**

```bash
python scripts/extract_features.py \
    --images-dir ./data/cropped_lips \
    --extractor vgg16 \
    --category words
```

**2. Train a sequence model:**

```bash
# LSTM
python scripts/train.py --features ./data/miracl_feature_maps_vgg16_words.pkl --model lstm

# LSTM with Attention
python scripts/train.py --features ./data/miracl_feature_maps_vgg16_words.pkl --model lstm_attention

# Transformer with traditional positional encoding
python scripts/train.py \
    --features ./data/miracl_feature_maps_resnet50_words.pkl \
    --model transformer \
    --pos-encoding traditional
```

**3. Hyperparameter tuning:**

```bash
python scripts/tune.py --features ./data/miracl_feature_maps_resnet50_words.pkl --max-epochs 30
```

**4. Run inference on a video:**

```bash
python scripts/predict.py \
    --video ./data/test_video.mp4 \
    --model-dir ./artifacts/vgg16_lstm_attention_words \
    --extractor vgg16 \
    --output ./outputs/captioned_video.mp4
```

## Key Design Decisions

- **VGG16 over ResNet50 for word classification**: VGG16 features consistently outperformed ResNet50 for word-level classification (91.3% vs 86.1% best), likely because the shallower architecture preserves fine-grained spatial details in the small 25×58 lip crops better than ResNet's aggressive downsampling.
- **LSTM-Attention for words, Transformer for phrases**: Attention-augmented LSTM achieved the highest word accuracy (91.3%), while the Transformer with traditional PE performed best on phrases (88.3%). This suggests phrases benefit more from the Transformer's ability to model longer-range dependencies.
- **Two positional encoding schemes**: Traditional sine-cosine PE consistently outperformed learned PE across both extractors, indicating that the fixed periodicity captures the sequential nature of lip movements more reliably than the learned 1D-CNN approach given the limited training data.
- **Zero-padding for variable-length sequences**: Frame sequences ranged from 7 to 27 frames. Pre-padding with zero vectors to a fixed length preserves the temporal alignment while keeping the implementation simple.

## References

- Viola & Jones (2001), "Rapid Object Detection using a Boosted Cascade of Simple Features"
- He et al. (2015), "Deep Residual Learning for Image Recognition" ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385))
- Lu & Li (2019), "Automatic Lip-Reading System Based on Deep CNN and Attention-Based LSTM"
- Chung et al. (2017), "Lip Reading Sentences in the Wild" ([arXiv:1611.05358](https://arxiv.org/abs/1611.05358))

## License

This project was developed as part of coursework at Northeastern University.
