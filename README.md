# Automatic Video Speech Recognition and Caption Generation System

## Objective
The project aims to develop an automated video speech recognition system, or a lip-reading model, capable of detecting and generating words spoken in short videos using only visual cues. The primary goal is to improve the quality of life for individuals with hearing impairments. The system also has potential applications in enhancing speech recognition in noisy environments, improving security and surveillance, and enabling device control through lip movements.

## System Design and Architecture
The system is designed to process video frames to detect and recognize spoken words using visual cues. The design involves several key components:
<p align="center">
  <img src="https://github.com/ashwin-sateesh/Video-Speech-Recognition-and-Caption-Generation/blob/main/assets/Video%20Speech%20Detection%20Caption%20Generation.png" alt="System Architecture">
</p>

1. **Face and Lip Detection**: 
   - The system uses a pre-trained Haar Cascade Classifier to detect faces and localize the lip region in each video frame.
   - The detected lip regions are cropped and resized to a uniform dimension (25x58x3) for further processing.

2. **Feature Extraction**:
   - Pre-trained convolutional neural networks (CNNs), specifically ResNet50 and VGG16, are employed to extract spatial features from the cropped lip images.
   - These features are represented as 512-dimensional vectors from VGG16 and 2048-dimensional vectors from ResNet50 after removing the softmax layer.

3. **Sequence Modeling**:
   - The extracted features are passed through sequence models such as Long Short-Term Memory (LSTM) networks, LSTM with Attention, and Transformers to learn the temporal dynamics of the image sequences.
   - Two types of positional encoding were used for the Transformer model: traditional sine-cosine based and learned positional encoding.

4. **Classification and Caption Generation**:
   - The sequence models predict the spoken word or phrase for each video instance.
   - The predicted words/phrases are then added as captions to the video.


## Methodology

### Dataset
- The MIRACL-VC1 dataset was used, consisting of video frames of 10 words and 10 phrases spoken by 15 individuals, totaling around 37,000 images.

### Preprocessing
- The face and lip regions were detected and cropped using the Haar Cascade Classifier.
- Image arrays were padded with zero vectors to maintain uniform dimensions across all instances.

### Feature Extraction
- ResNet50 and VGG16 models were used to extract feature vectors from the lip images.
- Global Average Pooling was applied to obtain 512-dimensional (VGG16) and 2048-dimensional (ResNet50) feature vectors.

### Model Training
- LSTM, LSTM with Attention, and Transformer models were trained on the extracted features to classify the spoken words and phrases.
- Hyperparameter tuning was conducted to optimize each model’s performance.

## Results
The results demonstrate the potential of sequence models in accurately classifying spoken words and phrases from image sequences. The performance of the models is summarized as follows:

| Model                              | Accuracy (Words) | Accuracy (Phrases) |
|------------------------------------|------------------|--------------------|
| ResNet50 + LSTM                    | 71.3%            | 72.3%              |
| ResNet50 + LSTM-Attn               | 86.1%            | 75.4%              |
| ResNet50 + Transformer (Traditional PE) | 82.4%            | 88.3%              |
| ResNet50 + Transformer (Learned PE) | 78.7%            | 82.3%              |
| VGG16 + LSTM                       | 89.1%            | 86.6%              |
| VGG16 + LSTM-Attn                  | 91.3%            | 77.6%              |
| VGG16 + Transformer (Traditional PE) | 86.6%            | 87.6%              |
| VGG16 + Transformer (Learned PE)   | 82.3%            | 80.6%              |

