# Automatic Video Speech Recognition and Caption Generation System

## Objective
The project aims to develop an automated video speech recognition system, or a lip-reading model, capable of detecting and generating words spoken in short videos using only visual cues. The primary goal is to improve the quality of life for individuals with hearing impairments. The system also has potential applications in enhancing speech recognition in noisy environments, improving security and surveillance, and enabling device control through lip movements.

## System Design and Architecture
The system is designed to process video frames to detect and recognize spoken words using visual cues. The design involves several key components:

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



