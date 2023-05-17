# Video-Speech-Recognition-and-Caption-Generation

Our goal in this project was to develop an automated video speech recognition system (or a lip-reading model) that can detect and generate words spoken in a short video using only visual cues, with the aim of improving the quality of life for individuals with hearing impairment.

To achieve this, we used pre-trained convolutional models to detect facial features in each image frame, which were then fed into sequence model layers such as LSTM, LSTM with attention and Transformer to learn how a sequence of image frames produces a certain word. We used the pre-trained Haar Cascade Classifier to detect the mouth and lips in each image, and cropped and resized the lip section of each image. To extract features from each image, we used VGG16 and Resnet50 convolutional models.


