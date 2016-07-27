# Emotion-Recognition-Using-STASM-and-OpenCV
Introduction:
The aim of this project is to detect emotions from video data. The detection will be performed on
front facing human faces.
Dataset:
The dataset used for this project is the Cohn-Kanade dataset. This dataset contains images
classified into 7 different categories. The categories are: neutral, anger, contempt, disgust, fear,
happy, sadness and surprise. Images from multiple subjects are collected. Multiple emotions are
extracted from each subject. Each such emotion contains a series of frames from the neutral frame
to the peak frame.
Toolset:
This project is executed using C++ and Visual Studio. OpenCV is used to perform the image
processing required for this project. The extraction of facial features (landmarks) is performed
using and implementation of Active Shape Models known as stasm. The SVM library from the ml
module of OpenCV is used to perform the training and classification of emotions.
Approach:
The following steps are used to perform emotion prediction on the data:
1) Identify peak frames and neutral frames for each emotion and place them in separate
folders.
2) Run stasm on each of these images and store the obtained landmark values in a csv file.
3) Assign labels to the extracted landmark instances. (0 – neutral, 1- peak)
4) Compute differences between selected landmark features and divide each feature by the xscale
and y-scale values
5) Store the computed ratios along with class labels in a csv file.
6) Train an SVM classifier with the computed feature information
7) Save the model created by the classifier.
8) Run stasm on the test video.
9) Extract landmark points for each frame and compute the ratios and store in a matrix.
10) Load the stored model for each emotion sequentially and check the prediction. If any of
the models predict an emotion, that emotion is displayed.
