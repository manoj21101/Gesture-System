# Sign Language to Text Conversion

## Overview

This GitHub repository contains a project focused on translating gestures captured in real-time through a camera into text using a Convolutional Neural Network (CNN). The CNN model has been trained to recognize American Sign Language gestures, providing a practical application for communication.

## Project Components

### 1. Gesture Recognition Model

- The model is implemented using the Keras library and consists of a Convolutional Neural Network trained on a dataset of American Sign Language gestures.
- The model architecture includes multiple convolutional layers, max-pooling layers, dropout layers to prevent overfitting, and fully connected layers.
- The training script involves preparing the dataset, augmenting images using ImageDataGenerator, and fitting the model to the training data.

### 2. Real-time Gesture Translation GUI

- The application utilizes the OpenCV library to capture real-time video input from the user's camera.
- A graphical user interface (GUI) built with Tkinter displays the camera feed, the recognized gesture symbol, current word being formed, and the complete sentence.
- Users can interact with the system by adding recognized symbols to the word, inserting spaces between words, deleting the last character, and clearing the current word.

## How to Use

1. Clone the repository to your local machine.
2. Run the main script (`main.py`) to start the application.
3. Point the camera towards your hand gestures, and the system will recognize and translate them into text in real-time.

## Directory Structure

- `gesture_System`: Contains the main application code, including the GUI and real-time gesture recognition.
- `model_training`: Contains the scripts used to train the CNN model.
- `dataSet`: Contains the dataset used for training, split into training and testing sets.

## Dependencies

- OpenCV
- Tkinter
- PIL (Pillow)
- Autocorrect
- Keras
- Theano (with CUDA support)

## Acknowledgments

The project utilizes the power of deep learning and computer vision to bridge the communication gap for individuals using American Sign Language. Special thanks to the contributors and libraries that made this project possible.

Feel free to contribute, report issues, or provide feedback!
