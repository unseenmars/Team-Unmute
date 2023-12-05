
# Installation and Deployment Guide for Team Unmute's Sign Language Recognition Project

## Introduction
This README guides you through setting up and running the Team Unmute's real-time sign language recognition project. This Python application integrates OpenCV, Keras, MediaPipe, gTTS, and Pygame for real-time gesture recognition and audio feedback.

## Prerequisites
- Python (Version 3.6 or later)
- Pip (Python package manager)

## Installation Steps

### 1. Clone the Repository
Clone the Team Unmute repository from GitHub.
```bash
git clone https://github.com/khemagarwal/Team-Unmute
cd Team-Unmute/Milestone\ 4\ Model
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
Create and activate a virtual environment for dependency management.
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
```

### 3. Install Required Libraries
Install the necessary Python libraries using pip.
```bash
pip install numpy opencv-python keras mediapipe pygame gTTS
```

### 4. Load Pre-trained Model
Ensure the pre-trained model (`actionsFinalFinal.h5`) is placed in the `Team-Unmute/Milestone 4 Model` directory.

### 5. Verify Webcam Access
Check that your development environment has access to a webcam for video input.

## Running the Application

1. Start the application by running the main Python script.
   ```bash
   python Team-Unmute/Milestone\ 4\ Model/main.py
   ```

2. The application will activate the webcam and start processing for gesture recognition.

3. Perform sign language gestures in front of the webcam to see the recognition and corresponding audio feedback.

4. To exit, press 'q' while the application window is active.

## Application Features and User Guide

### Sign Language Recognition
- The application detects specific sign language gestures in real-time.

### Audio Announcement
- Recognized gestures trigger an audio announcement of the corresponding sign.

### Timing and Cooldown
- A 2-second cooldown follows each gesture recognition, during which new gestures are not recognized.

### Notice Points
- Visual indicators on the screen show the recognized gesture and cooldown countdown.

### Best Practices for Usage
- Ensure clear visibility of hands and good lighting conditions.
- Wait for the cooldown to end before performing the next gesture.

## Troubleshooting
- **Webcam Access**: Ensure the webcam is properly connected and permitted to be used by the application.
- **Library Dependencies**: If there are issues with libraries, refer to their official documentation.
- **Model Compatibility**: Confirm that the Keras model is compatible with your version of Keras.

## Support
For help or feedback, please contact hgcai@cmu.edu.
