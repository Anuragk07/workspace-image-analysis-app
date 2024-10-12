# Workspace Image Analysis App

🏢 A Streamlit web application designed to analyze workspace images. This application leverages advanced computer vision techniques, utilizing YOLOv5 for object detection, MediaPipe for pose estimation, and transformer models for image captioning and visual question answering. Users can upload images of their workspaces to receive valuable insights on object counts, back support status, and estimated distances to screens.

<img width="1440" alt="Screenshot 2024-10-12 at 12 15 51 PM" src="https://github.com/user-attachments/assets/940195ed-902f-4d1f-8209-230a787531c3">


## Overview

This application employs cutting-edge computer vision and deep learning models to evaluate workspace images. It identifies objects such as screens, keyboards, and mice, assesses back support based on user posture, and estimates the distance from the user’s face to the nearest screen.

## Features

- **Object Detection**: Detects screens, laptops, keyboards, and mice in uploaded images using YOLOv5.
- **Back Support Analysis**: Analyzes posture with MediaPipe to evaluate if your back is adequately supported while working.
- **Distance Estimation**: Estimates the distance from your face to the nearest screen to encourage ergonomic practices.
- **Image Captioning**: Generates a descriptive caption for the uploaded image using the BLIP model.
- **Visual Question Answering**: Enables users to ask questions about the image with the ViLT model.

## Requirements

To run the application, you need the following Python packages:

```plaintext
streamlit
Pillow
torch
torchvision
opencv-python
mediapipe
numpy
pandas
transformers
ultralytics
yolov5 @ git+https://github.com/ultralytics/yolov5.git
