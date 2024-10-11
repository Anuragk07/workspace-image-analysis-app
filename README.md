# workspace-image-analysis-app
🏢 A Streamlit web application that analyzes workspace images. It uses YOLOv5 for object detection, MediaPipe for pose estimation, and transformer models for image captioning and visual question answering. Users can upload workspace images to receive insights on object counts, back support status, and estimated distances to screens.

![Uploading Screenshot 2024-10-12 at 12.18.15 AM.png…]()

## Overview

This application utilizes advanced computer vision techniques and deep learning models to evaluate images of workspaces. It detects objects like screens, keyboards, and mice, assesses the support status of your back based on pose estimation, and estimates the distance from your face to the nearest screen.

## Features

- **Object Detection**: Uses YOLOv5 to identify screens, laptops, keyboards, and mice in the uploaded image.
- **Back Support Analysis**: Utilizes MediaPipe for pose estimation to determine whether your back is properly supported while working.
- **Distance Estimation**: Estimates the distance from your face to the nearest screen to promote ergonomic practices.
- **Image Captioning**: Generates a description of the uploaded image using the BLIP model.
- **Visual Question Answering**: Allows users to ask questions about the image using the ViLT model.

## Requirements

- Python 3.x
- Streamlit
- Pillow
- Torch
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Transformers


