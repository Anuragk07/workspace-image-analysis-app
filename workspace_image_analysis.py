
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, ViltProcessor, ViltForQuestionAnswering
import torch
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Load YOLOv5 model from torch hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load MediaPipe for pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to estimate pixel distance between two points
def calculate_pixel_distance(nose_x, nose_y, screen_x_center, screen_y_center):
    return np.sqrt((nose_x - screen_x_center) ** 2 + (nose_y - screen_y_center) ** 2)

# Function to detect workstation objects using YOLOv5
def detect_workstation_objects_yolo(image):
    results = model(image)
    detections = results.pandas().xyxy[0]

    workstation_objects = {
        "Screens and Laptops": 0,
        "Keyboards": 0,
        "Mice": 0
    }
    
    for _, row in detections.iterrows():
        if row["name"] in ["tv", "monitor", "laptop"]:
            workstation_objects["Screens and Laptops"] += 1
        elif row["name"] == "keyboard":
            workstation_objects["Keyboards"] += 1
        elif row["name"] == "mouse":
            workstation_objects["Mice"] += 1

    return workstation_objects

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    ab = np.array(b) - np.array(a)
    bc = np.array(c) - np.array(b)
    angle = np.arctan2(np.linalg.det([ab, bc]), np.dot(ab, bc)) * (180.0 / np.pi)
    return abs(angle)

# Function to determine back support status
def determine_back_support(landmarks, image_height):
    # Extract y-coordinates of relevant landmarks
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

    # Calculate angles for back support analysis
    upper_back_angle = calculate_angle(
        [shoulder.x, shoulder.y],
        [elbow.x, elbow.y],
        [hip.x, hip.y]
    )

    # Define thresholds for support
    support_threshold_angle = 30  # Degrees

    support_status = {
        "Upper Back": "Supported" if upper_back_angle < support_threshold_angle else "Not Supported",
        "Mid Back": "Supported" if shoulder.y <= elbow.y <= hip.y else "Not Supported",
        "Lower Back": "Supported" if hip.y < 0.45 * image_height else "Not Supported"
    }
    
    return support_status

# Set up the Streamlit app
st.set_page_config(page_title="Workspace Image Analysis", layout="wide")

# App Title
st.title("🏢 Workspace Image Analysis App")
st.write("Upload an image of your workspace to get an accurate analysis of objects, back support, and distance.")

# Image upload widget
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, use_column_width=True)

    # Open the image
    image = Image.open(uploaded_file).convert("RGB")
    img_cv = np.array(image)

    # Object detection using YOLOv5 to detect workstation objects
    st.write("🔍 Detecting objects in the workspace...")
    workstation_objects = detect_workstation_objects_yolo(img_cv)

    # Display object counts in a visually appealing table
    st.write("### Detected Objects:")
    object_df = pd.DataFrame(list(workstation_objects.items()), columns=["Object", "Count"])
    st.table(object_df.style.set_table_attributes('style="width: 100%; border-collapse: collapse;"').set_properties(**{
        'border': '1px solid black',
        'text-align': 'center',
        'padding': '10px'
    }))

    # Pose estimation for back support analysis
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    image_height, image_width, _ = img_rgb.shape

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Back support analysis
        support_status = determine_back_support(landmarks, image_height)
        st.write("### Back Support Analysis:")
        for part, status in support_status.items():
            st.write(f"**{part}:** {status}")

        # Calculate distance from the face to the nearest screen
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        nose_x = int(nose.x * image_width)
        nose_y = int(nose.y * image_height)

        if workstation_objects["Screens and Laptops"] > 0:
            # Dummy distance calculation logic
            min_distance = 60  # Replace with actual logic
            distance_category = (
                "Less than one arm's length" if min_distance < 60 
                else "One arm's length" if 60 <= min_distance <= 70 
                else "More than one arm's length"
            )
            st.success(f"Estimated Distance from Face to Nearest Screen: **({distance_category})**")
        else:
            st.error("🚫 No screens or laptops detected in the image.")
    
    # Image captioning using BLIP
    st.write("🖼️ Generating Image Description...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = blip_processor(image, return_tensors="pt")
    output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    st.success("**Generated Image Description:** " + caption)

    # Question input for Visual Question Answering (ViLT)
    st.write("❓ Ask a question about the image:")
    question = st.text_input("Type your question here...")

    if question:
        vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        vilt_inputs = vilt_processor(image, question, return_tensors="pt")
        with torch.no_grad():
            outputs = vilt_model(**vilt_inputs)
            logits = outputs.logits
            answer_idx = logits.argmax(-1).item()
            answer = vilt_model.config.id2label[answer_idx]

        st.success("**Answer:** " + answer)
