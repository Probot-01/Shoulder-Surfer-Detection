# download_yolo.py
# A simple script to download and load the YOLOv8n model

# What is YOLOv8n?
# YOLO stands for "You Only Look Once" — it's a fast AI model that can detect
# objects in images/video in real time. The "v8" means version 8, and the "n"
# stands for "nano", which is the smallest and fastest version of the model.
# It trades a little accuracy for speed, making it great for live webcam use.

# Import the YOLO class from the ultralytics library
from ultralytics import YOLO

# Load the YOLOv8 nano model
# If "yolov8n.pt" is not already on your computer, ultralytics will
# automatically download it from the internet the first time you run this
model = YOLO("yolov8n.pt")

# Print a confirmation message once the model is loaded successfully
print("YOLO downloaded successfully!")
