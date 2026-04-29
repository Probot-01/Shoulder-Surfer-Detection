# live_person_detection.py
# Real-time person detection using YOLOv8n and a webcam.
# Draws green bounding boxes around detected people and shows a live count.

import cv2
from ultralytics import YOLO
import time  # Used to calculate frames per second (FPS)

# ------------------------------------------------------------------
# STEP 1: Load the YOLOv8 nano model
# ------------------------------------------------------------------
# "yolov8n.pt" is the smallest/fastest YOLO model — good for real-time use
# verbose=False suppresses the per-frame detection logs in the console
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("Model loaded! Starting webcam...\n")

# ------------------------------------------------------------------
# STEP 2: Open the webcam
# ------------------------------------------------------------------
# VideoCapture(0) opens the default (first) webcam on your system
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam working! Press 'q' to quit.")

# ------------------------------------------------------------------
# STEP 3: Performance tweak — reduce frame resolution for faster processing
# ------------------------------------------------------------------
# Smaller frames = YOLO has less pixels to analyze = higher FPS
# 640x480 is a good balance between quality and speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables to track FPS
prev_time = time.time()  # Time of the previous frame

# ------------------------------------------------------------------
# STEP 4: Main loop — read frames, detect people, draw boxes
# ------------------------------------------------------------------
while True:
    # Read one frame from the webcam
    # ret  -> True if the frame was captured successfully
    # frame -> the actual image as a NumPy array (height x width x 3 colors)
    ret, frame = cap.read()

    # If reading failed (e.g. camera disconnected), stop the loop
    if not ret:
        print("Error: Failed to read frame from webcam.")
        break

    # ------------------------------------------------------------------
    # STEP 5: Run YOLO detection on the current frame
    # ------------------------------------------------------------------
    # verbose=False stops YOLO from printing detection results every frame
    # stream=True makes it use less memory by processing as a generator
    results = model(frame, verbose=False, stream=False)

    # Grab the result for this single frame
    result = results[0]

    # ------------------------------------------------------------------
    # STEP 6: Filter detections — only keep "person" (class ID = 0)
    # ------------------------------------------------------------------
    person_count = 0  # How many people found in this frame

    for box in result.boxes:
        # Get the class ID (integer) — 0 means "person" in YOLO's COCO dataset
        class_id = int(box.cls[0])

        # Skip anything that is NOT a person
        if class_id != 0:
            continue

        person_count += 1  # Found one more person

        # Get the bounding box corners in pixel coordinates [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get the confidence score (e.g. 0.87 = 87% sure it's a person)
        confidence = float(box.conf[0])

        # ------------------------------------------------------------------
        # STEP 7: Draw a green rectangle around the detected person
        # ------------------------------------------------------------------
        # cv2.rectangle(image, top-left, bottom-right, color BGR, thickness)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ------------------------------------------------------------------
        # STEP 8: Write "Person X" label above the bounding box
        # ------------------------------------------------------------------
        label = f"Person {person_count} ({confidence:.0%})"

        # Draw a filled rectangle behind the text so it's easier to read
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w, y1), (0, 255, 0), -1)

        # Draw the label text in black over the green background
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),              # Position: just above the box top-left
            cv2.FONT_HERSHEY_SIMPLEX,   # Font style
            0.55,                        # Font size
            (0, 0, 0),                   # Text color: black (BGR)
            2                            # Text thickness
        )

    # ------------------------------------------------------------------
    # STEP 9: Show "People detected: N" in the top-left corner
    # ------------------------------------------------------------------
    summary = f"People detected: {person_count}"

    # Draw a dark semi-transparent banner behind the summary text
    cv2.rectangle(frame, (0, 0), (260, 40), (0, 0, 0), -1)

    cv2.putText(
        frame,
        summary,
        (10, 28),                   # Position: top-left of the frame
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,                         # Slightly larger font for the summary
        (0, 255, 0),                 # Green text
        2
    )

    # ------------------------------------------------------------------
    # STEP 10: Calculate and display FPS in the top-right corner
    # ------------------------------------------------------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)   # Frames per second = 1 / time per frame
    prev_time = curr_time

    fps_label = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_label,
        (frame.shape[1] - 120, 28),  # Top-right corner
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),               # Yellow text for FPS
        2
    )

    # ------------------------------------------------------------------
    # STEP 11: Show the frame in a window
    # ------------------------------------------------------------------
    cv2.imshow("Live Person Detection", frame)

    # ------------------------------------------------------------------
    # STEP 12: Check if the user pressed 'q' — if so, quit the loop
    # ------------------------------------------------------------------
    # cv2.waitKey(1) waits 1 ms for a key press (keeps the loop fast)
    # 0xFF masks the result to get just the key code
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("'q' pressed — stopping detection.")
        break

# ------------------------------------------------------------------
# STEP 13: Clean up when done
# ------------------------------------------------------------------
cap.release()           # Release the webcam so other apps can use it
cv2.destroyAllWindows() # Close the display window
print("Webcam closed. Goodbye!")
