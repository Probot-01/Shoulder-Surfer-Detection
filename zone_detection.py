# zone_detection.py
# Extends live_person_detection.py with role classification:
#   - The person with the LARGEST bounding box is labelled "USER" (blue box)
#   - All other detected people are labelled "OBSERVER?" (red box)
# This works on the assumption that the main user is closest to the camera
# and therefore appears largest in the frame.

import cv2
from ultralytics import YOLO
import time


# ===========================================================================
# HELPER FUNCTION: classify_persons
# ===========================================================================
def classify_persons(detections):
    """
    Classifies a list of bounding boxes into a USER and a list of OBSERVERs.

    Logic:
      - The person whose bounding box has the LARGEST area (width × height)
        is assumed to be the primary user (closest to the camera / screen).
      - Everyone else is a potential observer / shoulder surfer.

    Parameters:
      detections : list of tuples
          Each tuple is (x1, y1, x2, y2, confidence) for one detected person.

    Returns:
      user_box      : tuple (x1, y1, x2, y2, confidence) of the USER, or None
      observer_boxes: list of tuples for all other detected persons
    """

    # If nobody was detected, return empty results
    if not detections:
        return None, []

    # If only one person is detected, they are the user (no observers)
    if len(detections) == 1:
        return detections[0], []

    # Find the detection with the largest bounding box area
    # Area = width × height = (x2 - x1) × (y2 - y1)
    def box_area(det):
        x1, y1, x2, y2, conf = det
        return (x2 - x1) * (y2 - y1)

    # Sort detections by area in descending order (largest first)
    sorted_detections = sorted(detections, key=box_area, reverse=True)

    # The first item (largest area) is the USER
    user_box = sorted_detections[0]

    # All remaining detections are potential observers
    observer_boxes = sorted_detections[1:]

    return user_box, observer_boxes


# ===========================================================================
# DRAWING HELPER: draw_labeled_box
# ===========================================================================
def draw_labeled_box(frame, box, label, color):
    """
    Draws a colored bounding box and a label tag on the frame.

    Parameters:
      frame : the OpenCV image to draw on
      box   : tuple (x1, y1, x2, y2, confidence)
      label : text to show above the box (e.g. "USER" or "OBSERVER?")
      color : BGR color tuple for the box and tag background
    """
    x1, y1, x2, y2, conf = box

    # Draw the main bounding box rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Build the display text: label + confidence percentage
    display_text = f"{label} ({conf:.0%})"

    # Measure how wide and tall the text will be so we can size the tag
    (text_w, text_h), _ = cv2.getTextSize(
        display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )

    # Draw a filled rectangle as the tag background above the box
    cv2.rectangle(
        frame,
        (x1, y1 - text_h - 10),   # top-left of tag
        (x1 + text_w + 4, y1),     # bottom-right of tag
        color,
        -1                          # -1 means filled (not just outline)
    )

    # Write the label text in white over the colored tag
    cv2.putText(
        frame,
        display_text,
        (x1 + 2, y1 - 5),          # Slight padding inside the tag
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),            # White text
        2
    )


# ===========================================================================
# MAIN SCRIPT
# ===========================================================================

# Color constants in BGR format (OpenCV uses BGR, not RGB)
COLOR_USER     = (255, 100,   0)   # Blue  → the main user
COLOR_OBSERVER = (  0,   0, 220)   # Red   → potential shoulder surfer

# ------------------------------------------------------------------
# Load the YOLOv8 nano model
# ------------------------------------------------------------------
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("Model loaded! Starting webcam...\n")

# ------------------------------------------------------------------
# Open the webcam and set resolution for better FPS
# ------------------------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Webcam working! Press 'q' to quit.")

prev_time = time.time()  # For FPS calculation

# ------------------------------------------------------------------
# Main detection loop
# ------------------------------------------------------------------
while True:

    # Read one frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame. Exiting.")
        break

    # ------------------------------------------------------------------
    # Run YOLO detection on the current frame
    # ------------------------------------------------------------------
    results = model(frame, verbose=False)
    result  = results[0]

    # ------------------------------------------------------------------
    # Collect all "person" detections (class ID 0) into a list of tuples
    # ------------------------------------------------------------------
    detections = []  # Will hold (x1, y1, x2, y2, confidence) for each person

    for box in result.boxes:
        class_id = int(box.cls[0])

        # Only keep persons (class 0), ignore everything else
        if class_id != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence      = float(box.conf[0])

        detections.append((x1, y1, x2, y2, confidence))

    # ------------------------------------------------------------------
    # Classify persons: largest box → USER, rest → OBSERVERs
    # ------------------------------------------------------------------
    user_box, observer_boxes = classify_persons(detections)

    # ------------------------------------------------------------------
    # Draw USER box in BLUE
    # ------------------------------------------------------------------
    if user_box is not None:
        draw_labeled_box(frame, user_box, "USER", COLOR_USER)

    # ------------------------------------------------------------------
    # Draw each OBSERVER box in RED
    # ------------------------------------------------------------------
    for obs_box in observer_boxes:
        draw_labeled_box(frame, obs_box, "OBSERVER?", COLOR_OBSERVER)

    # ------------------------------------------------------------------
    # Build status strings for the top-left HUD
    # ------------------------------------------------------------------
    user_detected_str  = "User detected: YES" if user_box else "User detected: NO"
    observer_count     = len(observer_boxes)
    observer_count_str = f"Observers nearby: {observer_count}"

    # Warn visually if at least one observer is present
    hud_color = (0, 255, 0)              # Green by default (safe)
    if observer_count > 0:
        hud_color = (0, 100, 255)        # Orange if observers detected

    # ------------------------------------------------------------------
    # Draw the HUD panel in the top-left corner
    # ------------------------------------------------------------------
    # Dark background banner so text is readable on any background
    cv2.rectangle(frame, (0, 0), (300, 65), (0, 0, 0), -1)

    cv2.putText(frame, user_detected_str,  (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)
    cv2.putText(frame, observer_count_str, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)

    # ------------------------------------------------------------------
    # Calculate and show FPS in the top-right corner
    # ------------------------------------------------------------------
    curr_time = time.time()
    fps       = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (frame.shape[1] - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ------------------------------------------------------------------
    # Display the annotated frame
    # ------------------------------------------------------------------
    cv2.imshow("Zone Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("'q' pressed — stopping.")
        break

# ------------------------------------------------------------------
# Clean up
# ------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
print("Webcam closed. Goodbye!")
