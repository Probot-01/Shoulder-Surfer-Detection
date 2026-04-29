# face_cropper.py
# Combines YOLOv8n (person detection) with MediaPipe Face Detection.
# For each detected person:
#   - Draws a GREEN box around the full person
#   - Crops that region and looks for a face inside it using MediaPipe
#   - YELLOW box = face confirmed by MediaPipe
#   - ORANGE box = face ESTIMATED from the top-30% of the person box (fallback)

import cv2
import mediapipe as mp
from ultralytics import YOLO
import time

# =============================================================================
# SET UP MEDIAPIPE FACE DETECTION
# =============================================================================
# MediaPipe provides a ready-made face detector.
# model_selection=0 → short-range model (best for faces within ~2 metres)
# min_detection_confidence=0.5 → only report faces with ≥50% confidence
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# =============================================================================
# HELPER FUNCTION: extract_face_crop
# =============================================================================
def extract_face_crop(frame, person_bbox):
    """
    Looks for a face inside a person bounding box region.

    Two-stage approach:
      STAGE 1 — MediaPipe (preferred):
        Crops the person region and runs MediaPipe Face Detection on it.
        If a face is found, is_confirmed = True.

      STAGE 2 — Geometric fallback (when MediaPipe finds nothing):
        Takes the TOP 30% of the person bounding box height as a rough
        face estimate. Faces are almost always in the upper portion of a
        person's bounding box. is_confirmed = False to signal it's an estimate.

    Parameters:
      frame       : the full webcam frame (NumPy array, BGR)
      person_bbox : (x1, y1, x2, y2) pixel coordinates of the person box

    Returns:
      (face_img, face_coords, is_confirmed)
        face_img      : cropped face image (NumPy array) or None
        face_coords   : (fx1, fy1, fx2, fy2) in full-frame pixel coords, or None
        is_confirmed  : True  → MediaPipe detected a real face
                        False → fallback estimate used (orange box)
    """

    px1, py1, px2, py2 = person_bbox

    # ----- Safety clamp -----
    # Make sure coordinates don't go outside the frame edges
    frame_h, frame_w = frame.shape[:2]
    px1 = max(0, px1)
    py1 = max(0, py1)
    px2 = min(frame_w, px2)
    py2 = min(frame_h, py2)

    # If the crop is too small to be meaningful, skip it
    if px2 - px1 < 10 or py2 - py1 < 10:
        return None, None, False

    # Crop just the person region from the full frame
    person_crop = frame[py1:py2, px1:px2]

    # =========================================================================
    # STAGE 1: Try MediaPipe face detection inside the person crop
    # =========================================================================

    # MediaPipe needs RGB; OpenCV gives BGR → convert colour order
    crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

    # Run the face detector on the cropped region only
    face_results = face_detector.process(crop_rgb)

    if face_results.detections:
        # ---- MediaPipe found a face — convert its coords to full-frame ----

        # Take the FIRST (highest-confidence) detection
        detection = face_results.detections[0]

        crop_h = py2 - py1   # Height of the person crop in pixels
        crop_w = px2 - px1   # Width  of the person crop in pixels

        # MediaPipe returns bounding box as relative values (0.0 – 1.0)
        # relative to the CROP dimensions, not the full frame
        bb = detection.location_data.relative_bounding_box

        # Convert relative → absolute pixel coords within the crop
        face_x1_in_crop = int(bb.xmin               * crop_w)
        face_y1_in_crop = int(bb.ymin               * crop_h)
        face_x2_in_crop = int((bb.xmin + bb.width)  * crop_w)
        face_y2_in_crop = int((bb.ymin + bb.height) * crop_h)

        # Clamp to crop boundaries (MediaPipe can slightly exceed them)
        face_x1_in_crop = max(0, face_x1_in_crop)
        face_y1_in_crop = max(0, face_y1_in_crop)
        face_x2_in_crop = min(crop_w, face_x2_in_crop)
        face_y2_in_crop = min(crop_h, face_y2_in_crop)

        # Shift from crop-space → full-frame-space by adding the crop offset
        fx1 = px1 + face_x1_in_crop
        fy1 = py1 + face_y1_in_crop
        fx2 = px1 + face_x2_in_crop
        fy2 = py1 + face_y2_in_crop

        face_img = frame[fy1:fy2, fx1:fx2]

        # Safety check — if the resulting crop is empty, fall through to fallback
        if face_img.size > 0:
            return face_img, (fx1, fy1, fx2, fy2), True   # is_confirmed = True

    # =========================================================================
    # STAGE 2: Fallback — use the top 30% of the person box as a face estimate
    # =========================================================================
    # Why 30%? Human faces typically occupy the top quarter-to-third of a
    # standing person's silhouette. This is a rough but reliable heuristic.

    person_height = py2 - py1   # Total height of the person bounding box

    # Calculate the bottom of the top-30% region
    estimate_y2 = py1 + int(person_height * 0.30)

    # The face estimate spans the full width of the person box,
    # but only the top 30% of its height
    fx1, fy1 = px1, py1
    fx2, fy2 = px2, estimate_y2

    face_img = frame[fy1:fy2, fx1:fx2]

    # Guard: return None if the estimate crop is somehow empty
    if face_img.size == 0:
        return None, None, False

    return face_img, (fx1, fy1, fx2, fy2), False  # is_confirmed = False (estimate)


# =============================================================================
# LOAD YOLO MODEL
# =============================================================================
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("Model loaded! Starting webcam...\n")

# =============================================================================
# OPEN THE WEBCAM
# =============================================================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Reduce resolution slightly for better real-time performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Webcam working! Press 'q' to quit.\n")

prev_time = time.time()

# =============================================================================
# MAIN LOOP
# =============================================================================
while True:

    # Read one frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    # -------------------------------------------------------------------------
    # Run YOLO to detect all objects; keep only persons (class 0)
    # -------------------------------------------------------------------------
    results = model(frame, verbose=False)
    result  = results[0]

    person_count = 0  # total persons found this frame

    for box in result.boxes:

        # Skip anything that is not a person
        if int(box.cls[0]) != 0:
            continue

        person_count += 1

        # Get person bounding box in pixel coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence      = float(box.conf[0])

        # ---- Draw GREEN box around the full person ----
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Person label above the green box
        person_label = f"Person {person_count} ({confidence:.0%})"
        cv2.putText(frame, person_label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # -------------------------------------------------------------------------
        # Run face detection on just this person's crop
        # -------------------------------------------------------------------------
        # extract_face_crop now returns THREE values:
        #   face_img     : the face image crop (or None)
        #   face_coords  : (fx1, fy1, fx2, fy2) in full-frame coords (or None)
        #   is_confirmed : True = MediaPipe found a real face
        #                  False = geometric fallback estimate used
        face_img, face_coords, is_confirmed = extract_face_crop(frame, (x1, y1, x2, y2))

        if face_img is not None:
            fx1, fy1, fx2, fy2 = face_coords

            if is_confirmed:
                # ---- YELLOW box: MediaPipe confirmed a real face ----
                box_color  = (0, 255, 255)   # Yellow in BGR
                face_label = "Face Detected"
                print(f"Face found for person {person_count}")
            else:
                # ---- ORANGE box: geometric fallback estimate only ----
                box_color  = (0, 165, 255)   # Orange in BGR
                face_label = "Face Estimate"
                print(f"No face detected for person {person_count} — using estimate")

            # Draw the bounding box (yellow or orange depending on is_confirmed)
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), box_color, 2)

            # Draw the label above the box
            cv2.putText(frame, face_label, (fx1, fy1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)

            # Show the face crop in its own small popup window
            face_display = cv2.resize(face_img, (120, 120))
            cv2.imshow(f"Face - Person {person_count}", face_display)

        else:
            # Neither MediaPipe nor the fallback produced a usable crop
            print(f"No face detected for person {person_count}")

    # -------------------------------------------------------------------------
    # FPS counter (top-right)
    # -------------------------------------------------------------------------
    curr_time = time.time()
    fps       = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (frame.shape[1] - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # -------------------------------------------------------------------------
    # Show the annotated main frame
    # -------------------------------------------------------------------------
    cv2.imshow("Face Cropper", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("'q' pressed — stopping.")
        break

# =============================================================================
# CLEAN UP
# =============================================================================
cap.release()
face_detector.close()          # Properly release MediaPipe resources
cv2.destroyAllWindows()
print("Webcam closed. Goodbye!")
