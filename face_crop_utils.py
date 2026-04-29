# face_crop_utils.py
# ============================================================================
# PURPOSE:
#   Provides extract_padded_face_crop() — a robust helper that extracts a
#   clean, padded face crop from a person bounding box, suitable as input
#   to a 224×224 head pose classifier.
#
# WHY A SEPARATE FUNCTION?
#   The face extraction logic requires several validation steps, coordinate
#   conversions, and fallback strategies. Keeping it here makes main_system.py
#   cleaner and allows the function to be unit-tested independently.
#
# USAGE:
#   from face_crop_utils import extract_padded_face_crop
#   face_crop, reason = extract_padded_face_crop(frame, [x1, y1, x2, y2], face_detector)
#   if face_crop is not None:
#       result = predictor.predict(face_crop)
# ============================================================================

import cv2
import numpy as np


def extract_padded_face_crop(frame, person_box, face_detector):
    """
    Extracts a padded face crop from a person bounding box.

    Tries MediaPipe face detection first (precise). Falls back to a geometric
    estimate (top 28% of person crop) if MediaPipe finds nothing.
    Always returns a BGR numpy array ready to pass to HeadPosePredictor.

    Parameters:
        frame        (np.ndarray) : full webcam frame in BGR format
                                    shape: (height, width, 3)
        person_box   (list/tuple) : [x1, y1, x2, y2] from YOLO detection,
                                    in full-frame pixel coordinates
        face_detector             : initialized mediapipe FaceDetection object

    Returns:
        (face_crop, reason)
            face_crop (np.ndarray or None) : BGR crop ready for the model,
                                             or None if extraction failed
            reason    (str)                : describes what happened:
                "mediapipe_confirmed"  — MediaPipe found and confirmed a face
                "fallback_estimate"    — no face detected; used top-28% heuristic
                "box_too_small"        — person box area < 100px (bad detection)
                "empty_crop"           — numpy slice produced a 0-byte array
                "mediapipe_error"      — MediaPipe threw an unexpected exception
                "face_too_small"       — confirmed face crop < 40×40px
                "fallback_too_small"   — fallback crop < 40×40px
    """

    # =========================================================================
    # STEP 1 — Validate and clip the person bounding box
    # =========================================================================
    # YOLO sometimes returns boxes that slightly exceed the frame edges.
    # Clipping ensures all array slices stay within valid pixel ranges.

    frame_h, frame_w = frame.shape[:2]   # Full frame dimensions

    px1, py1, px2, py2 = person_box

    # Clip each coordinate to the frame boundaries
    px1 = max(0, min(int(px1), frame_w - 1))
    py1 = max(0, min(int(py1), frame_h - 1))
    px2 = max(0, min(int(px2), frame_w))
    py2 = max(0, min(int(py2), frame_h))

    # Reject boxes that are too tiny to contain a meaningful face
    # (width × height) — anything under 100 pixels is likely a false detection
    box_area = (px2 - px1) * (py2 - py1)
    if box_area < 100:
        print(f"  Face crop FAILED: box_too_small (area={box_area}px)")
        return None, "box_too_small"

    # =========================================================================
    # STEP 2 — Crop the person region from the full frame
    # =========================================================================
    # We work on a crop of the person rather than the full frame.
    # This means MediaPipe only searches a small region → faster and more
    # accurate (it doesn't accidentally find faces in the background).

    person_crop = frame[py1:py2, px1:px2]   # NumPy slice — no data is copied

    # Guard: if the slice somehow resulted in an empty array, bail out early
    if person_crop.size == 0:
        print(f"  Face crop FAILED: empty_crop")
        return None, "empty_crop"

    crop_h, crop_w = person_crop.shape[:2]  # Dimensions of the person crop

    # =========================================================================
    # STEP 3 — Run MediaPipe Face Detection on the person crop
    # =========================================================================
    # MediaPipe expects an RGB image; OpenCV gives us BGR — convert first.

    try:
        crop_rgb         = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        detection_result = face_detector.process(crop_rgb)
    except Exception as e:
        # MediaPipe can occasionally throw on malformed image data
        print(f"  Face crop FAILED: mediapipe_error ({e})")
        return None, "mediapipe_error"

    # =========================================================================
    # STEP 4a — MediaPipe found a face: extract with padding
    # =========================================================================

    if detection_result.detections:
        # Take the detection with the highest confidence score
        detection = detection_result.detections[0]

        # MediaPipe returns RELATIVE bounding box values, i.e. fractions of
        # the image dimensions (0.0–1.0). We need pixel values within the crop.
        bbox = detection.location_data.relative_bounding_box

        # Convert relative → absolute pixel coords inside the person crop
        raw_fx1 = int(bbox.xmin                   * crop_w)
        raw_fy1 = int(bbox.ymin                   * crop_h)
        raw_fx2 = int((bbox.xmin + bbox.width)    * crop_w)
        raw_fy2 = int((bbox.ymin + bbox.height)   * crop_h)

        # ---- Add 40% padding around the detected face box ----
        # WHY PADDING?
        #   MediaPipe tends to return a tight box around the facial landmarks.
        #   A head pose model trained on BIWI (which includes neck/shoulders)
        #   generally performs better when given a slightly wider crop that
        #   includes some hair and chin context — hence 40% expansion.
        face_w = raw_fx2 - raw_fx1
        face_h = raw_fy2 - raw_fy1

        pad_x = int(face_w * 0.4)   # 40% of face width added on each side
        pad_y = int(face_h * 0.4)   # 40% of face height added on top/bottom

        # Apply padding and clamp to crop boundaries (can't go outside the crop)
        fx1 = max(0,      raw_fx1 - pad_x)
        fy1 = max(0,      raw_fy1 - pad_y)
        fx2 = min(crop_w, raw_fx2 + pad_x)
        fy2 = min(crop_h, raw_fy2 + pad_y)

        # Extract the padded face from the PERSON CROP (not the full frame)
        face_crop = person_crop[fy1:fy2, fx1:fx2]

        # Reject if the resulting crop is too small for the model to use
        if face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
            print(f"  Face crop FAILED: face_too_small "
                  f"({face_crop.shape[1]}×{face_crop.shape[0]}px)")
            return None, "face_too_small"

        # Success — MediaPipe confirmed a face
        print(f"  Face crop: mediapipe_confirmed — size {face_crop.shape[:2]}")
        return face_crop, "mediapipe_confirmed"

    # =========================================================================
    # STEP 4b — No face detected: geometric fallback
    # =========================================================================
    # When MediaPipe can't find a face (looking sideways, back of head, blur),
    # we fall back to a simple anatomical heuristic:
    #   - Faces occupy roughly the top 28% of a standing person bounding box
    #   - We trim 15% of the width from each side to remove background
    # This is not guaranteed to contain a face but gives the model something
    # better than the full body crop.

    face_h  = int(crop_h * 0.28)        # Top 28% of person box height
    margin  = int(crop_w * 0.15)        # Trim 15% from left and right edges

    face_crop = person_crop[0:face_h, margin:crop_w - margin]

    # Reject if the fallback crop is too small
    if face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
        print(f"  Face crop FAILED: fallback_too_small "
              f"({face_crop.shape[1]}×{face_crop.shape[0]}px)")
        return None, "fallback_too_small"

    # Fallback success — return the estimate
    print(f"  Face crop: fallback_estimate — size {face_crop.shape[:2]}")
    return face_crop, "fallback_estimate"
