# main_system.py
# ============================================================================
# Shoulder Surfing Prevention System — Detection Only
# ============================================================================
# WHAT THIS DOES:
#   Opens the webcam, detects all persons in each frame using YOLOv8n,
#   classifies the largest person as the USER and everyone else as OBSERVERS,
#   extracts each observer's face, runs a head pose model (MobileNetV2) to
#   determine if they are LOOKING at the screen, and triggers a THREAT alert
#   via the ThreatDecisionEngine when a sustained stare is confirmed.
#
# Run with:  python main_system.py
# Quit with: press 'q' in the webcam window
# Screenshot: press 's'
# ============================================================================

# ═══════════════════════════════════════════════════════
# PERFORMANCE TUNING GUIDE
# ═══════════════════════════════════════════════════════
# pose_interval = 3
#   Run MediaPipe + MobileNetV2 every Nth frame.
#   Higher = faster FPS but slower reaction to gaze changes.
#
# yolo_interval = 2
#   Run YOLO person detection every Nth frame.
#   Higher = faster FPS but slower reaction to new persons entering frame.
#
# threat_threshold = 4  (set in ThreatDecisionEngine)
#   Consecutive THREAT frames required before declaring THREAT state.
#   Lower = faster response. (was 10)
#
# cooldown_frames = 20  (set in ThreatDecisionEngine)
#   Consecutive SAFE frames required before returning to SAFE. (was 30)
#
# observer_streak = 3  (set in ThreatDecisionEngine)
#   Consecutive LOOKING frames per observer needed to count as THREAT. (was 5)
#
# confidence_threshold = 75%  (set in ThreatDecisionEngine)
#   Minimum MobileNetV2 confidence for a LOOKING result to count. (was 80%)
#
# gap_threshold = 0.25  (set in head_pose_predictor.py)
#   Probability gap needed to confirm LOOKING. (was 0.30)
# ═══════════════════════════════════════════════════════

import cv2
import numpy as np
import time
from ultralytics import YOLO
import mediapipe as mp
from head_pose_predictor import HeadPosePredictor
from decision_engine import ThreatDecisionEngine
from face_crop_utils import extract_padded_face_crop


# ============================================================================
# HELPER FUNCTION
# ============================================================================

def resize_to_max_width(img, max_width=320):
    """
    OPTIMIZATION: Resize an image so its width does not exceed max_width,
    preserving aspect ratio. Used to speed up MediaPipe face detection by
    running it on a smaller person crop instead of the full-resolution crop.
    """
    if img.shape[1] <= max_width:
        return img
    scale = max_width / img.shape[1]
    new_h = int(img.shape[0] * scale)
    return cv2.resize(img, (max_width, new_h))


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():

    print("=" * 60)
    print("  Shoulder Surfing Prevention System — Starting")
    print("=" * 60)

    # ------------------------------------------------------------------
    # INITIALIZATION — runs ONCE before the main loop
    # ------------------------------------------------------------------

    # ONE VideoCapture, opened once, released in finally block
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check it is connected and not in use.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # YOLOv8n — person detection (pretrained on COCO, class 0 = person)
    model = YOLO("yolov8n.pt")

    # MobileNetV2-based head pose classifier
    predictor = HeadPosePredictor("best_head_pose_model.pth")

    # Threat decision engine — temporal smoothing to avoid flickering alerts
    engine = ThreatDecisionEngine(threat_threshold=4, cooldown_frames=20)

    # MediaPipe face detector — finds the face within each observer person crop
    mp_face       = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    # ------------------------------------------------------------------
    # SPEED OPTIMIZATION VARIABLES
    # ------------------------------------------------------------------

    # OPTIMIZATION A: YOLO frame-skip
    # Run YOLO every yolo_interval frames, reuse cached boxes in between.
    # Halves YOLO cost (~15-25ms saved every other frame).
    YOLO_INTERVAL   = 2
    last_yolo_boxes = []

    # OPTIMIZATION B: Head pose frame-skip
    # Run MediaPipe + MobileNetV2 every pose_interval frames.
    # Reuse last known result for each observer on skip frames.
    # Saves ~20-40ms per observer on skipped frames.
    POSE_INTERVAL         = 3
    last_observer_results = {}   # {observer_index: last predict() result}
    last_observer_count   = 0    # Used to detect when observer count changes

    frame_count   = 0
    fps_time      = time.time()
    prev_state        = None
    prev_person_count = -1

    print("System ready. Press 'q' to quit, 's' to screenshot.")

    # ------------------------------------------------------------------
    # MAIN LOOP — one iteration = one video frame
    # ------------------------------------------------------------------
    try:
        while True:

            # ----------------------------------------------------------------
            # A) Read frame from webcam
            # ----------------------------------------------------------------
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed — retrying...")
                continue

            frame_count  += 1
            # Draw on this copy so the original stays clean for model inference
            display_frame = frame.copy()

            # Resize if the camera delivers a frame larger than 640×480
            h_frame, w_frame = frame.shape[:2]
            if w_frame > 640 or h_frame > 480:
                frame         = cv2.resize(frame, (640, 480))
                display_frame = frame.copy()

            # ----------------------------------------------------------------
            # B) YOLO person detection — every YOLO_INTERVAL frames
            # ----------------------------------------------------------------
            if frame_count % YOLO_INTERVAL == 1:
                # YOLO frame: run full detection and cache result
                results   = model(frame, classes=[0], verbose=False)
                raw_boxes = []
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    if conf < 0.5:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    raw_boxes.append([x1, y1, x2, y2, conf])
                last_yolo_boxes = raw_boxes
            else:
                # Skip frame: reuse last known boxes
                raw_boxes = last_yolo_boxes

            # ----------------------------------------------------------------
            # C) Classify persons
            #    Largest bounding box area = USER (closest to camera)
            #    All others = potential OBSERVERs (shoulder surfers)
            # ----------------------------------------------------------------
            if len(raw_boxes) == 0:
                user_box, observer_boxes = None, []

            elif len(raw_boxes) == 1:
                # Only one person — must be the user, no threat possible
                user_box, observer_boxes = raw_boxes[0], []

            else:
                # Sort by area descending: largest box = user
                raw_boxes.sort(
                    key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                    reverse=True
                )
                user_box       = raw_boxes[0]
                observer_boxes = raw_boxes[1:]

            total_persons = len(raw_boxes)
            if total_persons != prev_person_count:
                print(f"Persons detected: {total_persons}")
                prev_person_count = total_persons

            # ----------------------------------------------------------------
            # D) Face crop + head pose — OBSERVERS only, every POSE_INTERVAL frames
            # ----------------------------------------------------------------

            # Clear cache if observer count changed (someone entered or left)
            current_observer_count = len(observer_boxes)
            if current_observer_count != last_observer_count:
                last_observer_results = {}
                last_observer_count   = current_observer_count

            observer_results = []
            run_pose = (frame_count % POSE_INTERVAL == 0)

            for i, obs_box in enumerate(observer_boxes):
                x1, y1, x2, y2, _ = obs_box

                # On skip frames: reuse cached result if available
                if not run_pose and i in last_observer_results:
                    observer_results.append(last_observer_results[i])
                    continue

                # ---- POSE FRAME: run full MediaPipe + MobileNetV2 ----

                # OPTIMIZATION: resize person crop to max 320px wide before
                # running MediaPipe — less data = faster detection
                person_crop_full = frame[y1:y2, x1:x2]
                if person_crop_full.size == 0:
                    result = {"label": "UNKNOWN", "confidence": 0.0,
                              "is_looking": False, "skip_reason": "empty_crop"}
                    last_observer_results[i] = result
                    observer_results.append(result)
                    continue

                person_crop_small = resize_to_max_width(person_crop_full, max_width=320)
                crop_h, crop_w    = person_crop_small.shape[:2]

                face_crop, reason = extract_padded_face_crop(
                    person_crop_small,
                    [0, 0, crop_w, crop_h],
                    face_detector
                )

                # Guard 1: face crop extraction failed
                if face_crop is None:
                    result = {"label": "UNKNOWN", "confidence": 0.0,
                              "is_looking": False, "skip_reason": reason}
                    last_observer_results[i] = result
                    observer_results.append(result)
                    continue

                # Guard 2: crop too small for meaningful inference
                if face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
                    result = {"label": "UNKNOWN", "confidence": 0.0,
                              "is_looking": False, "skip_reason": "crop_below_minimum"}
                    last_observer_results[i] = result
                    observer_results.append(result)
                    continue

                # OPTIMIZATION: resize face crop to 112×112 before MobileNetV2
                # (predictor upscales to 224×224 internally — this reduces data transfer)
                face_crop = cv2.resize(face_crop, (112, 112))

                result = predictor.predict(face_crop)
                result["is_estimate"] = (reason == "fallback_estimate")

                last_observer_results[i] = result
                observer_results.append(result)

            # ----------------------------------------------------------------
            # E) Decision engine → SAFE or THREAT
            # ----------------------------------------------------------------
            state       = engine.update(
                num_persons      = 1 + len(observer_boxes),
                observer_results = observer_results
            )
            status_info = engine.get_status_info()

            if state != prev_state:
                print(f"State changed → {state}")
                prev_state = state

            # ----------------------------------------------------------------
            # F) Draw UI on display_frame
            # ----------------------------------------------------------------

            # Status bar — green for SAFE, red for THREAT
            if state == "SAFE":
                cv2.rectangle(display_frame, (0, 0), (640, 50), (0, 180, 0), -1)
                cv2.putText(display_frame, "SAFE -- No Threat Detected",
                            (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(display_frame, (0, 0), (640, 50), (0, 0, 200), -1)
                cv2.putText(display_frame, "THREAT DETECTED -- Observer Looking!",
                            (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)

            # User box — BLUE
            if user_box is not None:
                ux1, uy1, ux2, uy2, u_conf = user_box
                cv2.rectangle(display_frame, (ux1, uy1), (ux2, uy2), (220, 100, 0), 2)
                cv2.putText(display_frame, f"USER ({u_conf:.0%})",
                            (ux1, max(uy1 - 8, 55)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 100, 0), 2, cv2.LINE_AA)

            # Observer boxes — RED + prediction label
            for obs_box, obs_result in zip(observer_boxes, observer_results):
                ox1, oy1, ox2, oy2, o_conf = obs_box
                cv2.rectangle(display_frame, (ox1, oy1), (ox2, oy2), (0, 40, 210), 2)
                cv2.putText(display_frame, f"OBSERVER ({o_conf:.0%})",
                            (ox1, max(oy1 - 8, 55)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 40, 210), 2, cv2.LINE_AA)

                pred_label  = obs_result.get("label", "UNKNOWN")
                pred_conf   = obs_result.get("confidence", 0.0)
                is_looking  = obs_result.get("is_looking", False)
                is_estimate = obs_result.get("is_estimate", False)

                if pred_label in ("UNKNOWN", "UNCERTAIN"):
                    ann_text  = pred_label
                    ann_color = (180, 180, 180)   # Grey
                elif is_looking:
                    ann_text  = f"LOOKING ({pred_conf:.1f}%)"
                    ann_color = (0, 40, 210)       # Red
                else:
                    ann_text  = f"NOT LOOKING ({pred_conf:.1f}%)"
                    ann_color = (40, 180, 40)       # Green

                if is_estimate:
                    ann_text += " (est.)"

                ann_y = min(oy2 + 22, display_frame.shape[0] - 6)
                cv2.putText(display_frame, ann_text, (ox1, ann_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, ann_color, 2, cv2.LINE_AA)

            # FPS counter — bottom-right corner
            fps      = 1.0 / max(time.time() - fps_time, 1e-6)
            fps_time = time.time()
            fps_text = f"FPS: {fps:.1f}"
            (fps_tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.putText(display_frame, fps_text,
                        (display_frame.shape[1] - fps_tw - 8, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            # Terminal FPS log every 30 frames
            if frame_count % 30 == 0:
                yolo_src = "YOLO"   if frame_count % YOLO_INTERVAL == 1 else "cached"
                pose_src = "live"   if frame_count % POSE_INTERVAL == 0 else "cached"
                print(f"[Frame {frame_count:>5}] FPS:{fps:.1f}  Persons:{total_persons}"
                      f"  State:{state}  YOLO:{yolo_src}  Pose:{pose_src}")

            # Person count — bottom-left corner
            cv2.putText(display_frame, f"Persons: {total_persons}",
                        (8, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            # ----------------------------------------------------------------
            # G) Display frame + key handling
            #    ONE imshow window — never recreated inside the loop
            # ----------------------------------------------------------------
            cv2.imshow("Shoulder Surfing Prevention System", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("'q' pressed — shutting down.")
                break

            if key == ord('s'):
                filename = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Screenshot saved: {filename}")

    finally:
        # Always runs — even if an exception crashes the loop
        cap.release()
        face_detector.close()
        cv2.destroyAllWindows()
        print("System shut down cleanly.")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    main()
