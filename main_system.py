# main_system.py
# ============================================================================
# Shoulder Surfing Prevention System — Full Integration with Screen Protection
# ============================================================================
# THREAD ARCHITECTURE:
#
#   MAIN THREAD  (required by tkinter)
#   └── runs the ScreenProtector overlay event loop
#       └── watches a shared flag: protection_active
#           → calls activate_protection() / deactivate_protection() when it changes
#
#   BACKGROUND THREAD  (webcam + AI)
#   └── reads webcam frames
#   └── runs YOLO → classify persons → extract face → head pose → decision engine
#   └── sets protection_active = True/False based on SAFE/THREAT state
#
# WHY TWO THREADS?
#   tkinter (the GUI toolkit used by ScreenProtector) has a strict rule:
#   ALL GUI operations MUST happen on the thread that created the window
#   — the "main thread". If another thread tries to touch tkinter, the app
#   crashes or behaves unpredictably.
#
#   But our webcam loop is a blocking while True — it would freeze the
#   tkinter event loop entirely if both ran on the main thread.
#
#   Solution: webcam loop → background thread.
#             tkinter GUI  → main thread.
#   They share one flag (protection_active) to communicate state.
#
# Run with:  python main_system.py
# Quit with: press 'q' in the YOLO window
# ============================================================================

# ═══════════════════════════════════════════════════════
# PERFORMANCE TUNING GUIDE
# ═══════════════════════════════════════════════════════
# Adjust these values to trade off speed vs. accuracy/responsiveness.
#
# pose_interval = 3
#   Run MediaPipe + MobileNetV2 every Nth frame.
#   Higher = faster FPS but slower reaction to gaze changes.
#   (e.g. 3 = inference runs ~20fps at 60fps camera)
#
# yolo_interval = 2
#   Run YOLO person detection every Nth frame.
#   Higher = faster FPS but slower reaction to new persons entering frame.
#
# threat_threshold = 4
#   Consecutive THREAT frames required before declaring THREAT state.
#   Lower = faster response, but more sensitive to false positives.
#   (was 10 — reduced for faster reaction time)
#
# cooldown_frames = 20
#   Consecutive SAFE frames required before returning to SAFE state.
#   Lower = faster recovery after threat leaves.
#   (was 30 — reduced to recover faster)
#
# observer_streak = 3  (set in ThreatDecisionEngine, shown here for reference)
#   Consecutive LOOKING frames per observer needed before counting as THREAT.
#   Lower = more sensitive, quicker to react.
#   (was 5 — reduced for faster reaction time)
#
# confidence_threshold = 75%  (set in ThreatDecisionEngine)
#   Minimum MobileNetV2 confidence required for a LOOKING result to count.
#   Lower = more sensitive, slightly higher false positive rate.
#   (was 80% — lowered slightly for responsiveness)
#
# gap_threshold = 0.25  (set in head_pose_predictor.py)
#   Probability gap needed between LOOKING and NOT_LOOKING to confirm LOOKING.
#   Lower = more sensitive, slightly more false positives on side faces.
#   (was 0.30 — lowered for faster THREAT detection)
# ═══════════════════════════════════════════════════════

import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
import mediapipe as mp
from head_pose_predictor import HeadPosePredictor
from decision_engine import ThreatDecisionEngine
from face_crop_utils import extract_padded_face_crop


# ============================================================================
# OPTIONAL: load ScreenProtector (graceful fallback if it fails)
# ============================================================================
try:
    from screen_protector import ScreenProtector
    _PROTECTOR_AVAILABLE = True
except Exception as e:
    print(f"[Warning] screen_protector.py could not be loaded: {e}")
    print("[Warning] System will run WITHOUT screen blur protection.")
    _PROTECTOR_AVAILABLE = False


# ============================================================================
# SHARED STATE — communication bridge between the two threads
# ============================================================================
_state_lock        = threading.Lock()
_protection_active = False
_should_quit       = False


def set_protection(active: bool):
    global _protection_active
    with _state_lock:
        _protection_active = active


def get_protection() -> bool:
    with _state_lock:
        return _protection_active


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def resize_to_max_width(img, max_width=320):
    """
    OPTIMIZATION 2 HELPER: Resize an image so its width does not exceed
    max_width, preserving the aspect ratio.

    WHY: MediaPipe FaceDetection runs on pixel data. A 640-wide person crop
    takes ~4× more work to process than a 320-wide crop. Since we only need
    approximate face coordinates (not pixel-perfect), half resolution is fine.

    Parameters:
        img       : numpy array (BGR image)
        max_width : maximum allowed width in pixels

    Returns:
        Resized image if wider than max_width, otherwise original unchanged.
    """
    if img.shape[1] <= max_width:
        return img   # Already small enough — no resize needed
    scale = max_width / img.shape[1]
    new_h = int(img.shape[0] * scale)
    return cv2.resize(img, (max_width, new_h))


# ============================================================================
# WEBCAM / AI THREAD — runs in the background
# ============================================================================

def webcam_thread():
    global _should_quit

    print("[Webcam Thread] Initializing detection pipeline...")

    cap = cv2.VideoCapture(0)     # ONE VideoCapture, opened once
    if not cap.isOpened():
        print("[Webcam Thread] ERROR: Could not open webcam.")
        _should_quit = True
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    model     = YOLO("yolov8n.pt")
    predictor = HeadPosePredictor("best_head_pose_model.pth")
    # threat_threshold and cooldown_frames updated — see TUNING GUIDE above
    engine    = ThreatDecisionEngine(threat_threshold=4, cooldown_frames=20)

    mp_face       = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    # ------------------------------------------------------------------
    # OPTIMIZATION 4: YOLO frame-skip
    # Run YOLO every yolo_interval frames instead of every frame.
    # Between YOLO frames, reuse the last known bounding boxes.
    # WHY: YOLO is the heaviest single step (~15-25ms on CPU). Cutting it
    # to every 2nd frame halves this cost with barely any visible degradation
    # since persons move slowly relative to frame rate.
    # ------------------------------------------------------------------
    YOLO_INTERVAL   = 2          # Was: ran every frame. Now: every 2nd frame.
    last_yolo_boxes = []         # Cache: last YOLO detections reused on skip frames

    # ------------------------------------------------------------------
    # OPTIMIZATION 1: Head pose frame-skip
    # Run MediaPipe + MobileNetV2 every pose_interval frames instead of every frame.
    # Between pose frames, reuse the last known result for each observer.
    # WHY: MobileNetV2 inference is expensive (~20-40ms per call on CPU).
    # Running it every 3rd frame reduces this cost by ~66% while still
    # updating the gaze classification ~20 times per second at 60fps.
    # ------------------------------------------------------------------
    POSE_INTERVAL         = 3   # Run head pose every 3rd frame
    last_observer_results = {}  # Cache: {observer_index: last predict() result}
    last_observer_count   = 0   # Track count to know when observers change

    frame_count   = 0
    fps_time      = time.time()
    prev_state        = None
    prev_person_count = -1

    print("[Webcam Thread] Ready. Starting detection loop...")

    try:
        while not _should_quit:

            # ----------------------------------------------------------------
            # A) Read frame
            # ----------------------------------------------------------------
            ret, frame = cap.read()
            if not ret:
                print("[Webcam Thread] Frame read failed — retrying...")
                continue

            frame_count  += 1
            display_frame = frame.copy()

            # Resize if camera gives a larger frame than 640×480
            h_frame, w_frame = frame.shape[:2]
            if w_frame > 640 or h_frame > 480:
                frame         = cv2.resize(frame, (640, 480))
                display_frame = frame.copy()

            # ----------------------------------------------------------------
            # B) YOLO detection — OPTIMIZATION 4: every YOLO_INTERVAL frames
            #
            # On YOLO frames: run full detection, store result in last_yolo_boxes
            # On skip frames: reuse last_yolo_boxes from the previous YOLO frame
            # ----------------------------------------------------------------
            if frame_count % YOLO_INTERVAL == 1:
                results   = model(frame, classes=[0], verbose=False)
                raw_boxes = []
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    if conf < 0.5:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    raw_boxes.append([x1, y1, x2, y2, conf])
                last_yolo_boxes = raw_boxes   # Update cache
            else:
                raw_boxes = last_yolo_boxes   # Reuse previous frame's boxes

            # ----------------------------------------------------------------
            # C) Classify persons: largest box = USER, rest = OBSERVERs
            # ----------------------------------------------------------------
            if len(raw_boxes) == 0:
                user_box, observer_boxes = None, []
            elif len(raw_boxes) == 1:
                user_box, observer_boxes = raw_boxes[0], []
            else:
                raw_boxes.sort(
                    key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                    reverse=True
                )
                user_box       = raw_boxes[0]
                observer_boxes = raw_boxes[1:]

            total_persons = len(raw_boxes)
            if total_persons != prev_person_count:
                print(f"[Webcam Thread] Persons detected: {total_persons}")
                prev_person_count = total_persons

            # ----------------------------------------------------------------
            # Clear the observer result cache when the observer count changes.
            # WHY: if observer A leaves and observer B enters, observer B gets
            # index 0 — we must not show observer A's stale result for B.
            # Also clear when count drops (someone left the frame).
            # ----------------------------------------------------------------
            current_observer_count = len(observer_boxes)
            if current_observer_count != last_observer_count:
                last_observer_results = {}   # Stale data — clear everything
                last_observer_count   = current_observer_count

            # ----------------------------------------------------------------
            # D) Face crop + head pose for each OBSERVER only
            #
            # OPTIMIZATION 1: Only run MediaPipe + MobileNetV2 on pose frames.
            # On skip frames, reuse last_observer_results[i] for each observer.
            #
            # OPTIMIZATION 2: Resize person crop to max 320px wide before
            # MediaPipe so it processes less data.
            #
            # OPTIMIZATION 3: Resize face crop to 112×112 before MobileNetV2.
            # The internal transform upscales to 224×224 anyway, so halving
            # the input data saves time in the data pipeline.
            # ----------------------------------------------------------------
            observer_results = []
            run_pose = (frame_count % POSE_INTERVAL == 0)  # True on pose frames

            for i, obs_box in enumerate(observer_boxes):
                x1, y1, x2, y2, _ = obs_box

                # On skip frames: reuse cached result if available
                if not run_pose and i in last_observer_results:
                    observer_results.append(last_observer_results[i])
                    continue   # Skip MediaPipe + MobileNetV2 this frame

                # ---- POSE FRAME: run full inference ----

                # OPTIMIZATION 2: resize person crop before MediaPipe
                person_crop_full = frame[y1:y2, x1:x2]
                if person_crop_full.size == 0:
                    result = {"label": "UNKNOWN", "confidence": 0.0,
                              "is_looking": False, "skip_reason": "empty_crop"}
                    last_observer_results[i] = result
                    observer_results.append(result)
                    continue

                # Scale the crop down so MediaPipe runs on fewer pixels
                person_crop_small = resize_to_max_width(person_crop_full, max_width=320)

                # Run face detection on the smaller crop, passing it as the frame.
                # extract_padded_face_crop expects the full frame + person box,
                # so we pass the small crop as the "frame" with a full-size box.
                crop_h_small, crop_w_small = person_crop_small.shape[:2]
                face_crop, reason = extract_padded_face_crop(
                    person_crop_small,
                    [0, 0, crop_w_small, crop_h_small],
                    face_detector
                )

                # Guard 1: crop extraction failed entirely
                if face_crop is None:
                    result = {"label": "UNKNOWN", "confidence": 0.0,
                              "is_looking": False, "skip_reason": reason}
                    last_observer_results[i] = result
                    observer_results.append(result)
                    continue

                # Guard 2: crop below 40×40 minimum
                if face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
                    result = {"label": "UNKNOWN", "confidence": 0.0,
                              "is_looking": False, "skip_reason": "crop_below_minimum"}
                    last_observer_results[i] = result
                    observer_results.append(result)
                    continue

                # OPTIMIZATION 3: resize face crop to 112×112 before MobileNetV2.
                # The predictor's internal transform.Resize((224,224)) will
                # upscale it — but the data transfer and initial processing
                # is faster with a smaller input.
                face_crop = cv2.resize(face_crop, (112, 112))

                # All guards passed — run head pose prediction
                result = predictor.predict(face_crop)
                result["is_estimate"] = (reason == "fallback_estimate")

                # Store in cache for reuse on skip frames
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

            set_protection(state == "THREAT")

            if state != prev_state:
                print(f"[Webcam Thread] State changed → {state}")
                prev_state = state

            # ----------------------------------------------------------------
            # F) Draw on display frame (unchanged)
            # ----------------------------------------------------------------

            # Status bar (green = SAFE, red = THREAT)
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

            # User box (BLUE)
            if user_box is not None:
                ux1, uy1, ux2, uy2, u_conf = user_box
                cv2.rectangle(display_frame, (ux1, uy1), (ux2, uy2), (220, 100, 0), 2)
                cv2.putText(display_frame, f"USER ({u_conf:.0%})",
                            (ux1, max(uy1 - 8, 55)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 100, 0), 2, cv2.LINE_AA)

            # Observer boxes (RED) + prediction labels
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
                    ann_text, ann_color = pred_label, (180, 180, 180)
                elif is_looking:
                    ann_text  = f"LOOKING ({pred_conf:.1f}%)"
                    ann_color = (0, 40, 210)
                else:
                    ann_text  = f"NOT LOOKING ({pred_conf:.1f}%)"
                    ann_color = (40, 180, 40)

                if is_estimate:
                    ann_text += " (est.)"

                ann_y = min(oy2 + 22, display_frame.shape[0] - 6)
                cv2.putText(display_frame, ann_text, (ox1, ann_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, ann_color, 2, cv2.LINE_AA)

            # FPS counter (bottom-right)
            fps      = 1.0 / max(time.time() - fps_time, 1e-6)
            fps_time = time.time()
            fps_text = f"FPS: {fps:.1f}"
            (fps_tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.putText(display_frame, fps_text,
                        (display_frame.shape[1] - fps_tw - 8, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            # FPS terminal print every 30 frames
            if frame_count % 30 == 0:
                yolo_src = "YOLO" if frame_count % YOLO_INTERVAL == 1 else "cached"
                pose_src = "live" if frame_count % POSE_INTERVAL == 0 else "cached"
                print(f"[Frame {frame_count:>5}] FPS:{fps:.1f}  Persons:{total_persons}"
                      f"  State:{state}  YOLO:{yolo_src}  Pose:{pose_src}")

            # Person count (bottom-left)
            cv2.putText(display_frame, f"Persons: {total_persons}",
                        (8, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            # ----------------------------------------------------------------
            # G) Show YOLO window + key handling
            # ----------------------------------------------------------------
            cv2.imshow("Shoulder Surfing Prevention System", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[Webcam Thread] 'q' pressed — signalling shutdown.")
                _should_quit = True
                break

            if key == ord('s'):
                filename = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"[Webcam Thread] Screenshot saved: {filename}")

            if key == ord('p'):
                current = get_protection()
                set_protection(not current)
                print(f"[Webcam Thread] Manual toggle → {'ON' if not current else 'OFF'}")

    finally:
        cap.release()
        face_detector.close()
        cv2.destroyAllWindows()
        print("[Webcam Thread] Cleaned up and exiting.")
        _should_quit = True


# ============================================================================
# MAIN THREAD — runs tkinter (ScreenProtector)
# ============================================================================

def main():
    global _should_quit

    print("=" * 60)
    print("  Shoulder Surfing Prevention System — Starting")
    print("=" * 60)

    protector       = None
    protector_ok    = False
    last_prot_state = False

    if _PROTECTOR_AVAILABLE:
        try:
            protector = ScreenProtector()
            protector.start()
            protector_ok = True
            print("[Main] ScreenProtector started.")
        except Exception as e:
            print(f"[Main] ScreenProtector start() failed: {e}")
            print("[Main] Continuing without screen blur.")
            protector_ok = False

    cam_thread = threading.Thread(target=webcam_thread, name="WebcamThread", daemon=True)
    cam_thread.start()
    print("[Main] Webcam thread started.")
    print("[Main] Controls: 'q' = quit  |  's' = screenshot  |  'p' = toggle protection")

    try:
        while not _should_quit:
            if protector_ok and protector is not None:
                current_prot = get_protection()

                if current_prot != last_prot_state:
                    if current_prot:
                        try:
                            protector.activate_protection()
                            print("[Main] Screen protection ACTIVATED.")
                        except Exception as e:
                            print(f"[Main] activate_protection() error: {e}")
                    else:
                        try:
                            protector.deactivate_protection()
                            print("[Main] Screen protection DEACTIVATED.")
                        except Exception as e:
                            print(f"[Main] deactivate_protection() error: {e}")

                    last_prot_state = current_prot

                try:
                    protector.update()
                except Exception as e:
                    print(f"[Main] protector.update() error: {e}")
                    protector_ok = False

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt — shutting down.")
        _should_quit = True

    finally:
        if protector is not None:
            try:
                protector.stop()
                print("[Main] ScreenProtector stopped.")
            except Exception:
                pass

        if cam_thread.is_alive():
            cam_thread.join(timeout=3.0)
            if cam_thread.is_alive():
                print("[Main] Webcam thread did not exit in time — forcing quit.")

        print("[Main] System shut down cleanly.")


if __name__ == "__main__":
    main()
