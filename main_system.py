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

import cv2
import numpy as np
import time
import threading                          # FIXED: merge conflict — HEAD version kept
from ultralytics import YOLO
import mediapipe as mp
from head_pose_predictor import HeadPosePredictor
from decision_engine import ThreatDecisionEngine
from face_crop_utils import extract_padded_face_crop


# ============================================================================
# OPTIONAL: load ScreenProtector (graceful fallback if it fails)
# ============================================================================
# We wrap the import and startup in try/except so that if screen_protector.py
# has a dependency problem (e.g. PIL not installed, permissions issue), the
# whole system still works — it just won't blur the screen on threat.

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
# threading.Lock() prevents both threads from reading/writing the flag at
# the exact same moment (which could cause corrupted or partially-updated data).

_state_lock        = threading.Lock()
_protection_active = False    # True = show blur overlay, False = hide it
_should_quit       = False    # True = user pressed 'q' → both threads exit


def set_protection(active: bool):
    """
    Thread-safe setter for the protection flag.
    Called from the BACKGROUND thread (webcam loop).
    """
    global _protection_active
    with _state_lock:
        _protection_active = active


def get_protection() -> bool:
    """
    Thread-safe getter for the protection flag.
    Called from the MAIN thread (tkinter loop).
    """
    with _state_lock:
        return _protection_active


# ============================================================================
# WEBCAM / AI THREAD — runs in the background
# ============================================================================

def webcam_thread():
    """
    Runs the entire YOLO + face crop + head pose + decision engine pipeline.
    Sets the shared _protection_active flag based on the result each frame.
    Exits cleanly when _should_quit becomes True.
    """
    global _should_quit

    # ------------------------------------------------------------------
    # Initialization — all the AI models are loaded HERE (in this thread)
    # ------------------------------------------------------------------
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
    engine    = ThreatDecisionEngine(threat_threshold=10, cooldown_frames=30)

    mp_face       = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    # Frame-skip optimisation: run YOLO every 3rd frame, cache the rest
    YOLO_INTERVAL = 3
    cached_boxes  = []
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
            # B) YOLO detection — every YOLO_INTERVAL frames
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
                cached_boxes = raw_boxes
            else:
                raw_boxes = cached_boxes

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
            # D) Face crop + head pose for each OBSERVER only
            #    NEVER run head pose on the user — skips one predictor call
            #    per frame (~10-20ms saved on CPU).
            # ----------------------------------------------------------------
            observer_results = []

            for obs_box in observer_boxes:
                x1, y1, x2, y2, _ = obs_box

                face_crop, reason = extract_padded_face_crop(
                    frame, [x1, y1, x2, y2], face_detector
                )

                # Guard 1: crop extraction failed entirely
                if face_crop is None:
                    observer_results.append({
                        "label": "UNKNOWN", "confidence": 0.0,
                        "is_looking": False, "skip_reason": reason
                    })
                    continue

                # Guard 2: crop present but below the 40×40 minimum
                # (model cannot produce meaningful predictions on tiny inputs)
                if face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
                    observer_results.append({
                        "label": "UNKNOWN", "confidence": 0.0,
                        "is_looking": False, "skip_reason": "crop_below_minimum"
                    })
                    continue

                # All guards passed — run head pose prediction
                result = predictor.predict(face_crop)
                result["is_estimate"] = (reason == "fallback_estimate")
                observer_results.append(result)

            # ----------------------------------------------------------------
            # E) Decision engine → SAFE or THREAT
            # ----------------------------------------------------------------
            state       = engine.update(
                num_persons      = 1 + len(observer_boxes),
                observer_results = observer_results
            )
            status_info = engine.get_status_info()

            # Update the shared flag so the main thread can react
            set_protection(state == "THREAT")

            if state != prev_state:
                print(f"[Webcam Thread] State changed → {state}")
                prev_state = state

            # ----------------------------------------------------------------
            # F) Draw on display frame
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

                # Colour-code the prediction label
                if pred_label in ("UNKNOWN", "UNCERTAIN"):
                    ann_text, ann_color = pred_label, (180, 180, 180)   # Grey
                elif is_looking:
                    ann_text  = f"LOOKING ({pred_conf:.1f}%)"
                    ann_color = (0, 40, 210)                             # Red
                else:
                    ann_text  = f"NOT LOOKING ({pred_conf:.1f}%)"
                    ann_color = (40, 180, 40)                            # Green

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
                det = "YOLO" if frame_count % YOLO_INTERVAL == 1 else "cached"
                print(f"[Frame {frame_count:>5}] FPS:{fps:.1f}  Persons:{total_persons}"
                      f"  State:{state}  Det:{det}")

            # Person count (bottom-left)
            cv2.putText(display_frame, f"Persons: {total_persons}",
                        (8, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            # ----------------------------------------------------------------
            # G) Show YOLO window + key handling
            #    ONE imshow window, never destroyed inside the loop.
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
                # Manual toggle: flip current protection state
                # 'p' = "protection" — useful for demo / testing
                current = get_protection()
                set_protection(not current)
                print(f"[Webcam Thread] Manual toggle → {'ON' if not current else 'OFF'}")

    finally:
        cap.release()                  # Release webcam — ONE release, here in finally
        face_detector.close()
        cv2.destroyAllWindows()        # Close the ONE imshow window
        print("[Webcam Thread] Cleaned up and exiting.")
        _should_quit = True            # Signal main thread to stop too


# ============================================================================
# MAIN THREAD — runs tkinter (ScreenProtector)
# ============================================================================

def main():
    global _should_quit

    print("=" * 60)
    print("  Shoulder Surfing Prevention System — Starting")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Create and start the ScreenProtector (must be on main thread)
    # ------------------------------------------------------------------
    protector       = None
    protector_ok    = False
    last_prot_state = False    # Track last known state to avoid redundant calls

    if _PROTECTOR_AVAILABLE:
        try:
            protector = ScreenProtector()
            protector.start()    # Creates the tkinter window on the main thread
            protector_ok = True
            print("[Main] ScreenProtector started.")
        except Exception as e:
            print(f"[Main] ScreenProtector start() failed: {e}")
            print("[Main] Continuing without screen blur.")
            protector_ok = False

    # ------------------------------------------------------------------
    # 2. Start the webcam/AI pipeline in a background thread
    # ------------------------------------------------------------------
    # daemon=True means: if the main thread exits, this thread is also
    # killed automatically — no orphaned processes left running.
    cam_thread = threading.Thread(target=webcam_thread, name="WebcamThread", daemon=True)
    cam_thread.start()
    print("[Main] Webcam thread started.")
    print("[Main] Controls: 'q' = quit  |  's' = screenshot  |  'p' = toggle protection")

    # ------------------------------------------------------------------
    # 3. Main thread loop — drives tkinter and syncs protection state
    # ------------------------------------------------------------------
    # This loop runs at ~100 fps (10ms sleep) to keep the overlay smooth.
    # The heavy AI work happens in cam_thread, so this loop is very fast.

    try:
        while not _should_quit:

            # Check the shared flag and update protector if it changed
            if protector_ok and protector is not None:
                current_prot = get_protection()

                if current_prot != last_prot_state:
                    # State changed since last frame — call activate or deactivate
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

                # Drive the tkinter event loop.
                # Without this call the overlay window would freeze.
                try:
                    protector.update()
                except Exception as e:
                    print(f"[Main] protector.update() error: {e}")
                    protector_ok = False

            time.sleep(0.01)    # ~100 Hz update rate for the overlay

    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt — shutting down.")
        _should_quit = True

    finally:
        # Clean up ScreenProtector
        if protector is not None:
            try:
                protector.stop()
                print("[Main] ScreenProtector stopped.")
            except Exception:
                pass

        # Wait for the webcam thread to finish
        if cam_thread.is_alive():
            cam_thread.join(timeout=3.0)
            if cam_thread.is_alive():
                print("[Main] Webcam thread did not exit in time — forcing quit.")

        print("[Main] System shut down cleanly.")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    main()
