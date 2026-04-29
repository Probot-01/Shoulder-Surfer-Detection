# main_system.py
# ============================================================================
# Shoulder Surfing Prevention System — Complete Integration Script
# ============================================================================
# Combines:
#   YOLO         → detect all persons in the webcam frame
#   MediaPipe    → find the face within each observer's bounding box
#   HeadPosePredictor → classify face as LOOKING or NOT_LOOKING
#   ThreatDecisionEngine → apply temporal smoothing → SAFE or THREAT state
#
# Run with:  python main_system.py
# Quit with: press 'q' in the video window
# Screenshot: press 's'
# ============================================================================

import cv2
import numpy as np
import time
from ultralytics import YOLO
import mediapipe as mp
from head_pose_predictor import HeadPosePredictor
from decision_engine import ThreatDecisionEngine
from face_crop_utils import extract_padded_face_crop


# ============================================================================
# WRAP EVERYTHING in try/finally so cleanup always runs,
# even if an unexpected error crashes the program mid-way.
# ============================================================================
try:

    # ========================================================================
    # INITIALIZATION — runs ONCE before the main loop
    # ========================================================================
    print("Initializing system...")

    # ------------------------------------------------------------------------
    # 1. Open the webcam
    #    VideoCapture(0) = first/default camera on the system.
    #    Created ONCE here, released ONCE in the finally block below.
    # ------------------------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check it is connected and not in use.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ------------------------------------------------------------------------
    # 2. Load the YOLO person detector
    #    YOLOv8n is the smallest/fastest variant — good for real-time use.
    #    It auto-downloads yolov8n.pt the first time if not already cached.
    # ------------------------------------------------------------------------
    model = YOLO("yolov8n.pt")

    # ------------------------------------------------------------------------
    # 3. Load the head pose classifier (MobileNetV2 trained on BIWI dataset)
    # ------------------------------------------------------------------------
    predictor = HeadPosePredictor("best_head_pose_model.pth")

    # ------------------------------------------------------------------------
    # 4. Create the threat decision engine
    #    threat_threshold=10 : needs 10 consecutive THREAT frames to alert
    #    cooldown_frames=30  : needs 30 consecutive SAFE frames to clear alert
    # ------------------------------------------------------------------------
    engine = ThreatDecisionEngine(threat_threshold=10, cooldown_frames=30)

    # ------------------------------------------------------------------------
    # 5. Initialize MediaPipe Face Detection — ONCE, before the loop
    #    model_selection=0 → short-range model (best for faces < ~2 metres away)
    #    min_detection_confidence=0.5 → only accept faces the model is ≥50% sure about
    #    Closed properly in the finally block using face_detector.close()
    # ------------------------------------------------------------------------
    mp_face      = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )

    # ------------------------------------------------------------------------
    # 6. State tracking variables (for "print only on change" logic)
    # ------------------------------------------------------------------------
    frame_count   = 0
    fps_time      = time.time()

    prev_state         = None   # Track last printed state so we don't spam
    prev_person_count  = -1     # Track last printed person count

    # -------------------------------------------------------------------------
    # SPEED OPTIMISATION 1: YOLO frame-skip
    #   Running YOLO on every frame is expensive (~15-30ms per call).
    #   Persons move slowly relative to frame rate, so we can safely reuse
    #   the last detections for 2 frames between each full YOLO call.
    #   YOLO_INTERVAL=3 → YOLO runs on frames 1, 4, 7, 10 ... (roughly 1/3 cost)
    # -------------------------------------------------------------------------
    YOLO_INTERVAL = 3     # Run YOLO every Nth frame
    cached_boxes  = []    # Last known bounding boxes (reused on skip frames)

    print("System ready. Press q to quit.")

    # ========================================================================
    # MAIN LOOP — one iteration = one video frame
    # ========================================================================
    while True:

        # ====================================================================
        # A) READ FRAME FROM WEBCAM
        # ====================================================================
        ret, frame = cap.read()
        if not ret:
            # This can happen if the camera is briefly unavailable — just
            # skip this frame and try again next iteration
            print("Frame read failed")
            continue

        frame_count   += 1
        # All drawing happens on this COPY so the original frame stays clean
        # for any pixel-reading operations (face crops, etc.)
        display_frame  = frame.copy()

        # ====================================================================
        # SPEED OPTIMISATION 2: Resize frame if larger than 640x480
        #   Some cameras ignore the cap.set() resolution hints and deliver
        #   larger frames. Resizing here ensures every downstream step
        #   (YOLO, MediaPipe, head pose) works on small pixels — faster.
        # ====================================================================
        h_frame, w_frame = frame.shape[:2]
        if w_frame > 640 or h_frame > 480:
            frame         = cv2.resize(frame, (640, 480))
            display_frame = frame.copy()  # Keep display copy in sync

        # ====================================================================
        # B) YOLO PERSON DETECTION — SPEED OPTIMISATION 3: frame-skip caching
        #
        #   On "YOLO frames" (frame_count % YOLO_INTERVAL == 1), run the full
        #   detector and save the result in cached_boxes.
        #   On all other frames, skip detection and reuse cached_boxes.
        #   Because persons move slowly vs frame rate, boxes stay accurate
        #   across 2-3 frames with no visible degradation.
        #
        #   classes=[0] → persons only; ignores cars, bags, etc. (faster YOLO)
        # ====================================================================
        if frame_count % YOLO_INTERVAL == 1:   # Frames 1, 4, 7, 10 ...
            results   = model(frame, classes=[0], verbose=False)
            raw_boxes = []
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                raw_boxes.append([x1, y1, x2, y2, conf])
            cached_boxes = raw_boxes          # Cache for the next N-1 frames
        else:
            raw_boxes = cached_boxes          # Reuse last known boxes

        # ====================================================================
        # C) CLASSIFY PERSONS
        #    The person CLOSEST to the camera appears LARGEST in the frame.
        #    We assume the primary user is closest → user = largest box.
        #    Everyone else is a potential observer/shoulder-surfer.
        # ====================================================================
        if len(raw_boxes) == 0:
            # Nobody visible
            user_box       = None
            observer_boxes = []

        elif len(raw_boxes) == 1:
            # Only one person → must be the user
            user_box       = raw_boxes[0]
            observer_boxes = []

        else:
            # Multiple people → sort by box area descending
            # Area = width × height = (x2-x1) × (y2-y1)
            raw_boxes.sort(
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                reverse=True   # largest first
            )
            user_box       = raw_boxes[0]       # Biggest box = user
            observer_boxes = raw_boxes[1:]      # Everything else = observers

        # Print person count only when it changes (avoid console spam)
        total_persons = len(raw_boxes)
        if total_persons != prev_person_count:
            print(f"Persons detected: {total_persons} total")
            prev_person_count = total_persons

        # ====================================================================
        # D) FACE DETECTION + HEAD POSE PREDICTION (observers only)
        #
        # SPEED OPTIMISATION 4: never run head pose on the main user.
        #   The user's own gaze does not affect the threat decision — only
        #   observer gaze matters. Skipping it saves one full predictor call
        #   per frame, which is significant (~10-20ms on CPU).
        # ====================================================================
        observer_results = []   # Will hold one dict per observer

        for i, obs_box in enumerate(observer_boxes):
            x1, y1, x2, y2, _ = obs_box

            print("Observer found — running head pose...")

            # ----------------------------------------------------------------
            # D1-D4) Extract a padded face crop using the utility function.
            #   extract_padded_face_crop() handles all of:
            #     - Person box clipping and size validation
            #     - MediaPipe face detection on the person crop
            #     - 40% padding around the detected face
            #     - Fallback to top-28% estimate if no face is found
            #   It returns (face_crop, reason_string) so we know what happened.
            # ----------------------------------------------------------------
            face_crop, reason = extract_padded_face_crop(
                frame, [x1, y1, x2, y2], face_detector
            )

            # TEMPORARY DEBUG — remove once the quality gate is confirmed working
            print(f"Observer {i}: crop={reason}, size={face_crop.shape[:2] if face_crop is not None else 'N/A'}")

            # ----------------------------------------------------------------
            # Guard 1: crop extraction failed entirely
            # ----------------------------------------------------------------
            if face_crop is None:
                print(f"  Skipping head pose — crop unavailable ({reason})")
                observer_results.append({
                    "label": "UNKNOWN", "confidence": 0.0,
                    "is_looking": False, "is_estimate": False,
                    "skip_reason": reason
                })
                continue   # Move on to the next observer

            # ----------------------------------------------------------------
            # Guard 2: crop is present but still too small for the model
            #   224×224 models struggle badly with tiny inputs — even if the
            #   crop technically exists, below 40×40 it contains too little
            #   information to produce a meaningful prediction.
            # ----------------------------------------------------------------
            if face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
                skip_reason = "crop_below_minimum"
                print(f"  Skipping head pose — crop unavailable ({skip_reason})")
                observer_results.append({
                    "label": "UNKNOWN", "confidence": 0.0,
                    "is_looking": False, "is_estimate": False,
                    "skip_reason": skip_reason
                })
                continue   # Move on to the next observer

            # ----------------------------------------------------------------
            # All guards passed — run head pose prediction
            # ----------------------------------------------------------------
            result = predictor.predict(face_crop)
            result["is_estimate"]  = (reason == "fallback_estimate")
            result["skip_reason"]  = None   # No issue — prediction ran normally

            observer_results.append(result)

        # ====================================================================
        # E) DECISION ENGINE
        #    Feed this frame's results in; get back the current SAFE/THREAT state.
        #    num_persons = 1 (user) + number of observers
        # ====================================================================
        state       = engine.update(
            num_persons      = 1 + len(observer_boxes),
            observer_results = observer_results
        )
        status_info = engine.get_status_info()

        # Print state only when it changes
        if state != prev_state:
            print(f"State: {state}")
            prev_state = state

        # ====================================================================
        # F) DRAW EVERYTHING on display_frame
        # ====================================================================

        # --------------------------------------------------------------------
        # F1) STATUS BAR — full-width coloured banner at the top (50px tall)
        # --------------------------------------------------------------------
        if state == "SAFE":
            cv2.rectangle(display_frame, (0, 0), (640, 50), (0, 180, 0), -1)
            cv2.putText(display_frame, "SAFE -- No Threat Detected",
                        (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
        else:  # THREAT
            cv2.rectangle(display_frame, (0, 0), (640, 50), (0, 0, 200), -1)
            cv2.putText(display_frame, "THREAT DETECTED -- Observer Looking!",
                        (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

        # --------------------------------------------------------------------
        # F2) USER BOX — BLUE rectangle + "USER" label
        # --------------------------------------------------------------------
        if user_box is not None:
            ux1, uy1, ux2, uy2, u_conf = user_box
            cv2.rectangle(display_frame, (ux1, uy1), (ux2, uy2),
                          (220, 100, 0), 2)   # Blue in BGR

            # Label above the box
            user_label = f"USER ({u_conf:.0%})"
            cv2.putText(display_frame, user_label,
                        (ux1, max(uy1 - 8, 55)),   # Don't draw over status bar
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (220, 100, 0), 2, cv2.LINE_AA)

        # --------------------------------------------------------------------
        # F3) OBSERVER BOXES — RED rectangle + role label + prediction label
        # --------------------------------------------------------------------
        for obs_box, obs_result in zip(observer_boxes, observer_results):
            ox1, oy1, ox2, oy2, o_conf = obs_box

            # Red bounding box (BGR: 0, 0, 210)
            cv2.rectangle(display_frame, (ox1, oy1), (ox2, oy2),
                          (0, 40, 210), 2)

            # "OBSERVER" label above the box
            obs_label = f"OBSERVER ({o_conf:.0%})"
            cv2.putText(display_frame, obs_label,
                        (ox1, max(oy1 - 8, 55)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 40, 210), 2, cv2.LINE_AA)

            # Prediction result below the box
            pred_label  = obs_result.get("label", "UNKNOWN")
            pred_conf   = obs_result.get("confidence", 0.0)
            is_looking  = obs_result.get("is_looking", False)
            is_estimate = obs_result.get("is_estimate", False)

            # Build annotation text
            if pred_label == "UNKNOWN":
                ann_text  = "UNKNOWN"
                ann_color = (180, 180, 180)   # Grey
            elif is_looking:
                ann_text  = f"LOOKING ({pred_conf:.1f}%)"
                ann_color = (0, 40, 210)       # Red — danger
            else:
                ann_text  = f"NOT LOOKING ({pred_conf:.1f}%)"
                ann_color = (40, 180, 40)       # Green — safe

            # Add "(est.)" suffix if this used the fallback crop
            if is_estimate:
                ann_text += " (est.)"

            # Draw below the bottom of the observer box
            ann_y = min(oy2 + 22, display_frame.shape[0] - 6)
            cv2.putText(display_frame, ann_text,
                        (ox1, ann_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                        ann_color, 2, cv2.LINE_AA)

        # --------------------------------------------------------------------
        # F4) FPS COUNTER — on screen (bottom-right) + terminal every 30 frames
        # --------------------------------------------------------------------
        fps      = 1.0 / max(time.time() - fps_time, 1e-6)
        fps_time = time.time()
        fps_text = f"FPS: {fps:.1f}"
        (fps_tw, _), _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        cv2.putText(display_frame, fps_text,
                    (display_frame.shape[1] - fps_tw - 8,
                     display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Print FPS to terminal every 30 frames so you can monitor performance
        # without it flooding the console every single frame.
        if frame_count % 30 == 0:
            det_source = "YOLO" if frame_count % YOLO_INTERVAL == 1 else "cached"
            print(f"[Frame {frame_count:>5}]  FPS: {fps:.1f}  |  "
                  f"Persons: {total_persons}  |  State: {state}  |  "
                  f"Detection: {det_source}")

        # --------------------------------------------------------------------
        # F5) PERSON COUNT — bottom-left corner
        # --------------------------------------------------------------------
        cv2.putText(display_frame, f"Persons: {total_persons}",
                    (8, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # ====================================================================
        # G) DISPLAY THE ANNOTATED FRAME
        #    The window is created on the first call and NEVER destroyed inside
        #    the loop — only in the finally block via cv2.destroyAllWindows().
        # ====================================================================
        cv2.imshow("Shoulder Surfing Prevention System", display_frame)

        # ====================================================================
        # H) KEY HANDLING
        #    waitKey(1) waits 1 ms for a keypress (keeps the loop fast).
        #    0xFF mask ensures compatibility across platforms.
        # ====================================================================
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # 'q' = quit: break out of the while loop → goes to finally block
            print("'q' pressed — shutting down.")
            break

        if key == ord('s'):
            # 's' = screenshot: save the current display frame to disk
            filename = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"Screenshot saved: {filename}")

# ============================================================================
# FINALLY BLOCK — cleanup always runs, even if an exception was raised above
# ============================================================================
finally:
    # Release the webcam so other applications can use it
    cap.release()

    # Close all OpenCV display windows
    cv2.destroyAllWindows()

    # Release MediaPipe face detector resources
    face_detector.close()

    print("System shut down cleanly.")
