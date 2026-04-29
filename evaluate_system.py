# evaluate_system.py
# ============================================================================
# PURPOSE:
#   Runs the shoulder surfing pipeline in "evaluation mode" to measure how
#   accurately the system detects real threats vs. false alarms.
#
# HOW IT WORKS:
#   For each frame, the system runs YOLO + face crop + head pose + decision
#   engine exactly like main_system.py. You then press a key to tell it the
#   GROUND TRUTH (what is ACTUALLY happening in the scene). It compares the
#   system's prediction against the truth and records everything to a CSV.
#
# GROUND TRUTH KEYS (press while the webcam window is focused):
#   L = "Observer is LOOKING at the screen" (a real threat)
#   N = "Observer is NOT looking"           (not a threat)
#   A = "No observer / Alone"              (just the user)
#   Q = quit and show final results
#
# RUN WITH:
#   python evaluate_system.py
# ============================================================================

import cv2
import csv
import time
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO
import mediapipe as mp
from head_pose_predictor import HeadPosePredictor
from decision_engine import ThreatDecisionEngine
from face_crop_utils import extract_padded_face_crop


# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_PATH      = "evaluation_results.csv"
CHART_PATH    = "evaluation_chart.png"
TARGET_FRAMES = 30     # Collect 30 labelled samples total
YOLO_INTERVAL = 3      # Run YOLO every Nth frame (speed optimisation)

# ============================================================================
# HELPER: compute metrics from collected rows
# ============================================================================

def compute_metrics(rows):
    """
    Computes classification metrics from the evaluation rows.

    WHAT ARE THESE METRICS?

    True Positive (TP):
        System said THREAT AND ground truth was 'L' (observer looking).
        → System correctly caught a real threat.

    False Positive (FP):
        System said THREAT BUT ground truth was 'N' or 'A' (no real threat).
        → System raised a false alarm (annoying to the user).

    True Negative (TN):
        System said SAFE AND ground truth was 'N' or 'A' (no real threat).
        → System correctly stayed quiet.

    False Negative (FN):
        System said SAFE BUT ground truth was 'L' (real threat missed).
        → System failed to catch a real shoulder surfer (dangerous!).

    True Positive Rate (TPR) = TP / (TP + FN)
        Also called Recall or Sensitivity.
        "Of all the real threats, what fraction did we catch?"
        Target: as high as possible (missing a threat is bad).

    False Positive Rate (FPR) = FP / (FP + TN)
        "Of all the safe situations, what fraction did we wrongly flag?"
        Target: as low as possible (false alarms damage user trust).

    True Negative Rate (TNR) = TN / (TN + FP) = 1 - FPR
        "Of all the safe situations, what fraction did we correctly ignore?"

    Parameters:
        rows: list of dicts, one per labelled frame

    Returns:
        dict of computed metrics
    """

    TP = FP = TN = FN = 0
    correct_confidences = []    # confidence values where prediction matched truth
    latencies           = []    # per-frame processing time in ms

    for row in rows:
        sys_state  = row["decision_engine_state"]   # "SAFE" or "THREAT"
        truth      = row["ground_truth"]             # 'L', 'N', or 'A'
        confidence = float(row["observer_confidence"]) if row["observer_confidence"] else 0.0
        latency    = float(row["latency_ms"])

        latencies.append(latency)

        # Real threat: observer was actually looking
        real_threat = (truth == "L")
        sys_threat  = (sys_state == "THREAT")

        if real_threat and sys_threat:
            TP += 1
            correct_confidences.append(confidence)
        elif not real_threat and sys_threat:
            FP += 1
        elif not real_threat and not sys_threat:
            TN += 1
            correct_confidences.append(confidence)
        else:  # real_threat and not sys_threat
            FN += 1

    total_positive = TP + FN   # All frames where truth = threat
    total_negative = FP + TN   # All frames where truth = safe

    TPR = TP / total_positive if total_positive > 0 else 0.0
    FPR = FP / total_negative if total_negative > 0 else 0.0
    TNR = TN / total_negative if total_negative > 0 else 0.0
    FNR = FN / total_positive if total_positive > 0 else 0.0
    avg_conf    = np.mean(correct_confidences) if correct_confidences else 0.0
    avg_latency = np.mean(latencies)           if latencies           else 0.0

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "TPR": TPR,   # True  Positive Rate (should be high)
        "FPR": FPR,   # False Positive Rate (should be low)
        "TNR": TNR,   # True  Negative Rate (should be high)
        "FNR": FNR,   # False Negative Rate (should be low)
        "avg_confidence_pct": avg_conf,
        "avg_latency_ms"    : avg_latency,
    }


# ============================================================================
# HELPER: draw bar chart
# ============================================================================

def save_chart(metrics, chart_path):
    """
    Saves a bar chart of the key evaluation metrics.
    Each bar represents one rate as a percentage (0–100%).
    """
    labels  = ["TPR\n(Catch Rate)", "FNR\n(Miss Rate)",
                "TNR\n(Correct Safe)", "FPR\n(False Alarm)"]
    values  = [metrics["TPR"]  * 100,
               metrics["FNR"]  * 100,
               metrics["TNR"]  * 100,
               metrics["FPR"]  * 100]
    # Green for good metrics, red for bad ones
    colours = ["#2ecc71", "#e74c3c", "#2ecc71", "#e74c3c"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Shoulder Surfing Detection System — Evaluation Results",
                 fontsize=14, fontweight="bold")

    # ---- Left chart: classification rates ----
    ax1 = axes[0]
    bars = ax1.bar(labels, values, color=colours, edgecolor="black", width=0.5)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Classification Rates")
    ax1.axhline(y=100, color="grey", linestyle="--", linewidth=0.8)

    # Add value labels on top of each bar
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 2,
                 f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Add legend
    green_patch = mpatches.Patch(color="#2ecc71", label="Good (want HIGH)")
    red_patch   = mpatches.Patch(color="#e74c3c", label="Bad  (want LOW)")
    ax1.legend(handles=[green_patch, red_patch], loc="upper right", fontsize=9)

    # ---- Right chart: confusion matrix as a table ----
    ax2 = axes[1]
    ax2.axis("off")

    table_data = [
        ["",            "Predicted SAFE",    "Predicted THREAT"],
        ["Truth: SAFE",  f"TN = {metrics['TN']}",  f"FP = {metrics['FP']}"],
        ["Truth: THREAT",f"FN = {metrics['FN']}",  f"TP = {metrics['TP']}"],
    ]

    table = ax2.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.4, 2.2)

    # Colour the cells
    cell_colors = {
        (0, 0): "#dddddd", (0, 1): "#dddddd", (0, 2): "#dddddd",
        (1, 0): "#dddddd",
        (1, 1): "#2ecc71",   # TN — correct
        (1, 2): "#e74c3c",   # FP — false alarm
        (2, 0): "#dddddd",
        (2, 1): "#e74c3c",   # FN — missed threat
        (2, 2): "#2ecc71",   # TP — correct detection
    }
    for (r, c), color in cell_colors.items():
        table[r, c].set_facecolor(color)

    ax2.set_title("Confusion Matrix", fontsize=12, fontweight="bold")

    # Add bottom summary text
    fig.text(0.5, 0.01,
             f"Avg Confidence (correct predictions): {metrics['avg_confidence_pct']:.1f}%   |   "
             f"Avg Latency: {metrics['avg_latency_ms']:.1f} ms",
             ha="center", fontsize=10, color="dimgray")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nChart saved to: {chart_path}")


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

def main():

    print("=" * 65)
    print("  Shoulder Surfing System — Evaluation Mode")
    print("=" * 65)
    print()
    print("GROUND TRUTH KEYS (press while webcam window is focused):")
    print("  L = Observer IS looking at screen  (real threat)")
    print("  N = Observer is NOT looking        (safe)")
    print("  A = Alone / no observer            (safe)")
    print("  Q = Quit evaluation and show results")
    print()
    print(f"Target: collect {TARGET_FRAMES} labelled frames.")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Initialize all components
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    model     = YOLO("yolov8n.pt")
    predictor = HeadPosePredictor("best_head_pose_model.pth")
    engine    = ThreatDecisionEngine(threat_threshold=10, cooldown_frames=30)

    mp_face       = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    YOLO_INTERVAL = 3
    cached_boxes  = []
    frame_count   = 0

    # ------------------------------------------------------------------
    # 2. Open CSV and write header
    # ------------------------------------------------------------------
    csv_file   = open(CSV_PATH, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "timestamp", "num_persons_detected",
        "observer_label", "observer_confidence",
        "decision_engine_state", "ground_truth", "latency_ms"
    ])
    csv_writer.writeheader()

    rows          = []
    ground_truth  = None    # Set when user presses L / N / A
    labelled_count = 0

    print(f"\nWebcam started. Press L / N / A in the window to label frames.")
    print(f"Progress will show on screen. Press Q when done.\n")

    # ------------------------------------------------------------------
    # 3. Main capture + label loop
    # ------------------------------------------------------------------
    try:
        while labelled_count < TARGET_FRAMES:

            t_start = time.time()   # Start measuring latency

            ret, frame = cap.read()
            if not ret:
                continue

            frame_count   += 1
            display_frame  = frame.copy()

            # Resize guard
            h, w = frame.shape[:2]
            if w > 640 or h > 480:
                frame         = cv2.resize(frame, (640, 480))
                display_frame = frame.copy()

            # ---- YOLO ----
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

            # ---- Classify persons ----
            if len(raw_boxes) == 0:
                user_box, observer_boxes = None, []
            elif len(raw_boxes) == 1:
                user_box, observer_boxes = raw_boxes[0], []
            else:
                raw_boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
                user_box       = raw_boxes[0]
                observer_boxes = raw_boxes[1:]

            # ---- Face crop + head pose ----
            observer_results = []
            for obs_box in observer_boxes:
                x1, y1, x2, y2, _ = obs_box
                face_crop, reason  = extract_padded_face_crop(
                    frame, [x1, y1, x2, y2], face_detector
                )
                if face_crop is None or face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
                    observer_results.append({
                        "label": "UNKNOWN", "confidence": 0.0,
                        "is_looking": False
                    })
                    continue
                result = predictor.predict(face_crop)
                result["is_estimate"] = (reason == "fallback_estimate")
                observer_results.append(result)

            # ---- Decision engine ----
            state = engine.update(
                num_persons      = 1 + len(observer_boxes),
                observer_results = observer_results
            )

            # ---- Latency ----
            latency_ms = (time.time() - t_start) * 1000.0

            # ---- Extract best observer result for logging ----
            # We log the result for the observer most likely to be a threat
            # (the one with highest LOOKING confidence, if any).
            best_obs_label = "NONE"
            best_obs_conf  = 0.0

            if observer_results:
                # Sort by LOOKING confidence descending
                looking_results = [r for r in observer_results if r.get("is_looking")]
                if looking_results:
                    best = max(looking_results, key=lambda r: r.get("confidence", 0))
                    best_obs_label = best.get("label", "UNKNOWN")
                    best_obs_conf  = best.get("confidence", 0.0)
                else:
                    best_obs_label = observer_results[0].get("label", "UNKNOWN")
                    best_obs_conf  = observer_results[0].get("confidence", 0.0)

            # ---- Draw UI on display_frame ----

            # Status bar
            bar_color = (0, 180, 0) if state == "SAFE" else (0, 0, 200)
            bar_text  = ("SAFE -- No Threat" if state == "SAFE"
                         else "THREAT DETECTED -- Observer Looking!")
            cv2.rectangle(display_frame, (0, 0), (640, 50), bar_color, -1)
            cv2.putText(display_frame, bar_text,
                        (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # User box
            if user_box is not None:
                ux1, uy1, ux2, uy2, _ = user_box
                cv2.rectangle(display_frame, (ux1, uy1), (ux2, uy2), (220, 100, 0), 2)
                cv2.putText(display_frame, "USER", (ux1, max(uy1-8, 55)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 100, 0), 2, cv2.LINE_AA)

            # Observer boxes
            for obs_box, obs_result in zip(observer_boxes, observer_results):
                ox1, oy1, ox2, oy2, _ = obs_box
                cv2.rectangle(display_frame, (ox1, oy1), (ox2, oy2), (0, 40, 210), 2)
                lbl   = obs_result.get("label", "UNKNOWN")
                conf  = obs_result.get("confidence", 0.0)
                color = (0, 40, 210) if obs_result.get("is_looking") else (40, 180, 40)
                cv2.putText(display_frame, f"{lbl} ({conf:.0f}%)",
                            (ox1, min(oy2+22, display_frame.shape[0]-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            # Progress counter + instruction overlay
            progress_text = (f"Labelled: {labelled_count}/{TARGET_FRAMES}  "
                             f"| Press  L / N / A  to label  |  Q = finish")
            cv2.rectangle(display_frame,
                          (0, display_frame.shape[0]-36),
                          (640, display_frame.shape[0]), (30, 30, 30), -1)
            cv2.putText(display_frame, progress_text,
                        (6, display_frame.shape[0]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (220, 220, 220), 1, cv2.LINE_AA)

            # Highlight if ground truth was just recorded this frame
            if ground_truth is not None:
                gt_map    = {"L": "LOOKING (L)", "N": "NOT LOOKING (N)", "A": "ALONE (A)"}
                gt_text   = f"Recorded: {gt_map.get(ground_truth, ground_truth)}"
                cv2.putText(display_frame, gt_text,
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 255, 200), 2, cv2.LINE_AA)

            cv2.imshow("Evaluation — Shoulder Surfing System", display_frame)

            # ---- Key handling ----
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("\n'Q' pressed — ending evaluation early.")
                break

            elif key in (ord('l'), ord('L'),
                         ord('n'), ord('N'),
                         ord('a'), ord('A')):

                ground_truth = chr(key).upper()

                # Write to CSV
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                row = {
                    "timestamp"             : timestamp,
                    "num_persons_detected"  : len(raw_boxes),
                    "observer_label"        : best_obs_label,
                    "observer_confidence"   : round(best_obs_conf, 2),
                    "decision_engine_state" : state,
                    "ground_truth"          : ground_truth,
                    "latency_ms"            : round(latency_ms, 2),
                }
                csv_writer.writerow(row)
                rows.append(row)
                labelled_count += 1

                print(f"  [{labelled_count:>2}/{TARGET_FRAMES}]  "
                      f"Truth={ground_truth}  State={state}  "
                      f"ObsLabel={best_obs_label}  Conf={best_obs_conf:.1f}%  "
                      f"Latency={latency_ms:.1f}ms")

                ground_truth = None   # Reset until next keypress

    finally:
        cap.release()
        face_detector.close()
        cv2.destroyAllWindows()
        csv_file.close()

    # ------------------------------------------------------------------
    # 4. Compute and print final metrics
    # ------------------------------------------------------------------
    if len(rows) == 0:
        print("No data collected — exiting.")
        return

    metrics = compute_metrics(rows)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total labelled frames : {len(rows)}")
    print(f"  True Positives  (TP)  : {metrics['TP']}")
    print(f"  False Positives (FP)  : {metrics['FP']}")
    print(f"  True Negatives  (TN)  : {metrics['TN']}")
    print(f"  False Negatives (FN)  : {metrics['FN']}")
    print()
    print(f"  True  Positive Rate (TPR/Recall)  : {metrics['TPR']*100:.1f}%  ← want HIGH")
    print(f"  False Positive Rate (FPR)         : {metrics['FPR']*100:.1f}%  ← want LOW")
    print(f"  True  Negative Rate (TNR/Spec.)   : {metrics['TNR']*100:.1f}%  ← want HIGH")
    print(f"  False Negative Rate (FNR/Miss)    : {metrics['FNR']*100:.1f}%  ← want LOW")
    print()
    print(f"  Avg confidence (correct preds)    : {metrics['avg_confidence_pct']:.1f}%")
    print(f"  Avg per-frame latency             : {metrics['avg_latency_ms']:.1f} ms")
    print(f"  Approx throughput                 : {1000/max(metrics['avg_latency_ms'],1):.1f} FPS")
    print(f"\n  CSV saved   : {CSV_PATH}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 5. Save chart
    # ------------------------------------------------------------------
    save_chart(metrics, CHART_PATH)


if __name__ == "__main__":
    main()
