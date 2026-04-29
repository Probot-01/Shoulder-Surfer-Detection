# AI-Based Shoulder Surfing Prevention System

> A real-time computer vision system that detects when someone is looking at your screen and triggers a protective blur overlay to prevent unauthorised viewing.

---

## 1. Project Overview

Shoulder surfing is a privacy attack where an unauthorised person looks at your screen in a public place. This system uses a webcam, deep learning, and computer vision to automatically detect when a nearby person is watching your screen and responds by blurring the display in real time. The user retains a small clear "spotlight" around their cursor so they can still work normally while the rest of the screen is hidden from the observer.

---

## 2. How It Works

The system runs a five-step pipeline on every webcam frame:

1. **Person Detection** — YOLOv8n scans the webcam feed and draws bounding boxes around every person in the frame.

2. **User vs. Observer Classification** — The person with the largest bounding box (closest to the camera) is labelled the **USER**. All other detected persons are labelled **OBSERVER**.

3. **Face Extraction** — For each observer, MediaPipe Face Detection locates the face within their bounding box. A 40 % padded crop is extracted. If no face is found, the top 28 % of the person box is used as a fallback estimate.

4. **Head Pose Classification** — A fine-tuned **MobileNetV2** model classifies each face crop as `LOOKING`, `NOT_LOOKING`, or `UNCERTAIN`. A probability-gap filter (gap ≥ 0.30 required to declare LOOKING) reduces false positives from ambiguous side-profile faces.

5. **Threat Decision** — A `ThreatDecisionEngine` tracks how many consecutive frames each observer has been looking. After **5 consecutive confirmed LOOKING frames** from one observer, the system declares a **THREAT** and activates the blur overlay.

---

## 3. Installation

> Tested on Python 3.10 · Windows 11

```bash
# 1. Clone or copy the project folder, then open a terminal inside it

# 2. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install all dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python ultralytics mediapipe Pillow numpy matplotlib mss

# 4. Download the YOLOv8n weights (first run only)
python download_yolo.py

# 5. Copy your trained model from Google Drive into this folder
#    File needed: best_head_pose_model.pth
```

> **GPU note:** Replace `cu118` with `cu121` for CUDA 12.1, or use `pip install torch torchvision` (CPU only) if no NVIDIA GPU is available.

---

## 4. How to Run

```bash
python run_demo.py
```

`run_demo.py` performs a full pre-flight check (webcam, model file, dependencies) and then launches the system automatically. It is the **only file the examiner needs to run**.

---

## 5. Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the system cleanly |
| `S` | Save a screenshot of the current frame |
| `P` | Manually toggle screen blur on / off (for testing) |
| `L` *(evaluate_system.py only)* | Label current frame as "Observer LOOKING" |
| `N` *(evaluate_system.py only)* | Label current frame as "Observer NOT LOOKING" |
| `A` *(evaluate_system.py only)* | Label current frame as "Alone / no observer" |

---

## 6. Files Description

| File | Purpose |
|------|---------|
| `run_demo.py` | **Start here.** Pre-flight checks + launches the system |
| `main_system.py` | Main integration loop — webcam thread + screen protector thread |
| `head_pose_predictor.py` | `HeadPosePredictor` class — loads MobileNetV2, runs inference on a face crop |
| `decision_engine.py` | `ThreatDecisionEngine` — temporal smoothing, per-observer streak tracking |
| `face_crop_utils.py` | `extract_padded_face_crop()` — MediaPipe face detection + 40 % padding + fallback |
| `screen_protector.py` | `ScreenProtector` — full-screen blur overlay with cursor spotlight (tkinter + PIL) |
| `evaluate_system.py` | Evaluation mode — collect ground-truth labels, compute TPR / FPR, save chart |
| `biwi_prepare_dataset.py` | *(Colab)* Parse BIWI dataset, compute yaw angles, generate CSV |
| `biwi_balance_dataset.py` | *(Colab)* Re-label with tighter threshold (±20°), oversample minority class |
| `biwi_dataset.py` | *(Colab)* `HeadPoseDataset` class + train / val data loaders |
| `biwi_model.py` | *(Colab)* `HeadPoseClassifier` — MobileNetV2 with frozen backbone + custom head |
| `biwi_train.py` | *(Colab)* Training loop — Adam, StepLR, saves best model, plots learning curves |
| `biwi_inference.py` | *(Colab)* Confusion matrix + 5-sample visual test on validation set |
| `download_yolo.py` | One-time download of `yolov8n.pt` |
| `zone_detection.py` | Early prototype — USER / OBSERVER zone classification (reference only) |
| `face_cropper.py` | Early prototype — MediaPipe face detection on person crops (reference only) |

---

## 7. Model Training

**Dataset:** [BIWI Kinect Head Pose Database](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html)
- 24 subjects, ~15,000 labelled head pose images
- Rotation matrices converted to yaw angles

**Labelling threshold:** `|yaw| ≤ 20°` → `LOOKING (1)` · `|yaw| > 20°` → `NOT_LOOKING (0)`

**Class balancing:** minority class (`NOT_LOOKING`) oversampled with replacement to achieve a 1 : 1 ratio (~30,000 balanced samples)

**Model architecture:**
- Backbone: **MobileNetV2** (ImageNet pretrained, fully frozen)
- Custom head: `Linear(1280 → 256) → ReLU → Dropout(0.3) → Linear(256 → 2)`
- Trainable parameters: ~329,000 (~9.4 % of total)

**Training setup (Google Colab T4 GPU):**
- Optimiser: Adam · LR = 0.0001
- Scheduler: StepLR — halves LR every 5 epochs
- Epochs: 20 · Batch size: 32

**Validation accuracy:** _______ % *(fill in after training)*

---

## 8. Results

> Fill in these values after running `python evaluate_system.py`

| Metric | Value |
|--------|-------|
| True Positive Rate (Threat correctly detected) | _______ % |
| False Positive Rate (False alarms) | _______ % |
| True Negative Rate (Safe correctly identified) | _______ % |
| False Negative Rate (Missed threats) | _______ % |
| Average prediction confidence | _______ % |
| Average per-frame latency | _______ ms |
| Effective frame rate | _______ FPS |

Evaluation chart: [`evaluation_chart.png`](./evaluation_chart.png)

---

## 9. Team

| Name | Role |
|------|------|
| *(Your Name)* | Solo developer — dataset preparation, model training, system integration, evaluation |

**Course:** *(Your Course Name)*  
**Institution:** *(Your College Name)*  
**Academic Year:** 2025 – 2026

---

## Acknowledgements

- [BIWI Kinect Head Pose Database](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html) — Fanelli et al., ETH Zurich
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://mediapipe.dev/) — Google
- [PyTorch](https://pytorch.org/) / [torchvision](https://pytorch.org/vision/)
