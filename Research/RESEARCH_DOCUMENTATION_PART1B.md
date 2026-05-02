# AI-Based Shoulder Surfing Prevention System
## Research Documentation — PART 1B OF 5
### Sections Covered: Literature Background · System Architecture

---

# SECTION 3: LITERATURE CONTEXT AND RESEARCH BACKGROUND

## 3.1 Computer Vision for Human Detection

### 3.1.1 Brief History Leading to YOLO

Object detection in computer vision has progressed through three broad generations. The first generation (pre-2012) relied on hand-crafted features: Haar cascades, HOG (Histogram of Oriented Gradients) descriptors, and SVM classifiers. These methods were fast but brittle — they required careful tuning for each object category and degraded sharply under lighting changes, occlusions, and scale variations.

The second generation began with AlexNet in 2012, which demonstrated that deep convolutional neural networks could learn features directly from data. Two-stage detectors such as R-CNN (2014), Fast R-CNN (2015), and Faster R-CNN (2015) achieved high accuracy by first proposing candidate regions and then classifying each region. However, their multi-stage design made them too slow for real-time applications — Faster R-CNN ran at approximately 7 FPS.

The third generation introduced single-stage detectors. YOLO (You Only Look Once), first published by Redmon et al. in 2016, reformulated object detection as a single regression problem: a single neural network predicts bounding boxes and class probabilities directly from the full image in one forward pass. This made YOLO dramatically faster than two-stage detectors while maintaining competitive accuracy.

YOLOv8, released by Ultralytics in 2023, is the eighth major iteration of this architecture. It introduces anchor-free detection, an improved backbone, and a decoupled head design. The nano variant (YOLOv8n) is the smallest and fastest configuration, optimised specifically for deployment on edge devices and CPU-only hardware.

### 3.1.2 Why YOLOv8n Was Chosen

| Criterion | YOLOv8n | YOLOv8s | YOLOv8m | HOG+SVM |
|---|---|---|---|---|
| CPU FPS (640×480) | 70–80 | 35–45 | 15–20 | 20–30 |
| Person detection mAP | Sufficient | Higher | Highest | Lower |
| Requires training | No (COCO) | No | No | Yes |
| Model size | 3.2 MB | 11.2 MB | 25.9 MB | N/A |
| Suitable for real-time | ✅ | ⚠️ | ❌ | ⚠️ |

YOLOv8n was chosen because it achieves 70–80 FPS on CPU-only hardware at 640×480 resolution — the only variant that leaves enough headroom for the rest of the pipeline (MediaPipe + MobileNetV2) while maintaining an overall system FPS above 55. Larger variants would make the system CPU-bound even before head pose inference runs.

YOLO is pretrained on the COCO dataset, which includes a "person" class (class index 0) with over 250,000 annotated person instances. This means YOLO requires zero additional training for person detection in this project.

**Why HOG+SVM was rejected:** While HOG+SVM person detectors run at comparable speed, they have substantially higher false positive rates in complex backgrounds, poorer handling of partial occlusions, and no confidence score suitable for threshold filtering.

**Why background subtraction was rejected:** Background subtraction (e.g. MOG2) assumes a static background and fails entirely when multiple people are present, when the camera moves, or when lighting changes — all common in real-world deployment.

## 3.2 Head Pose Estimation

### 3.2.1 What Head Pose Estimation Is

Head pose estimation is the task of determining the 3D orientation of a human head from a 2D image. A head's orientation is described by three rotation angles, which are named using aviation terminology:

- **Yaw:** Rotation around the vertical axis — the head turns left or right. A yaw of 0° means the face points directly forward; +30° means the face is turned 30° to the right.
- **Pitch:** Rotation around the horizontal axis — the head tilts up or down. Positive pitch means looking upward; negative means looking downward.
- **Roll:** Rotation around the depth axis — the head tilts to the left or right shoulder.

### 3.2.2 Why Only Yaw Matters for This Project

In the context of shoulder surfing detection, the critical question is: **is the observer's face oriented toward the user's screen?** The user's screen is approximately in front of the webcam. An observer who is looking at the screen will have a yaw angle close to 0° (facing forward, toward the camera/screen). An observer who is looking away will have a yaw angle significantly different from 0° (turned to the side).

Pitch (up-down tilt) and roll (head tilt) do not change whether an observer is looking at the screen — a person can look at a screen while their head is tilted, and a person can look away while their head is perfectly level. Therefore, this project uses only the yaw component of head pose as its classification signal.

### 3.2.3 Head Pose as a Proxy for Gaze

Head pose is not identical to gaze direction. The eyes can deviate from the direction the head is pointing by approximately ±15–20° through eye movement alone. A person whose head is pointed directly at the screen may be looking at the wall beside the screen, and a person whose head is slightly turned may still be reading the screen with a sideways glance.

Despite this limitation, head pose is a **sufficient proxy for gaze** in the shoulder surfing context for the following reasons:

1. **Sustained attention requires head orientation.** Reading text on a screen from more than one metre away requires sustained attention. Humans cannot comfortably read text from distance while their head is turned significantly away — the required eye deviation exceeds the comfortable range of eye movement. A genuine shoulder-surfer reading content from a screen will have their head oriented toward that screen.

2. **The cost of false negatives is low.** If the system misses a very brief sideways glance, the observer has not had time to read meaningful content. The threat model is sustained observation (several seconds), not momentary glances.

3. **Full gaze estimation is not viable at this resolution.** (See Section 3.3.)

This approach — using head orientation as a sufficient proxy for sustained screen-directed gaze — is supported by multiple academic studies in the access control and privacy surveillance literature.

## 3.3 Why Full Gaze Detection Was Not Used

### 3.3.1 What Full Gaze Detection Requires

Full gaze estimation (iris tracking, eye tracking) determines the precise direction of the user's gaze by locating the iris center within the eye and computing its deviation from the eye center. Accurate iris tracking requires:

- **High image resolution:** The iris must span at least 20–30 pixels in the image for reliable center estimation.
- **Close camera-to-face distance:** Most commercial eye trackers (Tobii, EyeLink) are used at 50–70 cm from the face with high-resolution sensors.
- **Controlled lighting:** Iris tracking often uses near-infrared illumination to produce consistent pupil contrast regardless of ambient lighting.

### 3.3.2 Why This Is Not Achievable with a Laptop Webcam

At a typical shoulder surfing distance (1–2 metres between observer and screen), the observer's entire face occupies approximately 40–80 pixels of width in a 640×480 frame. An eye within that face occupies roughly 8–15 pixels. An iris within that eye occupies 4–8 pixels. At this resolution, iris center localization has a positional uncertainty larger than the iris itself — making reliable gaze direction estimation impossible.

Furthermore, deep learning gaze estimation models (such as GazeNet, MPIIGaze, or ETH-XGaze) require substantially more computation than head pose models. Inference times of 50–150 ms per face are typical on CPU, compared to 15–30 ms for the head pose model used in this project.

**Conclusion:** Head pose estimation gives approximately 80% of the detection value at approximately 20% of the computational cost of full gaze estimation, making it the clearly correct choice for this deployment context.

## 3.4 The BIWI Kinect Head Pose Dataset

### 3.4.1 Dataset Description

The BIWI Kinect Head Pose Database was collected by Fanelli et al. at ETH Zurich. It contains recordings of 24 subjects (20 adults and 4 children) captured using a Microsoft Kinect depth camera. The dataset contains approximately 15,678 frames across all subjects, with each frame represented by:

- An RGB image (640×480) of the subject's face
- A corresponding depth map
- A pose annotation file (`.txt`) containing a 3×3 rotation matrix and a translation vector describing the precise head orientation

The dataset was designed specifically for head pose estimation research and is widely used as a benchmark in the computer vision literature.

### 3.4.2 Rotation Matrix to Yaw Extraction

Each `.txt` pose file contains a 3×3 rotation matrix R that encodes the full 3D orientation of the head. To extract the yaw angle (the only angle needed for this project), the following formula is used:

```python
import numpy as np

def extract_yaw(rotation_matrix):
    R = np.array(rotation_matrix).reshape(3, 3)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees(yaw)
```

`arctan2(R[1,0], R[0,0])` extracts the rotation angle around the vertical (Z) axis from the rotation matrix, which corresponds to yaw. The result in degrees gives the left-right head orientation: values close to 0° indicate the head is facing directly forward (toward the camera); values far from 0° (e.g. ±45° or more) indicate the head is turned significantly to the side.

### 3.4.3 Why BIWI Was Chosen

- Publicly available for academic use at no cost
- Large enough for meaningful transfer learning fine-tuning (~15,000 frames)
- Contains precise rotation matrix annotations (not just manually estimated angles)
- Widely used in head pose literature — results are comparable with prior work
- Includes sufficient variation in subjects, lighting, and head positions

### 3.4.4 Limitation: Frontal Collection Geometry

BIWI was collected with the camera placed directly in front of the subject at approximately 1 metre distance. In real-world shoulder surfing, the observer is viewed from the side (the webcam is on the laptop screen facing the user; the observer is beside or behind the user at an angle). This means the model trained on BIWI frontal images may not perfectly generalise to the angled webcam perspective of a real observer.

This limitation was addressed by:
1. Using a relatively wide yaw threshold (±20°) rather than a tight one, to account for perspective differences
2. Applying the probability gap filter to suppress uncertain predictions (see Section 7 in Part 2)
3. Adding generous face crop padding to provide more spatial context to the model

## 3.5 Transfer Learning and MobileNetV2

### 3.5.1 What Transfer Learning Is

Transfer learning is the practice of taking a neural network that has been trained on a large dataset for a general task, and adapting it for a specific task using a much smaller dataset. The rationale is that the general features learned on the large dataset — edges, textures, shapes, object parts — are useful starting points for the specific task, even though the tasks differ.

For image classification, ImageNet pretraining is the standard starting point. ImageNet contains 1.28 million images across 1,000 categories. A network trained to classify 1,000 ImageNet categories has learned rich, hierarchical visual features that transfer well to almost any image classification task.

Without transfer learning, training a convolutional neural network from scratch for binary head pose classification would require tens of thousands of training examples and days of training time — resources not available for a one-week solo project.

### 3.5.2 MobileNetV2 Architecture

MobileNetV2 (Howard et al., Google, 2018) is a convolutional neural network architecture designed for mobile and embedded deployment. Its key architectural innovations are:

- **Depthwise separable convolutions:** Split a standard convolution into a depthwise convolution (one filter per channel) and a pointwise convolution (1×1 cross-channel combination). This reduces computation by 8–9× with minimal accuracy loss.
- **Inverted residual blocks with linear bottlenecks:** Expand the feature space, apply depthwise convolution, then compress back. The residual skip connection ensures gradients flow well during training.
- **Final feature size:** The MobileNetV2 feature extractor outputs a 1280-dimensional feature vector after global average pooling.

The full MobileNetV2 has approximately 3.4 million parameters and runs inference in approximately 15–30 ms on a laptop CPU for a single 224×224 image.

### 3.5.3 Why MobileNetV2 Over Alternatives

| Model | Parameters | CPU Inference (ms) | Top-1 Accuracy (ImageNet) | Suitability |
|---|---|---|---|---|
| MobileNetV2 | 3.4M | 15–30 | 71.8% | ✅ Ideal |
| ResNet-50 | 25.6M | 80–120 | 76.1% | ❌ Too slow |
| EfficientNet-B0 | 5.3M | 25–45 | 77.1% | ⚠️ Acceptable |
| VGG-16 | 138M | 400–600 | 71.6% | ❌ Far too slow |
| Training from scratch | — | — | — | ❌ Not enough data |

MobileNetV2 was chosen because it is the fastest option that still provides sufficient ImageNet features for fine-tuning. The accuracy difference between MobileNetV2 and ResNet-50 on ImageNet (71.8% vs 76.1%) does not meaningfully translate to better head pose classification after fine-tuning, but the speed difference (15–30ms vs 80–120ms per inference) is decisive for maintaining real-time FPS.

### 3.5.4 Layer Freezing Strategy

This project freezes the entire MobileNetV2 backbone (all 18 convolutional blocks) and trains only a custom classification head added on top. This means 87.1% of the model parameters are frozen and only 12.9% (328,450 parameters) are updated during training.

**Why freeze the backbone entirely?**

The training dataset, even after oversampling, contains effectively only 1,448 unique NOT_LOOKING face images (repeated ~10× with augmentation). With only ~14,000 effectively unique training examples, unfreezing more backbone layers creates a high risk of overfitting — the model memorises training images rather than learning generalisable features. By freezing the backbone, the model is forced to use the general visual features already learned on ImageNet and can only adapt the final classification mapping. The result is a model that generalises better to new faces not seen during training.

The custom head consists of:
```
Linear(1280 → 256) → ReLU → Dropout(0.3) → Linear(256 → 2)
```
This design compresses the 1280-dimensional MobileNetV2 feature vector to a 256-dimensional intermediate representation, applies a non-linearity, uses dropout to prevent overfitting, and produces two output scores for NOT_LOOKING (class 0) and LOOKING (class 1).

---

# SECTION 4: COMPLETE SYSTEM ARCHITECTURE

## 4.1 Architecture Overview

The complete system is a sequential seven-stage pipeline. Each stage consumes the output of the previous stage and produces a structured output for the next stage.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  SYSTEM PIPELINE (per video frame)                  │
└─────────────────────────────────────────────────────────────────────┘

 [WEBCAM]
   │  BGR frame: numpy array (480, 640, 3)
   ▼
 [YOLO PERSON DETECTION]  ← yolov8n.pt (pretrained, COCO)
   │  List of bounding boxes: [[x1,y1,x2,y2,conf], ...]
   ▼
 [ZONE CLASSIFIER]  ← built into main_system.py
   │  user_box: [x1,y1,x2,y2,conf] or None
   │  observer_boxes: [[x1,y1,x2,y2,conf], ...]
   ▼
 [FOR EACH OBSERVER] ─────────────────────────────────────────────┐
   │                                                               │
   ▼                                                               │
 [FACE EXTRACTION]  ← face_crop_utils.py + MediaPipe              │
   │  face_crop: numpy array (H, W, 3)  or  None                  │
   │  reason: string                                               │
   ▼                                                               │
 [HEAD POSE CLASSIFICATION]  ← head_pose_predictor.py             │
   │  result dict:                                                 │
   │    label: "LOOKING" / "NOT_LOOKING" / "UNCERTAIN"            │
   │    confidence: float (0–100)                                  │
   │    is_looking: bool                                           │
   │    gap: float                                                 │
   └───────────────────────────────────────────────────────────────┘
   │  observer_results: [result_dict, ...]
   ▼
 [THREAT DECISION ENGINE]  ← decision_engine.py
   │  state: "SAFE" or "THREAT"
   ▼
 [DISPLAY / ALERT]  ← main_system.py drawing code
      Status bar (green/red) + bounding boxes + labels + FPS
```

## 4.2 Design Philosophy

### 4.2.1 Modular Design

The system is split into separate Python files, each with a single, well-defined responsibility:

- `main_system.py` orchestrates the pipeline — it contains no model logic itself, only the loop and drawing code
- `head_pose_predictor.py` encapsulates all model loading and inference — it can be imported independently
- `decision_engine.py` encapsulates all state machine logic — it has no dependency on OpenCV or any visual module
- `face_crop_utils.py` encapsulates all face extraction logic independently

This separation means each component can be tested, replaced, or improved independently. For example, the head pose model can be upgraded to a better architecture without changing the decision engine or main loop.

### 4.2.2 State Machine for Threat Detection

Rather than declaring a threat on any single frame where LOOKING is detected, the system uses a state machine with temporal memory:

```
SAFE ──(looking streak ≥ 3)──► accumulating threat frames
     ──(threat_frame_count ≥ 4)──► THREAT
THREAT ──(safe_frame_count ≥ 20)──► SAFE
```

A single LOOKING prediction does not trigger THREAT. Three consecutive LOOKING frames from the same observer increment a per-observer streak. When the global threat frame count reaches 4, the state flips to THREAT. The state remains THREAT until 20 consecutive SAFE frames confirm the observer has stopped looking.

### 4.2.3 Temporal Smoothing

At 60 FPS, even a 91.99% accurate model produces approximately 0.08 × 60 = ~5 incorrect predictions per second. Without temporal smoothing, the system would flicker between SAFE and THREAT multiple times per second on any observer, making it useless. Temporal smoothing converts these noisy per-frame signals into stable state transitions that only change when a genuine, sustained condition is met.

### 4.2.4 Fail-Safe Principle

Every uncertain or failed prediction defaults to SAFE, never to THREAT:
- Face crop extraction failure → UNKNOWN → treated as SAFE
- Crop below 40×40 pixels → UNKNOWN → treated as SAFE
- Model confidence below threshold → UNCERTAIN → treated as SAFE
- Probability gap between classes too small → UNCERTAIN → treated as SAFE

This means the system will sometimes miss a real threat (false negative) rather than falsely alert when no threat exists (false positive). For a privacy protection tool, a missed detection is less harmful than constant false alarms that cause the user to distrust and disable the system.

### 4.2.5 Asymmetric Thresholds

The probability gap threshold for triggering LOOKING (0.25) is higher than the threshold for triggering NOT_LOOKING (0.15). This means the model must have substantially stronger confidence to declare someone is looking than to declare they are not. The rationale is that false THREAT detections are more disruptive to the user experience than missed detections — a user who receives a false THREAT while sitting alone will lose confidence in the system.

## 4.3 File Structure and Responsibilities

### `main_system.py`
The central orchestrator. Contains the main loop that runs once per video frame, calling all other modules in sequence. Handles webcam initialisation and release, YOLO inference with frame-skip caching, zone classification (user vs observer), loop-level result caching for speed optimisation, UI drawing (bounding boxes, status bar, labels, FPS counter), and keyboard input handling (q=quit, s=screenshot).

Does NOT contain: any model weights, any training logic, any MediaPipe code, any decision logic.

### `head_pose_predictor.py`
Defines two classes: `HeadPoseClassifier` (the PyTorch model architecture) and `HeadPosePredictor` (the inference wrapper). `HeadPosePredictor.__init__()` loads the saved `.pth` checkpoint, rebuilds the architecture, loads the weights, and prepares the preprocessing pipeline. `HeadPosePredictor.predict(face_bgr)` takes a BGR face crop and returns a structured result dictionary including the label, confidence, is_looking flag, and the raw probability gap for debugging.

### `decision_engine.py`
Defines `ThreatDecisionEngine`. Contains all threshold parameters (threat_threshold, cooldown_frames, confidence_threshold), all internal counters (threat_frame_count, safe_frame_count, observer_looking_streak dict), the update() method called once per frame, and the get_status_info() method for display. Has zero dependency on OpenCV, PyTorch, or any other module — it is pure Python logic.

### `face_crop_utils.py`
Defines `extract_padded_face_crop(frame, person_box, face_detector)`. Handles all face extraction logic: validates the person bounding box, runs MediaPipe FaceDetection on the person crop, converts relative MediaPipe coordinates to absolute pixel coordinates, applies 40% padding to the face bounding box, clips coordinates to image boundaries, enforces the 40×40 minimum size gate, and implements the fallback (top 28% of person crop) when MediaPipe finds no face.

### `screen_protector.py`
*(Removed from current version — detection-only system.)* Was responsible for the tkinter-based transparent overlay with Gaussian blur and cursor spotlight. Removed because the project scope was narrowed to detection and alerting only.

### `evaluate_system.py`
Runs the full pipeline in evaluation mode. Allows the evaluator to label each scenario by pressing L (observer looking), N (observer not looking), or A (alone). Collects 30 labelled samples across three scenario types. Computes TPR, FPR, TNR, average confidence, and average latency. Saves results to `evaluation_results.csv` and `evaluation_chart.png`.

### `run_demo.py`
The single entry point for demonstration. Runs pre-flight checks: verifies that `best_head_pose_model.pth` exists, that `yolov8n.pt` is available (downloads if not), that the webcam is accessible, and that all required Python packages are importable. Prints a startup banner and then launches `main_system.py` if all checks pass.

### `requirements.txt`
Lists all required packages:
- `opencv-python`: webcam capture, image drawing, frame display
- `ultralytics`: YOLOv8n person detection
- `mediapipe==0.10.21`: face detection within person crops (pinned version for stability)
- `torch`, `torchvision`: MobileNetV2 model and inference
- `numpy`: array operations throughout the pipeline
- `pillow`: PIL Image conversion for PyTorch transforms
- `matplotlib`: evaluation chart generation
- `scikit-learn`: confusion matrix and metric computation in evaluate_system.py
- `mss`: fast screen capture (used by screen_protector, kept for completeness)

## 4.4 Data Flow: Tracing One Frame End-to-End

The following traces a single frame through the complete system, showing exactly what data exists at each step:

**Step 1 — Frame capture:**
```python
ret, frame = cap.read()
# frame: numpy array, shape (480, 640, 3), dtype uint8, color order BGR
display_frame = frame.copy()  # separate copy for drawing — never modified for inference
```

**Step 2 — YOLO detection (if frame_count % 2 == 1):**
```python
results = model(frame, classes=[0], verbose=False)
# results[0].boxes contains detected bounding boxes
# Output: raw_boxes = [[234, 89, 412, 478, 0.91], [51, 102, 178, 445, 0.74]]
# Format: [x1, y1, x2, y2, confidence]
```

**Step 3 — Zone classification:**
```python
# Sort by area descending: (412-234)*(478-89) = 178*389 = 69,242 → USER
#                          (178-51)*(445-102) = 127*343 = 43,561 → OBSERVER
user_box       = [234, 89, 412, 478, 0.91]
observer_boxes = [[51, 102, 178, 445, 0.74]]
```

**Step 4 — Face extraction for observer 0:**
```python
person_crop = frame[102:445, 51:178]  # shape (343, 127, 3)
# Resize to max 320px wide for MediaPipe speed
person_crop_small = resize_to_max_width(person_crop, 320)  # shape (343, 127, 3) — already <320
# MediaPipe runs on person_crop_small
# Returns relative bbox: xmin=0.12, ymin=0.03, width=0.76, height=0.38
# Convert to absolute within crop: x1=15, y1=10, x2=112, y2=140
# Add 40% padding: expand by 0.4 * width/height on each side
# Clip to crop boundaries
# face_crop shape: (163, 130, 3) → passes 40×40 guard
# Resize to 112×112 before inference
face_crop = cv2.resize(face_crop, (112, 112))  # shape (112, 112, 3)
```

**Step 5 — Head pose inference:**
```python
result = predictor.predict(face_crop)
# Inside predict():
#   face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
#   face_pil = Image.fromarray(face_rgb)
#   face_tensor = transform(face_pil)  # Resize to 224×224, normalize
#   logits = model(face_tensor.unsqueeze(0))  # shape [1, 2]
#   probs = softmax(logits)  # e.g. [0.08, 0.92]
#   gap = 0.92 - 0.08 = 0.84  ≥ 0.25 → LOOKING
# result = {"label": "LOOKING", "confidence": 92.0, "is_looking": True, "gap": 0.84, ...}
```

**Step 6 — Decision engine:**
```python
state = engine.update(num_persons=2, observer_results=[result])
# observer 0 is_looking=True, confidence=92.0 > 75 → streak[0] += 1 → streak[0] = 3
# streak[0] >= 3 → this_frame = "THREAT"
# threat_frame_count += 1 → threat_frame_count = 4
# threat_frame_count >= threat_threshold(4) → current_state = "THREAT"
# returns "THREAT"
```

**Step 7 — Draw and display:**
```python
# Red status bar at top of display_frame
cv2.rectangle(display_frame, (0,0), (640,50), (0,0,200), -1)
cv2.putText(display_frame, "THREAT DETECTED -- Observer Looking!", ...)
# Blue USER box at (234,89)-(412,478)
# Red OBSERVER box at (51,102)-(178,445)
# Label "LOOKING (92.0%)" below observer box
# FPS counter bottom-right
cv2.imshow("Shoulder Surfing Prevention System", display_frame)
```

The entire sequence from frame capture to display update completes in approximately 15–20 ms per frame at the configured frame-skip intervals, achieving approximately 55–65 FPS end-to-end.

---

*Continued in RESEARCH_DOCUMENTATION_PART2.md — Dataset Preparation, Model Training, Inference Pipeline, Face Extraction Module, Decision Engine, and Problems Faced.*
