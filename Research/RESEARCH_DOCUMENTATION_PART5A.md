# AI-Based Shoulder Surfing Prevention System
## Research Documentation — PART 5A OF 5
### Sections: Evaluation · Technology Stack · Deployment · Limitations

---

# SECTION 15: EVALUATION METHODOLOGY

## 15.1 Model-Level Evaluation (Google Colab)

The head pose classification model was evaluated on a held-out validation set of 5,692 samples representing 20% of the balanced dataset. This set was NOT oversampled and retains the natural distribution of the collected frames.

**Final metric:** Validation accuracy = **91.99%**

This means that on held-out BIWI frames the model correctly classifies the head pose (LOOKING or NOT_LOOKING at ±20° threshold) for 9,199 out of every 10,000 frames it has never seen during training.

**Confusion matrix interpretation:**

| | Predicted: LOOKING | Predicted: NOT_LOOKING |
|---|---|---|
| **Actual: LOOKING** | TP (true positive) | FN (false negative) |
| **Actual: NOT_LOOKING** | FP (false positive) | TN (true negative) |

At 91.99% accuracy, the combined error rate is 8.01%. The majority of errors are expected to fall in the boundary zone — frames where yaw is between 15° and 25°, near the ±20° decision boundary. These are inherently ambiguous: a face at 18° is visually very similar to a face at 22°, yet the first is labelled LOOKING and the second NOT_LOOKING.

**Why validation accuracy is meaningful but incomplete:**

Validation accuracy on BIWI tells us how well the model learned the BIWI labelling convention. It does not tell us how well it performs on real webcam captures of shoulder surfers — a different population with different camera optics, different distances, different lighting, and different image processing than the Kinect sensor used to collect BIWI. This gap (the domain gap) is partially addressed by augmentation during training, but not eliminated. System-level evaluation (Section 15.2) addresses real-world performance.

## 15.2 System-Level Evaluation (evaluate_system.py)

The system-level evaluation uses a controlled scenario methodology with human-labelled ground truth collected in real time.

**Protocol:**

30 test scenarios are performed, divided across three categories:

| Category | Count | Description |
|---|---|---|
| Observer LOOKING | 10 | Another person sits in view and looks directly at screen |
| Observer NOT LOOKING | 10 | Another person is in view but looks at wall, phone, or desk |
| No Observer | 10 | Evaluator sits alone in front of webcam |

For each scenario, the evaluator sets up the scene, presses the corresponding key (`L`, `N`, or `A`) while the webcam window is focused, and the system records the current detection state and all associated metrics.

**Metrics computed:**

- **True Positive Rate (TPR):** Of the 10 LOOKING scenarios, how many did the system correctly declare THREAT? Ideal = 100%.
- **False Positive Rate (FPR):** Of the 20 non-threat scenarios (N + A combined), how many did the system incorrectly declare THREAT? Ideal = 0%.
- **True Negative Rate (TNR):** Of the 20 non-threat scenarios, how many were correctly SAFE? TNR = 1 − FPR. Ideal = 100%.
- **Average confidence:** Mean confidence percentage across all LOOKING predictions made during THREAT-labelled scenarios.
- **Average latency:** Mean milliseconds from frame capture to decision engine state update.

**Output files:**
- `evaluation_results.csv`: Row per labelled sample, columns: timestamp, truth_label, system_state, observer_label, observer_confidence, latency_ms
- `evaluation_chart.png`: Bar chart of TPR/FPR/TNR + confusion matrix visualisation

## 15.3 FPS Performance Measurement

FPS is calculated per-frame:
```python
fps = 1.0 / max(time.time() - fps_time, 1e-6)
fps_time = time.time()
```

| Condition | FPS Before Optimization | FPS After Optimization |
|---|---|---|
| User alone | 70–80 | 70–80 (no change) |
| One observer present | 30–40 | **55–65** |

The alone-case FPS did not change because no observer means no MediaPipe or MobileNetV2 processing — YOLO runs every 2nd frame (unchanged). The observer-case improvement came from: YOLO frame-skip saving ~7.5ms/frame, pose frame-skip saving ~7ms/frame average, crop resize saving ~3ms, face resize saving ~3ms.

## 15.4 Latency Measurement

**Definition:** Wall-clock time from the moment an observer begins looking at the screen to the moment the status bar changes to red.

| Configuration | Minimum Frames Before THREAT | Wall-Clock Latency at 60 FPS |
|---|---|---|
| Before tuning (streak=5, threshold=10) | 5×3 + 10 = 25 frames | ~417ms |
| After tuning (streak=3, threshold=4) | 3×3 + 4 = 13 frames | **~217ms** |

**Why ~200ms is acceptable:** Human reaction time to visual stimuli is approximately 150–300ms. An observer who begins looking at a screen cannot read and comprehend a full sentence in less than 200ms. The system declares THREAT before the observer has processed meaningful content. For comparison, a person can type a password (8 characters at 60 WPM) in approximately 800ms — the system has 4× the time needed to respond before a credential could be captured.

Latency was measured during development by adding timestamps at the observer-streak update and state-change events and computing the difference.

---

# SECTION 16: COMPLETE TECHNOLOGY STACK

## 16.1 Full Stack Table

| Technology | Version | Role in Project |
|---|---|---|
| Python | 3.11 | Primary language |
| OpenCV (cv2) | 4.x | Webcam, image ops, display |
| Ultralytics YOLOv8n | 8.x | Person detection |
| MediaPipe | 0.10.21 | Face detection within person crops |
| PyTorch | 2.x | Model inference + training |
| TorchVision | 0.x | MobileNetV2, transforms |
| NumPy | 1.x | Array operations throughout |
| Pillow (PIL) | 10.x | Image conversion for torch transforms |
| scikit-learn | 1.x | Confusion matrix, resample, metrics |
| matplotlib | 3.x | Training curves, evaluation charts |
| pandas | 2.x | Dataset CSV handling (training only) |
| mss | 9.x | Fast screen capture (protector module) |
| Google Colab | — | Free GPU for training |
| Google Drive | — | Dataset storage, model hosting |
| GitHub | — | Version control, deployment |

## 16.2 Python 3.11

Python was chosen because it has the richest ecosystem of machine learning and computer vision libraries of any language. PyTorch, OpenCV, MediaPipe, and Ultralytics all provide first-class Python APIs. Rapid prototyping in Python is significantly faster than in C++ or Java — critical for a 7-day project.

Python 3.11 specifically was chosen because it is the latest stable version at the time of development with the widest library compatibility. MediaPipe 0.10.21 requires Python 3.8–3.11; Python 3.12 support was not yet guaranteed across all dependencies.

## 16.3 OpenCV (cv2)

OpenCV (Open Source Computer Vision Library) is the most widely used library for real-time computer vision. In this project it serves multiple roles:

- **Webcam capture:** `cv2.VideoCapture(0)`, `cap.read()`
- **Color conversion:** `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` for MediaPipe; `cv2.COLOR_BGR2RGB` for PyTorch
- **Image resizing:** `cv2.resize()` for crop resizing optimizations
- **Drawing:** `cv2.rectangle()`, `cv2.putText()`, `cv2.LINE_AA` for bounding boxes and labels
- **Display:** `cv2.imshow()`, `cv2.waitKey()` for the live display window
- **File I/O:** `cv2.imwrite()` for screenshots and debug image saving

OpenCV reads images in BGR channel order (not RGB) — this must be accounted for at every point where images are passed to other libraries (MediaPipe expects RGB, PyTorch models trained on PIL images expect RGB).

## 16.4 Ultralytics YOLOv8n

Ultralytics is the company that develops and maintains the YOLO series of object detectors. YOLOv8 (2023) is the eighth generation, introducing anchor-free detection, an improved backbone, and a unified Python API that simplifies deployment substantially compared to previous versions.

**Why YOLOv8 over YOLOv5 or YOLOv7:** YOLOv8 offers better accuracy at equivalent speeds due to architectural improvements. The Ultralytics API is cleaner and more stable than third-party YOLOv7 implementations. YOLOv8n (nano) is specifically designed for edge deployment, achieving 70–80 FPS on laptop CPUs — the highest FPS of any comparable detector.

**Why nano variant:** The nano variant has 3.2M parameters and runs at 70–80 FPS on CPU at 640×480. The small variant (11.2M parameters) runs at 35–45 FPS — already too slow when combined with the rest of the pipeline. For person detection at desk range (1–3 metres), YOLOv8n's accuracy is more than sufficient.

## 16.5 MediaPipe 0.10.21

MediaPipe is Google's open-source framework for real-time perception tasks. It provides pre-built, optimised ML pipelines for face detection, pose estimation, hand tracking, and more.

**Why FaceDetection and not FaceMesh:** FaceMesh detects 468 facial landmarks with sub-millimetre accuracy — but it requires the face to already be reasonably well-framed in the image. FaceDetection locates the face bounding box within a larger image, which is exactly what is needed here (finding the face within a person crop). FaceMesh would require running FaceDetection first anyway; using FaceDetection alone is simpler and faster.

**Why version 0.10.21 specifically:** MediaPipe's Python API changed significantly between versions. Version 0.10.21 is the last stable release that uses the legacy `mp.solutions.face_detection` API pattern. Later versions use a different Task API. This project was written against 0.10.21; using a different version will cause import errors or different API behaviour. This version is pinned in `requirements.txt`.

## 16.6 PyTorch + TorchVision

PyTorch is Facebook (Meta) AI's deep learning framework. It uses dynamic computation graphs (define-by-run), making it more intuitive for research and debugging than TensorFlow's static graph model.

**Why PyTorch over TensorFlow/Keras:** PyTorch has become the dominant framework in research settings, with the most current implementations of models like MobileNetV2 available via TorchVision. Debugging PyTorch models (examining tensors mid-computation) is simpler than debugging TensorFlow graphs. TorchVision provides `torchvision.models.mobilenet_v2(pretrained=True)` as a single-line pretrained model loader.

**TorchVision components used:**
- `torchvision.models.mobilenet_v2`: pretrained MobileNetV2 backbone
- `torchvision.transforms`: `Compose`, `Resize`, `ToTensor`, `Normalize`, `RandomHorizontalFlip`, `ColorJitter`, `RandomRotation` for preprocessing and augmentation

## 16.7 scikit-learn

scikit-learn is used only in two places: `sklearn.utils.resample` for random oversampling of the minority class during dataset preparation, and `sklearn.metrics.confusion_matrix` for generating the evaluation confusion matrix. It is not used in the inference pipeline.

## 16.8 Google Colab + Tesla T4

Google Colab provides browser-based Jupyter notebook access to NVIDIA GPU instances at no cost. The Tesla T4 GPU used for training has 16GB GDDR6 memory and approximately 8.1 TFLOPS of FP32 performance. Training the custom classification head (328,450 parameters) on 22,768 samples for 20 epochs takes approximately 60–90 minutes on T4 — compared to an estimated 12–18 hours on a laptop CPU.

---

# SECTION 17: GITHUB DEPLOYMENT AND PORTABILITY

## 17.1 Repository Structure

```
Shoulder Surfing Project/
│
├── main_system.py              # Main pipeline orchestrator — entry point
├── head_pose_predictor.py      # MobileNetV2 inference wrapper
├── decision_engine.py          # Threat state machine
├── face_crop_utils.py          # MediaPipe face extraction
├── evaluate_system.py          # System-level evaluation tool
├── run_demo.py                 # Pre-flight checks + system launcher
│
├── requirements.txt            # All pip dependencies with pinned versions
├── README.md                   # Setup guide for new device
│
├── RESEARCH_DOCUMENTATION_PART1.md   # Sections 1-2
├── RESEARCH_DOCUMENTATION_PART1B.md  # Sections 3-4
├── RESEARCH_DOCUMENTATION_PART2.md   # Sections 5-7
├── RESEARCH_DOCUMENTATION_PART3.md   # Sections 8-12
├── RESEARCH_DOCUMENTATION_PART4A.md  # Problems 1-7
├── RESEARCH_DOCUMENTATION_PART4B.md  # Problems 8-13 + Lessons
├── RESEARCH_DOCUMENTATION_PART5A.md  # Sections 15-18
├── RESEARCH_DOCUMENTATION_PART5B.md  # Sections 19-21
│
└── .gitignore                  # Excludes model, cache, datasets
```

Files NOT in repository (see Section 17.2):
- `best_head_pose_model.pth` — model weights
- `yolov8n.pt` — auto-downloaded
- `biwi_labels.csv`, `biwi_labels_balanced.csv` — dataset files
- `__pycache__/` — Python bytecode cache

## 17.2 What Is Not in the Repository and Why

**best_head_pose_model.pth (~9MB):**
GitHub recommends keeping files under 50MB and strongly discourages binary files in repositories — they cannot be diffed, they bloat repository size permanently (even after deletion, they remain in git history), and they slow down clone operations. The model is hosted on Google Drive with a public share link documented in README.md.

**yolov8n.pt:**
Ultralytics automatically downloads `yolov8n.pt` on the first call to `YOLO("yolov8n.pt")` if the file is not present. Including it in the repository would add 6MB of unnecessary binary data when the download is handled automatically.

**.gitignore contents:**
```
best_head_pose_model.pth
yolov8n.pt
biwi_labels.csv
biwi_labels_balanced.csv
__pycache__/
*.pyc
*.pth
*.pt
evaluation_results.csv
evaluation_chart.png
screenshot_*.jpg
```

## 17.3 Model Distribution Strategy

The trained model is the only artifact that cannot be recreated automatically. It is distributed via Google Drive:

1. After training in Colab, `best_head_pose_model.pth` is saved to Google Drive
2. A public share link is generated (anyone with link can download)
3. The README.md contains this link with step-by-step download instructions
4. Users place the downloaded `.pth` file in the project root directory

This approach is standard for academic ML projects where the model is too large for git but must be shareable for reproducibility.

## 17.4 Running on a New Device

**Why no retraining is needed:** The `.pth` file contains the complete state of the trained neural network — all 2,544,866 parameters at their trained values. Loading this file onto any device recreates the exact model that achieved 91.99% validation accuracy. The model's knowledge is entirely encoded in these weights; no access to the BIWI dataset or training code is required.

**Complete setup procedure:**

*Windows:*
```bash
git clone https://github.com/[username]/shoulder-surfing-project.git
cd shoulder-surfing-project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# Download best_head_pose_model.pth from Google Drive link in README
# Place it in the project root directory
python run_demo.py
```

*macOS/Linux:*
```bash
git clone https://github.com/[username]/shoulder-surfing-project.git
cd shoulder-surfing-project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Download best_head_pose_model.pth from Google Drive link in README
python run_demo.py
```

**Device-specific setting:** If `cv2.VideoCapture(0)` fails (camera not found), change the index to 1 or 2. This is the only line that may need adjustment on different devices.

**Why CPU is fast enough at runtime:** Training requires computing gradients across all parameters for tens of thousands of samples — an operation that benefits enormously from GPU parallelism. Inference runs the forward pass for a single image at a time, which requires far less computation and runs at 15–30ms per image on a modern laptop CPU — well within the real-time budget.

## 17.5 The Merge Conflict Incident

During development, a GitHub push failed because the remote repository had changes (from a previous session on a different machine) that the local repository did not have. Git was unable to automatically merge the diverging histories.

Git's conflict resolution leaves marker lines in affected files:
```
<<<<<<< HEAD
    # Local version
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
=======
    # Remote version
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FPS, 30)
>>>>>>> origin/main
```

These marker characters are not valid Python syntax. Every affected file crashed immediately with `SyntaxError: invalid syntax` pointing to the `<<<<<<<` line.

**Resolution:** `grep -rn "<<<<<<" *.py` identified all affected files and line numbers. Each conflict section was reviewed against project documentation to determine which version represented the intended behavior. Marker lines were deleted, the correct version was kept, and the files were committed cleanly.

**Prevention:**
- Always `git pull origin main` before beginning any session
- Use feature branches: `git checkout -b feature/optimization` for experimental work
- Commit small, frequent changes rather than large multi-file commits

---

# SECTION 18: KNOWN LIMITATIONS AND HONEST ASSESSMENT

## 18.1 Webcam Field of View: The Most Critical Limitation

A standard laptop webcam has a horizontal field of view of approximately 60–90°. The webcam is mounted at the top center of the screen, pointing directly forward. This means it can see persons sitting in front of the user — but not persons sitting directly beside the user.

The most common real-world shoulder surfing scenario is an observer seated or standing immediately to the left or right of the victim — at a café table, on a train, in a library. These observers are outside the webcam's field of view and are completely invisible to this system.

This is the single largest limitation of the design. Addressing it would require an external wide-angle camera or a camera mounted on the side of the laptop — neither of which is present on standard hardware.

## 18.2 Limited NOT_LOOKING Training Diversity

The NOT_LOOKING class in training consists of only 1,448 unique real images from BIWI, repeated approximately 10× through oversampling. This means the model has seen far less diversity in what "not looking" looks like compared to what "looking" looks like. Side faces from unusual angles (extreme downward pitch combined with yaw, faces partially occluded), in unusual lighting, or from subjects with distinctive facial features that weren't present in BIWI's 24 subjects may still produce incorrect LOOKING classifications.

The permanent fix for this limitation is collecting more real NOT_LOOKING training data — ideally from webcam captures at realistic shoulder-surfing distances, covering a wider range of subjects, angles, and lighting conditions.

## 18.3 Domain Gap: BIWI vs Real Webcam

BIWI was collected using a Microsoft Kinect RGB-D camera in a controlled laboratory environment with consistent, controlled lighting. A laptop webcam uses a small consumer CMOS sensor with a wide-angle lens, automatic exposure control, and varying lighting conditions. The image statistics (sharpness, color balance, noise level, depth of field) differ substantially between these two capture setups.

Augmentation (random brightness, rotation) partially bridges this gap but does not eliminate it. The most effective solution would be to collect a small webcam-captured dataset and use it for domain adaptation fine-tuning — a future work item described in Section 19.1.

## 18.4 Performance in Poor Lighting

In environments with low ambient light (dim café, evening use without artificial light), MediaPipe face detection reliability degrades significantly. The system falls back to the top-28% body crop heuristic more frequently. The fallback region is less precisely centered on the face and produces lower-quality head pose predictions.

A practical mitigation would be to add histogram equalization as a preprocessing step before MediaPipe:
```python
crop_yuv = cv2.cvtColor(crop, cv2.COLOR_BGR2YUV)
crop_yuv[:,:,0] = cv2.equalizeHist(crop_yuv[:,:,0])
crop_enhanced = cv2.cvtColor(crop_yuv, cv2.COLOR_YUV2BGR)
```
This was not implemented in the current version.

## 18.5 Face Occlusions

Observers wearing glasses, face masks, hats with large brims, or with hair covering significant portions of their face may cause MediaPipe to fail or produce inaccurate bounding boxes. The head pose model, trained on unoccluded BIWI faces, has no explicit handling for partial occlusion. Heavily occluded faces are more likely to be classified as UNCERTAIN and treated as SAFE — a fail-safe outcome but one that means genuine occluded threats may be missed.

## 18.6 Single-User Desk Setup Assumption

The system assumes the largest bounding box in the frame belongs to the user. This works well in a home office or solo desk setup. In an open office where multiple people legitimately share a desk area, or in a meeting room where colleagues are all at similar distances from a shared screen, the largest-box heuristic may misidentify colleagues as the user. This is a fundamental design assumption, not a bug — the system was designed for a single user at a personal laptop.

## 18.7 The UNCERTAIN Gap: A Deliberate Tradeoff

The probability gap filter (requiring gap ≥ 0.25 for LOOKING) means that any observer who consistently presents an ambiguous head angle — between 15° and 25° of yaw — will always produce UNCERTAIN predictions and will never trigger a THREAT. A sophisticated observer who knows about this threshold could theoretically maintain an ambiguous head angle and observe the screen without triggering the system.

This is a deliberate design decision, not an oversight. The alternative — requiring a smaller gap threshold — produces too many false positives from observers genuinely not looking at the screen. In a usability vs. security tradeoff, usability was prioritised because a system with too many false alarms is disabled by users and provides zero protection. A system with some false negatives on edge cases still provides protection in the vast majority of real scenarios.

---

*Continued in RESEARCH_DOCUMENTATION_PART5B.md — Future Work · Timeline · Conclusion*
