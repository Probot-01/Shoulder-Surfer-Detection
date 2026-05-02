# AI-Based Shoulder Surfing Prevention System
## Research Documentation — PART 5B OF 5 (FINAL)
### Sections: Future Work · Project Timeline · Conclusion

---

# SECTION 19: FUTURE WORK AND EXTENSIONS

## 19.1 Collect Real Webcam Perspective Dataset

**The problem it solves:** The BIWI dataset was collected with a camera directly in front of subjects. A shoulder surfer is observed from the side and slightly below (the webcam is at the bottom of the screen, angled upward). This perspective difference means the model has never seen training data from the actual deployment viewpoint.

**Proposed approach:**
1. Recruit 10 volunteers to sit at a laptop in various natural positions
2. Record their heads from the webcam perspective as they look in different directions
3. Collect approximately 500 images per subject across the full yaw range (−90° to +90°)
4. Label using MediaPipe FaceMesh (which provides yaw estimates from landmark geometry) as pseudo-labels — automated labelling without manual annotation
5. Mix approximately 30% real webcam data into the BIWI training set
6. Retrain the classification head with this combined dataset

**Expected outcome:** Substantially improved classification of side faces and intermediate-angle faces that the model currently classifies as UNCERTAIN. Reduction in fallback-to-UNCERTAIN rate during real deployment.

**Effort estimate:** 2–3 hours of data collection, 1–2 hours of processing, 90 minutes of retraining on Colab.

## 19.2 Iris Tracking as a Second-Order Signal

MediaPipe's FaceMesh solution provides 468 facial landmarks, including precise iris center coordinates (landmarks 468–477). The iris position relative to the eye center encodes gaze deviation — how far the eyes are looking away from the direction the head is pointing.

**Proposed integration:**
```python
# After head pose classification confirms LOOKING:
# Run FaceMesh on the same face crop
# Compute iris offset from eye center
iris_offset_x = iris_center_x - eye_center_x
# If iris_offset is large → eyes looking significantly away from head direction
# Downgrade LOOKING to UNCERTAIN
```

This would eliminate a category of false positives: a user who is looking at their own keyboard (head facing screen, eyes looking down) would generate a LOOKING head pose but NOT LOOKING iris signal — the combined result would correctly be NOT_LOOKING.

**Implementation complexity:** Medium. MediaPipe FaceMesh is already available in the installed library. The challenge is computing meaningful iris offsets robustly at the low resolution of cropped face images from webcam distance.

## 19.3 Window-Title-Aware Sensitivity

Currently, the system uses the same sensitivity settings regardless of what the user is doing. A user browsing social media faces the same THREAT threshold as a user entering their banking password.

**Proposed approach using `pygetwindow`:**
```python
import pygetwindow as gw

def get_sensitivity_mode():
    active = gw.getActiveWindow()
    if active is None:
        return "normal"
    title = active.title.lower()
    sensitive_keywords = ["bank", "password", "1password", "bitwarden",
                          "health", "medical", "tax", "finance", "payroll"]
    if any(kw in title for kw in sensitive_keywords):
        return "high"
    return "normal"
```

In normal mode: standard thresholds. In high-sensitivity mode: lower gap threshold (0.15 instead of 0.25), lower streak requirement (2 instead of 3). This makes the system more aggressive when the user is clearly doing something sensitive.

**Benefit:** Dramatically reduces false positives during casual use (browsing, watching videos) while maintaining strong protection during sensitive operations.

## 19.4 Screen Content Classification

Instead of blurring everything on THREAT detection, use OCR to identify what content is actually on screen and only obscure sensitive regions.

**Proposed approach:**
1. On THREAT detection, capture screenshot
2. Run PaddleOCR (fast, accurate, runs on CPU) on screenshot
3. Identify regions containing sensitive patterns: credit card numbers, passwords, email addresses, account numbers (using regex matching on OCR output)
4. Apply targeted blur only to those bounding boxes
5. Leave non-sensitive content (browser chrome, taskbar) visible

**Benefit:** The user can continue working normally with most of their screen visible; only genuinely sensitive content is obscured. Far less disruptive than full-screen blur.

## 19.5 Adaptive Threshold Learning Per User

The current thresholds (gap=0.25, streak=3, confidence=75%) are fixed values tuned to work reasonably for most setups. Different users have different typical webcam positions, different environments, and different faces. A user with a webcam positioned at an unusual angle might consistently produce intermediate gap values, causing many UNCERTAIN predictions.

**Proposed approach:**
- After 50 confirmed true positives and 50 confirmed true negatives (user-validated), compute the optimal gap threshold for this specific user/camera/environment combination
- Store per-user calibration profile in `~/.shoulder_surfing/calibration.json`
- Load profile on startup and use calibrated thresholds

## 19.6 Multi-Camera Support

**The fundamental field-of-view problem** (Section 18.1) can only be solved with additional cameras. A second wide-angle USB webcam positioned to the side of the laptop would cover the most common shoulder surfing scenario (observer seated beside the user).

**Architecture change required:**
```python
cap_front = cv2.VideoCapture(0)   # Built-in webcam — covers front
cap_side  = cv2.VideoCapture(1)   # External — covers side

# Run full pipeline on both simultaneously (background thread per camera)
# Fuse decisions: THREAT if either camera reports THREAT
```

This is a hardware requirement (costs ~$15–30 for a basic USB webcam) but requires only moderate software changes.

## 19.7 Mobile Deployment

Smartphones have powerful front-facing cameras and run neural networks efficiently on-device via mobile ML frameworks.

**Proposed architecture:**
- Convert MobileNetV2 model to TensorFlow Lite (`.tflite`) or Core ML (`.mlmodel`) format using standard conversion tools
- Build a mobile app (Android: Kotlin + ML Kit, iOS: Swift + Core ML)
- Use the front-facing camera as the observer-detection webcam
- Use the rear-facing camera to confirm what is on the screen
- The compact MobileNetV2 (3.4MB) is well-suited for on-device inference

MobileNetV2 was specifically designed with mobile deployment in mind — its depthwise separable convolutions reduce computation to a level that runs at 30+ FPS on mobile NPUs.

## 19.8 Research Paper Extension

This project contains the core of a publishable research contribution:

**The novel contribution:** Intent-based observer classification (LOOKING/NOT_LOOKING) rather than presence-based detection (person nearby/not nearby). Combined with the probability gap filter and temporal smoothing architecture, this represents a complete, evaluatable system design.

**Proposed study design:**
- 20 participants as "users" (persons being observed)
- 5 trained "observers" performing standardised observation scenarios
- Metrics: FAR (False Acceptance Rate), FRR (False Rejection Rate), EER (Equal Error Rate) — standard biometric security metrics
- Baseline comparison: presence-only detection (alert whenever a second person is nearby)
- Result: intent-based detection expected to have significantly lower FPR while maintaining comparable TPR

**Suitable venues:** ACM CHI (Human Factors in Computing), IEEE Security & Privacy Workshop, UIST (User Interface Software and Technology), or a dedicated privacy/security journal.

---

# SECTION 20: PROJECT TIMELINE RECONSTRUCTION

The complete system was built in approximately 7 working days. The following reconstructs the development sequence based on the problems encountered and the order in which modules were built and integrated.

## Day 1: Foundation and Person Detection

**Morning:**
- Python virtual environment setup
- `pip install ultralytics opencv-python numpy`
- Verified webcam opens correctly with a minimal `cap.read()` test script
- Confirmed YOLO download and first inference call works

**Afternoon:**
- Implemented person detection loop: YOLO → parse boxes → draw on frame
- Implemented zone classification: sort by area, assign USER/OBSERVER
- First successful display of blue USER box and red OBSERVER boxes
- (Note: Webcam-opens-twice and no-red-boxes bugs both occurred here and were fixed on this day)

**End of Day 1 state:** Working person detector with correct USER/OBSERVER classification, live display at 70–80 FPS.

## Day 2: Face Extraction Module

**Morning:**
- Research into head pose estimation approaches
- Decision to use MediaPipe FaceDetection over manual face detector
- `pip install mediapipe==0.10.21`
- Explored MediaPipe API, wrote minimal face detection test

**Afternoon:**
- Implemented `face_crop_utils.py` first version
- Discovered MediaPipe NoneType errors (Problem 5) — fixed with None check
- Discovered coordinate mapping bug (Problem 7) — required debug print analysis to diagnose
- Fixed coordinate conversion: relative → absolute within crop

**End of Day 2 state:** Working face extraction, returns padded crops for detected faces.

## Day 3: Dataset Preparation and Training Setup

**Morning:**
- Downloaded BIWI Kinect Head Pose Database from ETH Zurich
- Explored file structure: `frame_XXXXX_rgb.png` + `frame_XXXXX_pose.txt`
- Wrote pose file parser, extracted rotation matrices
- Implemented yaw extraction: `arctan2(R[1,0], R[0,0])`

**Afternoon:**
- Applied ±30° threshold: discovered 96.4%/3.6% imbalance (Problem 1)
- Tightened to ±20°: 90.8%/9.2% — better but still imbalanced
- Implemented random oversampling to 14,230/14,230 balance
- Set up Google Colab notebook, mounted Drive, uploaded BIWI data
- Wrote PyTorch Dataset class, DataLoader, MobileNetV2 architecture

**End of Day 3 state:** Balanced dataset ready, training script ready in Colab.

## Day 4: Model Training and Architecture Fix

**Morning:**
- First training run on Colab T4
- Discovered overfitting: train_acc=99%, val_acc=78% (Problem 2)
- Identified layer freezing bug — 60.1% trainable was too many
- Fixed: freeze ALL backbone → 12.9% trainable

**Afternoon:**
- Second training run with corrected freezing
- 20 epochs, ~75 minutes on T4
- Val accuracy reached 91.99% — checkpoint saved
- Downloaded `best_head_pose_model.pth` from Drive

**End of Day 4 state:** Trained model achieving 91.99% val accuracy, stored locally.

## Day 5: HeadPosePredictor Integration

**Morning:**
- Implemented `head_pose_predictor.py`: `HeadPoseClassifier` + `HeadPosePredictor`
- Implemented preprocessing: BGR→RGB, resize 224×224, ImageNet normalize
- Implemented argmax classification (initial — no gap filtering yet)
- First integration test: ran predictor on saved debug face crops

**Afternoon:**
- Integrated predictor into main_system.py loop
- First full system run: webcam → YOLO → face crop → head pose → display
- Discovered label inversion (Problem 9): system working backwards
- Fixed: `is_looking = (predicted_class == 1)`

**End of Day 5 state:** Full pipeline running, predictions correct but gaps and streak not yet implemented.

## Day 6: Bug Fixes and Quality Improvements

**Morning:**
- Discovered face crops too small (Problem 6): debug prints showed (20,18,3) crops
- Added 40×40 minimum size gate
- Added 40% padding to MediaPipe bounding box
- Discovered fallback region wrong (Problem 8): chest instead of face
- Fixed: top 28% with 15% horizontal margin

**Afternoon:**
- Tested with observer at extreme angles: model too lenient on side faces (Problem 10)
- Implemented probability gap filtering (gap ≥ 0.25 / ≤ -0.15)
- Added UNCERTAIN class with is_looking=False
- Implemented per-observer streak counter (streak ≥ 3)
- Raised confidence threshold to 75%

**End of Day 6 state:** Accurate threat detection, no more side-face false positives.

## Day 7: Performance, Tuning, and Deployment

**Morning:**
- Measured FPS drop with observer: 70–80 → 30–40 (Problem 11)
- Implemented YOLO frame-skip (interval=2), pose frame-skip (interval=3)
- Added crop resize to 320px, face resize to 112×112
- FPS recovered to 55–65 with observer

**Afternoon:**
- Measured threat latency: 400ms+ (Problem 12)
- Tuned thresholds: streak=3, threshold=4, cooldown=20, confidence=75%, gap=0.25
- Latency reduced to ~200ms
- Pushed to GitHub → merge conflict (Problem 13)
- Systematic scan, resolved conflicts, clean push
- Wrote README.md with setup instructions
- Final evaluation runs using evaluate_system.py

**End of Day 7 state:** Complete, functional, optimised system ready for demonstration.

---

# SECTION 21: CONCLUSION

## 21.1 What Was Built

The AI-Based Shoulder Surfing Prevention System is a complete, self-contained, real-time privacy monitoring application that runs on any standard laptop with a webcam. It continuously analyses the webcam feed, identifies all persons in view, classifies each non-user person's head orientation using a trained neural network, and issues an alert when a sustained, confident LOOKING classification is detected from any observer.

The system achieves:
- **91.99% model validation accuracy** on the BIWI head pose dataset
- **55–65 FPS** sustained performance with an observer present
- **~200ms threat response latency** from observer beginning to stare to system alert
- **Zero-GPU runtime** — all inference runs on laptop CPU
- **Modular, documented codebase** deployable on any device in under 10 minutes

## 21.2 What Was Achieved vs What Was Originally Planned

The original concept was straightforward: detect if a second person is near the user and blur the screen. This presence-based approach would have been trivial to implement (YOLO → count persons → blur) and would have been equally trivial to defeat (the system would alert whenever any person walked past, making it useless in any shared space).

The final system is substantially more sophisticated:
- **Intent-based classification:** does not alert because a person is nearby, but only because they are actively looking
- **Probabilistic uncertainty quantification:** the gap filter and UNCERTAIN class prevent false positives from ambiguous inputs
- **Temporal smoothing:** the streak + threshold state machine prevents single-frame noise from triggering alerts
- **Performance optimisation:** frame-skip caching, crop resizing, and face resizing enable real-time operation on consumer hardware

This evolution from presence detection to intent detection is the core intellectual contribution of the project, and it arose organically from encountering and solving real problems during development — not from a pre-planned design.

## 21.3 What Was Learned

### Technical Skills
- **PyTorch training pipeline:** dataset classes, data loaders, optimizers, schedulers, checkpointing
- **Transfer learning:** understanding what frozen vs trainable layers mean, how to choose the right configuration
- **OpenCV real-time processing:** VideoCapture lifecycle, drawing, display, keyboard handling
- **MediaPipe integration:** API patterns, coordinate systems, None handling, lifecycle management
- **State machine design:** temporal smoothing, asymmetric thresholds, hysteresis

### AI Engineering Lessons
- **Class balance must be verified before training, not after.** Imbalanced training silently produces useless models that pass accuracy metrics.
- **Generated code must be verified.** Layer freezing percentage must be explicitly computed and checked.
- **Domain gaps are real.** A model trained on lab data performs differently on consumer hardware in real conditions.
- **Uncertainty quantification matters.** Raw argmax classification is insufficient for safety-relevant outputs.

### Debugging Methodology
The systematic approach used throughout — observe symptom precisely, add targeted debug prints, identify root cause from data, apply minimal fix, verify with same prints — proved reliable across all 13 problems encountered. This methodology is independent of the specific technologies and is directly applicable to any future AI or software project.

### Software Engineering
- **Modular design enables fast debugging.** Each module could be tested independently.
- **Version control is essential even for solo projects.** The merge conflict was a direct consequence of editing the same file on two different machines without syncing.
- **Documentation written during development is far more accurate than documentation written after.** Every decision in this document was recorded close to the time it was made, preserving the reasoning that would otherwise be lost.

## 21.4 What Makes This Project Meaningful

**It solves a real problem.** Shoulder surfing affects millions of people daily. The absence of automated software protection for this threat is a genuine gap that this project addresses. The system built here could realistically be packaged as a background application and used by real users.

**It demonstrates the complete ML pipeline.** From raw academic dataset (BIWI) through data preparation, model training, integration into a real-time system, performance optimisation, evaluation, and deployment — this project covers the full lifecycle of an applied machine learning system. Most student projects cover only a portion of this pipeline.

**It combines multiple AI disciplines.** Object detection (YOLO), face detection (MediaPipe BlazeFace), image classification (MobileNetV2), temporal reasoning (state machine), and human-computer interaction (real-time visual feedback) are all present and integrated. Each discipline informed the others.

**It has a clear path to publishable research.** The intent-based observer classification approach — particularly the probability gap filter as an uncertainty quantification mechanism — is novel in the context of shoulder surfing prevention. A controlled user study following this implementation would produce a complete paper suitable for submission to privacy and HCI venues.

## 21.5 Final Honest Assessment

This is a working system built by one person in seven days using free tools, a free dataset, and free cloud compute. It runs in real time, it detects real observers with high reliability under good lighting conditions, and it is documented thoroughly enough that a developer unfamiliar with the codebase could reproduce it completely from these documents alone.

It is not perfect. The webcam field-of-view limitation means the most common shoulder surfing scenario — someone sitting directly beside you — is invisible to the system. The BIWI domain gap means that unusual faces, unusual lighting, or unusual angles may be classified incorrectly. The fallback face extraction is a heuristic that fails for non-standard body proportions.

But every limitation is documented. Every design decision has a recorded rationale. Every bug encountered was diagnosed to its root cause and fixed systematically. The result is not a polished product but it is a principled engineering effort — one that demonstrates the kind of methodical, problem-solving approach that real AI systems development requires.

The project was built, tested, evaluated, documented, and deployed in seven days. It works.

---

# DOCUMENT COMPLETION NOTICE

This concludes the five-part research documentation for the AI-Based Shoulder Surfing Prevention System. To compile the complete document, read the following files in order:

| Order | File | Sections |
|---|---|---|
| 1 | `RESEARCH_DOCUMENTATION_PART1.md` | Abstract, Introduction |
| 2 | `RESEARCH_DOCUMENTATION_PART1B.md` | Literature Background, System Architecture |
| 3 | `RESEARCH_DOCUMENTATION_PART2.md` | Dataset Preparation, Model Training, Inference |
| 4 | `RESEARCH_DOCUMENTATION_PART3.md` | Webcam, YOLO, Face Extraction, Decision Engine |
| 5 | `RESEARCH_DOCUMENTATION_PART4A.md` | Problems 1–7 |
| 6 | `RESEARCH_DOCUMENTATION_PART4B.md` | Problems 8–13, Lessons Learned |
| 7 | `RESEARCH_DOCUMENTATION_PART5A.md` | Evaluation, Technology Stack, Deployment, Limitations |
| 8 | `RESEARCH_DOCUMENTATION_PART5B.md` | Future Work, Timeline, Conclusion |

All files are located in the project root directory. The complete documentation covers every aspect of this project with no significant omissions — from the initial motivation through every technical decision, every bug, every solution, and every lesson learned.

**Total coverage:**
- Section 1: Abstract
- Section 2: Introduction
- Sections 3–4: Literature and Architecture
- Sections 5–7: Dataset, Training, Inference
- Sections 8–12: All System Modules
- Section 13: 13 Problems with complete post-mortems
- Section 14: Consolidated Lessons Learned
- Sections 15–18: Evaluation, Stack, Deployment, Limitations
- Sections 19–21: Future Work, Timeline, Conclusion
