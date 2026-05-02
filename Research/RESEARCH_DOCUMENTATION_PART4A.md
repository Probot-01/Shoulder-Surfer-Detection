# AI-Based Shoulder Surfing Prevention System
## Research Documentation — PART 4A OF 5
### Section 13: Problems 1–7 — Complete Engineering Post-Mortem

---

# SECTION 13: COMPLETE PROBLEMS AND SOLUTIONS RECORD

Each problem is documented with: symptoms, root cause, debug process, fix, and lessons learned.

---

## Problem 1: Severe Class Imbalance (96.4% vs 3.6%)

**Summary:** The BIWI dataset, when labelled with a ±30° yaw threshold, produced a nearly single-class dataset that would cause the model to learn nothing useful.

**When discovered:** During initial dataset preparation, immediately after running the label generation script and printing class distribution statistics.

**Symptoms observed:**
```
Class distribution:
  LOOKING:     15,119 samples (96.4%)
  NOT_LOOKING:    559 samples  (3.6%)
```
The numbers appeared in the terminal after processing all BIWI frames.

**Technical root cause:** The BIWI dataset was collected by instructing subjects to move their heads naturally while being recorded by a camera placed directly in front of them. The subjects naturally spent most of their time facing the camera — after all, they were being recorded and the camera was at eye level in front of them. A ±30° yaw window is wide enough to capture any face orientation from "looking directly at camera" all the way to "looking 30° to the left or right." Because subjects rarely turned their heads beyond 30° during the recording sessions, almost all frames fell within this window and were labelled LOOKING.

**Why this symptom produces this specific failure:** A neural network optimises its loss function (CrossEntropyLoss) over all training samples. The gradient of the loss with respect to model parameters is computed as a weighted average over the batch. If 96.4% of samples belong to one class, that class dominates every gradient update. The model quickly learns that predicting LOOKING for every input minimises loss to approximately `−log(0.964) = 0.037` — an extremely low value — without ever learning to distinguish the two classes. The model achieves 96.4% accuracy while being completely useless.

**Debug process:** Print class counts after labelling. Check model predictions on a held-out set: if all predictions are LOOKING regardless of input, this confirms the lazy classifier problem. Also check per-class accuracy (NOT_LOOKING recall = 0% is definitive).

**The solution — two-step fix:**

*Step 1: Tighten threshold to ±20°*
```python
label = "LOOKING" if abs(yaw_degrees) <= 20 else "NOT_LOOKING"
```
Result: 14,230 LOOKING vs 1,448 NOT_LOOKING. Still imbalanced (90.8% vs 9.2%) but minority class grew from 559 to 1,448 — enough real images to work with.

*Step 2: Random oversampling*
```python
from sklearn.utils import resample
not_looking_balanced = resample(
    not_looking_samples, replace=True, n_samples=14230, random_state=42
)
```
Final dataset: 14,230 + 14,230 = 28,460 samples, perfectly balanced.

**Why this fixes the root cause:** With equal class counts, every gradient update gives equal weight to both classes. The model cannot achieve low loss by predicting one class always — it must correctly distinguish them. The tighter threshold also means NOT_LOOKING samples genuinely look different (faces clearly turned away) from LOOKING samples, giving the model learnable visual signal.

**Behavior after fix:** Model learns both classes, achieves 91.99% validation accuracy with balanced per-class performance.

**General lesson:** Class imbalance is extremely common in real-world datasets — medical diagnosis datasets have 99%+ healthy samples, fraud detection datasets have 99%+ legitimate transactions. Always print class distribution before training. Never trust accuracy alone on imbalanced datasets; always check per-class recall and precision.

---

## Problem 2: Too Many Layers Unfrozen (60.1% trainable)

**Summary:** Code generation produced a MobileNetV2 setup where 60.1% of parameters were trainable, creating severe overfitting risk with the heavily oversampled dataset.

**When discovered:** After the first training attempt, when training loss reached near-zero but validation accuracy plateaued at 78% — a textbook overfitting signature.

**Symptoms observed:**
```
Epoch 8:  train_loss=0.08  train_acc=97.2%  val_acc=78.4%
Epoch 12: train_loss=0.03  train_acc=99.1%  val_acc=77.9%
Epoch 16: train_loss=0.01  train_acc=99.8%  val_acc=76.2%  ← val declining
```
Training accuracy approaching 100% while validation accuracy stagnates or declines is the definitive overfitting signature.

**Technical root cause:** The initial freezing code used an index-based approach:
```python
# WRONG
for i, (name, param) in enumerate(backbone.named_parameters()):
    if i < 40:
        param.requires_grad = False
```
MobileNetV2 has approximately 120 named parameter tensors. Freezing only the first 40 left the latter 80 (60.1% of parameters) trainable. With 1,448 unique NOT_LOOKING images repeated ~10× each, the model has enough free parameters to memorise every training image's exact pixel values — rather than learning generalisable head orientation features.

**Why 60% trainable + oversampled data = overfitting:** Each oversampled NOT_LOOKING image appears 10 times in training. If the model has 60% of its backbone free to adapt, it can learn to identify each specific image by subtle pixel-level features (compression artifacts, specific lighting on specific subjects) rather than the semantic concept of "face turned away." This works perfectly on training data but fails on any new face it has never seen.

**Debug process:** Print parameter counts:
```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable/total*100:.1f}%")
# Output: Trainable: 60.1%
```
Plotting training vs validation loss curves visually confirms overfitting.

**Solution:**
```python
# CORRECT — freeze ALL backbone parameters
for param in backbone.parameters():
    param.requires_grad = False
# Custom head parameters are still trainable (defined separately)
```
Result: Trainable parameters = 328,450 (12.9% of total).

**Why this fixes overfitting:** The frozen backbone's 87.1% of parameters cannot change at all — they are fixed at their ImageNet-pretrained values. Only the 328,450 parameters in the custom head can adapt. With a relatively small trainable space, the model cannot memorise individual training images. It must use the general ImageNet features (edges, textures, shapes) as input and only learns how to map those abstract features to LOOKING/NOT_LOOKING labels.

**Behavior after fix:** Training and validation accuracy converge closely. Final val_acc = 91.99%.

**General lesson:** When using oversampled or augmented data, always calculate and print the percentage of trainable parameters before starting training. For small fine-tuning datasets (<50,000 unique examples), a trainable percentage above 20% is a warning sign.

---

## Problem 3: Webcam Opens Twice

**Summary:** The system opened two separate webcam windows in sequence — the first missing UI elements, the second closing after 2–3 seconds.

**When discovered:** During the first integration test of the full system, when `main_system.py` was first run.

**Symptoms observed:**
- Terminal showed normal initialization messages
- Webcam window appeared with person detection boxes but no status bar and no head pose labels
- After pressing `q`, the window closed — then immediately reopened
- Second window had the correct UI but closed automatically within 2–3 seconds
- No error messages in terminal

**Technical root cause:** Two separate `VideoCapture(0)` instances were being created — one in the main function and one in a background thread, both attempting to capture from the same camera device. On most operating systems, only one process/thread can hold exclusive access to a webcam. The second instance either silently failed or grabbed frames from a buffer, causing the second window to show then immediately fail when the buffer was exhausted.

Separately, `main_system.py` was structured with the main pipeline in a called function, and a second call to this function existed elsewhere in the file — possibly from a generated `if __name__ == "__main__"` block that called `main()` twice, or from a `run_demo.py` that called `subprocess.run(["python", "main_system.py"])` while also calling the `main()` function directly.

**Debug process:**
```python
# Added at VideoCapture creation:
import traceback
print(f"VideoCapture created at:")
traceback.print_stack()
```
This revealed two separate call stacks both reaching `cv2.VideoCapture(0)`.

**Solution:** Enforce a single VideoCapture instance:
```python
cap = cv2.VideoCapture(0)   # Created ONCE, before the while loop
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    return
try:
    while True:
        ret, frame = cap.read()
        # ... all processing ...
finally:
    cap.release()   # Released ONCE, in finally block
```
Removed all duplicate calls and ensured `run_demo.py` launches `main_system.py` via `subprocess.run()` only, never importing and calling it directly.

**Behavior after fix:** Single window opens, stays open, all UI elements present from first frame.

**General lesson:** Resource ownership must be clear and singular. VideoCapture, file handles, and network sockets must have one clear owner. Use try/finally to guarantee release regardless of how the program exits.

---

## Problem 4: No RED Observer Boxes Appearing

**Summary:** The terminal showed multiple persons detected but only the USER blue box appeared in the display window — no red observer boxes.

**When discovered:** During first live test with two people in front of the camera.

**Symptoms observed:**
```
Persons detected: 2
[Frame   30] FPS:63.2  Persons:2  State:SAFE
```
Terminal confirmed two persons detected. Display showed only one blue USER box with no second box.

**Technical root cause:** The zone classification result (`observer_boxes`) was computed correctly but never passed to the drawing code. The drawing section had its own internal variable also named `observer_boxes` (initialised to `[]`) that was never populated from the zone classification output. This was a variable scoping / copy-paste error where two independent code blocks each had their own `observer_boxes` variable.

```python
# Zone classification section:
user_box = raw_boxes[0]
observer_boxes = raw_boxes[1:]   # ← Correct

# ... other processing ...

# Drawing section (BUG):
observer_boxes = []   # ← This shadows the correct variable!
for obs_box in observer_boxes:  # Always empty
    # draw red box
```

**Debug process:**
```python
# Added immediately before drawing loop:
print(f"DEBUG drawing: user_box={user_box is not None}, observers={len(observer_boxes)}")
# Output: DEBUG drawing: user_box=True, observers=0
# Contradiction: terminal said 2 persons, but drawing sees 0 observers
```
This isolated the bug to the drawing section's own variable.

**Solution:** Remove the shadowing `observer_boxes = []` line from the drawing section. Ensure the drawing loop uses the variable populated by zone classification.

**Behavior after fix:** Red OBSERVER boxes appear immediately when a second person enters the frame.

**General lesson:** Variable shadowing is a silent, difficult-to-spot bug. Python does not warn about variable reuse within the same scope. Always use distinct variable names for each logical stage of a pipeline, or structure code so that each stage's output is the direct input to the next stage.

---

## Problem 5: MediaPipe NoneType Errors

**Summary:** The system crashed with NoneType errors whenever a frame had no face detected, and occasionally on the first frame after startup.

**When discovered:** During integration testing, whenever the camera showed an empty scene or a person from behind.

**Symptoms observed:**
```
AttributeError: 'NoneType' object has no attribute 'close'
  face_detector.close()

AttributeError: 'NoneType' object is not subscriptable
  detections = result.detections[0]
```

**Technical root cause — two separate bugs:**

*Bug A:* `face_detector` was referenced in the `finally` block before it was assigned. If the `FaceDetection()` constructor threw an exception, `face_detector` remained undefined (or was `None`) but the `finally` block still tried to call `face_detector.close()`.

*Bug B:* `result.detections` can be `None` (not an empty list) when MediaPipe finds no face. The code attempted to subscript `result.detections[0]` without first checking `if result.detections:`.

**Debug process:** Read the full traceback — the line numbers pointed directly to the problematic lines. The AttributeError messages are self-explanatory once the None origin is understood.

**Solution:**

*Fix A:*
```python
face_detector = None   # Initialize to None before try block
try:
    face_detector = mp_face.FaceDetection(...)
    # ... loop ...
finally:
    if face_detector is not None:
        face_detector.close()   # Only call if successfully created
```

*Fix B:*
```python
result = face_detector.process(crop_rgb)
if result.detections is None or len(result.detections) == 0:
    return None, "no_face_detected"   # Handle gracefully
best_detection = result.detections[0]
```

**Behavior after fix:** Empty frames and frames with no visible face are handled gracefully, returning the fallback crop instead of crashing.

**General lesson:** Any value that can be None must be checked before use. MediaPipe specifically documents that `result.detections` is None (not empty list) when no detections are found — always read library documentation for None vs empty distinction.

---

## Problem 6: Face Crops Too Small (20–42 Pixels)

**Summary:** MediaPipe returned face bounding boxes without padding, producing crops too small for meaningful inference. The model generated confident but wrong LOOKING predictions from blurry noise.

**When discovered:** After adding debug prints to the face extraction pipeline. The system was producing constant THREAT alerts even when the observer was clearly looking away.

**Symptoms observed:**
```
Face crop shape: (41, 42, 3)   ← borderline
Face crop shape: (20, 18, 3)   ← far too small
State: THREAT  Label: LOOKING  Confidence: 87.3%
State: THREAT  Label: LOOKING  Confidence: 91.1%
```
The system declared THREAT at high confidence regardless of observer head direction.

**Technical root cause:** MediaPipe's face bounding box is intentionally tight — it encloses only the face oval with minimal margin. For a person detected at 1.5 metres, the face region in a 640×480 frame is approximately 60×80 pixels. MediaPipe's tight box within this 60×80 region might be only 35×40 pixels. The preprocessing step then upscales this 35×40 crop to 224×224 — an approximately 6× magnification. At this magnification, each original pixel becomes a 6×6 block of identical color. The result is a highly pixelated, blurry image that contains no useful edge or texture information.

**Why the model is confidently wrong on noise:** The model was trained on clean 224×224 images but never on heavily pixelated ones. When it receives a pixelated image, its features extract unusual activation patterns that happen to correlate with the LOOKING class (possibly because ImageNet features respond to certain frequency patterns differently). The confidence is high because the model has never seen anything like this in training and its calibration breaks down for out-of-distribution inputs.

**Debug process:**
```python
face_crop, reason = extract_padded_face_crop(...)
if face_crop is not None:
    print(f"Face crop shape: {face_crop.shape}")
    cv2.imwrite(f"debug_face_{frame_count}.jpg", face_crop)
```
Saving the actual crops to disk made the problem immediately visible — the saved images were unrecognisably pixelated blobs.

**Solution — two parts:**

*Part 1: Add 40% padding to MediaPipe bbox:*
```python
pad_x = int(face_w * 0.4)
pad_y = int(face_h * 0.4)
fx1 = max(0, raw_fx1 - pad_x)
fy1 = max(0, raw_fy1 - pad_y)
fx2 = min(crop_w, raw_fx2 + pad_x)
fy2 = min(crop_h, raw_fy2 + pad_y)
```

*Part 2: Enforce 40×40 minimum size gate:*
```python
if face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
    return None, "face_too_small"
```

**Behavior after fix:** Small crops are rejected and handled as UNKNOWN (safe). Remaining crops are larger, better quality, and produce meaningful predictions. False THREAT rate from noisy inputs drops to near zero.

**General lesson:** Always inspect intermediate artifacts (crops, preprocessed images) visually during debugging. Numbers alone (shape, size) do not reveal quality. This problem class — confident wrong predictions on out-of-distribution inputs — is extremely common in deployed neural networks. Input validation gates are essential.

---

## Problem 7: Coordinate Mapping Bug

**Summary:** MediaPipe's relative face coordinates (0.0–1.0) were treated as absolute pixel coordinates, causing face crops to point at completely wrong regions.

**When discovered:** After the face crop size bug was partially fixed. Debug prints showed face crop coordinates that didn't match where faces were visible on screen.

**Symptoms observed:**
```
Person box: x1=128, y1=200, x2=342, y2=413  (size: 214×213)
MediaPipe bbox: xmin=0.610, ymin=0.174, w=0.195, h=0.196
Buggy face coords: fx1=0, fy1=0, fx2=0, fy2=0
  (int(0.610)=0, int(0.174)=0)
```
Face crop was always the top-left corner of the person box — no matter where the face was.

**Technical root cause:** MediaPipe's `relative_bounding_box` returns coordinates as fractions of the input image dimensions. If MediaPipe was given a person crop of size 214×213 pixels, then:
- `xmin=0.610` means 61.0% of the crop width = pixel 130 within the crop
- `ymin=0.174` means 17.4% of the crop height = pixel 37 within the crop

The buggy code applied `int()` directly to these values:
```python
# WRONG
fx1 = int(bbox.xmin)   # int(0.610) = 0
fy1 = int(bbox.ymin)   # int(0.174) = 0
fx2 = int(bbox.xmin + bbox.width)   # int(0.805) = 0
fy2 = int(bbox.ymin + bbox.height)  # int(0.370) = 0
```
Every face crop started at pixel (0,0) within the person crop.

**Debug process:**
```python
print(f"Relative bbox: xmin={bbox.xmin:.3f}, ymin={bbox.ymin:.3f}")
print(f"Crop size: {crop_w}×{crop_h}")
print(f"Computed face: ({fx1},{fy1})-({fx2},{fy2})")
```
The computed coordinates being (0,0)-(0,0) immediately revealed the integer truncation bug.

**Solution:**
```python
# CORRECT — multiply by crop dimensions to convert to pixels
crop_h, crop_w = person_crop.shape[:2]
raw_fx1 = int(bbox.xmin * crop_w)
raw_fy1 = int(bbox.ymin * crop_h)
raw_fx2 = int((bbox.xmin + bbox.width) * crop_w)
raw_fy2 = int((bbox.ymin + bbox.height) * crop_h)
```

**After fix, same example:**
```
Relative bbox: xmin=0.610, ymin=0.174, w=0.195, h=0.196
Crop size: 214×213
Correct face: (130, 37) - (172, 79)  ← accurate face location within crop
```

**Behavior after fix:** Face crops now contain actual faces rather than random corner regions. Head pose predictions immediately became meaningful.

**General lesson:** Coordinate system mismatch is one of the most common bugs in computer vision pipelines. Every coordinate in a multi-stage pipeline must be labelled with its reference frame (full frame space? crop space? relative?). Adding explicit comments like `# coords in person_crop space, not full frame` prevents this class of bug entirely. When debugging spatial bugs, always visualise: draw the crop rectangle on the display frame to verify it encloses the expected region.

---

*Continued in RESEARCH_DOCUMENTATION_PART4B.md — Problems 8–13 and Consolidated Lessons Learned*
