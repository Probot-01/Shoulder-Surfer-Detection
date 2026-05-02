# AI-Based Shoulder Surfing Prevention System
## Research Documentation — PART 4B OF 5
### Section 13 (continued): Problems 8–13 · Section 14: Lessons Learned

---

## Problem 8: Fallback Region Incorrect (Chest Instead of Face)

**Summary:** The fallback face extraction (used when MediaPipe finds no face) cropped the chest/neck area instead of the face, producing useless input for the head pose model.

**When discovered:** When testing with observers at extreme angles (nearly profile view), where MediaPipe reliably fails. Debug images showed cropped fabric and neck regions.

**Symptoms observed:**
```
reason=fallback_estimate  Label=LOOKING  Confidence=78.4%
reason=fallback_estimate  Label=LOOKING  Confidence=81.2%
```
Fallback crops consistently predicted LOOKING even when the observer was clearly in profile. Saving the crops to disk revealed they contained the observer's shirt and neck, not their face.

**Technical root cause:** The original fallback logic:
```python
# WRONG — too much of the bounding box
face_crop = person_crop[0 : int(crop_h * 0.35), :]
```
In a typical upper-body bounding box (waist to top of head), 35% of the height from the top reaches approximately the upper chest. The face is typically in the top 15–25% of such a box. Taking 35% includes the forehead down through the neck and into the chest — predominantly non-face content.

**Debug process:** Save fallback crops with visible filename indicating they are fallbacks:
```python
if reason == "fallback_estimate":
    cv2.imwrite(f"fallback_crop_{frame_count}.jpg", face_crop)
```
Opening these files immediately revealed the body/chest content.

**Solution:**
```python
# CORRECT — tighter crop with horizontal centering
margin_x = int(crop_w * 0.15)
face_crop = person_crop[
    0 : int(crop_h * 0.28),   # Top 28% of height
    margin_x : crop_w - margin_x   # 15% margin each side
]
```

**Why 28%:** Empirical testing on the actual webcam setup showed that 28% of the upper-body bounding box height reliably captures from the top of the head to approximately the chin. The 15% horizontal margin removes the shoulder areas and narrows the crop to the head-width region.

**Behavior after fix:** Fallback crops contain actual head/face regions. Fallback predictions remain less reliable than MediaPipe-confirmed crops (they still receive the UNCERTAIN treatment from the gap filter) but no longer systematically predict LOOKING for chest regions.

**General lesson:** Fallback logic requires independent testing with real data. A fallback that works on average anatomy may fail for specific body proportions or bounding box configurations. Always inspect fallback outputs separately from primary outputs.

---

## Problem 9: Label Inversion (LOOKING/NOT_LOOKING Reversed)

**Summary:** The entire system was working correctly except that the labels were perfectly inverted — looking away triggered THREAT, looking at the screen was SAFE.

**When discovered:** During the first live functional test after all previous bugs were fixed. The system responded correctly to observers but with reversed semantics.

**Symptoms observed:**
```
[Observer turns away from screen]
  State changed → THREAT
  Label: LOOKING  Confidence: 89.3%

[Observer looks directly at screen]
  State changed → SAFE
  Label: NOT_LOOKING  Confidence: 84.1%
```
Everything responded correctly except backward — the system was a perfectly inverted detector.

**Technical root cause:** The class mapping was defined correctly:
```python
CLASS_NAMES = {0: "NOT_LOOKING", 1: "LOOKING"}
```
But the `is_looking` field was computed using the wrong class index:
```python
# WRONG
predicted_class = torch.argmax(probs).item()
is_looking = (predicted_class == 0)   # ← Should be == 1
label = CLASS_NAMES[predicted_class]
```
Class 0 is NOT_LOOKING. The condition `predicted_class == 0` sets `is_looking=True` when NOT_LOOKING wins — the exact opposite of the intended logic.

**Debug process:** Add a comprehensive debug print:
```python
print(f"argmax={predicted_class}, label={label}, is_looking={is_looking}")
# Output: argmax=1, label=LOOKING, is_looking=False  ← wrong!
# LOOKING predicted but is_looking=False
```
The contradiction between `label=LOOKING` and `is_looking=False` immediately identified the off-by-one error.

**Solution:**
```python
# CORRECT
predicted_class = torch.argmax(probs).item()
label      = CLASS_NAMES[predicted_class]
is_looking = (predicted_class == 1)   # Class 1 is LOOKING
```

**Behavior after fix:** System correctly identifies observers looking at screen (LOOKING, is_looking=True) and rewards NOT_LOOKING observers with SAFE state.

**General lesson:** In binary classification with 0/1 class indices, always explicitly document which index corresponds to which class, and always verify with a manual test: feed a known-LOOKING image and check all four fields (label, confidence, is_looking, gap) manually. This type of inversion bug — where the system works perfectly in the wrong direction — is particularly insidious because it passes many unit tests and looks operational.

---

## Problem 10: Model Too Lenient on Side Faces

**Summary:** Even faces clearly in profile (90° from camera) were classified as LOOKING with 65–75% confidence, causing false THREAT alerts for observers looking in completely different directions.

**When discovered:** After fixing the inversion bug, during realistic testing with an observer deliberately looking at the side wall.

**Symptoms observed:**
```
[Observer looking at side wall, nearly full profile]
  Label: LOOKING  Confidence: 68.4%  Gap: +0.11
  Label: LOOKING  Confidence: 71.2%  Gap: +0.17
  State: THREAT
```
The system incorrectly classified a profile face as LOOKING.

**Technical root cause — three compounding factors:**

*Factor 1: Weak NOT_LOOKING features.* The NOT_LOOKING class had only 1,448 unique real images (profiles, side faces). Despite augmentation, the model saw far less visual diversity in NOT_LOOKING than in LOOKING. The resulting feature space for NOT_LOOKING was less well-defined.

*Factor 2: Profile faces are ambiguous.* A face photographed at 60–80° of yaw shows the nose, one eye, and one cheek. This appearance is relatively rare in ImageNet training data. The frozen MobileNetV2 backbone has weaker, less discriminative features for this viewpoint. The model defaults toward the majority class (LOOKING) when uncertain.

*Factor 3: Confidence threshold too low.* The original threshold of 70% confidence meant that a 71% LOOKING prediction triggered THREAT — yet 71% is barely above chance for this two-class problem. A model that is 71% confident in a binary decision is substantially uncertain.

**Debug process:**
```python
print(f"gap={result['gap']:.3f}  prob_looking={result['prob_looking']:.1f}%"
      f"  prob_not_looking={result['prob_not_looking']:.1f}%")
# gap=+0.11: model is 55.5% vs 44.5% — barely above chance
```
Printing the gap values revealed that most profile face predictions had small positive gaps (+0.05 to +0.20) — the model was barely leaning toward LOOKING, not confidently predicting it.

**Solution — four combined changes:**

*1. Probability gap filter:*
```python
gap = prob_looking - prob_not_looking
if gap >= 0.25:     label = "LOOKING",    is_looking = True
elif gap <= -0.15:  label = "NOT_LOOKING", is_looking = False
else:               label = "UNCERTAIN",   is_looking = False, confidence = 0.0
```

*2. Confidence threshold raised to 75%:* Only LOOKING predictions with confidence > 75% contribute to threat streak.

*3. UNCERTAIN class added:* Predictions with gap between -0.15 and +0.25 are now silently ignored.

*4. Per-observer streak counter (3 frames):* Even if a side face occasionally passes the gap filter, it must do so for 3 consecutive pose-frames to contribute to THREAT.

**Behavior after fix:** Profile faces produce `gap ≈ +0.05 to +0.18` — below the 0.25 threshold. They are classified as UNCERTAIN and contribute nothing to threat counts. Only faces clearly oriented toward the camera (gap ≥ 0.25) register as LOOKING.

**General lesson:** Raw argmax classification is insufficient for safety-critical applications. Confidence calibration and uncertainty quantification are important properties in deployed AI systems. The probability gap filter is a practical, effective way to implement input-based uncertainty rejection without changing the model architecture.

---

## Problem 11: FPS Drop When Observer Enters (70–80 → 30–40 FPS)

**Summary:** Adding one observer to the scene caused FPS to drop from 70–80 to 30–40 — making the system feel sluggish and reducing the effective update rate of all threshold-count mechanisms.

**When discovered:** During the first integrated test with an observer present.

**Symptoms observed:**
```
[User alone]
[Frame   30] FPS:76.3  Persons:1  State:SAFE
[Observer enters]
[Frame   60] FPS:34.7  Persons:2  State:SAFE
[Frame   90] FPS:31.2  Persons:2  State:SAFE
```
FPS halved when the second person was detected.

**Technical root cause:** Every frame with an observer present ran:
1. MediaPipe FaceDetection on the full-resolution (640×480 or larger) person crop — approximately 8–12ms
2. MobileNetV2 inference on the face crop (upscaled to 224×224) — approximately 18–25ms

Combined with YOLO (~15ms per frame), total per-frame processing reached 41–52ms, limiting FPS to approximately 19–24 FPS theoretical maximum. The measured 30–40 FPS reflected the system spending roughly 25–33ms per frame on these operations.

**Debug process:**
```python
t0 = time.time()
face_crop, reason = extract_padded_face_crop(...)
t1 = time.time()
result = predictor.predict(face_crop)
t2 = time.time()
print(f"MediaPipe: {(t1-t0)*1000:.1f}ms  MobileNetV2: {(t2-t1)*1000:.1f}ms")
# Output: MediaPipe: 9.3ms  MobileNetV2: 22.7ms
```

**Solution — four optimizations:**

*Optimization 1: YOLO frame-skip (interval=2)*
YOLO runs every 2nd frame, caching results for the skipped frame. Average YOLO cost: 7.5ms/frame (was 15ms).

*Optimization 2: Pose frame-skip (interval=3)*
MediaPipe + MobileNetV2 run every 3rd frame; cached results reused. Average pose cost: 3–4ms/frame (was 10ms+).

*Optimization 3: Person crop resized to 320px max before MediaPipe*
```python
person_crop_small = resize_to_max_width(person_crop, max_width=320)
```
Smaller input → MediaPipe processes ~6ms/frame (was 9–12ms).

*Optimization 4: Face crop resized to 112×112 before MobileNetV2*
```python
face_crop = cv2.resize(face_crop, (112, 112))
```
Reduces data transfer to the model preprocessing pipeline. Combined with the internal 224×224 resize, saves 3–5ms.

**Behavior after fix:**
```
[Frame  30] FPS:73.1  Persons:2  State:SAFE  YOLO:YOLO  Pose:live
[Frame  60] FPS:68.4  Persons:2  State:SAFE  YOLO:cached  Pose:cached
```
55–65 FPS sustained with observer present.

**General lesson:** Profile per-operation timing before optimising. The 3ms YOLO call doesn't need optimisation; the 22ms MobileNetV2 call does. Frame-skip caching is a powerful technique for real-time systems where the cost of an operation exceeds the time budget but its result changes slowly.

---

## Problem 12: Threat Declared Too Late (400ms+ Delay)

**Summary:** After all previous bugs were fixed, the system correctly detected observers but reacted too slowly — the user could see the observer for several hundred milliseconds before any alert appeared.

**When discovered:** During final system evaluation. Measured by pressing a stopwatch at the moment of deliberate screen staring and recording when the status bar changed to red.

**Symptoms:** Observer looks directly at screen → status bar stays GREEN for 400ms+ before turning red.

**Technical root cause — cascade of thresholds:**

With the original threshold values:
- `pose_interval = 3` (head pose runs every 3rd frame)
- `observer_streak = 5` consecutive LOOKING pose-frames required
- `threat_threshold = 10` consecutive THREAT frames required

Minimum real frames before THREAT:
```
streak of 5 pose-frames × pose_interval(3) = 15 real frames
+ threat_threshold(10) = 25 real frames minimum
At 60 FPS = 25/60 = 417ms
```
This was the minimum — actual delay could be longer if the first few pose-frames were UNCERTAIN.

**Debug process:** Print timestamp at each milestone:
```python
print(f"[{time.time():.3f}] Observer streak: {engine.observer_looking_streak}")
print(f"[{time.time():.3f}] Threat frame count: {engine.threat_frame_count}")
print(f"[{time.time():.3f}] State → {state}")
```
The timestamps confirmed the cascade: streak took ~250ms to reach 5, then threshold took another ~167ms to reach 10. Total: ~417ms.

**Solution — reduce all thresholds:**
```python
# Before → After
observer_streak    = 5 → 3     (in decision_engine.py)
threat_threshold   = 10 → 4    (in ThreatDecisionEngine constructor)
cooldown_frames    = 30 → 20   (in ThreatDecisionEngine constructor)
confidence_thresh  = 80 → 75%  (in ThreatDecisionEngine)
gap_threshold      = 0.30 → 0.25  (in head_pose_predictor.py)
```

New minimum delay:
```
streak of 3 pose-frames × pose_interval(3) = 9 real frames
+ threat_threshold(4) = 13 real frames minimum
At 60 FPS = 13/60 ≈ 217ms
```

**Behavior after fix:** Observer looks at screen → THREAT declared within ~200ms. Perceptibly faster — the system reacts before the observer has had time to read a complete sentence.

**Why cooldown was also reduced:** The original cooldown of 30 frames at 60 FPS = 500ms. After an observer looked away, the screen remained in THREAT state for half a second unnecessarily. Reducing to 20 frames = 333ms, still preventing oscillation but recovering faster.

**General lesson:** In real-time systems, all time constants must be calculated in actual wall-clock milliseconds, not abstract frame counts. The relationship between frame counts and real time depends on FPS — always compute and document the wall-clock equivalents of every threshold.

---

## Problem 13: Git Merge Conflict Breaking Python Files

**Summary:** A failed GitHub push left git conflict markers inside Python source files, causing immediate `SyntaxError` crashes on every run.

**When discovered:** After attempting to push local changes while a remote version existed with different code.

**Symptoms observed:**
```
  File "main_system.py", line 147
    <<<<<<< HEAD
    ^
SyntaxError: invalid syntax
```
The entire project became unrunnable simultaneously.

**Technical root cause:** Git's merge conflict resolution inserts marker lines into files when it cannot automatically merge two versions:
```python
<<<<<<< HEAD
# version A: local code
cap = cv2.VideoCapture(0)
=======
# version B: remote code  
cap = cv2.VideoCapture(0)
frame_width = 640
>>>>>>> origin/main
```
These marker characters (`<`, `>`, `=`) are valid in some file types (JSON, text) but are completely invalid Python syntax. The Python interpreter encounters `<<<<<<<` on line 147 and throws `SyntaxError: invalid syntax` before executing a single line.

**Why multiple files were affected:** The merge conflict occurred on a commit that touched multiple files simultaneously (`main_system.py`, `decision_engine.py`, `face_crop_utils.py`, `screen_protector.py`). Git inserted conflict markers in every file that had diverging changes between local and remote.

**Debug process — systematic scan:**
```bash
# Search all Python files for conflict markers
grep -rn "<<<<<<" *.py
grep -rn ">>>>>>>" *.py
grep -rn "=======" *.py
```
This revealed which files were affected and on which lines.

**Resolution strategy:** For each conflicted section, identify which version is correct using project documentation and intended behavior:
1. If both versions are functionally equivalent: keep either one
2. If one version is clearly newer/better: keep that version, delete the other
3. If both versions have needed changes: manually merge them, keeping all needed lines
4. Delete all three marker lines (`<<<<<<<`, `=======`, `>>>>>>>`)

**In this project:** `main_system.py` had the most severe conflict — the entire two-thread architecture (HEAD version) vs. the original single-thread version were interleaved. Resolution: keep the HEAD version throughout, delete the old version entirely.

**Behavior after fix:** All files parse correctly. System runs normally.

**Lesson on prevention:**
- Always pull before pushing: `git pull origin main` before `git push`
- Use feature branches: develop on a branch, merge only when complete
- Commit frequently with small changes: smaller commits produce smaller conflicts
- Never edit the same file on two separate machines without syncing first

**General lesson:** Merge conflicts are a normal part of collaborative development (even solo development across multiple machines). The critical skill is systematic conflict identification (grep for markers) and resolution using documented intended behavior as the source of truth.

---

# SECTION 14: CONSOLIDATED LESSONS LEARNED

## 14.1 What This Project Teaches About AI System Debugging

This project encountered bugs across every layer of the stack: dataset statistics, model training configuration, coordinate geometry, variable scoping, library API semantics, performance timing, and version control. The problems did not come from complex algorithms but from integration failures between correct individual components.

The overarching lesson: **an AI system is a pipeline, and each stage's output is the next stage's input.** A bug at stage N appears as a symptom at stage N+3. Understanding which stage produced the incorrect data requires instrumenting each stage independently.

## 14.2 Testing in Realistic Conditions

Problems 8, 10, 11, and 12 were invisible during controlled single-person tests and only appeared during realistic multi-person tests. The fallback crop bug (Problem 8) only manifested when MediaPipe actually failed — which only happened with extreme angles. The FPS bug (Problem 11) only appeared when a second person was actually present. Always test with real-world conditions, not just the happy path.

## 14.3 Check Class Balance Before Training

Problem 1 (class imbalance) is the first thing to check when preparing any classification dataset. The check is trivial:
```python
print(collections.Counter(all_labels))
```
Add this check as the first step of every dataset preparation script. Never proceed to training without confirming reasonable class balance.

## 14.4 Verify Generated Code — Never Trust Blindly

Problem 2 (layer freezing) resulted from accepting generated code without manually verifying the trainable parameter count. For any training configuration, always verify:
```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {100*trainable/total:.1f}%")
```
Run this before starting any training run and compare against expected values.

## 14.5 Document Coordinate Systems Explicitly

Problems 6 and 7 both stemmed from coordinate system confusion. Every pixel coordinate in a CV pipeline must be labelled with its reference frame. A practical convention:

```python
# CLEAR naming convention
face_x1_in_crop = int(bbox.xmin * crop_w)    # in crop space
face_x1_in_frame = x1 + face_x1_in_crop      # in full-frame space
```
Use variable names that encode the coordinate space. Add comments that explicitly state which coordinate space each variable uses.

## 14.6 State Machines Over Threshold Decisions

Problems 12 and the general flickering issue demonstrate why state machines with temporal hysteresis are superior to simple per-frame thresholds. A threshold decision produces binary, instantaneous outputs that oscillate under noise. A state machine with separate entry and exit conditions produces stable, debounced outputs that only change when sustained evidence warrants it. For any real-time AI application that interacts with a user, always use temporal smoothing.

## 14.7 Fail-Safe Defaults

The decision to treat UNKNOWN, UNCERTAIN, and all failed predictions as SAFE (not THREAT) proved critical for usability. Early versions that defaulted ambiguous cases to THREAT produced constant false alarms that made the system unusable within minutes. The principle: in a privacy protection tool, a missed detection is less harmful than a false alarm that causes the user to disable the system entirely.

## 14.8 Modular Architecture Enables Fast Debugging

The clean separation between `main_system.py`, `head_pose_predictor.py`, `decision_engine.py`, and `face_crop_utils.py` allowed each module to be tested and debugged independently. The label inversion bug (Problem 9) was caught by running `head_pose_predictor.py` directly on a test image — without needing to run the entire webcam pipeline. The decision engine's state transitions were verified by running `decision_engine.py` standalone with simulated inputs. Modular design reduces the search space for debugging from "the whole system" to "this specific module."

## 14.9 The Debugging Methodology Used

Every bug in this project was resolved using the same systematic process:

1. **Observe symptom precisely:** What is visually wrong? What does the terminal show? What exactly is incorrect?
2. **Add targeted debug prints:** Insert `print()` statements at the boundary between the suspected stage and its downstream consumer. Print the actual data being passed, not just whether an operation ran.
3. **Identify root cause from data:** The debug prints reveal which stage produces wrong data. Trace backward from there.
4. **Apply a minimal targeted fix:** Change as little as possible. Do not refactor unrelated code during debugging.
5. **Verify with the same debug prints:** After fixing, the debug prints should show correct data. Remove prints only after verification.

This methodology is applicable to any AI system, any programming language, and any scale of project. The investment in adding targeted debug prints always pays back many times over in reduced debugging time.

---

*Continued in RESEARCH_DOCUMENTATION_PART5.md — Results, Evaluation Methodology, Limitations, Future Work, Tools Table, Project Timeline, and Conclusion*
