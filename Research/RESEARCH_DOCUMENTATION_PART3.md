# AI-Based Shoulder Surfing Prevention System
## Research Documentation — PART 3 OF 5
### Sections: Webcam Module · YOLO Detection · Face Extraction · Decision Engine · Screen Protection

---

# SECTION 8: WEBCAM INPUT MODULE

## 8.1 OpenCV VideoCapture

```python
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

`VideoCapture(0)` opens the device at index 0 — the default system camera. On most laptops this is the built-in webcam. If an external USB webcam is connected and preferred, index 1 or 2 is used instead. The index corresponds to the order in which the operating system enumerates camera devices.

Resolution is set to 640×480 because: (a) it is sufficient for YOLO to detect persons at desk range, (b) it matches the resolution that produces 70–80 FPS on CPU, and (c) BIWI images are 640×480, making this a natural fit for the inference pipeline. Higher resolutions (1280×720) increase YOLO processing time significantly without adding useful detail for this task.

`cap.read()` returns a tuple `(ret, frame)`:
- `ret` — boolean: True if the frame was read successfully, False if the camera disconnected or the stream ended
- `frame` — numpy array of shape `(480, 640, 3)`, dtype uint8, color order BGR

**Why `frame.copy()`:** All drawing operations (bounding boxes, text, status bar) happen on `display_frame = frame.copy()`. The original `frame` is passed unchanged to all model inference calls. Drawing on the original would corrupt the pixel values used for YOLO and MediaPipe, producing subtle inference errors that are very hard to debug.

**Single instance rule:** VideoCapture must be created exactly once before the loop and released exactly once in the `finally` block. Creating multiple VideoCapture instances for the same camera causes the second instance to fail silently or produce black frames. Releasing inside the loop would close and reopen the camera every iteration — each reopen takes ~200–500ms and causes visible frame drops.

```python
try:
    while True:
        ret, frame = cap.read()
        # ... processing ...
finally:
    cap.release()          # Always runs, even if loop crashes
    cv2.destroyAllWindows()
```

The `try/finally` pattern guarantees the webcam is released even if an exception occurs anywhere in the loop. Without it, a crash would leave the camera locked — requiring the user to restart their system or kill the process to free the device.

## 8.2 Frame Rate and Performance

FPS is calculated using the time between consecutive frames:

```python
fps = 1.0 / max(time.time() - fps_time, 1e-6)
fps_time = time.time()
```

`max(..., 1e-6)` prevents division by zero if two frames are captured in the same instant.

| Scenario | FPS (before optimization) | FPS (after optimization) |
|---|---|---|
| User alone (YOLO only) | 70–80 | 70–80 (unchanged) |
| Observer present | 30–40 | 55–65 |

FPS matters because the threat detection pipeline relies on frame counts. The streak counter and threat threshold are expressed in frames, not seconds. At 30 FPS, a streak of 3 frames represents 100ms. At 60 FPS, the same streak represents 50ms. The thresholds were tuned assuming approximately 60 FPS operation — running significantly slower would require retuning all frame-count thresholds.

---

# SECTION 9: PERSON DETECTION MODULE (YOLOv8n)

## 9.1 YOLO Detection Call

```python
results = model(frame, classes=[0], verbose=False)
for box in results[0].boxes:
    conf = float(box.conf[0])
    if conf < 0.5:
        continue
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    raw_boxes.append([x1, y1, x2, y2, conf])
```

**`classes=[0]`:** Restricts YOLO to detecting only class 0 (person) from the COCO dataset. This eliminates all other detections (chairs, laptops, bags) and reduces unnecessary post-processing. Without this filter, YOLO returns detections for all 80 COCO classes, most of which are irrelevant and add processing overhead.

**`verbose=False`:** YOLO prints inference statistics to the terminal by default (processing time, number of detections). At 60 FPS this creates thousands of terminal lines per minute, slowing down the terminal and obscuring useful log messages from the system.

**Confidence threshold 0.5:** A detection with confidence below 50% is more likely a false positive (YOLO seeing a person-like shape in background texture) than a real person. 0.5 is the standard threshold for YOLO deployments. Lowering to 0.3 increases false positives; raising to 0.7 causes real persons at the edges of the frame to be missed.

**Output format:** `box.xyxy[0]` gives absolute pixel coordinates `[x1, y1, x2, y2]` where (x1,y1) is the top-left corner and (x2,y2) is the bottom-right corner. `box.conf[0]` is the detection confidence as a float 0–1.

## 9.2 Zone Classification Algorithm

```python
if len(raw_boxes) == 0:
    user_box, observer_boxes = None, []
elif len(raw_boxes) == 1:
    user_box, observer_boxes = raw_boxes[0], []
else:
    raw_boxes.sort(key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
    user_box       = raw_boxes[0]   # Largest area = user
    observer_boxes = raw_boxes[1:]  # All others = observers
```

**Why largest bounding box = user:** In a standard laptop usage setup, the user sits closest to the webcam (which is on the laptop screen). Physical proximity to the camera directly corresponds to larger apparent size in the image — a fundamental property of perspective projection. An observer standing or sitting further away appears smaller. This heuristic requires no calibration, no registration, and no prior knowledge of who the user is.

**Edge case — 0 persons:** No processing needed. The decision engine receives `num_persons=0` and returns SAFE immediately.

**Edge case — 1 person:** Must be the user. No observer exists. Skip all face extraction and head pose steps.

**Known limitation:** If the user leans back (increasing their distance from the camera), their bounding box shrinks. If an observer simultaneously leans forward, the observer's box may temporarily become larger, causing them to be misclassified as the user. This is an edge case that rarely occurs in practice but represents a fundamental limitation of the distance-proxy heuristic.

## 9.3 YOLO Frame Skipping

```python
YOLO_INTERVAL   = 2
last_yolo_boxes = []

if frame_count % YOLO_INTERVAL == 1:
    # Run YOLO and update cache
    results = model(frame, classes=[0], verbose=False)
    # ... parse boxes ...
    last_yolo_boxes = raw_boxes
else:
    raw_boxes = last_yolo_boxes  # Reuse previous frame's result
```

YOLO is the most expensive single operation in the pipeline at ~15–25ms per call on CPU. Running it every 2nd frame halves this cost from 15ms/frame to 7.5ms/frame average. At 60 FPS, persons move approximately 3–8 pixels between consecutive frames — the cached bounding box from the previous frame remains accurate enough for zone classification and cropping.

**Why interval=2 not 3 or 4:** At interval=3, persons moving quickly (walking across the room) can move 10–15 pixels before the next YOLO update, causing the crop region to miss the face slightly. At interval=2, the maximum positional error is 5–8 pixels — within the tolerance of the 40% face padding. Higher intervals also delay detection of new persons entering the frame.

---

# SECTION 10: FACE EXTRACTION MODULE

## 10.1 Why Face Extraction Is Necessary

YOLO provides a bounding box around the entire person — from head to feet (or from head to waist if the lower body is off-screen). The head pose model expects a face crop, not a full-body crop. Feeding a full-body image to MobileNetV2 would:
1. Waste most of the 224×224 pixel budget on non-face content (clothing, background)
2. Make the face region extremely small within the input image (~10–20% of width)
3. Produce consistently poor and unreliable head pose predictions

A dedicated face extraction step focuses the model on the relevant region.

## 10.2 MediaPipe FaceDetection

```python
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,           # Short-range model: optimized for 0-2 metres
    min_detection_confidence=0.5
)
```

MediaPipe is Google's open-source framework for real-time perception tasks. Its FaceDetection module uses a lightweight BlazeFace model that can detect faces in approximately 2–5ms on CPU — fast enough to run every pose frame without becoming a bottleneck.

**model_selection=0 (short-range) vs model_selection=1 (full-range):** The short-range model is optimised for faces at 0–2 metres from the camera. At desk/shoulder-surfing distances, this is the correct choice. The full-range model (model_selection=1) is designed for faces at 2–5 metres and has lower accuracy at close range.

**Single initialization:** The FaceDetection object is created once before the main loop and closed in the `finally` block. Creating it per-frame would add ~10–50ms of initialization overhead per call — completely eliminating any performance gain from face-specific processing.

**Input must be RGB:** MediaPipe expects RGB channel order. OpenCV provides BGR. Conversion is mandatory:
```python
crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
result = face_detector.process(crop_rgb)
```

**Output — relative coordinates:** MediaPipe returns face locations as relative values in [0.0, 1.0] relative to the input image dimensions. `xmin=0.1` means the face starts at 10% of the image width from the left.

## 10.3 The Coordinate Conversion Bug (and Fix)

The most significant implementation bug in this project was treating MediaPipe's relative coordinates as absolute coordinates in the full frame.

**The bug:**
```python
# WRONG — treats relative coords as absolute full-frame coords
bbox = result.detections[0].location_data.relative_bounding_box
x1 = int(bbox.xmin)          # Returns ~0.12 → int() → 0 pixels
y1 = int(bbox.ymin)          # Returns ~0.05 → int() → 0 pixels
```

This produced face crops starting at pixel (0,0) in the full frame — the top-left corner — regardless of where the person was. The crops contained background, not faces.

**The fix:**
```python
# CORRECT — convert relative → absolute WITHIN the crop
crop_h, crop_w = person_crop.shape[:2]
bbox = result.detections[0].location_data.relative_bounding_box

raw_fx1 = int(bbox.xmin * crop_w)
raw_fy1 = int(bbox.ymin * crop_h)
raw_fx2 = int((bbox.xmin + bbox.width) * crop_w)
raw_fy2 = int((bbox.ymin + bbox.height) * crop_h)
```

The key insight: MediaPipe ran on the person crop, so its relative coordinates are relative to the crop, not the full frame. Multiplying by `crop_w` and `crop_h` converts them to absolute pixel positions within the crop. The crop is then indexed at these positions to extract the face region.

## 10.4 40% Padding

After converting coordinates, padding is added:

```python
face_w = raw_fx2 - raw_fx1
face_h = raw_fy2 - raw_fy1
pad_x  = int(face_w * 0.4)
pad_y  = int(face_h * 0.4)

fx1 = max(0,      raw_fx1 - pad_x)
fy1 = max(0,      raw_fy1 - pad_y)
fx2 = min(crop_w, raw_fx2 + pad_x)
fy2 = min(crop_h, raw_fy2 + pad_y)

face_crop = person_crop[fy1:fy2, fx1:fx2]
```

**Why 40% padding:** MediaPipe's face bounding box tightly encloses the face oval (forehead to chin, ear to ear). This tight crop misses information critical for head pose estimation: the outline of the head, the ears (which reveal lateral orientation), the top of the head (which reveals vertical orientation), and the neck (which gives context for head-body alignment). Expanding by 40% on each side captures this contextual information while keeping the face as the central focus of the crop.

**Clipping with `max(0, ...)` and `min(crop_w, ...)`:** The padding calculation may produce coordinates outside the crop boundaries (negative values or values exceeding crop dimensions). Clipping prevents invalid numpy array slicing that would produce empty arrays.

## 10.5 Minimum Size Gate: 40×40 Pixels

```python
if face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
    return None, "face_too_small"
```

The head pose model (MobileNetV2) expects a 224×224 input. The preprocessing pipeline resizes all inputs to this size using `transforms.Resize((224, 224))`. If the face crop is 20×20 pixels, this resize applies an 11× magnification factor. At 11× upscaling, individual pixel blocks become 11×11 areas — the result is a heavily pixelated, blocky image that looks nothing like the training data. The model produces near-random predictions on such inputs.

**Why 40×40 specifically:** Empirical testing showed that crops below 40×40 produce confident but wrong predictions (the model is certain, but wrong). At 40×40 and above, the model's uncertainty correctly increases for ambiguous inputs. The 40×40 threshold was chosen as the point below which the quality of predictions is consistently unreliable.

## 10.6 Fallback Mechanism

When MediaPipe fails to detect a face (poor lighting, extreme angle, occlusion, person too small):

```python
# Fallback: use top 28% of person crop with 15% horizontal margin
crop_h, crop_w = person_crop.shape[:2]
margin_x = int(crop_w * 0.15)
face_crop = person_crop[
    0 : int(crop_h * 0.28),
    margin_x : crop_w - margin_x
]
```

**Why 28% height:** In a full-body or upper-body person crop, the face typically occupies the top 20–30% of the bounding box height. Using 28% centers this range. Using 35% would include the neck and upper chest — non-face content that reduces prediction quality. Using 20% might clip the chin and forehead.

**Why 15% horizontal margin:** Removes the lateral edges of the person bounding box (shoulder areas, background beside the head) while keeping the central face region.

**Return tag "fallback_estimate":** This tag distinguishes fallback crops from confirmed MediaPipe detections. In the UI, fallback results are annotated with "(est.)" to indicate lower confidence. In the decision engine, fallback results are treated identically to confirmed results — the gap filter handles quality control.

## 10.7 All Return Codes

| Reason String | Meaning |
|---|---|
| `mediapipe_confirmed` | Face detected by MediaPipe, padded, passed size gate |
| `fallback_estimate` | MediaPipe found no face, used top-28% position estimate |
| `box_too_small` | Person bounding box area < 100 pixels — too small to crop |
| `empty_crop` | Crop region was an empty array (coordinates outside frame) |
| `mediapipe_error` | MediaPipe threw an exception (caught by try/except) |
| `face_too_small` | MediaPipe found face but result < 40×40 after padding |
| `fallback_too_small` | Fallback region < 40×40 pixels (very small person box) |

All cases except `mediapipe_confirmed` and `fallback_estimate` return `(None, reason_string)`. The calling code in `main_system.py` checks for None and returns an UNKNOWN result dict, which the decision engine ignores.

---

# SECTION 11: THREAT DECISION ENGINE

## 11.1 Why a State Machine, Not a Simple Threshold

Without temporal smoothing, the system would declare THREAT on any single frame where an observer's prediction is LOOKING. At 60 FPS, even a 92% accurate model produces approximately 5 wrong predictions per second. The result would be constant flickering between SAFE and THREAT — the status bar would flash red dozens of times per minute even with no observer present, making the system completely unusable.

A state machine requires sustained evidence before transitioning states, and sustained counter-evidence before transitioning back. This converts noisy per-frame signals into stable, reliable state changes.

## 11.2 State Transitions

```
           observer detected
SAFE ─────────────────────────► MONITORING
         (num_persons > 1)
                │
                │ per-observer streak >= 3
                ▼
        THREAT ACCUMULATING
                │
                │ threat_frame_count >= 4
                ▼
             THREAT
                │
                │ safe_frame_count >= 20
                ▼
              SAFE
```

In the actual implementation, the state machine is simplified to two public states (SAFE, THREAT) with internal counters tracking progress:

- `threat_frame_count`: increments every frame any observer's streak ≥ 3; resets when no observer is threatening
- `safe_frame_count`: increments every frame no observer is threatening; resets when a threat is accumulating

## 11.3 Per-Observer Streak Tracking

```python
self.observer_looking_streak = {}  # {observer_index: consecutive_frame_count}

for i, result in enumerate(observer_results):
    if (result["label"] not in ("UNKNOWN", "UNCERTAIN")
            and result["is_looking"]
            and result["confidence"] > self.confidence_threshold):
        self.observer_looking_streak[i] = self.observer_looking_streak.get(i, 0) + 1
    else:
        self.observer_looking_streak[i] = 0  # Reset on any non-LOOKING frame

    if self.observer_looking_streak[i] >= 3:
        this_frame = "THREAT"
        break
```

**Why per-observer, not global:** Consider two observers: A glances at the screen once (streak 1), B glances at the screen twice (streak 2). A global counter might combine these to reach 3 and declare THREAT — but neither observer has actually sustained a gaze. Per-observer tracking requires one person to stare continuously for 3 consecutive pose-frames.

**Streak reset on non-LOOKING:** Any frame where an observer is NOT_LOOKING, UNCERTAIN, or UNKNOWN resets their streak to 0. This ensures the streak represents consecutive frames, not cumulative frames.

**Clear streaks when observer count changes:**
```python
if current_observer_count != self.prev_observer_count:
    self.observer_looking_streak = {}
```
When an observer leaves the frame (count decreases), all streak data is cleared. When a new observer enters (count increases), all data is cleared. This prevents the streak of observer A from being incorrectly attributed to observer B who took A's position index.

## 11.4 Two-Layer Counting Rationale

The system uses two separate counting mechanisms:
1. **Per-observer streak** (≥ 3 consecutive pose-frames): confirms the observer is actively looking
2. **Global threat threshold** (≥ 4 consecutive THREAT frames): confirms the threat is sustained, not a one-frame artifact

Combined with `pose_interval=3`, the minimum real frames before THREAT is declared:
```
streak of 3 pose-frames × pose_interval(3) = 9 real frames to confirm observer
+ threat_threshold(4) real frames = 13 real frames minimum
At 60 FPS = approximately 200ms total latency
```

This is significantly faster than the original configuration (streak=5, threshold=10) which required approximately 400ms.

## 11.5 Cooldown Mechanism

```python
if self.safe_frame_count >= self.cooldown_frames:  # 20 frames
    self.current_state = "SAFE"
```

The cooldown prevents the system from rapidly oscillating between SAFE and THREAT when an observer is intermittently looking — for example, someone who glances at the screen, looks away for a moment, then looks back. Without cooldown, the system would activate and deactivate the alert every few frames, creating a distracting flashing effect.

**Why 20 frames:** At 60 FPS, 20 frames = 330ms. This gives the observer approximately one-third of a second to look away consistently before the alert clears. Long enough to prevent oscillation; short enough to respond quickly when a genuine threat resolves.

---

# SECTION 12: SCREEN PROTECTION MODULE

> **Note:** The screen protection module (screen_protector.py) was removed from the current version of the system, which focuses on detection and alerting only. This section documents the original design for completeness.

## 12.1 The Transparency Challenge

A standard operating system window captures all mouse events within its bounds. If a full-screen opaque overlay is placed over the screen, the user cannot click anything beneath it — their workflow is completely blocked. This is unacceptable for a privacy protection tool that the user runs continuously.

The solution is a **click-through overlay** — a window that is visible (shows the blur effect) but is transparent to mouse events (clicks pass through to whatever is beneath). This requires platform-specific system calls:

**Windows (using ctypes/Win32 API):**
```python
import ctypes
hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE
ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x20)  # WS_EX_TRANSPARENT
```

**macOS (using tkinter attributes):**
```python
root.wm_attributes("-transparent", True)
root.wm_attributes("-topmost", True)
```

## 12.2 The Blur Effect

When a THREAT is detected:
1. A screenshot of the current screen is captured using `mss` (fast screen capture library, faster than PIL's ImageGrab)
2. The screenshot is converted to a PIL Image and Gaussian blur is applied:
   ```python
   blurred = screenshot.filter(ImageFilter.GaussianBlur(radius=25))
   ```
3. The blurred image is displayed as the background of the tkinter canvas

**Blur radius 25:** At radius=25, text on screen becomes illegible from a distance (as seen by an observer) but the overall layout of windows remains dimly visible to the user. The user can sense that they are still working without being able to read content. Lower values (radius=10) leave text too readable; higher values (radius=40) create a completely opaque grey rectangle that provides no contextual information.

## 12.3 Cursor Spotlight

```python
# Every 50ms
cursor_x, cursor_y = get_cursor_position()
canvas.create_oval(
    cursor_x - 80, cursor_y - 80,
    cursor_x + 80, cursor_y + 80,
    fill="",       # Transparent interior — shows original unblurred content
    outline=""     # No border
)
```

A circular clear region of radius 80 pixels is drawn at the cursor's current position, revealing the unblurred screen beneath. The user can read content immediately around the cursor while everything else remains blurred.

**Spotlight radius 80px:** Large enough to comfortably read a line of text or click a UI element; small enough that an observer at 1+ metres cannot read the content (the angular resolution at that distance makes 160px too small to read).

**Update frequency 50ms (20 Hz):** The spotlight position is refreshed 20 times per second. This is fast enough for smooth cursor tracking during normal work. Running faster would increase CPU usage on the main thread without perceptible improvement.

## 12.4 Thread Architecture (Original Design)

The original design used two threads because tkinter has a hard requirement: all tkinter operations must run on the same thread that created the tkinter window — the main thread. The webcam loop is a blocking `while True` — if it runs on the main thread, tkinter freezes and the overlay never updates.

```
MAIN THREAD:
  - Creates ScreenProtector (tkinter window)
  - Calls protector.update() every 10ms to drive event loop
  - Reads shared _protection_active flag
  - Calls activate/deactivate based on flag changes

BACKGROUND THREAD (daemon=True):
  - Runs entire webcam + YOLO + face crop + head pose + engine pipeline
  - Sets _protection_active = True/False based on state
  - Sends 'q' keypress signal via _should_quit flag

SHARED STATE:
  _protection_active: bool (read by main, written by background)
  _should_quit: bool (written by background on 'q' press)
  _state_lock: threading.Lock() (prevents race conditions)
```

`daemon=True` on the background thread means it is automatically killed when the main thread exits — preventing orphaned background processes from keeping the webcam locked after the application closes.

After removal of the screen protection module, this two-thread architecture was replaced with a simple single-thread `while True` loop, eliminating all threading complexity.

---

*Continued in RESEARCH_DOCUMENTATION_PART4.md — All 11 Problems Faced with Complete Analysis · Evaluation Methodology · Results · Limitations · Future Work · Tools Table · Project Timeline · Conclusion*
