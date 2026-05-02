# AI-Based Shoulder Surfing Prevention System
## Research Documentation — PART 2 OF 5
### Sections: Dataset Preparation · Model Training · Inference Pipeline

---

# SECTION 5: DATASET — BIWI KINECT HEAD POSE DATABASE

## 5.1 Dataset Description

The BIWI Kinect Head Pose Database was produced by Fanelli et al. at ETH Zurich. It is one of the most widely cited academic datasets for head pose estimation.

| Property | Value |
|---|---|
| Source | ETH Zurich (free for academic use) |
| Collection device | Microsoft Kinect depth camera |
| Subjects | 24 (20 adults, 4 children) |
| Total frames | 15,678 |
| Image format | RGB PNG, 640×480 |
| Annotation format | 3×3 rotation matrix + translation vector per frame |

Each subject folder contains pairs of files:
- `frame_XXXXX_rgb.png` — the RGB face image
- `frame_XXXXX_pose.txt` — the pose annotation (rotation matrix, 3 rows × 3 columns + translation)

The dataset was collected in controlled indoor sessions where subjects were asked to move their heads naturally while the Kinect camera recorded their precise 3D pose. Because subjects were recorded facing the camera (the typical setup for head pose research), the majority of frames contain faces oriented approximately toward the camera — a fact that created the class imbalance problem described in Section 5.3.

## 5.2 Extracting the Yaw Angle from the Rotation Matrix

### What a Rotation Matrix Is

A rotation matrix is a 3×3 grid of numbers that encodes a 3D rotation. Think of it as a compact way to describe how much something has turned in three-dimensional space. Every possible orientation of a rigid object (like a head) in 3D space can be described by one unique rotation matrix.

The BIWI pose files contain this matrix for each frame. Loading it looks like:

```python
import numpy as np

def load_pose(pose_file):
    with open(pose_file, 'r') as f:
        lines = f.readlines()
    # First 3 lines: rotation matrix rows
    R = np.array([
        [float(x) for x in lines[0].split()],
        [float(x) for x in lines[1].split()],
        [float(x) for x in lines[2].split()]
    ])
    return R
```

### Extracting Yaw

A head's orientation is described by three angles:
- **Yaw** — left/right turn (rotating around vertical axis)
- **Pitch** — up/down tilt (rotating around horizontal axis)
- **Roll** — shoulder tilt (rotating around depth axis)

For shoulder surfing detection, only yaw matters. An observer looking at the screen will have yaw ≈ 0° (facing directly toward the camera/screen). An observer looking away will have large yaw values (turned left or right).

Yaw is extracted from the rotation matrix using:

```python
yaw_radians = np.arctan2(R[1, 0], R[0, 0])
yaw_degrees = np.degrees(yaw_radians)
```

**What `arctan2(R[1,0], R[0,0])` does:** The first column of a rotation matrix `[R[0,0], R[1,0], R[2,0]]` describes where the object's X-axis is pointing after rotation. `arctan2(y, x)` computes the angle of a 2D vector from its x and y components. Here it gives the angle the X-axis has rotated around the Z-axis — which is the yaw angle.

**Physical interpretation:**
- yaw = 0° → head faces directly forward (toward screen)
- yaw = +30° → head turned 30° to the right
- yaw = −30° → head turned 30° to the left
- yaw = ±90° → head in full profile (definitely not looking at screen)

Pitch and roll are ignored because they do not change whether an observer is looking at the screen. A person can look at a screen with their head tilted or slightly up/down.

## 5.3 Label Conversion — First Attempt (FAILED: ±30° threshold)

The first approach applied this rule:
```python
if abs(yaw_degrees) <= 30:
    label = "LOOKING"
else:
    label = "NOT_LOOKING"
```

**Result:**
| Class | Count | Percentage |
|---|---|---|
| LOOKING | 15,119 | **96.4%** |
| NOT_LOOKING | 559 | **3.6%** |

This distribution is catastrophic for training. Here is why:

### The "Lazy Model" Problem

A neural network optimises for whatever loss function it is given. CrossEntropyLoss measures how wrong the model's predictions are averaged across all training samples. If 96.4% of samples are LOOKING, the model can achieve 96.4% accuracy by simply predicting LOOKING for every single input regardless of what it sees. The loss would be low, the accuracy would be high, and the model would have learned absolutely nothing about head orientation.

This is called the **trivial classifier** or **majority class bias** problem. The model is not "lazy" — it is correctly optimising its objective. The problem is that accuracy is a misleading metric when classes are severely imbalanced.

To understand why this is useless: the actual system needs to detect NOT_LOOKING reliably. A model that predicts LOOKING always would declare THREAT for every observer regardless of their head direction — the exact opposite of the desired behaviour.

The ±30° threshold was too wide. Because BIWI subjects mostly face the camera, nearly all yaw angles fell within ±30°, creating a nearly single-class dataset.

## 5.4 Label Conversion — Second Attempt (SUCCESS: ±20°)

Tightening the threshold to ±20°:

```python
if abs(yaw_degrees) <= 20:
    label = "LOOKING"
else:
    label = "NOT_LOOKING"
```

**Result:**
| Class | Count | Percentage |
|---|---|---|
| LOOKING | 14,230 | **90.8%** |
| NOT_LOOKING | 1,448 | **9.2%** |

**Why ±20° is more appropriate:** At 20° of yaw, the head is turning noticeably away from center. A person at ±20° to ±45° is clearly oriented away from the screen — not in a position to comfortably read it. The ±30° zone was too generous, including faces that are clearly not looking at the screen within the LOOKING class.

The distribution is still imbalanced (90.8% vs 9.2%) but represents a real improvement. The NOT_LOOKING class grew from 559 to 1,448 examples — enough real image diversity to work with after oversampling.

## 5.5 Solving Class Imbalance: Random Oversampling

### What Oversampling Is

Random oversampling duplicates samples from the minority class (NOT_LOOKING) until both classes have equal counts. Unlike undersampling (which discards majority class samples and loses information), oversampling retains all original data.

```python
from sklearn.utils import resample

not_looking_oversampled = resample(
    not_looking_samples,
    replace=True,            # sample with replacement (allows duplicates)
    n_samples=14230,         # match the majority class count
    random_state=42
)
```

**What happens:**
- 1,448 real NOT_LOOKING images are duplicated to reach 14,230
- Each image is used approximately 9.8 times on average
- Combined with 14,230 LOOKING images → 28,460 total, perfectly balanced

### Why Oversampling Is Valid

Oversampling is a standard, widely-accepted technique in machine learning for imbalanced datasets, used in medical diagnosis, fraud detection, and other domains where minority class examples are rare but critical. The fundamental validity of the approach is that the model is being shown that both classes matter equally during training — the loss contribution from NOT_LOOKING frames is no longer swamped by the majority class.

The concern with oversampling is that the model may memorise the 1,448 repeated images rather than learning generalizable features. This concern is directly addressed by data augmentation (Section 6.2), which applies random transformations to each image every time it is used, making each repetition visually distinct.

## 5.6 Final Dataset Statistics and Split

| Stage | LOOKING | NOT_LOOKING | Total |
|---|---|---|---|
| Raw BIWI (±30°) | 15,119 (96.4%) | 559 (3.6%) | 15,678 |
| After ±20° threshold | 14,230 (90.8%) | 1,448 (9.2%) | 15,678 |
| After oversampling | 14,230 (50%) | 14,230 (50%) | **28,460** |
| Training (80%) | — | — | **22,768** |
| Validation (20%) | — | — | **5,692** |

**Why 80/20 split:** The 80/20 train/validation split is the standard in machine learning. With 28,460 samples, 5,692 validation samples is sufficient for statistically meaningful accuracy measurement. PyTorch's `random_split` is used, which assigns samples randomly to each split.

**Critical: The validation set was NOT oversampled.** Oversampling was applied only to the training data. The validation set retains the natural 90.8%/9.2% distribution. This is essential: validation accuracy must reflect real-world conditions. If the validation set were balanced artificially, the reported accuracy would not represent how the model performs on real-world inputs.

---

# SECTION 6: MODEL TRAINING

## 6.1 Architecture Decisions

### Why MobileNetV2

MobileNetV2 was chosen because it is the lightest ImageNet-pretrained backbone that provides sufficient feature quality for fine-tuning at CPU-compatible inference speed. Full comparison:

| Model | Params | CPU Inference | ImageNet Acc | Choice |
|---|---|---|---|---|
| MobileNetV2 | 3.4M | 15–30ms | 71.8% | ✅ Selected |
| ResNet-50 | 25.6M | 80–120ms | 76.1% | ❌ Too slow |
| EfficientNet-B0 | 5.3M | 25–45ms | 77.1% | ⚠️ Acceptable |
| VGG-16 | 138M | 400–600ms | 71.6% | ❌ Unusable |
| Custom CNN | — | — | — | ❌ Needs 10× more data |

Training from scratch was rejected because the dataset has only 1,448 unique NOT_LOOKING images. Deep CNNs trained from scratch on small datasets overfit severely — they memorise training examples rather than learning generalisable features.

### Layer Freezing — First Attempt (WRONG: 60.1% trainable)

The initial code incorrectly left a large portion of the MobileNetV2 backbone unfrozen:

```python
# WRONG — only froze early layers
for i, (name, param) in enumerate(backbone.named_parameters()):
    if i < 40:   # Only froze first 40 parameter tensors
        param.requires_grad = False
```

This left 60.1% of parameters trainable. With a heavily oversampled dataset (1,448 unique images repeated ~10×), having 60% of parameters free to update creates severe overfitting risk — the model has enough capacity to memorise every training image and will not generalise to new faces.

### Layer Freezing — Correct Version (12.9% trainable)

```python
# CORRECT — freeze entire backbone
for param in backbone.parameters():
    param.requires_grad = False
# Only the custom head (added separately) is trainable
```

With only 12.9% (328,450 parameters) trainable, the model is forced to:
1. Use the rich ImageNet visual features already learned in the frozen backbone
2. Only learn how to map those features to LOOKING/NOT_LOOKING labels
3. Not overfit to the repeated training images

**What the frozen backbone already knows:** edges, textures, shapes, face-like structures, hair, skin tones — all learned from 1.28 million ImageNet images. This knowledge transfers directly to classifying head orientation.

### Custom Head Design

```
MobileNetV2 features (1280-dim)
         ↓
Linear(1280 → 256)   — compress feature vector
         ↓
ReLU                  — non-linearity: allows complex mappings
         ↓
Dropout(0.3)          — randomly zero 30% of neurons during training
         ↓
Linear(256 → 2)       — 2 output scores: [NOT_LOOKING, LOOKING]
```

**Why Linear(1280→256) not Linear(1280→2) directly:** An intermediate layer allows the model to learn a non-linear transformation of the features before classification. Without it, the classifier is purely linear — limited in what it can distinguish. The 256-dimensional intermediate representation allows the model to reorganise the 1280 features into a more discriminative space.

**Why Dropout(0.3):** Dropout randomly disables 30% of neurons during each training step. This forces the network not to rely on any single neuron and encourages redundant representations — a proven regularisation technique. It is especially important here because the training data has many repeated images.

**Why 2 outputs instead of 1:** For a binary problem, two output nodes with softmax (one per class) allows the decision engine to compare both class probabilities and compute the confidence gap. A single sigmoid output only gives one probability and does not allow the gap-based uncertainty filtering implemented in the inference pipeline.

## 6.2 Data Augmentation

Augmentation is applied only during training, never during validation.

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),           # Required by MobileNetV2
    transforms.RandomHorizontalFlip(),       # Mirror the face
    transforms.ColorJitter(brightness=0.2),  # ±20% brightness variation
    transforms.RandomRotation(10),           # ±10° random tilt
    transforms.ToTensor(),                   # [0,255] uint8 → [0,1] float32
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],          # ImageNet channel means
        std= [0.229, 0.224, 0.225]           # ImageNet channel stds
    )
])
```

**Resize(224, 224):** MobileNetV2 was designed for 224×224 input. Using a different size changes the spatial resolution of intermediate feature maps and degrades accuracy.

**RandomHorizontalFlip:** Mirrors the image left-right with 50% probability. A face looking left becomes a face looking right. This is valid for head pose because left-looking and right-looking have symmetric semantics relative to the ±20° threshold.

**ColorJitter(brightness=0.2):** Simulates different lighting conditions — a bright office versus a dim cafe. Prevents the model from relying on absolute pixel brightness.

**RandomRotation(10):** Simulates slight camera tilt or head roll variation (±10°). Prevents the model from relying on precise vertical alignment of the face.

**Normalize:** This is mandatory. MobileNetV2 was pretrained on data normalized with ImageNet statistics. If inference data is not normalized identically, the model receives pixel value ranges it has never seen, producing nonsensical outputs. Every future inference call must apply identical normalization.

**Why augmentation is critical for oversampled data:** Without augmentation, the 1,448 NOT_LOOKING images are shown to the model 9.8 times each — identically. The model memorises their specific pixel patterns rather than learning head orientation. With augmentation, each repetition is randomly transformed, making the 1,448 images effectively appear as ~14,000 distinct images. This is the mechanism that makes oversampling work in practice.

## 6.3 Training Configuration

| Hyperparameter | Value | Justification |
|---|---|---|
| Loss function | CrossEntropyLoss | Standard for multi-class classification with softmax output |
| Optimizer | Adam | Adaptive learning rate, faster convergence than SGD for fine-tuning |
| Learning rate | 0.0001 | Low LR prevents overwriting pretrained weights in the head |
| LR Scheduler | StepLR(step=5, gamma=0.5) | Halves LR every 5 epochs for stable late-stage convergence |
| Epochs | 20 | Sufficient for fine-tuning a pretrained model |
| Batch size | 32 | Standard; fits in Colab GPU memory with room to spare |
| Checkpoint | Save on val_acc improvement | Best model kept, not the final epoch model |

**Why Adam over SGD:** Adam maintains per-parameter adaptive learning rates. For fine-tuning (where different parameters may need different update scales), Adam converges faster and more stably than vanilla SGD with a fixed learning rate.

**Why learning rate 0.0001:** The backbone is frozen, so only the custom head is updated. The head starts with random weights and needs to find the correct values without overshooting. 0.001 (a common default) can cause training instability early on. 0.0001 gives smooth, stable convergence.

**Why StepLR:** In early epochs, the head weights are far from optimal — large updates are needed. In later epochs, the weights are near-optimal — large updates overshoot and cause oscillation. Halving the learning rate every 5 epochs transitions from coarse to fine optimisation automatically.

**Why save best checkpoint, not last:** The last epoch is not necessarily the best. With 20 epochs and a decaying learning rate, validation accuracy may reach its peak at epoch 14 or 16 and then plateau or slightly degrade. Saving only when validation accuracy improves guarantees the saved model is the best one seen during training.

## 6.4 Training Results

| Metric | Value |
|---|---|
| Final validation accuracy | **91.99%** |
| Training duration | ~60–90 minutes (Tesla T4) |
| Output file | `best_head_pose_model.pth` |
| Convergence behaviour | Steady increase in val_acc through epochs 1–12, plateau at 13–20 |

**Why 91.99% is a strong result:** Binary classification of LOOKING vs NOT_LOOKING on BIWI is a well-studied task. Published results on similar configurations range from 88% to 95%. Achieving 91.99% with a frozen backbone, 1,448 unique minority-class examples, and approximately 90 minutes of training demonstrates that the approach — transfer learning + oversampling + augmentation — is highly effective even with constrained resources.

The model is not perfect (8.01% error rate), and the remaining errors are analysed in Part 3 (Problems Faced). Most errors occur on faces at intermediate yaw angles (15°–25°) where the visual difference between LOOKING and NOT_LOOKING is subtle and the decision boundary is inherently ambiguous.

## 6.5 Google Colab Setup

Google Colab provides free access to NVIDIA GPU instances (Tesla T4 for standard tier) via a browser-based Jupyter notebook interface. It was used for training because:

**Training speed comparison:**
| Hardware | Estimated training time (20 epochs, 28,460 samples) |
|---|---|
| Laptop CPU (Intel i5) | ~12–18 hours |
| Google Colab T4 GPU | ~60–90 minutes |
| Local NVIDIA RTX 3060 | ~20–30 minutes |

Training on laptop CPU is technically possible but impractical for a one-week project timeline. Colab provides T4 access at zero cost with no setup beyond a Google account.

**Setup steps in Colab:**
1. Mount Google Drive to access the BIWI dataset and save the model
2. Install required packages (`pip install torch torchvision`)
3. Run the training script (`biwi_model.py`)
4. Download `best_head_pose_model.pth` from Drive to the local project folder

The trained `.pth` file is the only artifact that needs to be transferred from Colab to the deployment machine. All inference code runs entirely on CPU without any GPU or Colab dependency.

---

# SECTION 7: MODEL INFERENCE — HeadPosePredictor

## 7.1 Loading the Model at Runtime

```python
checkpoint = torch.load("best_head_pose_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

**Why `map_location=device`:** The model was saved on a GPU (Colab T4). When loading on a CPU-only machine, `map_location` tells PyTorch to remap all tensor locations to CPU. Without this, PyTorch attempts to load GPU tensors and crashes if no GPU is present.

**Why `model.eval()`:** This switches the model from training mode to evaluation mode, which has two critical effects:
- **Dropout is disabled:** During training, Dropout randomly zeros neurons. During inference, all neurons must be active for deterministic, consistent predictions. Forgetting `model.eval()` means predictions randomly change between identical inputs.
- **BatchNorm uses stored statistics:** If BatchNorm layers exist, they use running mean/variance computed during training rather than batch statistics. (MobileNetV2 uses BatchNorm in its backbone, though the backbone is frozen.)

**Why the architecture must match exactly:** PyTorch saves only the weight tensors, not the architecture. When loading, the architecture is rebuilt in code and then populated with saved weights. If the architecture definition differs by even one layer, the key names in the state dictionary won't match and `load_state_dict` will raise an error. This is why `HeadPoseClassifier` in `head_pose_predictor.py` must be identical to the class used in the training script.

## 7.2 Preprocessing Pipeline

The preprocessing at inference must exactly match the validation transforms used during training:

```python
self.transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    )
])
```

Note: no augmentation at inference (no flips, no brightness changes, no rotation). Augmentation is a training-only tool — at inference we want the model to see the actual image, not a randomly transformed version.

**BGR to RGB conversion:** OpenCV reads images in BGR channel order. PyTorch and PIL expect RGB. This conversion is mandatory:
```python
face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
```
Skipping this causes the model to receive blue channel values where it expects red, and vice versa — producing systematically wrong predictions that appear random.

**What happens if normalization is skipped or uses wrong values:** The normalized pixel range during training was approximately [-2.1, 2.6] (after subtracting ImageNet mean and dividing by std). Without normalization, pixel values are in [0, 1] — a completely different range. The model's learned weights were calibrated to operate on normalized values; receiving un-normalized values is equivalent to feeding it garbage input. Predictions will be either always one class or randomly distributed.

## 7.3 Probability Gap Filtering — The Key Innovation

### Why Argmax Alone Is Insufficient

After the forward pass, `softmax(logits)` produces two probabilities that sum to 1.0 — for example, `[0.42, 0.58]`. The argmax of this is class 1 (LOOKING) with 58% confidence. A naive implementation would declare LOOKING for this prediction and contribute to the threat streak.

But `[0.42, 0.58]` is a model in genuine doubt. The difference between the two probabilities is only 0.16 — the model has barely more evidence for LOOKING than NOT_LOOKING. On an ambiguous side-profile face, this kind of near-50/50 split is common and does not represent meaningful evidence of threat intent.

### The Gap Metric

```python
gap = prob_looking - prob_not_looking
# [0.08, 0.92] → gap = +0.84  (clearly LOOKING)
# [0.42, 0.58] → gap = +0.16  (uncertain)
# [0.55, 0.45] → gap = -0.10  (uncertain, slight lean toward NOT_LOOKING)
# [0.85, 0.15] → gap = -0.70  (clearly NOT_LOOKING)
```

### Threshold Rules

```python
if gap >= 0.25:       # Strong evidence of LOOKING
    label = "LOOKING"
    is_looking = True
    confidence = prob_looking * 100

elif gap <= -0.15:    # Evidence of NOT_LOOKING
    label = "NOT_LOOKING"
    is_looking = False
    confidence = prob_not_looking * 100

else:                 # Insufficient evidence
    label = "UNCERTAIN"
    is_looking = False
    confidence = 0.0
```

### Why Asymmetric Thresholds (0.25 for LOOKING vs 0.15 for NOT_LOOKING)

The two thresholds deliberately make it harder to confirm LOOKING than to confirm NOT_LOOKING. This encodes the system's design philosophy: **a false THREAT alert is worse than a missed detection.**

A false positive (falsely triggering THREAT) disrupts the user, erodes trust in the system, and may cause the user to disable it entirely. A false negative (missing a real observer) means the user's screen is visible for slightly longer — a less severe consequence.

By requiring a gap of 0.25 for LOOKING but only 0.15 for NOT_LOOKING, the system is biased toward safety. When the model is uncertain, it defaults to NOT_LOOKING (treated as safe).

### The UNCERTAIN Class

UNCERTAIN predictions are treated identically to NOT_LOOKING by the decision engine — they do not contribute to the threat streak. The `confidence = 0.0` field signals to the decision engine that this prediction should be ignored (the engine requires `confidence > 75` to count a LOOKING result).

UNCERTAIN fills the gap that previously caused false positives: side-profile faces and intermediate yaw angles that the model is genuinely unsure about are now silently ignored rather than contributing to THREAT counts.

## 7.4 UNKNOWN vs UNCERTAIN

| Return Value | Cause | Meaning |
|---|---|---|
| UNKNOWN | Input is None, or face image < 20×20 px | Face extraction completely failed — no image to classify |
| UNCERTAIN | Image is valid but model is not confident | Model ran but evidence is insufficient to decide |

Both are treated identically by the decision engine: ignored, contributing 0 to the threat streak. The distinction matters only for debugging — UNKNOWN indicates a face extraction failure, UNCERTAIN indicates a model uncertainty issue. When reviewing terminal logs or evaluation CSVs, seeing many UNKNOWN entries suggests a face detection problem; many UNCERTAIN entries suggest the observer's face is consistently at an ambiguous angle.

### Complete Output Dictionary

```python
{
    "label"           : "LOOKING",   # or "NOT_LOOKING" / "UNCERTAIN" / "UNKNOWN"
    "confidence"      : 92.0,        # 0–100, percentage certainty of the winning class
    "is_looking"      : True,        # True only for LOOKING label
    "gap"             : 0.84,        # prob_looking - prob_not_looking (-1.0 to +1.0)
    "prob_looking"    : 92.0,        # probability of LOOKING class × 100
    "prob_not_looking": 8.0,         # probability of NOT_LOOKING class × 100
}
```

The `gap` and `prob_*` fields are included for debugging and threshold tuning. When evaluating the system or adjusting thresholds, these values allow examination of exactly what the model "thinks" about each observer on each frame.

---

*Continued in RESEARCH_DOCUMENTATION_PART3.md — Face Extraction Module, Threat Decision Engine, All 11 Problems Faced with Solutions*
