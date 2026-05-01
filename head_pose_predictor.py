# head_pose_predictor.py
# ============================================================================
# A self-contained module for head pose inference.
# Defines HeadPosePredictor — a class you can import into any other script
# (e.g. zone_detection.py, live_person_detection.py) to classify whether
# a detected face is LOOKING or NOT_LOOKING.
#
# USAGE (from another script):
#   from head_pose_predictor import HeadPosePredictor
#   predictor = HeadPosePredictor("best_head_pose_model.pth")
#   result = predictor.predict(face_crop_bgr)
#   print(result)   # {"label": "LOOKING", "confidence": 94.3, "is_looking": True}
# ============================================================================

import cv2                               # For reading webcam frames (BGR format)
import numpy as np                       # Numerical operations
import torch
import torch.nn as nn
import torch.nn.functional as F          # softmax
from torchvision import models, transforms
from PIL import Image                    # Convert numpy array → PIL for transforms


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
# This must EXACTLY match the architecture used during training in biwi_model.py.
# When you load saved weights (state_dict), PyTorch matches each weight tensor
# to a layer by name. If the architecture differs even slightly, loading fails.

class HeadPoseClassifier(nn.Module):
    """
    MobileNetV2-based binary classifier:
        - Entire feature backbone: FROZEN (all 18 blocks use ImageNet weights)
        - Custom classifier head: trained on BIWI head pose data
            Linear(1280 → 256) → ReLU → Dropout(0.3) → Linear(256 → 2)
    """

    def __init__(self):
        super(HeadPoseClassifier, self).__init__()

        # Load MobileNetV2 with pretrained ImageNet weights
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

        # Freeze ALL backbone parameters
        # requires_grad=False → PyTorch will not calculate or store gradients
        # for these, so they are never updated (their ImageNet values are locked)
        for param in backbone.parameters():
            param.requires_grad = False

        # Keep only the feature extractor part of MobileNetV2
        # backbone.features: 18 convolutional blocks, outputs [B, 1280, 7, 7]
        self.features = backbone.features

        # Global Average Pooling: reduces [B, 1280, 7, 7] → [B, 1280, 1, 1]
        # Takes the average of each feature map, giving one value per channel
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Custom classifier head — these are the trainable layers
        # Input layer (1280) matches MobileNetV2's final feature size
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),   # Compress 1280-dim feature → 256-dim
            nn.ReLU(),              # Non-linearity: allows learning complex patterns
            nn.Dropout(0.3),        # Drop 30% of neurons randomly (reduces overfitting)
            nn.Linear(256, 2)       # 2 output scores: [NOT_LOOKING, LOOKING]
        )

    def forward(self, x):
        """
        Defines the forward pass (how data flows through layers).
        PyTorch calls this automatically when you do model(input).

        x shape in  : [batch, 3, 224, 224]
        x shape out : [batch, 2]  (raw scores / logits for each class)
        """
        x = self.features(x)       # Extract features: [B, 1280, 7, 7]
        x = self.avgpool(x)        # Pool: [B, 1280, 1, 1]
        x = torch.flatten(x, 1)   # Flatten: [B, 1280]
        x = self.classifier(x)    # Classify: [B, 2]
        return x


# ============================================================================
# HeadPosePredictor CLASS
# ============================================================================

class HeadPosePredictor:
    """
    A ready-to-use wrapper around HeadPoseClassifier for real-time inference.

    Simply create one instance at startup, then call .predict() on every
    face crop you want to classify.

    Example:
        predictor = HeadPosePredictor("best_head_pose_model.pth")
        result = predictor.predict(face_bgr_crop)
        if result["is_looking"]:
            print("User is looking at the screen")
    """

    # Class-level constants
    CLASS_NAMES = {0: "NOT_LOOKING", 1: "LOOKING"}

    # Minimum face crop size to bother running inference on
    # Images smaller than this are likely noise or detection errors
    MIN_SIZE = 20   # pixels (width AND height must exceed this)

    def __init__(self, model_path):
        """
        Loads the trained model from disk and prepares it for inference.

        Parameters:
            model_path (str): path to the saved "best_head_pose_model.pth" file
        """

        # ------------------------------------------------------------------
        # Detect and store the compute device (GPU preferred, else CPU)
        # ------------------------------------------------------------------
        # torch.cuda.is_available() returns True only if:
        #   - You have an NVIDIA GPU
        #   - CUDA drivers are correctly installed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[HeadPosePredictor] Using device: {self.device}")

        # ------------------------------------------------------------------
        # Build the model architecture and load the trained weights
        # ------------------------------------------------------------------
        self.model = HeadPoseClassifier()

        # torch.load reads the checkpoint dictionary saved during training
        # map_location=self.device handles the case where the model was
        # trained on GPU but you're now running on CPU (or vice versa)
        checkpoint = torch.load(model_path, map_location=self.device)

        # Fill the model's layers with the saved weight values
        # strict=True (default): every key in the checkpoint must match
        # a layer in the model — any mismatch raises an error immediately
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Move all model tensors to the selected device
        self.model = self.model.to(self.device)

        # Switch to evaluation mode — critical for correct inference:
        #   Dropout layers are DISABLED (use all neurons for stable output)
        #   BatchNorm layers use running statistics, not batch statistics
        # If you forget this, predictions will randomly vary each call!
        self.model.eval()

        # Print which epoch this checkpoint came from (useful for debugging)
        epoch   = checkpoint.get("epoch", "unknown")
        val_acc = checkpoint.get("val_accuracy", 0.0)
        print(f"[HeadPosePredictor] Model loaded from: {model_path}")
        print(f"[HeadPosePredictor] Checkpoint: epoch={epoch}, val_acc={val_acc*100:.2f}%")

        # ------------------------------------------------------------------
        # Preprocessing pipeline — must match val_transform from training
        # ------------------------------------------------------------------
        # Using the same normalization as training is crucial.
        # If training used ImageNet stats, inference MUST use the same.
        # Different stats → the model sees completely different pixel ranges
        # than it was trained on → bad predictions guaranteed.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),    # Scale to model's expected input
            transforms.ToTensor(),            # [0,255] uint8 → [0,1] float32 tensor
            transforms.Normalize(             # Center each channel around 0
                mean=[0.485, 0.456, 0.406],   # ImageNet channel means (R, G, B)
                std= [0.229, 0.224, 0.225]    # ImageNet channel stds  (R, G, B)
            )
        ])

        print("[HeadPosePredictor] Ready for inference.\n")

    # -------------------------------------------------------------------------

    def predict(self, face_image):
        """
        Classifies a single face image as LOOKING or NOT_LOOKING.

        Parameters:
            face_image (np.ndarray): a BGR face crop from OpenCV
                                     shape: (height, width, 3)

        Returns:
            dict with keys:
                "label"      (str)  : "LOOKING", "NOT_LOOKING", or "UNKNOWN"
                "confidence" (float): model certainty 0–100 (e.g. 94.3)
                "is_looking" (bool) : True if label == "LOOKING"
        """

        # ------------------------------------------------------------------
        # Input validation — guard against bad inputs early
        # ------------------------------------------------------------------
        if face_image is None:
            # No image was passed at all (e.g. extract_face_crop returned None)
            return {"label": "UNKNOWN", "confidence": 0.0, "is_looking": False}

        h, w = face_image.shape[:2]   # shape is (height, width, channels)

        if h < self.MIN_SIZE or w < self.MIN_SIZE:
            # Image is too small — likely a detection artefact, skip it
            return {"label": "UNKNOWN", "confidence": 0.0, "is_looking": False}

        # ------------------------------------------------------------------
        # Preprocessing
        # ------------------------------------------------------------------
        # OpenCV loads images in BGR order; PIL and PyTorch expect RGB
        # cv2.cvtColor rearranges the channel order: [B,G,R] → [R,G,B]
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Convert the numpy array to a PIL Image so torchvision transforms work
        # (transforms.ToTensor() expects PIL format as input)
        face_pil = Image.fromarray(face_rgb)

        # Apply the preprocessing pipeline: resize → tensor → normalize
        face_tensor = self.transform(face_pil)    # Shape: [3, 224, 224]

        # Add batch dimension — models always expect [batch, C, H, W]
        # .unsqueeze(0) inserts a new dimension at position 0
        # [3, 224, 224] → [1, 3, 224, 224]  (batch of 1 image)
        face_tensor = face_tensor.unsqueeze(0)

        # Move the tensor to the same device as the model
        face_tensor = face_tensor.to(self.device)

        # ------------------------------------------------------------------
        # Inference
        # ------------------------------------------------------------------
        # torch.no_grad() disables gradient computation during inference:
        #   - We're not training, so we don't need gradients
        #   - Saves memory (~50% less) and runs faster
        with torch.no_grad():
            logits = self.model(face_tensor)      # Raw scores: [1, 2]

        # ------------------------------------------------------------------
        # Convert raw scores → probabilities
        # ------------------------------------------------------------------
        # Softmax converts raw scores (logits) into probabilities that:
        #   - Are all between 0 and 1
        #   - Sum to exactly 1.0
        # e.g. logits [1.2, 3.8] → probabilities [0.077, 0.923]
        probs = F.softmax(logits, dim=1)          # Shape: [1, 2]

        # Extract individual class probabilities (as plain Python floats)
        prob_not_looking = probs[0][0].item()     # Probability of class 0 (NOT_LOOKING)
        prob_looking     = probs[0][1].item()     # Probability of class 1 (LOOKING)

        # ------------------------------------------------------------------
        # PROBABILITY GAP FILTERING
        # ------------------------------------------------------------------
        # WHY NOT JUST USE argmax?
        #   argmax always picks a winner, even when the model is nearly 50/50.
        #   A side profile face might give probabilities [0.42, 0.58] — argmax
        #   says "LOOKING" with 58% confidence, which used to trigger THREAT.
        #   But 58% vs 42% is barely a signal — the model is very uncertain.
        #
        # THE FIX — measure the GAP between the two probabilities:
        #   gap = prob_looking - prob_not_looking
        #   Positive gap → model leans toward LOOKING
        #   Negative gap → model leans toward NOT_LOOKING
        #   Near-zero gap → model is genuinely uncertain → treat as SAFE
        #
        # WHY ASYMMETRIC THRESHOLDS (0.30 vs 0.15)?
        #   We want LOOKING to be HARD to trigger → needs a strong lean (≥0.30)
        #   We want NOT_LOOKING to be EASY to trigger → only needs a mild lean (≤-0.15)
        #   "When in doubt, assume the observer is NOT a threat" = safer for the user.
        #
        # EXAMPLES:
        #   probs [0.10, 0.90] → gap = +0.80 → clearly LOOKING   ✅
        #   probs [0.40, 0.60] → gap = +0.20 → UNCERTAIN (safe)  ⚠️
        #   probs [0.55, 0.45] → gap = -0.10 → UNCERTAIN (safe)  ⚠️
        #   probs [0.80, 0.20] → gap = -0.60 → clearly NOT_LOOKING ✅
        gap = prob_looking - prob_not_looking

        if gap >= 0.25:   # was 0.30 — lowered for faster THREAT response
            # Positive gap ≥ 0.25 → model is confident the face is LOOKING
            label      = "LOOKING"
            is_looking = True
            confidence = prob_looking * 100

        elif gap <= -0.15:
            # Negative gap (or mildly negative) → model leans NOT_LOOKING
            label      = "NOT_LOOKING"
            is_looking = False
            confidence = prob_not_looking * 100

        else:
            # Gap is between -0.15 and +0.30 → model is uncertain.
            # We treat UNCERTAIN as NOT_LOOKING (give the benefit of the doubt).
            # confidence = 0.0 signals to the decision engine to ignore this result.
            label      = "UNCERTAIN"
            is_looking = False
            confidence = 0.0

        return {
            "label"           : label,
            "confidence"      : round(confidence, 2),
            "is_looking"      : is_looking,
            # --- Debug fields (useful for tuning thresholds) ---
            "gap"             : round(gap, 3),
            "prob_looking"    : round(prob_looking * 100, 1),
            "prob_not_looking": round(prob_not_looking * 100, 1),
        }


# ============================================================================
# QUICK SELF-TEST
# ============================================================================
# This block only runs when you execute this file directly:
#   python head_pose_predictor.py
#
# It does NOT run when the file is imported by another script.
# (That's what "if __name__ == '__main__'" means.)

if __name__ == "__main__":

    print("=" * 50)
    print("HeadPosePredictor — Self Test")
    print("=" * 50)

    # Create a predictor — loads the saved model
    predictor = HeadPosePredictor("best_head_pose_model.pth")

    # Create a dummy 224×224 black image to test the pipeline
    # np.zeros creates an array filled with zeros (black pixels)
    # dtype=np.uint8 → values in range [0, 255] like a real image
    # Shape: (height=224, width=224, channels=3)
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)

    # Run prediction on the dummy image
    result = predictor.predict(dummy_image)

    print("\nDummy image prediction result:")
    print(f"  Label      : {result['label']}")
    print(f"  Confidence : {result['confidence']}%")
    print(f"  Is looking : {result['is_looking']}")

    # Test the small-image guard
    tiny_image = np.zeros((10, 10, 3), dtype=np.uint8)
    result_tiny = predictor.predict(tiny_image)
    print(f"\nTiny (10×10) image result  : {result_tiny}")

    # Test the None guard
    result_none = predictor.predict(None)
    print(f"None image result          : {result_none}")

    print("\nModel loaded and working!")
