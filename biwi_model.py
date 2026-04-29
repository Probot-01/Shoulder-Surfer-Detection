# ============================================================================
# biwi_model.py
# ============================================================================
# PURPOSE:
#   Define a MobileNetV2-based binary classifier for head pose detection.
#   Uses Transfer Learning: start from a model already trained on ImageNet
#   (millions of images) and fine-tune only the end layers for our task.
#
# HOW TO USE:
#   Paste into a Google Colab cell (after biwi_dataset.py).
# ============================================================================


# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn                         # Neural network building blocks
from torchvision import models                # Pretrained model zoo (MobileNetV2, ResNet, etc.)


# ============================================================================
# BACKGROUND: What is Transfer Learning and Layer Freezing?
# ============================================================================
#
# TRANSFER LEARNING:
#   Training a neural network from scratch needs millions of images and hours
#   of GPU time. Instead, we start with MobileNetV2 already trained on
#   ImageNet (1.2 million images, 1000 categories). It already knows how to
#   detect edges, textures, shapes, and facial features. We just need to
#   re-train the final layers to answer a new question: "LOOKING or NOT?"
#
# FREEZING LAYERS:
#   Each layer in a neural network has "parameters" (numbers that get updated
#   during training). "Freezing" a layer means setting requires_grad=False,
#   which tells PyTorch: "don't update these numbers — keep the pretrained
#   knowledge as-is."
#
#   Think of it like hiring an experienced chef (pretrained model):
#     - Frozen layers = their existing cooking skills (don't change these)
#     - Unfrozen layers = teaching them YOUR specific restaurant's menu
#
# WHY FREEZE MOST LAYERS?
#   1. SPEED: Only a few thousand parameters update instead of millions
#   2. PREVENT OVERFITTING: Less room for the model to memorise noise
#   3. PRESERVE KNOWLEDGE: Early layers detect universal features (edges,
#      textures) that are useful for ANY vision task — no need to relearn them
#
# MobileNetV2 STRUCTURE (simplified):
#   model.features[0]  ← very first conv layer (detects raw edges)
#   model.features[1]  ← ...
#   ...                    (18 feature blocks total, indexed 0–17)
#   model.features[17] ← last feature block (detects high-level patterns)
#   model.classifier   ← final decision layer (we REPLACE this entirely)
#
#   STRATEGY: Freeze the ENTIRE features backbone (all 18 blocks).
#   Only the brand-new classifier head will be trained.
#   This keeps 100% of ImageNet knowledge untouched and trains only
#   ~10–15% of total parameters — faster, less overfitting.
# ============================================================================


# ============================================================================
# HeadPoseClassifier — The Model Class
# ============================================================================

class HeadPoseClassifier(nn.Module):
    """
    Binary classifier for head pose (LOOKING vs NOT_LOOKING).

    Architecture:
        MobileNetV2 backbone (pretrained on ImageNet)
            └── features[0–17]  : ALL FROZEN  (no backbone params updated)
        Custom classifier head (ALL trainable — newly created):
            Linear(1280 → 256) → ReLU → Dropout(0.3) → Linear(256 → 2)
    """

    def __init__(self):
        """
        Called once when you create the model object.
        Loads MobileNetV2, freezes most layers, replaces the classifier head.
        """
        # Always call the parent class constructor first in PyTorch models
        super(HeadPoseClassifier, self).__init__()

        # ------------------------------------------------------------------
        # STEP A: Load MobileNetV2 with pretrained ImageNet weights
        # ------------------------------------------------------------------
        # weights=DEFAULT → uses the best available pretrained weights
        # This downloads ~14 MB the first time and caches it locally
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # ------------------------------------------------------------------
        # STEP B: FREEZE ALL backbone layers — every single one
        # ------------------------------------------------------------------
        # Loop through every parameter in MobileNetV2 and disable gradient
        # computation. requires_grad=False means PyTorch will not compute
        # or store gradients for these values during the backward pass,
        # so they will NEVER change during training. We preserve the full
        # ImageNet pretraining from first layer to last.
        for param in backbone.parameters():
            param.requires_grad = False
        # At this point: 0% of backbone params are trainable.
        # Only the new classifier head (created below) will be trained.

        # ------------------------------------------------------------------
        # STEP C: Extract the fully-frozen feature extractor
        # ------------------------------------------------------------------
        # backbone.features contains all 18 convolutional blocks.
        # All parameters inside are frozen (requires_grad=False).
        # It takes an image [B, 3, 224, 224] and outputs [B, 1280, 7, 7].
        self.features = backbone.features

        # MobileNetV2 uses global average pooling to collapse spatial dimensions
        # Input:  [batch, 1280, 7, 7]  (spatial feature maps)
        # Output: [batch, 1280, 1, 1]  (one value per channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ------------------------------------------------------------------
        # STEP E: Replace the final classifier with our custom head
        # ------------------------------------------------------------------
        # MobileNetV2's original classifier: Linear(1280 → 1000) for ImageNet
        # We replace it with a custom 2-class head.
        #
        # WHY THIS SHAPE?
        #   1280 → 256: compress the rich feature vector into a compact form
        #   ReLU: non-linearity (adds ability to learn complex patterns)
        #   Dropout(0.3): randomly zeros 30% of neurons during training
        #                 → forces the network not to rely on any single neuron
        #                 → reduces overfitting
        #   256 → 2: final scores for [NOT_LOOKING, LOOKING]
        #            (raw scores, called "logits" — softmax applied during loss)

        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),   # Compress 1280 → 256
            nn.ReLU(),              # Non-linear activation
            nn.Dropout(0.3),        # Regularisation (30% dropout)
            nn.Linear(256, 2)       # Output: 2 class scores
        )
        # The new classifier layers have requires_grad=True by default
        # since they are freshly created (not loaded from pretrained weights)

    # -------------------------------------------------------------------------
    def forward(self, x):
        """
        Defines how data flows through the model (the forward pass).
        PyTorch calls this automatically when you do: output = model(input)

        Parameters:
            x (Tensor): input image batch, shape [batch_size, 3, 224, 224]

        Returns:
            logits (Tensor): raw class scores, shape [batch_size, 2]
                             Use argmax to get the predicted class.
        """

        # Pass through the fully-frozen feature extractor (no gradients computed here)
        x = self.features(x)        # [B, 1280, 7, 7]

        # Global average pool: reduce spatial dimensions to 1×1
        x = self.avgpool(x)         # [B, 1280, 1, 1]

        # Flatten: convert [B, 1280, 1, 1] → [B, 1280] for the linear layers
        x = torch.flatten(x, 1)     # [B, 1280]

        # Pass through the custom classifier head
        x = self.classifier(x)      # [B, 2]

        return x


# ============================================================================
# Helper Function: count_trainable_params
# ============================================================================

def count_trainable_params(model):
    """
    Counts and prints how many parameters will actually be trained.

    "Trainable" means requires_grad=True — these get updated by gradient descent.
    "Frozen"    means requires_grad=False — these stay fixed during training.

    Parameters:
        model: any PyTorch nn.Module

    Returns:
        trainable (int): number of trainable parameters
        total     (int): total number of parameters
    """

    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen     = total - trainable

    print("\nModel Parameter Summary:")
    print(f"  Total parameters      : {total:>10,}")
    print(f"  Trainable parameters  : {trainable:>10,}  ← will be updated")
    print(f"  Frozen parameters     : {frozen:>10,}  ← kept from ImageNet pretraining")
    print(f"  Training overhead     : {trainable/total*100:.1f}% of total params")

    return trainable, total


# ============================================================================
# Instantiate Model and Move to Device
# ============================================================================

# Detect whether a GPU is available
# torch.cuda.is_available() → True if an NVIDIA GPU + CUDA drivers are present
# In Colab: Runtime → Change runtime type → GPU  (to enable GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  No GPU found — training on CPU (will be slower)")

# Create the model
model = HeadPoseClassifier()

# Move model to the detected device (GPU or CPU)
# All parameters and buffers are moved; you must also move your data batches
# to the same device during training (see training script)
model = model.to(device)

# Print trainable vs frozen parameter counts
count_trainable_params(model)

print("\nModel is ready for training!")
print("Next step: run biwi_train.py to start the training loop.")
