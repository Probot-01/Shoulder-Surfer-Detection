# ============================================================================
# biwi_inference.py
# ============================================================================
# PURPOSE:
#   Load the trained HeadPoseClassifier model and use it for inference.
#   - predict_head_pose() : classifies a single image
#   - Test on 5 validation samples with visual output
#   - Draw a confusion matrix over the full validation set
#
# PREREQUISITES:
#   Run biwi_dataset.py and biwi_model.py first so that
#   val_subset, val_loader, HeadPoseClassifier, and device are available.
#
# HOW TO USE:
#   Paste into a Google Colab cell and run it.
# ============================================================================


# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F          # For softmax (converts logits → probabilities)
from torchvision import transforms
from PIL import Image                    # Open image files
import matplotlib.pyplot as plt          # Show images and plots
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm                    # Progress bar for validation loop


# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = "/content/drive/MyDrive/shoulder_surfing/best_head_pose_model.pth"

# Human-readable class names indexed by label integer
# label 0 = NOT_LOOKING,  label 1 = LOOKING
CLASS_NAMES = {0: "NOT_LOOKING", 1: "LOOKING"}

# Validation preprocessing — must be IDENTICAL to val_transform in biwi_dataset.py
# Using the same normalization ensures the model sees the same pixel range
# it was trained on. Different normalization = garbage predictions.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),       # Resize to the model's expected input size
    transforms.ToTensor(),               # Convert PIL image → tensor, scale [0,255]→[0,1]
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # Normalize
])


# ============================================================================
# HELPER — Load the Model from a Checkpoint
# ============================================================================

def load_model(model_path, device):
    """
    Creates a fresh HeadPoseClassifier and loads the saved weights into it.

    Why do we need to re-create the model before loading?
      torch.save() only saves the WEIGHTS (numbers), not the model code.
      So we must build the same architecture first, then fill it with the
      saved weights using load_state_dict().

    Parameters:
        model_path (str): path to the .pth checkpoint file
        device: torch.device (cuda or cpu)

    Returns:
        model: HeadPoseClassifier with trained weights, set to eval mode
    """

    # Create a new model with the same architecture as during training
    loaded_model = HeadPoseClassifier()

    # Load the checkpoint dictionary that was saved during training
    # map_location=device ensures it loads correctly whether or not
    # the current machine has a GPU
    checkpoint = torch.load(model_path, map_location=device)

    # Fill the model with the saved weights
    # state_dict is a dict of {layer_name: weight_tensor}
    loaded_model.load_state_dict(checkpoint["model_state_dict"])

    # Move the model to the correct device (GPU or CPU)
    loaded_model = loaded_model.to(device)

    # Switch to evaluation mode:
    #   - Dropout layers are DISABLED (use all neurons for stable output)
    #   - BatchNorm uses running statistics (not batch statistics)
    # IMPORTANT: Always call .eval() before inference, or results will vary!
    loaded_model.eval()

    epoch    = checkpoint.get("epoch", "?")
    val_acc  = checkpoint.get("val_accuracy", 0)
    print(f"Model loaded from: {model_path}")
    print(f"  Saved at epoch: {epoch}  |  Val accuracy: {val_acc*100:.2f}%")

    return loaded_model


# ============================================================================
# MAIN FUNCTION — predict_head_pose
# ============================================================================

def predict_head_pose(image_path, model_path):
    """
    Predicts whether the person in an image is LOOKING or NOT_LOOKING.

    How it works:
      1. Load the image from disk and convert to RGB
      2. Preprocess: resize → tensor → normalize (same as val_transform)
      3. Add a "batch" dimension — models expect [batch, channels, H, W],
         but we have just [channels, H, W], so we use .unsqueeze(0)
      4. Run through the model → get raw scores (logits) for each class
      5. Apply softmax to convert logits → probabilities that sum to 1.0
      6. Pick the class with the highest probability

    Parameters:
        image_path (str): absolute path to the image file
        model_path (str): path to the saved .pth model checkpoint

    Returns:
        prediction_label    (str)  : "LOOKING" or "NOT_LOOKING"
        confidence_percent  (float): e.g. 94.3  (the model's certainty)
    """

    # ---- Load and preprocess the image ----
    image_pil = Image.open(image_path).convert("RGB")  # Always use RGB (3 channels)

    # Apply the same preprocessing as during validation
    image_tensor = inference_transform(image_pil)       # Shape: [3, 224, 224]

    # Add batch dimension: [3, 224, 224] → [1, 3, 224, 224]
    # Neural networks always process batches; here our "batch" has 1 image
    image_tensor = image_tensor.unsqueeze(0)            # Shape: [1, 3, 224, 224]

    # Move to same device as the model
    image_tensor = image_tensor.to(device)

    # ---- Run inference ----
    # torch.no_grad(): disable gradient tracking — we are not training,
    # so we don't need gradients. This saves memory and runs faster.
    with torch.no_grad():
        logits = inference_model(image_tensor)          # Raw scores: [1, 2]

    # ---- Convert logits → probabilities ----
    # softmax turns raw scores into probabilities between 0 and 1
    # dim=1 → apply across the class dimension
    probabilities = F.softmax(logits, dim=1)            # Shape: [1, 2]

    # Get the index (0 or 1) with the highest probability
    predicted_class_idx = torch.argmax(probabilities, dim=1).item()

    # Get the confidence for the predicted class as a percentage
    confidence_percent = probabilities[0][predicted_class_idx].item() * 100

    # Map the class index to a human-readable label
    prediction_label = CLASS_NAMES[predicted_class_idx]

    return prediction_label, confidence_percent


# ============================================================================
# LOAD THE MODEL ONCE (reused by all predictions below)
# ============================================================================

inference_model = load_model(MODEL_PATH, device)


# ============================================================================
# TEST ON 5 VALIDATION SAMPLES
# ============================================================================
# We'll grab 5 images from val_subset, predict on them,
# and compare the prediction to the true label.

print("\n" + "=" * 60)
print("Testing on 5 Validation Samples")
print("=" * 60)

# Pick 5 evenly spaced indices from the validation set
num_val     = len(val_subset)
test_indices = [int(i * num_val / 5) for i in range(5)]

# Set up a matplotlib figure: 1 row, 5 columns
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Head Pose Predictions on Validation Samples", fontsize=13, fontweight="bold")

for plot_col, idx in enumerate(test_indices):

    # Get the image path and true label directly from the CSV DataFrame
    # (val_subset.dataset.data holds the full DataFrame)
    row        = val_subset.dataset.data.iloc[val_subset.indices[idx]]
    image_path = row["image_path"]
    true_label = CLASS_NAMES[int(row["label"])]

    # Run prediction
    pred_label, confidence = predict_head_pose(image_path, MODEL_PATH)

    # Is the prediction correct?
    is_correct = (pred_label == true_label)
    status_str = "✅ CORRECT" if is_correct else "❌ WRONG"

    # Print to console
    print(f"\nSample {plot_col + 1}:")
    print(f"  Image      : {image_path.split('/')[-1]}")
    print(f"  True label : {true_label}")
    print(f"  Predicted  : {pred_label}  ({confidence:.1f}% confidence)")
    print(f"  Result     : {status_str}")

    # Show the image in the subplot
    img_pil = Image.open(image_path).convert("RGB")
    axes[plot_col].imshow(img_pil)
    axes[plot_col].axis("off")

    # Color the title green if correct, red if wrong
    title_color = "green" if is_correct else "red"
    axes[plot_col].set_title(
        f"Pred: {pred_label}\n{confidence:.1f}%  {status_str}",
        fontsize=8,
        color=title_color
    )

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/shoulder_surfing/sample_predictions.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("\n5-sample preview saved to Google Drive.")


# ============================================================================
# CONFUSION MATRIX OVER THE FULL VALIDATION SET
# ============================================================================
# A confusion matrix shows, for each TRUE class, how many images the model
# predicted as each possible class. The diagonal = correct predictions.
#
#                   Predicted NOT_LOOKING  |  Predicted LOOKING
#  True NOT_LOOKING        TN              |        FP
#  True LOOKING            FN              |        TP
#
# TN = True Negative  (not looking, predicted not looking) ✅
# TP = True Positive  (looking, predicted looking)         ✅
# FP = False Positive (not looking but predicted looking)  ❌
# FN = False Negative (looking but predicted not looking)  ❌

print("\n" + "=" * 60)
print("Computing Confusion Matrix (full validation set)...")
print("=" * 60)

all_preds  = []   # Predicted class indices for every val image
all_labels = []   # True class indices for every val image

inference_model.eval()

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Running validation"):

        images = images.to(device)

        # Get model outputs (logits) for the whole batch
        outputs = inference_model(images)

        # Get the predicted class (index with highest score)
        _, predicted = torch.max(outputs, dim=1)

        # Collect predictions and true labels as plain Python lists
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Build the confusion matrix using sklearn
cm = confusion_matrix(all_labels, all_preds)

# Display it as a nice annotated heatmap
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix = cm,
    display_labels   = ["NOT_LOOKING (0)", "LOOKING (1)"]
)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — Validation Set", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/shoulder_surfing/confusion_matrix.png",
            dpi=150, bbox_inches="tight")
plt.show()

# ---- Print a text version of the confusion matrix ----
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

TN = cm[0, 0]   # True Negative  : NOT_LOOKING predicted as NOT_LOOKING
FP = cm[0, 1]   # False Positive : NOT_LOOKING predicted as LOOKING
FN = cm[1, 0]   # False Negative : LOOKING predicted as NOT_LOOKING
TP = cm[1, 1]   # True Positive  : LOOKING predicted as LOOKING

overall_acc = (TP + TN) / (TP + TN + FP + FN)

print("\nConfusion Matrix Summary:")
print(f"  True Positives  (LOOKING → LOOKING)         : {TP:>5}")
print(f"  True Negatives  (NOT_LOOKING → NOT_LOOKING) : {TN:>5}")
print(f"  False Positives (NOT_LOOKING → LOOKING)     : {FP:>5}  ← false alarm")
print(f"  False Negatives (LOOKING → NOT_LOOKING)     : {FN:>5}  ← missed detection")
print(f"\n  Overall Accuracy : {overall_acc*100:.2f}%")
print(f"\nConfusion matrix saved to Google Drive.")
