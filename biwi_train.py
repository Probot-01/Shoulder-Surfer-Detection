# ============================================================================
# biwi_train.py
# ============================================================================
# PURPOSE:
#   Complete training loop for the HeadPoseClassifier (MobileNetV2-based).
#   Trains for 20 epochs, saves the best model, and plots learning curves.
#
# PREREQUISITES (run these first in Colab):
#   1. biwi_balance_dataset.py  → creates biwi_labels_balanced.csv
#   2. biwi_dataset.py          → creates train_loader, val_loader
#   3. biwi_model.py            → creates model, device
#
# HOW TO USE:
#   Paste this entire script into a Google Colab cell and run it.
# ============================================================================


# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn                     # Loss functions, layers
from torch.optim import Adam              # Adam optimizer
from torch.optim.lr_scheduler import StepLR  # Learning rate scheduler
import matplotlib.pyplot as plt           # For plotting loss/accuracy curves
import os                                 # For building file paths

# tqdm shows a live progress bar — works great in Colab
# If tqdm isn't installed: !pip install tqdm
from tqdm import tqdm


# ============================================================================
# WHAT IS "LOSS"?
# ============================================================================
# Loss is a single number that measures "how wrong the model is right now".
# A high loss = many wrong predictions or very uncertain predictions.
# A low loss = mostly correct, confident predictions.
#
# During training, PyTorch:
#   1. Feeds a batch of images through the model → gets predictions
#   2. Compares predictions to the true labels → computes the loss number
#   3. Works backwards (backpropagation) to figure out which parameters
#      caused the error
#   4. Nudges those parameters slightly in the direction that reduces loss
#      (this is gradient descent / the optimizer's job)
#
# We use CrossEntropyLoss, which is the standard loss for classification tasks.
# It penalises confidently wrong predictions much more than uncertain ones.
# ============================================================================


# ============================================================================
# CONFIGURATION
# ============================================================================

# Where to save the best model weights
SAVE_PATH = "/content/drive/MyDrive/shoulder_surfing/best_head_pose_model.pth"

NUM_EPOCHS    = 20        # Total number of training passes through the dataset
LEARNING_RATE = 0.0001    # How big each parameter update step is
                          # Too large → overshoots, unstable training
                          # Too small → learns very slowly

# StepLR settings: multiply LR by GAMMA every STEP_SIZE epochs
LR_STEP_SIZE  = 5         # Reduce LR every 5 epochs
LR_GAMMA      = 0.5       # Halve the LR each time (0.5 = ×½)


# ============================================================================
# STEP 1 — Loss function, Optimizer, Scheduler
# ============================================================================

# ---- Loss Function ----
# CrossEntropyLoss is standard for multi-class classification.
# It combines log-softmax + negative log-likelihood in one efficient step.
criterion = nn.CrossEntropyLoss()

# ---- Optimizer ----
# Adam is an adaptive optimizer — it adjusts the step size per parameter.
# We only pass the TRAINABLE parameters (classifier head) to the optimizer.
# Passing frozen parameters would waste memory and time (they don't update anyway).
optimizer = Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

# ---- Learning Rate Scheduler ----
# The learning rate controls how big each update step is.
# Starting with a larger LR is good for early training (fast learning).
# Reducing it later helps the model settle into a good minimum (fine tuning).
# StepLR multiplies the LR by gamma every step_size epochs:
#   Epoch 1–5  : LR = 0.0001
#   Epoch 6–10 : LR = 0.00005
#   Epoch 11–15: LR = 0.000025
#   Epoch 16–20: LR = 0.0000125
scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

print("Loss function : CrossEntropyLoss")
print(f"Optimizer     : Adam  (lr={LEARNING_RATE})")
print(f"LR scheduler  : StepLR  (step={LR_STEP_SIZE}, gamma={LR_GAMMA})")
print(f"Epochs        : {NUM_EPOCHS}")
print(f"Model save to : {SAVE_PATH}\n")


# ============================================================================
# STEP 2 — Helper functions: train_one_epoch and evaluate
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Runs the model through the entire training set ONCE (one epoch).

    For each batch:
      1. Move images + labels to GPU/CPU
      2. Zero the gradients from the previous batch
      3. Forward pass: predict class scores
      4. Compute loss
      5. Backward pass: compute gradients
      6. Optimizer step: update parameters
      7. Track running loss and correct predictions

    Returns:
        avg_loss (float): average loss across all batches
        accuracy (float): fraction of correct predictions (0.0 – 1.0)
    """

    model.train()   # Switch to training mode:
                    # - Dropout randomly deactivates neurons (regularisation)
                    # - BatchNorm uses batch statistics

    running_loss    = 0.0
    correct         = 0
    total           = 0

    # tqdm wraps the loader to show a live progress bar
    loop = tqdm(loader, desc="  Training", leave=False)

    for images, labels in loop:

        # Move data to the same device as the model (GPU or CPU)
        images = images.to(device)
        labels = labels.to(device)

        # ---- Zero gradients ----
        # PyTorch ACCUMULATES gradients by default. We must reset them
        # before each batch, otherwise they pile up from previous batches.
        optimizer.zero_grad()

        # ---- Forward pass ----
        # Feed the image batch through the model → get raw class scores (logits)
        # Shape: [batch_size, 2]  (one score per class)
        outputs = model(images)

        # ---- Compute loss ----
        # Compare predicted scores to true labels
        loss = criterion(outputs, labels)

        # ---- Backward pass ----
        # Compute how much each trainable parameter contributed to the loss
        # (this fills in the .grad attribute of each trainable parameter)
        loss.backward()

        # ---- Optimizer step ----
        # Nudge each trainable parameter slightly to reduce the loss
        optimizer.step()

        # ---- Track metrics ----
        running_loss += loss.item()

        # torch.max returns the highest score and its index (= predicted class)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)

        # Update the progress bar with current loss
        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


# ----------------------------------------------------------------------------

def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set WITHOUT updating parameters.

    Key differences from train_one_epoch:
      - model.eval() disables Dropout and uses running BatchNorm stats
      - torch.no_grad() skips gradient computation (saves memory + speed)

    Returns:
        avg_loss (float): average validation loss
        accuracy (float): validation accuracy (0.0 – 1.0)
    """

    model.eval()    # Switch to evaluation mode:
                    # - Dropout is DISABLED (use all neurons for stable output)
                    # - BatchNorm uses running statistics

    running_loss = 0.0
    correct      = 0
    total        = 0

    # torch.no_grad() tells PyTorch not to build the computation graph
    # (we don't need gradients during validation → saves memory and time)
    with torch.no_grad():

        loop = tqdm(loader, desc="  Validation", leave=False)

        for images, labels in loop:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


# ============================================================================
# STEP 3 — Main Training Loop (20 epochs)
# ============================================================================

# Lists to record metrics after each epoch (used for plotting later)
train_losses     = []
val_losses       = []
train_accuracies = []
val_accuracies   = []

# Keep track of the best validation accuracy we've seen so far
# We only save the model when this improves
best_val_accuracy = 0.0

print("=" * 65)
print("Starting Training")
print("=" * 65)

for epoch in range(1, NUM_EPOCHS + 1):

    print(f"\nEpoch {epoch:>2}/{NUM_EPOCHS}  |  LR: {scheduler.get_last_lr()[0]:.6f}")
    print("-" * 45)

    # ---- Train for one full pass through the training set ----
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    # ---- Evaluate on the validation set ----
    val_loss, val_acc = evaluate(
        model, val_loader, criterion, device
    )

    # ---- Step the learning rate scheduler ----
    # This checks whether it's time to reduce the LR (every 5 epochs)
    scheduler.step()

    # ---- Record metrics ----
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc * 100)   # Convert to percentage
    val_accuracies.append(val_acc * 100)

    # ---- Print epoch summary ----
    print(f"  Train  |  Loss: {train_loss:.4f}  |  Acc: {train_acc*100:.2f}%")
    print(f"  Val    |  Loss: {val_loss:.4f}  |  Acc: {val_acc*100:.2f}%")

    # ---- Save model if validation accuracy improved ----
    # We only save when we beat the previous best — this way the saved model
    # is always the best one, even if later epochs overfit and get worse.
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc

        # Make sure the save directory exists
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

        # torch.save saves the model's learned parameters (state_dict)
        # We save the entire dict so we know which epoch and accuracy it came from
        torch.save({
            "epoch"          : epoch,
            "model_state_dict": model.state_dict(),
            "val_accuracy"   : val_acc,
            "val_loss"       : val_loss,
        }, SAVE_PATH)

        print(f"  ✅ New best! Saved model  (val_acc = {val_acc*100:.2f}%)")
    else:
        print(f"  — No improvement  (best so far: {best_val_accuracy*100:.2f}%)")

print("\n" + "=" * 65)
print(f"Training complete!  Best validation accuracy: {best_val_accuracy*100:.2f}%")
print(f"Best model saved to: {SAVE_PATH}")
print("=" * 65)


# ============================================================================
# STEP 4 — Plot Learning Curves
# ============================================================================
# Learning curves show how the model improved over epochs.
# We want to see both lines trending DOWN (loss) or UP (accuracy).
# A large gap between train and val curves = overfitting.

epochs_range = range(1, NUM_EPOCHS + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("HeadPoseClassifier — Training History", fontsize=14, fontweight="bold")

# ---- Plot 1: Loss ----
ax1.plot(epochs_range, train_losses, label="Train Loss",      color="royalblue",  linewidth=2)
ax1.plot(epochs_range, val_losses,   label="Validation Loss", color="tomato", linewidth=2, linestyle="--")
ax1.set_title("Loss over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Mark LR reduction points with vertical lines (every 5 epochs)
for step in range(LR_STEP_SIZE, NUM_EPOCHS, LR_STEP_SIZE):
    ax1.axvline(x=step, color="gray", linestyle=":", alpha=0.6, label="LR reduced" if step == LR_STEP_SIZE else "")

# ---- Plot 2: Accuracy ----
ax2.plot(epochs_range, train_accuracies, label="Train Accuracy",      color="royalblue",  linewidth=2)
ax2.plot(epochs_range, val_accuracies,   label="Validation Accuracy", color="tomato", linewidth=2, linestyle="--")
ax2.set_title("Accuracy over Epochs (%)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim([0, 100])
ax2.legend()
ax2.grid(True, alpha=0.3)

# Mark LR reduction points
for step in range(LR_STEP_SIZE, NUM_EPOCHS, LR_STEP_SIZE):
    ax2.axvline(x=step, color="gray", linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/shoulder_surfing/training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Learning curve plot saved to Google Drive.")


# ============================================================================
# STEP 5 — Final Summary
# ============================================================================

print("\n" + "=" * 65)
print("FINAL TRAINING SUMMARY")
print("=" * 65)
print(f"  Epochs trained          : {NUM_EPOCHS}")
print(f"  Best validation accuracy: {best_val_accuracy*100:.2f}%")
print(f"  Final train loss        : {train_losses[-1]:.4f}")
print(f"  Final val loss          : {val_losses[-1]:.4f}")
print(f"  Final train accuracy    : {train_accuracies[-1]:.2f}%")
print(f"  Final val accuracy      : {val_accuracies[-1]:.2f}%")
print(f"\nModel checkpoint : {SAVE_PATH}")
print(f"Learning curves  : /content/drive/MyDrive/shoulder_surfing/training_curves.png")
