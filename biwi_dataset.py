# ============================================================================
# biwi_dataset.py
# ============================================================================
# PURPOSE:
#   Define a PyTorch Dataset class for the BIWI head pose binary classification
#   task, configure image transforms, and create DataLoaders for training and
#   validation.
#
# HOW TO USE:
#   Paste into a Google Colab cell (after running biwi_balance_dataset.py).
# ============================================================================


# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import pandas as pd                          # Read the CSV file
from PIL import Image                        # Open image files as RGB

import torch                                 # Core PyTorch library
from torch.utils.data import Dataset, DataLoader, random_split
                                             # Dataset  → base class we inherit from
                                             # DataLoader → wraps dataset into batches
                                             # random_split → splits dataset into subsets

import torchvision.transforms as transforms  # Standard image preprocessing tools


# ============================================================================
# HeadPoseDataset — Custom PyTorch Dataset Class
# ============================================================================
# In PyTorch, a Dataset is any class that:
#   1. Inherits from torch.utils.data.Dataset
#   2. Implements __len__()    → tells PyTorch "how many samples do I have?"
#   3. Implements __getitem__() → tells PyTorch "give me sample at index i"
#
# PyTorch's DataLoader will call these methods automatically when training.

class HeadPoseDataset(Dataset):
    """
    Custom Dataset for loading BIWI head pose images and their binary labels.

    Parameters:
        csv_path  (str)             : Path to the CSV file with columns
                                      [image_path, yaw_angle, label]
        transform (callable, None)  : Optional image transforms to apply.
                                      Pass train_transform or val_transform.
    """

    def __init__(self, csv_path, transform=None):
        """
        __init__ is called ONCE when you create the dataset object.
        It loads the CSV into memory and stores the transform.
        """

        # Load the CSV into a pandas DataFrame
        # Each row represents one image: [image_path, yaw_angle, label]
        self.data = pd.read_csv(csv_path)

        # Store the transform pipeline (could be None if no transforms needed)
        self.transform = transform

        print(f"Dataset loaded: {len(self.data):,} samples from {csv_path}")

    # -------------------------------------------------------------------------
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        PyTorch calls this to know how many items exist.
        """
        return len(self.data)

    # -------------------------------------------------------------------------
    def __getitem__(self, index):
        """
        Returns one sample from the dataset at position 'index'.
        PyTorch calls this repeatedly to build each batch.

        Parameters:
            index (int): Row number to fetch (0 to len-1)

        Returns:
            image (Tensor): Image as a PyTorch tensor, shape [3, 224, 224]
            label (int)   : 0 = NOT_LOOKING, 1 = LOOKING
        """

        # Get the row at this index from the DataFrame
        row = self.data.iloc[index]

        # ---- Load the image ----
        # PIL.Image.open() reads the image file from disk
        # .convert("RGB") ensures it's always 3 channels (R, G, B)
        # This handles edge cases like grayscale or RGBA images
        image_path = row["image_path"]
        image = Image.open(image_path).convert("RGB")

        # ---- Apply transforms (resize, normalize, augment, etc.) ----
        # If a transform pipeline was provided, run the image through it
        # This converts the PIL image into a PyTorch tensor
        if self.transform is not None:
            image = self.transform(image)

        # ---- Get the label ----
        # int() ensures it's a plain Python integer (not a numpy int)
        label = int(row["label"])

        # Return the (image_tensor, label) pair
        return image, label


# ============================================================================
# Image Transforms
# ============================================================================
# Transforms are a pipeline of operations applied to every image before
# it's fed into the model.
#
# WHY NORMALIZE?
#   Neural networks train much faster when pixel values are roughly centered
#   around 0 with similar spread. ImageNet mean/std are standard values
#   used when working with pretrained models (like ResNet, MobileNet, etc.)
#   because those models were originally trained with this normalization.
#
# WHY AUGMENT TRAINING DATA?
#   Augmentations (random flips, brightness changes, rotations) artificially
#   expand the training set by creating slightly different versions of each
#   image. This helps the model generalize better and not memorize exact pixels.

# ImageNet normalization values (used by almost all pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # Mean per R, G, B channel
IMAGENET_STD  = [0.229, 0.224, 0.225]   # Std  per R, G, B channel

# ---- Training transforms ----
# Include random augmentations to prevent overfitting
train_transform = transforms.Compose([

    # Resize the image to 224×224 pixels
    # (standard input size for most pretrained CNNs)
    transforms.Resize((224, 224)),

    # Randomly flip the image LEFT↔RIGHT with 50% probability
    # A flipped face is still a valid LOOKING/NOT_LOOKING example
    transforms.RandomHorizontalFlip(p=0.5),

    # Randomly adjust brightness and contrast by ±20%
    # Simulates different lighting conditions (bright room, dark webcam, etc.)
    transforms.ColorJitter(brightness=0.2, contrast=0.2),

    # Randomly rotate the image up to ±10 degrees
    # Simulates a tilted head or camera angle
    transforms.RandomRotation(degrees=10),

    # Convert the PIL image to a PyTorch tensor
    # Also rescales pixel values from [0, 255] to [0.0, 1.0]
    transforms.ToTensor(),

    # Normalize each channel using ImageNet mean and std
    # Formula: output = (input - mean) / std
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---- Validation transforms ----
# NO augmentations — we want deterministic results when evaluating the model
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ============================================================================
# Create the Full Dataset and Split into Train / Validation
# ============================================================================

CSV_PATH = "/content/drive/MyDrive/shoulder_surfing/biwi_labels_balanced.csv"

# Create the full dataset (we'll apply transforms after splitting)
# Use no transform here because train and val need different transforms
full_dataset = HeadPoseDataset(csv_path=CSV_PATH, transform=None)

total_size = len(full_dataset)

# Calculate split sizes: 80% train, 20% val
train_size = int(0.80 * total_size)
val_size   = total_size - train_size   # Remainder goes to validation

# random_split shuffles and divides the dataset
# generator ensures the split is reproducible (same split every run)
train_subset, val_subset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"\nDataset split:")
print(f"  Total  : {total_size:,}")
print(f"  Train  : {train_size:,}  (80%)")
print(f"  Val    : {val_size:,}  (20%)")


# ============================================================================
# Apply Correct Transforms to Each Subset
# ============================================================================
# random_split gives us Subset objects that share the same underlying dataset.
# We need to wrap each Subset with its own transform.
# The cleanest way: override the dataset transform on each subset.

# Apply train_transform to the training subset
train_subset.dataset = HeadPoseDataset(csv_path=CSV_PATH, transform=train_transform)

# Apply val_transform to the validation subset
val_subset.dataset   = HeadPoseDataset(csv_path=CSV_PATH, transform=val_transform)


# ============================================================================
# Create DataLoaders
# ============================================================================
# A DataLoader wraps a Dataset and handles:
#   - Batching  : groups individual samples into batches (e.g. 32 at a time)
#   - Shuffling : randomises the order each epoch (training only)
#   - num_workers: parallel processes to load data faster (set to 2 for Colab)

BATCH_SIZE = 32

# Training DataLoader — shuffle=True so the model sees a random order each epoch
train_loader = DataLoader(
    train_subset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,        # Shuffle training data every epoch
    num_workers = 2,           # 2 background workers to load images faster
    pin_memory  = True         # Speeds up CPU→GPU transfer if using a GPU
)

# Validation DataLoader — shuffle=False for consistent, repeatable evaluation
val_loader = DataLoader(
    val_subset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,       # Keep val order consistent
    num_workers = 2,
    pin_memory  = True
)

print(f"\nDataLoaders created (batch_size = {BATCH_SIZE}):")
print(f"  Train batches : {len(train_loader):,}")
print(f"  Val batches   : {len(val_loader):,}")


# ============================================================================
# Quick Sanity Check — peek at one batch
# ============================================================================
# Grab the first batch from the training loader and check shapes
sample_images, sample_labels = next(iter(train_loader))

print(f"\nSanity check — first batch:")
print(f"  Image batch shape : {sample_images.shape}")
        # Expected: torch.Size([32, 3, 224, 224])
        #   32  = batch size
        #    3  = RGB channels
        #  224  = height
        #  224  = width
print(f"  Label batch shape : {sample_labels.shape}")
        # Expected: torch.Size([32])
print(f"  Unique labels in batch : {sample_labels.unique().tolist()}")
        # Expected: [0, 1]  (both classes present)
print(f"\nDataset is ready for model training!")
