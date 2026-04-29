# ============================================================================
# biwi_balance_dataset.py
# ============================================================================
# PURPOSE:
#   Fix the severe class imbalance in biwi_labels.csv using two techniques:
#     1. Re-labeling with a tighter yaw threshold (±20° instead of ±30°)
#     2. Random oversampling of the minority class
#
# WHAT IS CLASS IMBALANCE?
#   Imagine teaching a child to recognize cats vs dogs, but you only show
#   them 1 dog photo and 27 cat photos. They would just learn to guess "cat"
#   for everything and still be "96% accurate" — but totally useless for dogs.
#   That's class imbalance: when one class has far more samples than another.
#   A model trained on imbalanced data learns to ignore the rare class.
#
# WHY DOES OVERSAMPLING HELP?
#   Oversampling duplicates the minority class (NOT_LOOKING) until both
#   classes have equal counts. The model then sees each class equally often
#   during training and learns to distinguish them properly.
#
# HOW TO USE:
#   Paste this entire script into a Google Colab cell and run it.
# ============================================================================


# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import pandas as pd  # For loading, transforming, and saving the CSV
import numpy as np   # For random operations (shuffle, seed)

# Set a random seed so results are reproducible
# (every run will produce the same shuffled order)
np.random.seed(42)

print("Libraries loaded.")


# ----------------------------------------------------------------------------
# Configuration — change paths here if needed
# ----------------------------------------------------------------------------

INPUT_CSV  = "/content/drive/MyDrive/shoulder_surfing/biwi_labels.csv"
OUTPUT_CSV = "/content/drive/MyDrive/shoulder_surfing/biwi_labels_balanced.csv"

# New (tighter) yaw threshold for re-labeling
NEW_YAW_THRESHOLD = 20.0   # degrees


# ============================================================================
# STEP 0 — Load the original CSV
# ============================================================================

print("\n" + "=" * 55)
print("STEP 0: Loading original CSV")
print("=" * 55)

df_original = pd.read_csv(INPUT_CSV)

# Capture the original class counts for the final summary
original_counts = df_original["label"].value_counts().sort_index()

print(f"Loaded {len(df_original):,} rows from:\n  {INPUT_CSV}\n")
print("Original class distribution:")
print(f"  label=1  LOOKING     : {original_counts.get(1, 0):>6,}")
print(f"  label=0  NOT_LOOKING : {original_counts.get(0, 0):>6,}")


# ============================================================================
# STEP 1 — Re-label using a tighter yaw threshold (±20° instead of ±30°)
# ============================================================================
# WHY TIGHTEN THE THRESHOLD?
#   With ±30°, almost everything gets labelled LOOKING because only
#   very turned heads (beyond 30°) are NOT_LOOKING. Tightening to ±20°
#   means any yaw between 20°–30° now counts as NOT_LOOKING, which gives
#   us more NOT_LOOKING samples without collecting new data.

print("\n" + "=" * 55)
print(f"STEP 1: Re-labeling with tighter threshold (±{NEW_YAW_THRESHOLD}°)")
print("=" * 55)

# Work on a copy so we don't accidentally change the original
df = df_original.copy()

# Apply the new labels using .apply() which runs a function on every row
# Lambda is a one-line function: if |yaw| <= 20 → 1 (LOOKING), else → 0
df["label"] = df["yaw_angle"].apply(
    lambda yaw: 1 if abs(yaw) <= NEW_YAW_THRESHOLD else 0
)

# Count after re-labeling
relabeled_counts = df["label"].value_counts().sort_index()

print(f"\nAfter re-labeling (threshold = ±{NEW_YAW_THRESHOLD}°):")
print(f"  label=1  LOOKING     : {relabeled_counts.get(1, 0):>6,}")
print(f"  label=0  NOT_LOOKING : {relabeled_counts.get(0, 0):>6,}")

# How much did the minority class grow?
old_minority = original_counts.get(0, 0)
new_minority = relabeled_counts.get(0, 0)
print(f"\n  NOT_LOOKING grew from {old_minority:,} → {new_minority:,} "
      f"(+{new_minority - old_minority:,} samples from tighter threshold)")


# ============================================================================
# STEP 2 — Balance the dataset using random oversampling
# ============================================================================
# WHAT IS OVERSAMPLING?
#   We take the smaller class (NOT_LOOKING) and randomly duplicate its rows
#   until it has the same size as the larger class (LOOKING).
#   "With replacement" means the same row can be picked more than once —
#   like drawing cards from a deck and putting each card back before drawing
#   the next one.

print("\n" + "=" * 55)
print("STEP 2: Oversampling minority class to balance dataset")
print("=" * 55)

# Separate the two classes into their own DataFrames
df_looking     = df[df["label"] == 1]   # All LOOKING rows
df_not_looking = df[df["label"] == 0]   # All NOT_LOOKING rows

n_looking     = len(df_looking)
n_not_looking = len(df_not_looking)

print(f"\nBefore oversampling:")
print(f"  LOOKING     : {n_looking:,}")
print(f"  NOT_LOOKING : {n_not_looking:,}")
print(f"  Target      : {n_looking:,} samples per class")

# Oversample the minority class (NOT_LOOKING) to match the majority (LOOKING)
# replace=True → allows the same row to be sampled more than once
# random_state=42 → makes the sampling reproducible
df_not_looking_oversampled = df_not_looking.sample(
    n           = n_looking,    # Sample exactly as many as the majority class
    replace     = True,         # Allow duplicates (oversampling)
    random_state = 42
)

print(f"\n  Oversampled NOT_LOOKING: {len(df_not_looking_oversampled):,} rows "
      f"(duplicated from {n_not_looking:,} originals)")

# Combine the majority class with the oversampled minority class
df_balanced = pd.concat([df_looking, df_not_looking_oversampled])

# Shuffle the combined DataFrame so LOOKING and NOT_LOOKING rows are
# interleaved randomly (not all LOOKING first, then all NOT_LOOKING)
# frac=1 means "shuffle 100% of the rows"
# reset_index=True → give the shuffled DataFrame fresh row numbers (0, 1, 2 ...)
# drop=True → discard the old index column (we don't need the old row numbers)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Verify the final counts
balanced_counts = df_balanced["label"].value_counts().sort_index()

print(f"\nAfter oversampling (balanced):")
print(f"  label=1  LOOKING     : {balanced_counts.get(1, 0):>6,}")
print(f"  label=0  NOT_LOOKING : {balanced_counts.get(0, 0):>6,}")
print(f"  TOTAL                : {len(df_balanced):>6,}")


# ============================================================================
# STEP 3 — Save the balanced CSV
# ============================================================================

print("\n" + "=" * 55)
print("STEP 3: Saving balanced CSV")
print("=" * 55)

# index=False → don't write the row numbers as a column in the CSV
df_balanced.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved balanced dataset to:\n  {OUTPUT_CSV}")


# ============================================================================
# STEP 4 — Final summary
# ============================================================================

print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)

print(f"\n{'Stage':<30} {'LOOKING':>10} {'NOT_LOOKING':>12} {'TOTAL':>8}")
print("-" * 62)

orig_l  = original_counts.get(1, 0)
orig_nl = original_counts.get(0, 0)
print(f"{'Original (±30° threshold)':<30} "
      f"{orig_l:>10,} {orig_nl:>12,} {orig_l + orig_nl:>8,}")

rel_l  = relabeled_counts.get(1, 0)
rel_nl = relabeled_counts.get(0, 0)
print(f"{'After re-labeling (±20°)':<30} "
      f"{rel_l:>10,} {rel_nl:>12,} {rel_l + rel_nl:>8,}")

bal_l  = balanced_counts.get(1, 0)
bal_nl = balanced_counts.get(0, 0)
print(f"{'After oversampling (balanced)':<30} "
      f"{bal_l:>10,} {bal_nl:>12,} {bal_l + bal_nl:>8,}")

print("-" * 62)
print(f"\nBalanced CSV confirmed saved at:\n  {OUTPUT_CSV}")
print("\nDataset is ready for model training!")
