# ============================================================================
# biwi_prepare_dataset.py
# ============================================================================
# PURPOSE:
#   Parse the BIWI Kinect Head Pose dataset, calculate the yaw angle for
#   every image, and classify each frame as:
#       LOOKING     (label = 1)  →  |yaw| <= 30°   (person facing forward/screen)
#       NOT_LOOKING (label = 0)  →  |yaw|  > 30°   (person looking away)
#
# OUTPUT:
#   biwi_labels.csv saved to Google Drive
#
# HOW TO USE:
#   Copy-paste this entire script into a Google Colab notebook cell and run it.
#   (Runtime → Run All, or press Shift+Enter on the cell)
# ============================================================================


# ----------------------------------------------------------------------------
# CELL 1 — Mount Google Drive
# ----------------------------------------------------------------------------
# This gives Colab access to your personal Google Drive files.
# A popup will ask you to sign in and grant permission — click "Allow".
from google.colab import drive
drive.mount('/content/drive')
print("Google Drive mounted successfully!")


# ----------------------------------------------------------------------------
# CELL 2 — Imports
# ----------------------------------------------------------------------------
import os          # For walking through folder structures
import numpy as np # For maths (rotation matrix → yaw angle)
import pandas as pd # For building and saving the DataFrame / CSV

print("Libraries imported successfully!")


# ----------------------------------------------------------------------------
# CELL 3 — Configuration
# ----------------------------------------------------------------------------

# Root folder of the BIWI dataset on your Google Drive
DATASET_ROOT = "/content/drive/MyDrive/shoulder_surfing/head_pose_dataset"

# Where to save the output CSV
OUTPUT_CSV = "/content/drive/MyDrive/shoulder_surfing/biwi_labels.csv"

# Yaw threshold in degrees:
#   |yaw| <= YAW_THRESHOLD → LOOKING (label 1)
#   |yaw|  > YAW_THRESHOLD → NOT_LOOKING (label 0)
YAW_THRESHOLD = 30.0

print(f"Dataset root : {DATASET_ROOT}")
print(f"Output CSV   : {OUTPUT_CSV}")
print(f"Yaw threshold: ±{YAW_THRESHOLD}°")


# ----------------------------------------------------------------------------
# CELL 4 — Helper function: parse_pose_file
# ----------------------------------------------------------------------------

def parse_pose_file(pose_path):
    """
    Reads a BIWI pose.txt file and extracts the yaw angle in degrees.

    BIWI pose.txt layout:
        Line 1:  R[0][0]  R[0][1]  R[0][2]   ← first row of rotation matrix
        Line 2:  R[1][0]  R[1][1]  R[1][2]   ← second row
        Line 3:  R[2][0]  R[2][1]  R[2][2]   ← third row
        Line 4:  tx  ty  tz                   ← translation vector (we ignore this)

    What is a rotation matrix?
        It's a 3×3 grid of numbers that describes how something is rotated
        in 3D space. We only need it to figure out which direction the head
        is pointing (yaw = left/right rotation).

    Yaw formula:
        yaw = arctan2(R[1][0], R[0][0])
        arctan2 is like arctan but smarter — it handles all four quadrants
        correctly and gives a result in radians, which we convert to degrees.

    Parameters:
        pose_path (str): absolute path to the pose.txt file

    Returns:
        yaw_deg (float): yaw angle in degrees, or None if the file is bad
    """

    try:
        with open(pose_path, 'r') as f:
            lines = f.readlines()

        # We need at least 3 lines for the rotation matrix
        if len(lines) < 3:
            print(f"  WARNING: Too few lines in {pose_path} — skipping.")
            return None

        # Parse each row of the 3×3 rotation matrix
        # Each line has 3 space-separated float values
        row0 = list(map(float, lines[0].split()))   # [R00, R01, R02]
        row1 = list(map(float, lines[1].split()))   # [R10, R11, R12]
        row2 = list(map(float, lines[2].split()))   # [R20, R21, R22]

        # Make sure each row has exactly 3 values
        if len(row0) < 3 or len(row1) < 3 or len(row2) < 3:
            print(f"  WARNING: Malformed rotation matrix in {pose_path} — skipping.")
            return None

        # Reconstruct the 3×3 rotation matrix as a NumPy array
        # (makes it easy to index like R[row][col])
        R = np.array([row0, row1, row2])

        # ---------- Calculate the yaw angle ----------
        # Yaw is the left/right rotation of the head (rotation around the Y-axis).
        # The formula arctan2(R[1,0], R[0,0]) extracts this from the rotation matrix.
        # np.arctan2 returns a value in radians → multiply by 180/π to get degrees.
        yaw_rad = np.arctan2(R[1, 0], R[0, 0])
        yaw_deg = np.degrees(yaw_rad)   # Convert radians → degrees

        return yaw_deg

    except Exception as e:
        # Catch-all: print what went wrong and skip this file
        print(f"  ERROR reading {pose_path}: {e}")
        return None


# ----------------------------------------------------------------------------
# CELL 5 — Main loop: walk dataset, parse poses, build DataFrame
# ----------------------------------------------------------------------------

print("\nScanning dataset...\n")

# Lists to accumulate results — we'll build the DataFrame at the end
records = []   # Each item will be a dict { image_path, yaw_angle, label }

# os.listdir returns all items in the dataset root folder
# Each item should be a subject folder: "01", "02", ..., "24"
subject_folders = sorted(os.listdir(DATASET_ROOT))

for subject in subject_folders:

    subject_path = os.path.join(DATASET_ROOT, subject)

    # Skip anything that is NOT a folder (e.g. stray files)
    if not os.path.isdir(subject_path):
        continue

    print(f"Processing subject: {subject}")

    # List all files in this subject's folder
    all_files = os.listdir(subject_path)

    # Keep only the RGB image files
    # BIWI naming convention: frame_XXXXX_rgb.png
    image_files = [f for f in all_files if f.endswith("_rgb.png")]

    for image_file in sorted(image_files):

        # Build the full path to the image
        image_path = os.path.join(subject_path, image_file)

        # Derive the matching pose filename by replacing "_rgb.png" with "_pose.txt"
        # Example: "frame_00001_rgb.png" → "frame_00001_pose.txt"
        pose_file = image_file.replace("_rgb.png", "_pose.txt")
        pose_path = os.path.join(subject_path, pose_file)

        # Skip if the pose file doesn't exist (shouldn't happen, but just in case)
        if not os.path.exists(pose_path):
            print(f"  WARNING: No pose file for {image_file} — skipping.")
            continue

        # Parse the yaw angle from the pose file
        yaw = parse_pose_file(pose_path)

        # Skip this frame if parsing failed
        if yaw is None:
            continue

        # ---------- Binary classification ----------
        # |yaw| <= 30° → person is roughly facing forward → LOOKING (1)
        # |yaw|  > 30° → person is turned away           → NOT_LOOKING (0)
        if abs(yaw) <= YAW_THRESHOLD:
            label = 1   # LOOKING
        else:
            label = 0   # NOT_LOOKING

        # Store the result for this frame
        records.append({
            "image_path": image_path,
            "yaw_angle" : round(yaw, 4),   # Round to 4 decimal places
            "label"     : label
        })

print(f"\nDone scanning. {len(records)} frames processed.")


# ----------------------------------------------------------------------------
# CELL 6 — Build the DataFrame and print statistics
# ----------------------------------------------------------------------------

# Convert our list of dicts into a pandas DataFrame
# A DataFrame is like an Excel spreadsheet — rows and named columns
df = pd.DataFrame(records, columns=["image_path", "yaw_angle", "label"])

# Count how many frames fall into each class
total          = len(df)
looking_count  = (df["label"] == 1).sum()   # How many are LOOKING
not_look_count = (df["label"] == 0).sum()   # How many are NOT_LOOKING

print("=" * 50)
print("DATASET SUMMARY")
print("=" * 50)
print(f"Total images found    : {total}")
print(f"LOOKING     (label=1) : {looking_count}  ({looking_count/total*100:.1f}%)")
print(f"NOT_LOOKING (label=0) : {not_look_count}  ({not_look_count/total*100:.1f}%)")
print("=" * 50)

# Show a random sample of 5 rows to visually verify things look right
print("\n5 sample rows from the DataFrame:")
print(df.sample(5).to_string(index=False))


# ----------------------------------------------------------------------------
# CELL 7 — Save the DataFrame as a CSV to Google Drive
# ----------------------------------------------------------------------------

# Make sure the output folder exists (create it if not)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Save the DataFrame to a CSV file
# index=False → don't write row numbers as a column
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nCSV saved to: {OUTPUT_CSV}")
print("You can now use biwi_labels.csv in your model training script.")
