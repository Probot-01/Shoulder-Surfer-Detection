# test_yolo_photo.py
# Tests YOLOv8 object detection on a single photo and highlights detected people

# Import the YOLO class from the ultralytics library
from ultralytics import YOLO

# Import OpenCV for drawing rectangles and displaying the image
import cv2

# ------------------------------------------------------------------
# STEP 1: Load the YOLOv8 nano model
# ------------------------------------------------------------------
# "yolov8n.pt" is the pre-trained model file downloaded earlier
model = YOLO("yolov8n.pt")

# ------------------------------------------------------------------
# STEP 2: Run detection on a photo
# ------------------------------------------------------------------
# model() runs the detection and returns a list of Result objects
# Each Result contains all detected objects found in that image
results = model("test_photo.jpg")

# We only passed one image, so we grab the first (and only) result
result = results[0]

# ------------------------------------------------------------------
# STEP 3: Load the original image with OpenCV so we can draw on it
# ------------------------------------------------------------------
image = cv2.imread("test_photo.jpg")

# ------------------------------------------------------------------
# STEP 4: Filter detections to only keep "person" (class ID = 0)
# ------------------------------------------------------------------
# result.boxes contains all detected bounding boxes
# Each box has:
#   .cls  -> the class ID (0 = person, 1 = bicycle, 2 = car, etc.)
#   .xyxy -> the box coordinates [x1, y1, x2, y2] in pixels
#   .conf -> confidence score (how sure the model is)

person_count = 0  # We'll count how many people were found

for box in result.boxes:
    # Get the class ID as an integer (e.g. 0 for person)
    class_id = int(box.cls[0])

    # Only process detections where class ID is 0 (person)
    if class_id == 0:
        person_count += 1  # Increment our person counter

        # Extract the bounding box coordinates and convert to integers
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get the confidence score as a percentage (e.g. 0.92 -> "92%")
        confidence = float(box.conf[0])
        label = f"Person {confidence:.0%}"

        # ------------------------------------------------------------------
        # STEP 5: Draw a green rectangle around the detected person
        # ------------------------------------------------------------------
        # cv2.rectangle(image, top-left corner, bottom-right corner, color, thickness)
        # Color is in BGR format: (0, 255, 0) = green
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw a label showing "Person" and the confidence score above the box
        cv2.putText(
            image,          # image to draw on
            label,          # text to display
            (x1, y1 - 10), # position: just above the top-left of the box
            cv2.FONT_HERSHEY_SIMPLEX,  # font style
            0.6,            # font size
            (0, 255, 0),    # font color (green, BGR)
            2               # font thickness
        )

# ------------------------------------------------------------------
# STEP 6: Print how many people were detected
# ------------------------------------------------------------------
print(f"People detected: {person_count}")

# ------------------------------------------------------------------
# STEP 7: Save the result image with the green boxes drawn on it
# ------------------------------------------------------------------
cv2.imwrite("test_yolo_result.jpg", image)
print("Result saved as test_yolo_result.jpg")

# ------------------------------------------------------------------
# STEP 8: Display the result in a window
# ------------------------------------------------------------------
cv2.imshow("YOLO Detection Result", image)

# Wait until the user presses any key before closing the window
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
