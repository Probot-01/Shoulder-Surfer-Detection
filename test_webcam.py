# test_webcam.py
# A simple script to test if your webcam is working using OpenCV

# Import the OpenCV library (cv2 is the Python name for OpenCV)
import cv2

# Open the default webcam (index 0 = first/default camera)
# VideoCapture(0) tells OpenCV to use the first camera it finds
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    # If the camera couldn't be opened, print an error and stop
    print("Error: Could not open webcam.")
else:
    # Camera opened successfully!
    print("Webcam working!")

    # Start an infinite loop to keep reading frames from the webcam
    while True:
        # Read one frame from the webcam
        # ret  -> True if the frame was captured successfully, False otherwise
        # frame -> the actual image (as a NumPy array)
        ret, frame = cap.read()

        # If reading the frame failed, break out of the loop
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Display the current frame in a window titled "Webcam Test"
        cv2.imshow("Webcam Test", frame)

        # Wait 1 millisecond for a key press and check which key was pressed
        # cv2.waitKey() returns the ASCII value of the key pressed
        # ord('q') gives the ASCII value of the letter 'q'
        # If the user presses 'q', break out of the loop and stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Print a message when the user has quit
    print("Webcam closed")

    # Release the webcam so other programs can use it
    cap.release()

    # Close all OpenCV windows that were opened
    cv2.destroyAllWindows()
