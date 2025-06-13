import cv2  # This imports the OpenCV library

cap = cv2.VideoCapture(0)  # This tries to open the default webcam (usually the built-in or first USB camera)

if cap.isOpened():  # Checks if the camera was successfully opened
    print("Camera works!")  # If yes, print this message
else:
    print("Camera not available.")  # If not, print this error message

cap.release()  # Close the camera properly after testing
