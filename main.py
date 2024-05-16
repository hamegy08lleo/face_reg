import cv2
import numpy as np

# Load the Haar cascade file
face_cascade = cv2.CascadeClassifier('haar_cascade_files/haarcascade_frontalface_default.xml')

# Check if the cascade file has been loaded correctly
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Create an LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load your trained model
# Make sure to train your model first
recognizer.read('model.yml')

# Define the scaling factor
scaling_factor = 1

# Iterate until the user hits the 'Esc' key
while True:
    # Capture the current frame
    _, frame = cap.read()

    # Resize the frame
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run the face detector on the grayscale image
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each face detected
    for (x,y,w,h) in face_rects:
        # Extract the ROI of the face from the grayscale image
        roi_gray = gray[y:y+h, x:x+w]
        # Try to recognize the face
        label, confidence = recognizer.predict(roi_gray)
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        # Display the label and confidence score
        text = f'{label}, {confidence:.2f}'
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Display the output
    cv2.imshow('Face Detector', frame)

    # Check if the user hit the 'Esc' key
    c = cv2.waitKey(1)
    if c == 27:
        break

# Release the video capture object
cap.release()

# Close all the windows
cv2.destroyAllWindows()
