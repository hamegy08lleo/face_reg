import cv2
import os
import tkinter as tk
from tkinter import simpledialog, Button
import time

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Define the directory to store the images
dataset_dir = 'dataset'

# Create a directory for the label if it doesn't exist
os.makedirs(dataset_dir, exist_ok=True)

# Define the scaling factor
scaling_factor = 1


def capture_images():
    # Create a simple dialog to enter the label
    start_time = time.time()
    
    root = tk.Tk()
    root.withdraw()
    label = simpledialog.askstring("Input", "Enter the label for the images",
                                   parent=root)
    #os.makedirs(os.path.join(dataset_dir, label), exist_ok=True)

    # Iterate until the user hits the 'Esc' key
    noLoop = 5
    while noLoop > 0:
        # Capture the current frame
        _, frame = cap.read()
        cv2.imshow('Webcam', frame)
        
        
        if time.time() - start_time > 5:
            # Save the captured image to the dataset directory
            img_name = os.path.join(dataset_dir, f'{label}.{int(time.time())}.png')
            cv2.imwrite(img_name, frame)
            # Update the start time
            noLoop -= 1
            start_time = time.time()

        # Display the output

        # Check if the user hit the 'Esc' key
        c = cv2.waitKey(1)
        if c == 27:
            break
        

    # Release the video capture object
    cap.release()

    # Close all the windows
    cv2.destroyAllWindows()

# Create a button to start capturing images
root = tk.Tk()
Button(root, text="Start Capturing", command=capture_images).pack()
root.mainloop()
