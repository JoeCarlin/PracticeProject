import numpy as np
import cv2

# load a video capture device
cap = cv2.VideoCapture(0)

# while loop that will go until key is pressed
while True:
    # read a frame from the video capture device
    ret, frame = cap.read()

    # if the frame was not read correctly, break the loop
    if not ret:
        break

    # display the frame in a window
    cv2.imshow('frame', frame)

    # wait for 1 ms and check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break