import cv2
import numpy as np

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at pixel (Column: {x}, Row: {y})")
        # You can perform additional actions here if needed

# Load the image
img = cv2.imread('./worlds/sauga_map.png')

# Create a window and bind the mouse callback function
cv2.namedWindow('image')
cv2.setMouseCallback('image', click_event)

# Display the image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
