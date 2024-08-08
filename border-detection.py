import os
os.system('cd border_detection && python scan.py --image ../test.jpg')
import cv2
import numpy as np
import json

# Function to resize and reshape the cheque
def transform_cheque(image_path, corners):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not open or find the image: {image_path}")
        return
    
    # Define the points of the corners in the order:
    # top-left, top-right, bottom-right, bottom-left
    pts1 = np.array(corners, dtype="float32")

    # Compute the width of the new image, which is the maximum distance between
    # bottom-right and bottom-left x-coordinates or the top-right and top-left x-coordinates
    widthA = np.linalg.norm(pts1[2] - pts1[3])
    widthB = np.linalg.norm(pts1[1] - pts1[0])
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image, which is the maximum distance between
    # top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.linalg.norm(pts1[1] - pts1[2])
    heightB = np.linalg.norm(pts1[0] - pts1[3])
    maxHeight = max(int(heightA), int(heightB))

    # Define the destination points which will be the points of the new image
    pts2 = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

    return transformed

image_path = 'test.jpg'
corners = json.load(open('output_contour.json'))
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape
ratio = image_height / 500
corners = [[value *ratio for value in corner] for corner in corners]
transformed_image = transform_cheque(image_path, corners)
cv2.imwrite('transformed_cheque.jpg', transformed_image)
cv2.imshow('Transformed Cheque', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()