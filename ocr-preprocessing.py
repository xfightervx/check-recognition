import pytesseract
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('assets./testwritten.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding
binary = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
)

# Perform morphological operations to clean the image
cv2.imwrite('binary.jpg', binary)
# Display the images
# Apply morphological operations to remove noise
kernel = np.ones((1, 1), np.uint8)
image = cv2.dilate(binary, kernel, iterations=1)
image = cv2.erode(image, kernel, iterations=1)

# Invert the image for Tesseract
image = cv2.bitwise_not(image)

# Deskewing the image
coords = np.column_stack(np.where(image > 0))
angle = cv2.minAreaRect(coords)[-1]

# Correct the angle
if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle

(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle+90, 1.0)
image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
cv2.imwrite('assets/deskewed.jpg', image) 