import cv2
import numpy as np
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt

# Load the deskewed image
image_path = 'assets/deskewed.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply morphological operations to remove noise
kernel = np.ones((1, 1), np.uint8)
image = cv2.dilate(image, kernel, iterations=1)
image = cv2.erode(image, kernel, iterations=1)

# Improve the image contrast and brightness
alpha = 1.5 # Simple contrast control
beta = 20   # Simple brightness control
image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Invert the image for Tesseract
image = cv2.bitwise_not(image)

# Save the cleaned image for OCR
output_path = "/mnt/data/cleaned_for_ocr.png"

# Perform OCR on the cleaned image
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/., "'
extracted_text = pytesseract.image_to_string(image, lang='eng+ara', config=custom_config)


with open("assets/extracted.txt", "w" , encoding="utf-8") as f:
    f.write(extracted_text)