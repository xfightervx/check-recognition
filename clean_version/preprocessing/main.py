import cv2
import numpy as np

def preprocess_check_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding with better parameters
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,  # Increased block size for better text detection
        10   # Increased constant for better contrast
    )
    
    # Apply light morphological operations
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.dilate(binary, kernel, iterations=1)
    cleaned = cv2.erode(cleaned, kernel, iterations=1)
    
    # Deskew the image
    try:
        # Find coordinates of text pixels
        coords = np.column_stack(np.where(cleaned > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        # Correct the angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        # Rotate the image
        (h, w) = cleaned.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle+90, 1.0)
        deskewed = cv2.warpAffine(
            cleaned, 
            M, 
            (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Invert to get black text on white background
        result = cv2.bitwise_not(deskewed)
        
    except:
        # If deskewing fails, return the cleaned image
        result = cv2.bitwise_not(cleaned)
    
    return result

def main():
    image_path = "assets/testwritten.jpg"
    
    # Process the image
    processed_image = preprocess_check_image(image_path)
    
    # Save the processed image in assets folder
    cv2.imwrite("assets/preprocessed.jpg", processed_image)
    
    # Display the original and processed images (optional)
    original = cv2.imread(image_path)
    cv2.imshow("Original", original)
    cv2.imshow("Processed", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()