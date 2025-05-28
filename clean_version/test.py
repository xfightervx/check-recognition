import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_cramped_region(image, min_box_area=150):
    """
    Automatically detect a dense/cramped text region based on contour clustering.
    Returns cropped region of interest (ROI).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of potential text blobs
    contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > min_box_area:
            candidates.append((x, y, x + w, y + h))

    # Merge bounding boxes
    if not candidates:
        return None, image

    candidates = np.array(candidates)
    x_min = np.min(candidates[:, 0])
    y_min = np.min(candidates[:, 1])
    x_max = np.max(candidates[:, 2])
    y_max = np.max(candidates[:, 3])

    roi = image[y_min:y_max, x_min:x_max]
    return roi, image[y_min:y_max, x_min:x_max]

def segment_cramped_rows(image, min_line_height=5, spacing_threshold=2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal_proj = np.sum(binary, axis=1)

    rows = []
    in_text = False
    start_row = 0

    for y, val in enumerate(horizontal_proj):
        if val > spacing_threshold and not in_text:
            start_row = y
            in_text = True
        elif val <= spacing_threshold and in_text:
            if y - start_row >= min_line_height:
                rows.append((start_row, y))
            in_text = False

    if in_text and (len(binary) - start_row >= min_line_height):
        rows.append((start_row, len(binary)))

    row_images = [image[y1:y2, :] for y1, y2 in rows]
    return row_images, horizontal_proj

# Load and process full image
image_path = "assets/deskewed.jpg"
full_image = cv2.imread(image_path)

# Detect cramped region automatically
cramped_crop, extracted = detect_cramped_region(full_image)

if cramped_crop is not None:
    row_images, proj_profile = segment_cramped_rows(cramped_crop)

    plt.figure(figsize=(12, 4))
    plt.plot(proj_profile)
    plt.title("Horizontal Projection Profile")
    plt.xlabel("Row index")
    plt.ylabel("Sum of white pixels")
    plt.grid(True)
    plt.show()

    if row_images:
        fig, axes = plt.subplots(len(row_images), 1, figsize=(12, 2.5 * len(row_images)))
        if len(row_images) == 1:
            axes = [axes]
        for ax, img in zip(axes, row_images):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis("off")
        plt.tight_layout()
        plt.show()
    else:
        print("No text rows detected after segmentation.")
else:
    print("No cramped region found in the image.")
