import os
import cv2
import torch
import numpy as np
from craft import CRAFT
from craft_utils import getDetBoxes
from imgproc import loadImage, normalizeMeanVariance

# ========================
# --- Model Utilities ---
# ========================

def load_craft_model(trained_model_path='weights/craft_mlt_25k.pth', cuda=False):
    net = CRAFT()
    state_dict = torch.load(trained_model_path, map_location='cuda' if cuda else 'cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net.load_state_dict(new_state_dict)
    net.eval()
    return net.cuda() if cuda else net

# ========================
# --- Image Utilities ---
# ========================

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, _ = img.shape
    target_size = mag_ratio * max(height, width)
    if target_size > square_size:
        target_size = square_size
    ratio = target_size / max(height, width)
    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation)
    target_h32 = target_h if target_h % 32 == 0 else target_h + (32 - target_h % 32)
    target_w32 = target_w if target_w % 32 == 0 else target_w + (32 - target_w % 32)
    padded = np.zeros((target_h32, target_w32, 3), dtype=np.uint8)
    padded[:target_h, :target_w, :] = proc
    return padded, ratio, (target_w32, target_h32)

# ========================
# --- CRAFT Inference ---
# ========================

def run_craft_on_image(image_path, net, cuda=False):
    image = loadImage(image_path)
    img_resized, target_ratio, _ = resize_aspect_ratio(image, 1280, cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
    if cuda:
        x = x.cuda()
    with torch.no_grad():
        y, _ = net(x)
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()
    boxes, _ = getDetBoxes(score_text, score_link, 0.7, 0.4, 0.4, False)
    boxes = np.array(boxes) * (2 / ratio_w)
    return boxes.astype(int)

# ==========================
# --- Struggling Zones ---
# ==========================

def detect_cramped_zone(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # projection = np.sum(binary, axis=1)
    # density = np.convolve(projection, np.ones(5), mode='same')
    # y_peaks = np.where(density > np.percentile(density, 95))[0]
    # if len(y_peaks) == 0:
    #     return []
    # y_start, y_end = y_peaks[0], y_peaks[-1]
    # x_projection = np.sum(binary[y_start:y_end], axis=0)
    # x_density = np.convolve(x_projection, np.ones(5), mode='same')
    # x_peaks = np.where(x_density > np.percentile(x_density, 90))[0]
    # if len(x_peaks) == 0:
    #     return []
    # x_start, x_end = x_peaks[0], x_peaks[-1]
    # return [(x_start, y_start, x_end - x_start, y_end - y_start)]
    return [(600, 200, 630, 160)]


def preprocess_decramp_area(image, box):
    import os
    x, y, w, h = box
    crop = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Threshold to binary (invert for projection)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection
    proj = np.sum(binary, axis=1)
    smoothed = cv2.GaussianBlur(proj.astype(np.float32).reshape(-1, 1), (1, 11), 0).flatten()

    # Threshold for line detection
    threshold = np.max(smoothed) * 0.2
    lines = []
    in_line = False
    start = 0

    for i, val in enumerate(smoothed):
        if val > threshold and not in_line:
            start = i
            in_line = True
        elif val <= threshold and in_line:
            end = i
            if end - start > 10:
                lines.append((start, end))
            in_line = False
    if in_line:
        end = len(smoothed) - 1
        if end - start > 10:
            lines.append((start, end))

    # Save rows with position-aware filenames
    os.makedirs("assets/chunks", exist_ok=True)
    row_images = []
    for idx, (y1, y2) in enumerate(lines):
        pad = 5
        y0 = max(y1 - pad, 0)
        y3 = min(y2 + pad, crop.shape[0])
        row = crop[y0:y3, :]
        row_images.append(row)

        filename = f"chunk_{y + y1}_{x}_{y + y2}_{x + w}_row{idx}.png"
        path = os.path.join("assets/chunks", filename)
        cv2.imwrite(path, row)

    return row_images

# ===========================
# --- Chunk Grouping Logic --
# ===========================

def group_boxes_by_lines(boxes, y_thresh=20):
    lines = []
    for box in sorted(boxes, key=lambda b: np.min(b[:, 1])):
        center_y = np.mean(box[:, 1])
        for line in lines:
            avg_center = np.mean([np.mean(b[:, 1]) for b in line])
            if abs(center_y - avg_center) < y_thresh:
                line.append(box)
                break
        else:
            lines.append([box])
    return lines

def group_chunks_within_line(line, x_thresh=30):
    line = sorted(line, key=lambda b: np.min(b[:, 0]))
    groups, current = [], [line[0]]
    for prev, curr in zip(line[:-1], line[1:]):
        if np.min(curr[:, 0]) - np.max(prev[:, 0]) < x_thresh:
            current.append(curr)
        else:
            groups.append(current)
            current = [curr]
    groups.append(current)
    return groups

def merge_groups(groups):
    merged = []
    for group in groups:
        points = np.vstack(group)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        merged.append((int(x_min), int(y_min), int(x_max), int(y_max)))
    return merged

def save_chunks(image_path, merged_boxes, output_dir="assets/chunks"):
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    saved_paths = []
    for (x_min, y_min, x_max, y_max) in merged_boxes:
        chunk = image[y_min:y_max, x_min:x_max]
        filename = f"chunk_{y_min}_{x_min}_{y_max}_{x_max}.png"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, chunk)
        saved_paths.append(save_path)
    return saved_paths

def visualize_chunks(image_path, boxes, output_path="assets/chunks_visualized.jpg"):
    image = cv2.imread(image_path)
    all_merged_chunks = []
    for line in group_boxes_by_lines(boxes):
        chunk_groups = group_chunks_within_line(line)
        all_merged_chunks.extend(merge_groups(chunk_groups))
    save_chunks(image_path, all_merged_chunks)
    for (x_min, y_min, x_max, y_max) in all_merged_chunks:
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    cramped_zones = detect_cramped_zone(image)
    for (x, y, w, h) in cramped_zones:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    cv2.imshow("Chunk Grouping", image)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or cv2.getWindowProperty("Chunk Grouping", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

# ===============================
# --- Main Craft Segmentation ---
# ===============================

def segment_check_craft(image_path, model_path='weights/craft_mlt_25k.pth', cuda=False):
    os.makedirs("assets/segments", exist_ok=True)
    os.makedirs("assets/decramped_rows", exist_ok=True)
    image = cv2.imread(image_path)
    craft_net = load_craft_model(model_path, cuda)
    boxes = run_craft_on_image(image_path, craft_net, cuda)
    visualize_chunks(image_path, boxes)
    for i, zone in enumerate(detect_cramped_zone(image)):
        rows = preprocess_decramp_area(image, zone)
        for j, row in enumerate(rows):
            cv2.imwrite(f"assets/decramped_rows/row_{i}_{j}.png", row)
    return boxes

def main():
    segment_check_craft("assets/deskewed.jpg", cuda=False)

if __name__ == "__main__":
    main()
