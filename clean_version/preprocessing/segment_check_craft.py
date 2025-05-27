import os
import cv2
import torch
import numpy as np
from craft import CRAFT
from craft_utils import getDetBoxes
from imgproc import loadImage, normalizeMeanVariance

# Load CRAFT model
def load_craft_model(trained_model_path='weights/craft_mlt_25k.pth', cuda=False):
    net = CRAFT()
    net.load_state_dict(torch.load(trained_model_path, map_location='cuda' if cuda else 'cpu'))
    net.eval()
    if cuda:
        net = net.cuda()
    return net

# Resize helper
def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, _ = img.shape
    target_size = mag_ratio * max(height, width)
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)
    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation)

    # Pad to multiples of 32
    target_h32 = target_h if target_h % 32 == 0 else target_h + (32 - target_h % 32)
    target_w32 = target_w if target_w % 32 == 0 else target_w + (32 - target_w % 32)
    padded = np.zeros((target_h32, target_w32, 3), dtype=np.uint8)
    padded[:target_h, :target_w, :] = proc
    return padded, ratio, (target_w32, target_h32)

# Run CRAFT
def run_craft_on_image(image_path, net, cuda=False):
    image = loadImage(image_path)
    canvas_size = 1280
    mag_ratio = 1.5
    img_resized, target_ratio, _ = resize_aspect_ratio(image, canvas_size, cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y, _ = net(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    boxes, _ = getDetBoxes(score_text, 0.7, 0.4, 0.4)
    boxes = np.array(boxes) * (1 / ratio_w)
    return boxes.astype(int)

# Segment with CRAFT
def segment_check_craft(image_path, model_path='weights/craft_mlt_25k.pth', cuda=False):
    os.makedirs("assets/segments", exist_ok=True)
    image = cv2.imread(image_path)
    original = image.copy()

    craft_net = load_craft_model(model_path, cuda)
    boxes = run_craft_on_image(image_path, craft_net, cuda)

    segments = []
    for i, box in enumerate(boxes):
        x_min = np.min(box[:, 0])
        y_min = np.min(box[:, 1])
        x_max = np.max(box[:, 0])
        y_max = np.max(box[:, 1])
        roi = original[y_min:y_max, x_min:x_max]
        cv2.rectangle(original, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        segments.append({'roi': roi, 'position': (x_min, y_min, x_max - x_min, y_max - y_min)})
        cv2.imwrite(f"assets/segments/segment_{i}.jpg", roi)

    cv2.imwrite("assets/segmented_check.jpg", original)
    cv2.imshow("CRAFT Segmentation", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return original, segments

def main():
    image_path = "assets/preprocessed.jpg"
    segment_check_craft(image_path, cuda=False)

if __name__ == "__main__":
    main()
