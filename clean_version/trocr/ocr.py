import os
import re
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re

def is_useful_text(text, confidence, min_conf=0.005):
    text = text.strip()
    if not text:
        return False

    # If confidence is abnormally low, discard unless text looks meaningful
    if confidence < min_conf:
        # Check for useful patterns: numbers, dates, currency, long words
        if re.fullmatch(r"[0-9,\.]+ ?[#\$]?", text):  # amount or currency
            return True
        if re.fullmatch(r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]?\d{2,4}", text):  # date-like
            return True
        if len(text) >= 4 and re.search(r"[a-zA-Z]", text):  # meaningful word
            return True
        return False
    
    # If not gibberish (at least one letter or digit)
    if not re.search(r"[a-zA-Z0-9]", text):
        return False

    # Too short and no structure
    if len(text) < 2 and confidence < 0.9:
        return False

    # Likely useful
    return True

# Initialize model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
model.eval()

# Force CPU or CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Folder with cropped chunks
chunks_dir = "../assets/chunks"
output_txt = "chunk_ocr_filtered.txt"
confidence_threshold = 0.85  # keep only predictions with >85% average token confidence

# Coordinate pattern: chunk_ymin_xmin_ymax_xmax.png
coord_pattern = re.compile(r"chunk_(\d+)_(\d+)_(\d+)_(\d+)\.png")

def calculate_confidence(logits, predicted_ids):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidences = []
    for i, token_id in enumerate(predicted_ids):
        if token_id != processor.tokenizer.pad_token_id:
            token_confidence = probs[0, i, token_id].item()
            confidences.append(token_confidence)
    return sum(confidences) / len(confidences) if confidences else 0.0

# Run OCR with filtering
with open(output_txt, "w", encoding="utf-8") as f_out:
    for filename in sorted(os.listdir(chunks_dir)):
        if not filename.endswith(".png"):
            continue

        match = coord_pattern.match(filename)
        if not match:
            continue
        y_min, x_min, y_max, x_max = map(int, match.groups())

        image_path = os.path.join(chunks_dir, filename)
        image = Image.open(image_path).convert("RGB")

        # Prepare input
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        # Get prediction + logits
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)
            output_ids = generated_ids.sequences[0]
            logits = model(pixel_values, decoder_input_ids=output_ids.unsqueeze(0)).logits

        # Decode prediction
        text = processor.batch_decode(output_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()

        # Confidence
        avg_conf = calculate_confidence(logits, output_ids)
        print(f"{filename} → Confidence: {avg_conf:.2%} → Text: {text}")

        if is_useful_text(text, avg_conf):
            f_out.write(f"{y_min},{x_min},{y_max},{x_max}\t{text}\n")
        else:
            print(f"[SKIPPED] Low-quality: {text} ({avg_conf:.2%})")
