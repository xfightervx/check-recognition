import os
import re
import torch
from PIL import Image,ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re



processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



image = Image.open("../assets/decramped_rows/row_0_1.png").convert("RGB")
image2 = ImageOps.invert(image)


pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
pixel_values2 = processor(images=image2, return_tensors="pt").pixel_values.to(device)


with torch.no_grad():
    generated_ids = model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)
    output_ids = generated_ids.sequences[0]
    logits = model(pixel_values, decoder_input_ids=output_ids.unsqueeze(0)).logits
    text = processor.batch_decode(output_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()
    generated_ids2 = model.generate(pixel_values2, output_scores=True, return_dict_in_generate=True)
    output_ids2 = generated_ids2.sequences[0]
    logits2 = model(pixel_values2, decoder_input_ids=output_ids2.unsqueeze(0)).logits
    text2 = processor.batch_decode(output_ids2.unsqueeze(0), skip_special_tokens=True)[0].strip()
    print(text)
    print(text2)