from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr

# 1) Move model to device
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.eval()

# 2) Fixed caption function
def generate_caption(img):
    # Gradio gives you a numpy array by default
    img_pil = Image.fromarray(img).convert("RGB")
    # Preprocess & send to device
    inputs = processor(images=img_pil, return_tensors="pt").to(device)
    # Generate token IDs
    out_ids = model.generate(**inputs, max_new_tokens=32)
    # Decode into a string
    caption = processor.decode(out_ids[0], skip_special_tokens=True).strip()
    return caption

# 3) Build the Gradio interface
demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(label="Upload Image"),
    outputs=gr.Textbox(label="Caption"),
    title="BLIP-v1 Caption Generator",
    description="Upload an image and get a caption from BLIP-v1.",
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)