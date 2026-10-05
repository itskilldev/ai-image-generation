import os
from datetime import datetime

import streamlit as st
import torch
from diffusers import DiffusionPipeline
from PIL import Image

# -----------------------------
# Config
# -----------------------------
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ID = "segmind/tiny-sd"

# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    return pipe

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="AI Image Generator", page_icon="🎨")
st.title("🎨 AI Image Generator")
st.write("Generate images from text prompts using Segmind's tiny-sd model.")

prompt = st.text_input("Enter your prompt", placeholder="e.g. a futuristic city at sunset")

with st.sidebar:
    st.header("Settings")
    num_inference_steps = st.slider(
        "Inference Steps", min_value=10, max_value=100, value=25, step=5
    )
    guidance_scale = st.slider(
        "Guidance Scale", min_value=1.0, max_value=15.0, value=7.5, step=0.5
    )

generate = st.button("Generate Image")

# -----------------------------
# Generate
# -----------------------------
if generate:
    if not prompt.strip():
        st.warning("Please enter a prompt before generating.")
    else:
        try:
            with st.spinner("Generating image... this may take a moment."):
                pipe = load_model()
                result = pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
                image = result.images[0]

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_{timestamp}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                image.save(filepath)

            st.success("Image generated successfully!")
            st.image(image, caption=prompt, use_column_width=True)

            with open(filepath, "rb") as f:
                st.download_button(
                    label="Download Image",
                    data=f,
                    file_name=filename,
                    mime="image/png",
                )

        except Exception as e:
            st.error(f"Error during image generation: {e}")
