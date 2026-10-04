import streamlit as st
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered"
)

# -----------------------------
# Create output folder
# -----------------------------
os.makedirs("generated_images", exist_ok=True)


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    model_id = "segmind/tiny-sd"

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )

    pipe = pipe.to("cpu")

    return pipe


# -----------------------------
# App UI
# -----------------------------
st.title("🎨 AI Image Generator")
st.write("Generate an image from a text prompt using an open-source AI model.")

st.divider()

prompt = st.text_area(
    "Enter your prompt",
    placeholder="Example: A futuristic city at sunset, cinematic and highly detailed",
    height=120
)

col1, col2 = st.columns(2)

with col1:
    steps = st.slider(
        "Quality / Steps",
        min_value=5,
        max_value=30,
        value=15
    )

with col2:
    guidance = st.slider(
        "Prompt Guidance",
        min_value=1.0,
        max_value=12.0,
        value=7.5
    )


generate = st.button(
    "✨ Generate Image",
    use_container_width=True
)


# -----------------------------
# Generate Image
# -----------------------------
if generate:

    if not prompt.strip():
        st.warning("Please enter a prompt first.")

    else:

        try:

            with st.spinner("Generating image... This may take some time on CPU."):

                pipe = load_model()

                result = pipe(
                    prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance
                )

                image = result.images[0]

                # Save image
                file_path = "generated_images/generated_image.png"
                image.save(file_path)

            st.success("Image generated successfully!")

            st.image(
                image,
                caption="Generated Image",
                use_container_width=True
            )

            # Download button
            with open(file_path, "rb") as file:

                st.download_button(
                    label="⬇️ Download Image",
                    data=file,
                    file_name="generated_image.png",
                    mime="image/png",
                    use_container_width=True
                )

        except Exception as e:

            st.error(f"Error generating image: {e}")