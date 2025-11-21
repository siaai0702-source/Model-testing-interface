import streamlit as st
import boto3
import json
import base64
import time
from datetime import datetime
from io import BytesIO
from PIL import Image
import os

# -----------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------
ENDPOINT_NAME = "sia-lora-fashion-20251121-115224"
OUTPUT_DIR = "./generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# AWS Runtime client
runtime = boto3.client(
    "sagemaker-runtime",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)

# -----------------------------------------------------
# MAIN UI
# -----------------------------------------------------
def main():
    st.set_page_config(
        page_title="SIA",
        layout="centered"
    )

    st.title("Fashion Design Image Generator")
    st.markdown("Create AI-generated fashion designs using your prompts.")

    # Sidebar settings
    with st.sidebar:
        st.header("Generation Settings")
        num_inference_steps = st.slider("Inference Steps", 10, 50, 4)
        guidance_scale = st.slider("Guidance Scale", 1.0, 10.0, 7.5)
        lora_scale = st.slider("LoRA Scale", 0.1, 2.0, 1.5)
        seed = st.number_input("Seed (optional)", value=42)

    # Prompt Input
    prompt = st.text_input(
        "Enter your prompt:",
        placeholder="e.g., Technical flat sketch of a women's modern denim jacket"
    )

    generate_button = st.button("Generate Image")

    # -----------------------------------------------------
    # CALL SAGEMAKER ENDPOINT
    # -----------------------------------------------------
    if generate_button and prompt:
        try:
            st.info("Generating image... Please wait.")

            payload = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "height": 512,
        		"width": 512,
                "lora_scale": lora_scale,
                "seed": int(seed)
            }

            start_time = time.time()

            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload)
            )

            elapsed = time.time() - start_time
            response_body = json.loads(response["Body"].read().decode("utf-8"))
            images = response_body.get("images", [])

            st.success(f"Generated {len(images)} image(s) in {elapsed:.2f}s")

            for idx, img_data in enumerate(images):
                image_bytes = base64.b64decode(img_data["image"])
                img = Image.open(BytesIO(image_bytes))

                filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                img.save(filepath)

                st.image(img, caption=f"Result {idx + 1}")
                st.download_button(
                    label="Download Image",
                    data=image_bytes,
                    file_name=filename
                )

        except Exception as e:
            st.error(f"Generation failed: {e}")

    elif generate_button and not prompt:
        st.warning("Please enter a prompt before generating.")

if __name__ == "__main__":
    main()
