import streamlit as st
import boto3
import json
import base64
import time
from datetime import datetime
from io import BytesIO
from PIL import Image
import os
import uuid
from botocore.exceptions import ClientError

# -----------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------
ENDPOINT_NAME = "Sia-FD-01-endpoint"
INPUT_BUCKET = "sia-comfyui-async-379632383058-us-east-1"
T2I_WORKFLOW_PATH = "workflow/original-t2i.json"
I2I_WORKFLOW_PATH = "workflow/IMG2IMG.json"
VTO_WORKFLOW_PATH = "workflow/VTRO.json"
OUTPUT_DIR = "./generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Placeholder paths for internal model images (replace with actual paths)
MALE_MODEL_PATH = "sia-app/model_images/05584261250-a1.jpg"  # Replace with actual male model image path
FEMALE_MODEL_PATH = "sia-app/model_images/pexels-jimmyelizarraras-20069973.jpg"  # Replace with actual female model image path

# AWS clients
runtime = boto3.client(
    "sagemaker-runtime",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)

# -----------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------
def load_workflow_from_s3(workflow_key):
    """Load workflow JSON from S3"""
    try:
        response = s3_client.get_object(Bucket=INPUT_BUCKET, Key=workflow_key)
        workflow = json.loads(response['Body'].read().decode('utf-8'))
        print(f" Loaded workflow from s3://{INPUT_BUCKET}/{workflow_key}")
        return workflow
    except Exception as e:
        raise Exception(f"Failed to load workflow {workflow_key}: {str(e)}")

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def encode_uploaded_image(uploaded_file) -> str:
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

def customize_t2i_workflow(workflow: dict, params: dict) -> dict:
    # Positive prompt (node 12)
    if "prompt" in params:
        workflow["12"]["inputs"]["text"] = params["prompt"]
    
    return workflow

def customize_i2i_workflow(workflow: dict, params: dict, image_filename: str) -> dict:
    workflow["17"]["inputs"]["image"] = image_filename
    
    if "prompt" in params:
        workflow["19"]["inputs"]["prompt"] = params["prompt"]
    
    if "negative_prompt" in params:
        workflow["15"]["inputs"]["prompt"] = params["negative_prompt"]
    
    if "denoise" in params:
        workflow["5"]["inputs"]["denoise"] = params["denoise"]
    
    # Seed (node 5) - optional
    if "seed" in params:
        workflow["5"]["inputs"]["seed"] = params["seed"]
    
    return workflow

def customize_vto_workflow(workflow: dict, params: dict, product_filename: str, model_filename: str) -> dict:
    # Set product image (node 17 - the garment/product)
    workflow["17"]["inputs"]["image"] = product_filename
    
    # Set model image (node 27 - the person wearing the garment)
    workflow["27"]["inputs"]["image"] = model_filename
    
    # Positive prompt (node 19)
    if "prompt" in params:
        workflow["19"]["inputs"]["prompt"] = params["prompt"]

    # Negative prompt (node 15)
    if "negative_prompt" in params:
        workflow["15"]["inputs"]["prompt"] = params["negative_prompt"]

    if "seed" in params:
        workflow["11"]["inputs"]["seed"] = params["seed"]
    
    if "denoise" in params:
        workflow["11"]["inputs"]["denoise"] = params["denoise"]
    
    if "cfg" in params:
        workflow["11"]["inputs"]["cfg"] = params["cfg"]
    
    if "steps" in params:
        workflow["11"]["inputs"]["steps"] = params["steps"]
    
    return workflow

def poll_for_result(bucket, key, max_wait=1800, poll_interval=5):
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                raise
        time.sleep(poll_interval)
    raise TimeoutError("Timed out waiting for result")

def parse_s3_uri(uri):
    parts = uri.replace("s3://", "").split("/")
    bucket = parts[0]
    key = "/".join(parts[1:])
    return bucket, key

def poll_for_generated_image(bucket, unique_id, max_wait=1800, poll_interval=5):
    """
    Poll S3 for output/{unique_id}.png
    """
    start_time = time.time()
    image_key = f"output/{unique_id}.png"

    while time.time() - start_time < max_wait:
        try:
            response = s3_client.get_object(
                Bucket=bucket,
                Key=image_key
            )
            image_bytes = response["Body"].read()
            img = Image.open(BytesIO(image_bytes))
            return img, image_bytes

        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchKey":
                raise

        time.sleep(poll_interval)

    raise TimeoutError("Timed out waiting for generated image")

# -----------------------------------------------------
# MAIN UI
# -----------------------------------------------------
def main():
    st.set_page_config(
        page_title="SIA",
        layout="centered"
    )

    st.title("Fashion Design Image Generator")
    st.markdown("Create AI-generated fashion designs using text-to-image, image-to-image, or virtual try-on.")

    # Sidebar settings
    with st.sidebar:
        st.header("Generation Settings")
        mode = st.selectbox("Mode", ["Text-to-Image (t2i)", "Image-to-Image (i2i)", "Virtual Try-On (vto)"])
        # steps = st.slider("Inference Steps", 10, 50, 4)
        # guidance_scale = st.slider("Guidance Scale (CFG)", 1.0, 10.0, 7.5)
        # lora_scale = st.slider("LoRA Scale", 0.1, 2.0, 1.5)
        # if mode in ["Image-to-Image (i2i)", "Virtual Try-On (vto)"]:
        #     denoise = st.slider("Denoise Strength", 0.1, 1.0, 0.75)
        # seed = st.number_input("Seed (optional)", value=42)
        if mode == "Text-to-Image (t2i)":
            width = st.number_input("Width", value=1024)
            height = st.number_input("Height", value=1024)

    # Prompt Inputs
    prompt = st.text_area("Positive Prompt", placeholder="e.g., A high-quality deep red cotton T-shirt...")
    negative_prompt = st.text_area("Negative Prompt", placeholder="e.g., blur, distortion, artifacts")

    # Mode-specific inputs
    uploaded_image = None
    model_image_b64 = None
    product_image_b64 = None
    if mode == "Image-to-Image (i2i)":
        uploaded_image = st.file_uploader("Upload Input Image", type=["png", "jpg", "jpeg"])
    elif mode == "Virtual Try-On (vto)":
        uploaded_image = st.file_uploader("Upload Product Image", type=["png", "jpg", "jpeg"])
        gender = st.selectbox("Select Model Gender", ["Female", "Male"])
        model_path = FEMALE_MODEL_PATH if gender == "Female" else MALE_MODEL_PATH
        if os.path.exists(model_path):
            model_image_b64 = encode_image(model_path)
        else:
            st.error(f"Model image not found at {model_path}")
            return

    generate_button = st.button("Generate Image")

    # -----------------------------------------------------
    # PROCESS GENERATION
    # -----------------------------------------------------
    if generate_button:
        if not prompt:
            st.warning("Please enter a positive prompt.")
            return

        MODE_TO_WORKFLOW = {
            "Text-to-Image (t2i)": "t2i",
            "Image-to-Image (i2i)": "i2i",
            "Virtual Try-On (vto)": "vto"
        }

        workflow_type = MODE_TO_WORKFLOW[mode]

        unique_id = str(uuid.uuid4())
        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt
        }
        if mode == "Text-to-Image (t2i)":
            params["width"] = width
            params["height"] = height
  

        try:
            st.info("Preparing workflow...")

            if workflow_type == "t2i":
                workflow = load_workflow_from_s3(T2I_WORKFLOW_PATH)
                workflow = customize_t2i_workflow(workflow, params)
            elif workflow_type == "i2i":
                if not uploaded_image:
                    st.warning("Please upload an input image for i2i.")
                    return
                image_b64 = encode_uploaded_image(uploaded_image)
                image_filename = f"{unique_id}.png"
                workflow = load_workflow_from_s3(I2I_WORKFLOW_PATH)
                workflow = customize_i2i_workflow(workflow, params, image_filename)
                workflow["uploads"] = {image_filename: image_b64}
            elif workflow_type == "vto":
                if not uploaded_image:
                    st.warning("Please upload a product image for vto.")
                    return
                product_image_b64 = encode_uploaded_image(uploaded_image)
                product_filename = f"{unique_id}_product.png"
                model_filename = f"{unique_id}_model.png"
                workflow = load_workflow_from_s3(VTO_WORKFLOW_PATH)
                workflow = customize_vto_workflow(workflow, params, product_filename, model_filename)
                workflow["uploads"] = {
                    product_filename: product_image_b64,
                    model_filename: model_image_b64
                }

            workflow["request_id"] = unique_id

            payload = json.dumps(workflow).encode("utf-8")
            input_key = f"inputs/{workflow_type}-{unique_id}.json"

            s3_client.put_object(
                Bucket=INPUT_BUCKET,
                Key=input_key,
                Body=payload,
                ContentType="application/json"
            )

            input_location = f"s3://{INPUT_BUCKET}/{input_key}"

            st.info("Invoking endpoint asynchronously...")

            response = runtime.invoke_endpoint_async(
                EndpointName=ENDPOINT_NAME,
                InputLocation=input_location,
                ContentType="application/json",
                InvocationTimeoutSeconds=1800,
            )
            
            st.info("Waiting for generated image...")

            try:
                img, image_bytes = poll_for_generated_image(
                    bucket=INPUT_BUCKET,
                    unique_id=unique_id
                )

                filename = f"{unique_id}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                img.save(filepath)

                st.success("Image generated successfully!")
                st.image(img, caption="Generated Image", width=420)

                st.download_button(
                    label="Download Image",
                    data=image_bytes,
                    file_name=filename,
                    mime="image/png"
                )

            except TimeoutError:
                st.error("Image generation timed out. Please try again.")
        except Exception as e:
            st.error(f"Generation failed: {e}")

if __name__ == "__main__":
    main()