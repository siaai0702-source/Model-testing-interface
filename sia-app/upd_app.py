import streamlit as st
import boto3
import time
import re
from datetime import datetime
from io import BytesIO
from PIL import Image
import os
import uuid
from botocore.exceptions import ClientError
from urllib.parse import urlparse

# -----------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------
AGENT_ID = os.getenv("AGENT_ID", "I7O3ST8K4M")
AGENT_ALIAS_ID = os.getenv("AGENT_ALIAS_ID", "TSTALIASID")
BUCKET_NAME = "sia-comfyui-async-379632383058-us-east-1"
UPLOAD_PREFIX = "user-uploads/"
OUTPUT_PREFIX = "output/"
MAX_POLL_TIME = 300
POLL_INTERVAL = 5

session = boto3.session.Session()
region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
bedrock_agent_runtime = session.client("bedrock-agent-runtime", region_name=region)
s3_client = session.client("s3", region_name=region)

# -----------------------------------------------------
# HELPERS
# -----------------------------------------------------
def upload_image_to_s3(image_file):
    """Upload image to S3 and return key and S3 URI"""
    unique_filename = f"{uuid.uuid4()}.png"
    key = f"{UPLOAD_PREFIX}{unique_filename}"
    image_file.seek(0)
    s3_client.upload_fileobj(image_file, BUCKET_NAME, key, ExtraArgs={'ContentType': 'image/png'})
    
    s3_uri = f"s3://{BUCKET_NAME}/{key}"
    
    https_url = s3_client.generate_presigned_url('get_object', 
                                          Params={'Bucket': BUCKET_NAME, 'Key': key}, 
                                          ExpiresIn=604800)
    
    return key, s3_uri, https_url

def convert_https_to_s3_uri(https_url: str) -> str:
    """Convert HTTPS S3 URL to s3:// URI format"""
    try:
        parsed = urlparse(https_url)
        if '.s3.amazonaws.com' in parsed.netloc:
            host_parts = parsed.netloc.split('.')
            bucket_name = host_parts[0]
            key = parsed.path.lstrip('/')
        elif 's3.' in parsed.netloc and 'amazonaws.com' in parsed.netloc:
            path_parts = parsed.path.lstrip('/').split('/', 1)
            bucket_name = path_parts[0]
            key = path_parts[1] if len(path_parts) > 1 else ''
        else:
            path_parts = parsed.path.lstrip('/').split('/', 1)
            if len(path_parts) >= 2:
                bucket_name = path_parts[0]
                key = path_parts[1]
            else:
                bucket_name = BUCKET_NAME
                key = parsed.path.lstrip('/')
        
        if '?' in key:
            key = key.split('?')[0]
        
        return f"s3://{bucket_name}/{key}"
    except Exception as e:
        st.error(f"Error converting URL to S3 URI: {str(e)}")
        return f"s3://{BUCKET_NAME}/user-uploads/unknown.png"

def invoke_bedrock_agent(user_input: str, session_id: str) -> str:
    """Invoke Bedrock Agent with error handling"""
    try:
        response = bedrock_agent_runtime.invoke_agent(
            agentId=AGENT_ID,
            agentAliasId=AGENT_ALIAS_ID,
            sessionId=session_id,
            inputText=user_input
        )
        agent_response = ""
        for event in response.get('completion', []):
            if 'chunk' in event and 'bytes' in event['chunk']:
                agent_response += event['chunk']['bytes'].decode('utf-8')
        return agent_response.strip()
    except Exception as e:
        st.error(f"Agent error: {str(e)}")
        return ""

def extract_request_id(text: str) -> str | None:
    """Improved extraction of request/unique ID (supports short 8-char and full UUID)"""
    patterns = [
        r'Request ID[:\s]*\[?([a-zA-Z0-9\-]+)\]?',
        r'ID[:\s]*\[?([a-zA-Z0-9\-]+)\]?',
        r'([a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12})',
        r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',
        r'([a-f0-9]{8})',  # Critical: catches Lambda's short unique_id
        r'unique_id["\s:]*([a-f0-9]{8})',
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def is_generation_in_progress(text: str) -> bool:
    """Check if agent indicates image generation started"""
    keywords = [
        "generating", "processing", "request id", "unique id", "operation:",
        "calling the", "image generation", "design is being", "virtual try-on",
        "try-on", "started generating", "now creating"
    ]
    return any(k in text.lower() for k in keywords)

def check_for_generated_image(unique_id: str):
    """Check S3 for generated image"""
    try:
        key = f"{OUTPUT_PREFIX}{unique_id}.png"
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        bytes_data = obj['Body'].read()
        return Image.open(BytesIO(bytes_data)), bytes_data
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            try:
                key = f"{OUTPUT_PREFIX}{unique_id}"
                obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
                bytes_data = obj['Body'].read()
                return Image.open(BytesIO(bytes_data)), bytes_data
            except:
                return None, None
        st.error(f"S3 Error: {str(e)}")
        return None, None

def poll_for_image(unique_id: str, placeholder):
    """Poll S3 with progress UI"""
    start_time = time.time()
    with placeholder.container():
        status = st.empty()
        timer = st.empty()
        progress = st.progress(0)
        
        status.markdown("### 🎨 Generating your design...")
        timer.markdown("**⏱ Starting the image generation process**")
        
        while time.time() - start_time < MAX_POLL_TIME:
            elapsed = int(time.time() - start_time)
            progress.progress(min(elapsed / MAX_POLL_TIME, 1.0))
            timer.markdown(f"**⏱ Time elapsed: {elapsed} seconds**")
            
            img, bytes_data = check_for_generated_image(unique_id)
            if img:
                progress.empty()
                status.empty()
                timer.empty()
                return img, bytes_data
            
            if elapsed < 30:
                status.markdown("### 🎨 Generating your design... (Initializing)")
            elif elapsed < 60:
                status.markdown("### 🎨 Generating your design... (Processing)")
            else:
                status.markdown(f"### 🎨 Still generating... (This may take a few minutes)")
                timer.markdown(f"**⏱ Time elapsed: {elapsed} seconds** | **Track ID:** `{unique_id}`")
            
            time.sleep(POLL_INTERVAL)
        
        progress.empty()
        status.empty()
        timer.empty()
        return None, None

def display_image(img, max_width=600):
    """Display image with scaling"""
    w, h = img.size
    if w > max_width:
        ratio = max_width / w
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    st.image(img, use_column_width=False)

# -----------------------------------------------------
# SESSION STATE
# -----------------------------------------------------
def initialize_session_state():
    if 'chats' not in st.session_state:
        st.session_state.chats = {}
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None
    if 'current_uploaded_image_uri' not in st.session_state:
        st.session_state.current_uploaded_image_uri = None
    if 'current_uploaded_image_url' not in st.session_state:
        st.session_state.current_uploaded_image_url = None

def create_new_chat():
    chat_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        'session_id': session_id,
        'messages': [],
        'created_at': datetime.now(),
        'title': 'New Chat',
        'pending_request_id': None,
        'uploaded_image_uri': None,
        'uploaded_image_url': None
    }
    st.session_state.current_chat_id = chat_id
    st.session_state.current_uploaded_image_uri = None
    st.session_state.current_uploaded_image_url = None
    return chat_id

def add_message(role: str, content: str, image=None, image_bytes=None):
    if not st.session_state.current_chat_id:
        return
    chat = st.session_state.chats[st.session_state.current_chat_id]
    has_image = (role == 'user' and st.session_state.current_uploaded_image_uri is not None)
    display_content = content + (" 🖼️" if has_image else "")
    
    chat['messages'].append({
        'role': role,
        'content': content,
        'display_content': display_content,
        'image': image,
        'image_bytes': image_bytes,
        'has_image': has_image,
        'timestamp': datetime.now()
    })
    
    if role == 'user' and len(chat['messages']) == 1:
        title = content.strip()[:40]
        chat['title'] = title + "..." if len(title) > 40 else title or "New Chat"

# -----------------------------------------------------
# MAIN UI
# -----------------------------------------------------
def main():
    st.set_page_config(page_title="SIA Fashion Design Agent", layout="wide", initial_sidebar_state="expanded")
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("💬 Chat History")
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            create_new_chat()
            st.rerun()
        
        st.divider()
        
        if st.session_state.chats:
            for chat_id, chat in sorted(st.session_state.chats.items(), key=lambda x: x[1]['created_at'], reverse=True):
                active = chat_id == st.session_state.current_chat_id
                col1, col2 = st.columns([4, 1])
                with col1:
                    btn_text = f"{'🟢' if active else '⚪'} {chat['title']}"
                    if st.button(btn_text, key=f"chat_{chat_id}", use_container_width=True):
                        st.session_state.current_chat_id = chat_id
                        st.session_state.current_uploaded_image_uri = chat.get('uploaded_image_uri')
                        st.session_state.current_uploaded_image_url = chat.get('uploaded_image_url')
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{chat_id}"):
                        del st.session_state.chats[chat_id]
                        if st.session_state.current_chat_id == chat_id:
                            st.session_state.current_chat_id = None
                            create_new_chat()
                        st.rerun()
        else:
            st.info("Start a new chat!")
        
        st.divider()
        st.subheader("📎 Image Upload")
        st.markdown("Upload an image for editing or virtual try-on")
        
        uploaded = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"], key="uploader")
        
        if uploaded:
            st.image(uploaded, caption="Uploaded Image Preview", width=200)
            if st.button("Clear Uploaded Image", use_container_width=True):
                st.session_state.current_uploaded_image_uri = None
                st.session_state.current_uploaded_image_url = None
                if st.session_state.current_chat_id:
                    st.session_state.chats[st.session_state.current_chat_id]['uploaded_image_uri'] = None
                    st.session_state.chats[st.session_state.current_chat_id]['uploaded_image_url'] = None
                st.rerun()
            
            if st.button("Confirm & Upload to S3", type="primary", use_container_width=True):
                with st.spinner("Uploading to S3..."):
                    _, s3_uri, https_url = upload_image_to_s3(uploaded)
                    st.session_state.current_uploaded_image_uri = s3_uri
                    st.session_state.current_uploaded_image_url = https_url
                    if st.session_state.current_chat_id:
                        current_chat = st.session_state.chats[st.session_state.current_chat_id]
                        current_chat['uploaded_image_uri'] = s3_uri
                        current_chat['uploaded_image_url'] = https_url
                    st.success("✅ Image uploaded and ready!")
                    st.info(f"**S3 URI:** `{s3_uri}`")
                    st.rerun()
        
        if st.session_state.current_uploaded_image_uri:
            st.divider()
            st.markdown("**📎 Currently Attached Image:**")
            st.success("✅ Image attached")
            st.code(st.session_state.current_uploaded_image_uri, language="text")
            if st.button("Remove Attachment", key="remove_attachment"):
                st.session_state.current_uploaded_image_uri = None
                st.session_state.current_uploaded_image_url = None
                if st.session_state.current_chat_id:
                    chat = st.session_state.chats[st.session_state.current_chat_id]
                    chat['uploaded_image_uri'] = None
                    chat['uploaded_image_url'] = None
                st.rerun()
    
    # Main content
    st.title("🎨 SIA Fashion Design Agent")
    st.markdown("*Create and edit fashion designs with AI*")
    
    if not st.session_state.current_chat_id:
        create_new_chat()
    
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    
    # Display messages
    for msg in current_chat['messages']:
        with st.chat_message(msg['role']):
            st.markdown(msg.get('display_content', msg['content']))
            if msg.get('image'):
                display_image(msg['image'])
                if msg.get('image_bytes'):
                    st.download_button(
                        "💾 Download Image",
                        data=msg['image_bytes'],
                        file_name=f"design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
    
    # Chat input
    chat_input_key = f"chat_input_{st.session_state.current_chat_id}"
    if prompt := st.chat_input("Describe your design or changes...", key=chat_input_key):
        # Build proper input for agent
        user_input = prompt
        
        if st.session_state.current_uploaded_image_uri:
            s3_uri = st.session_state.current_uploaded_image_uri
            user_input = f'{prompt}\n[uploads image url] "{s3_uri}"'
            
            prompt_lower = prompt.lower()
            if any(word in prompt_lower for word in ["model", "try on", "wear", "fit", "on model", "on person", "virtual try", "show on"]):
                user_input += "\nOperation: vto (virtual try-on on a female model)"
            elif any(word in prompt_lower for word in ["edit", "modify", "change", "adjust", "recolor", "alter", "make it", "turn it"]):
                user_input += "\nOperation: i2i (image edit)"
            else:
                user_input += "\nOperation: t2i (new design)"
        else:
            user_input += "\nOperation: t2i (new design)"
        
        add_message('user', prompt)
        
        with st.chat_message("user"):
            display_text = prompt + (" 🖼️" if st.session_state.current_uploaded_image_uri else "")
            st.markdown(display_text)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            poll_placeholder = st.empty()
            
            pending_id = current_chat.get('pending_request_id')
            if pending_id and any(word in prompt.lower() for word in ["ok", "generate", "yes", "show", "ready", "go", "continue"]):
                response_placeholder.markdown("🔄 Continuing with your image generation...")
                image, img_bytes = poll_for_image(pending_id, poll_placeholder)
                if image:
                    response_placeholder.success("✨ Your design is ready!")
                    display_image(image)
                    st.download_button("💾 Download Image", data=img_bytes, file_name=f"design_{pending_id}.png", mime="image/png")
                    add_message('assistant', "Here's your generated design!", image, img_bytes)
                    current_chat['pending_request_id'] = None
                    st.session_state.current_uploaded_image_uri = None
                    st.session_state.current_uploaded_image_url = None
                else:
                    response_placeholder.error(f"⏰ Generation timed out.\n**Track ID:** `{pending_id}`")
                    add_message('assistant', f"Image generation timed out. Track ID: {pending_id}")
                return
            
            with st.spinner("🤖 Processing your request..."):
                agent_response = invoke_bedrock_agent(user_input, current_chat['session_id'])
            
            if not agent_response:
                response_placeholder.error("No response from agent.")
                add_message('assistant', "Sorry, I couldn't process your request.")
                return
            
            if is_generation_in_progress(agent_response):
                request_id = extract_request_id(agent_response)
                if request_id:
                    current_chat['pending_request_id'] = request_id
                    response_placeholder.empty()
                    image, img_bytes = poll_for_image(request_id, poll_placeholder)
                    if image:
                        response_placeholder.success("✨ Your design is ready!")
                        display_image(image)
                        st.download_button("💾 Download Image", data=img_bytes, file_name=f"design_{request_id}.png", mime="image/png")
                        add_message('assistant', "Here's your generated design!", image, img_bytes)
                        current_chat['pending_request_id'] = None
                        st.session_state.current_uploaded_image_uri = None
                        if st.session_state.current_chat_id:
                            st.session_state.chats[st.session_state.current_chat_id]['uploaded_image_uri'] = None
                    else:
                        response_placeholder.markdown(
                            f"🔄 Your design is being generated!\n\n**Track ID:** `{request_id}`\n\n"
                            f"*Type 'ok' or 'show' when ready to see it*"
                        )
                        add_message('assistant', f"Design generating (ID: {request_id}). Say 'ok' when ready.")
                else:
                    response_placeholder.markdown(f"{agent_response}\n\n*Generation in progress...*")
                    add_message('assistant', agent_response)
            else:
                response_placeholder.markdown(agent_response)
                add_message('assistant', agent_response)

if __name__ == "__main__":
    main()