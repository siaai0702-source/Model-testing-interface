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
    """Upload image to S3 and return key and S3 URI (not HTTPS URL)"""
    unique_filename = f"{uuid.uuid4()}.png"
    key = f"{UPLOAD_PREFIX}{unique_filename}"
    image_file.seek(0)
    s3_client.upload_fileobj(image_file, BUCKET_NAME, key, ExtraArgs={'ContentType': 'image/png'})
    
    # Return S3 URI format that the Lambda function expects
    s3_uri = f"s3://{BUCKET_NAME}/{key}"
    
    # Also generate a presigned URL for display purposes
    https_url = s3_client.generate_presigned_url('get_object', 
                                          Params={'Bucket': BUCKET_NAME, 'Key': key}, 
                                          ExpiresIn=604800)
    
    return key, s3_uri, https_url

def convert_https_to_s3_uri(https_url: str) -> str:
    """Convert HTTPS S3 URL to s3:// URI format"""
    try:
        parsed = urlparse(https_url)
        # Extract bucket and key from URL
        # Format: https://bucket-name.s3.region.amazonaws.com/key
        # or: https://s3.region.amazonaws.com/bucket-name/key
        
        if '.s3.amazonaws.com' in parsed.netloc:
            # Standard S3 URL format
            host_parts = parsed.netloc.split('.')
            bucket_name = host_parts[0]
            # Remove leading slash from path
            key = parsed.path.lstrip('/')
        elif 's3.' in parsed.netloc and 'amazonaws.com' in parsed.netloc:
            # Path-style URL
            path_parts = parsed.path.lstrip('/').split('/', 1)
            bucket_name = path_parts[0]
            key = path_parts[1] if len(path_parts) > 1 else ''
        else:
            # Custom domain or other format
            # Try to extract from the path
            path_parts = parsed.path.lstrip('/').split('/', 1)
            if len(path_parts) >= 2:
                bucket_name = path_parts[0]
                key = path_parts[1]
            else:
                # Fallback: use BUCKET_NAME from config
                bucket_name = BUCKET_NAME
                key = parsed.path.lstrip('/')
        
        # Remove query parameters from key
        if '?' in key:
            key = key.split('?')[0]
        
        return f"s3://{bucket_name}/{key}"
    except Exception as e:
        st.error(f"Error converting URL to S3 URI: {str(e)}")
        # Fallback: construct from known bucket
        return f"s3://{BUCKET_NAME}/user-uploads/unknown.png"

def invoke_bedrock_agent(user_input: str, session_id: str) -> str:
    """Invoke Bedrock Agent with proper error handling"""
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
    """Extract Request ID from agent response"""
    # Pattern for Request ID in various formats
    patterns = [
        r'Request ID[:\s]*\[?([a-zA-Z0-9\-]+)\]?',
        r'ID[:\s]*\[?([a-zA-Z0-9\-]+)\]?',
        r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})',  # UUID pattern
        r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'   # Another UUID pattern
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def is_generation_in_progress(text: str) -> bool:
    """Check if agent response indicates image generation started"""
    generation_keywords = [
        "generating",
        "processing",
        "request id",
        "unique id",
        "operation:",
        "tool to generate",
        "positive_prompt",
        "i will now call",
        "calling the",
        "please wait",
        "image generation",
        "design is being",
        "your design is being",
        "virtual try-on",
        "try-on"
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in generation_keywords)

def check_for_generated_image(unique_id: str):
    """Check S3 for generated image by unique ID"""
    try:
        # Try with .png extension first
        key = f"{OUTPUT_PREFIX}{unique_id}.png"
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        bytes_data = obj['Body'].read()
        return Image.open(BytesIO(bytes_data)), bytes_data
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            # Try without extension
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
    """Poll S3 for generated image with progress indicator"""
    start_time = time.time()
    with placeholder.container():
        status = st.empty()
        timer = st.empty()
        progress = st.progress(0)
        
        # Initial message
        status.markdown("### 🎨 Generating your design...")
        timer.markdown("**⏱ Starting the image generation process**")
        
        while time.time() - start_time < MAX_POLL_TIME:
            elapsed = int(time.time() - start_time)
            progress_percentage = min(elapsed / MAX_POLL_TIME, 1.0)
            progress.progress(progress_percentage)
            
            # Update timer
            timer.markdown(f"**⏱ Time elapsed: {elapsed} seconds**")
            
            # Check for image
            img, bytes_data = check_for_generated_image(unique_id)
            if img:
                # Success - clear placeholders and return image
                progress.empty()
                status.empty()
                timer.empty()
                return img, bytes_data
            
            # Update status message based on elapsed time
            if elapsed < 30:
                status.markdown("### 🎨 Generating your design... (Initializing)")
            elif elapsed < 60:
                status.markdown("### 🎨 Generating your design... (Processing)")
            else:
                status.markdown(f"### 🎨 Still generating... (This may take a few minutes)")
                timer.markdown(f"**⏱ Time elapsed: {elapsed} seconds**\n\n**Track ID:** `{unique_id}`")
            
            time.sleep(POLL_INTERVAL)
        
        # Timeout
        progress.empty()
        status.empty()
        timer.empty()
        return None, None

def display_image(img, max_width=600):
    """Display image with proper scaling"""
    w, h = img.size
    if w > max_width:
        ratio = max_width / w
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    st.image(img, use_column_width=False)

# -----------------------------------------------------
# SESSION STATE
# -----------------------------------------------------
def initialize_session_state():
    """Initialize session state variables"""
    if 'chats' not in st.session_state:
        st.session_state.chats = {}
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None
    if 'current_uploaded_image_uri' not in st.session_state:
        st.session_state.current_uploaded_image_uri = None
    if 'current_uploaded_image_url' not in st.session_state:
        st.session_state.current_uploaded_image_url = None

def create_new_chat():
    """Create a new chat session"""
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
    """Add a message to current chat"""
    if not st.session_state.current_chat_id:
        return
    
    chat = st.session_state.chats[st.session_state.current_chat_id]
    
    # Determine if user message includes image
    has_image = (role == 'user' and st.session_state.current_uploaded_image_uri is not None)
    display_content = content + (" 🖼️" if has_image else "")
    
    chat['messages'].append({
        'role': role,
        'content': content,  # Store original content
        'display_content': display_content,  # Store display content
        'image': image,
        'image_bytes': image_bytes,
        'has_image': has_image,
        'timestamp': datetime.now()
    })
    
    # Update chat title if this is the first user message
    if role == 'user' and len(chat['messages']) == 1:
        title = content.strip()[:40]
        chat['title'] = title + "..." if len(title) > 40 else title or "New Chat"

# -----------------------------------------------------
# MAIN UI
# -----------------------------------------------------
def main():
    st.set_page_config(
        page_title="SIA Fashion Design Agent", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("💬 Chat History")
        
        # New Chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            create_new_chat()
            st.rerun()
        
        st.divider()
        
        # Chat history list
        if st.session_state.chats:
            for chat_id, chat in sorted(
                st.session_state.chats.items(), 
                key=lambda x: x[1]['created_at'], 
                reverse=True
            ):
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
                            st.session_state.current_uploaded_image_uri = None
                            st.session_state.current_uploaded_image_url = None
                        st.rerun()
        else:
            st.info("Start a new chat!")
        
        st.divider()
        
        # Image Upload Section
        st.subheader("📎 Image Upload")
        st.markdown("Upload an image for editing, modification, or virtual try-on")
        
        uploaded = st.file_uploader(
            "Choose an image file", 
            type=["png", "jpg", "jpeg"], 
            key="uploader"
        )
        
        if uploaded:
            # Display uploaded image preview
            st.image(uploaded, caption="Uploaded Image Preview", width=200)
            
            # Clear image button
            if st.button("Clear Uploaded Image", use_container_width=True):
                st.session_state.current_uploaded_image_uri = None
                st.session_state.current_uploaded_image_url = None
                if st.session_state.current_chat_id:
                    current_chat = st.session_state.chats[st.session_state.current_chat_id]
                    current_chat['uploaded_image_uri'] = None
                    current_chat['uploaded_image_url'] = None
                st.rerun()
            
            # Upload to S3 if not already uploaded
            if st.button("Confirm & Upload to S3", type="primary", use_container_width=True):
                with st.spinner("Uploading to S3..."):
                    _, s3_uri, https_url = upload_image_to_s3(uploaded)
                    st.session_state.current_uploaded_image_uri = s3_uri
                    st.session_state.current_uploaded_image_url = https_url
                    
                    # Store in current chat
                    if st.session_state.current_chat_id:
                        current_chat = st.session_state.chats[st.session_state.current_chat_id]
                        current_chat['uploaded_image_uri'] = s3_uri
                        current_chat['uploaded_image_url'] = https_url
                    
                    st.success("✅ Image uploaded to S3 and ready!")
                    st.info(f"**S3 URI:** `{s3_uri}`")
                    st.rerun()
        
        # Show current image status
        if st.session_state.current_uploaded_image_uri:
            st.divider()
            st.markdown("**📎 Currently Attached Image:**")
            st.success("✅ Image is attached and ready")
            st.code(st.session_state.current_uploaded_image_uri, language="text")
            if st.button("Remove Attachment", key="remove_attachment"):
                st.session_state.current_uploaded_image_uri = None
                st.session_state.current_uploaded_image_url = None
                if st.session_state.current_chat_id:
                    current_chat = st.session_state.chats[st.session_state.current_chat_id]
                    current_chat['uploaded_image_uri'] = None
                    current_chat['uploaded_image_url'] = None
                st.rerun()
    
    # Main content area
    st.title("🎨 SIA Fashion Design Agent")
    st.markdown("*Create and edit fashion designs with AI*")
    
    # Create new chat if none exists
    if not st.session_state.current_chat_id:
        create_new_chat()
    
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    
    # Display chat history
    for msg in current_chat['messages']:
        with st.chat_message(msg['role']):
            # Display message content
            st.markdown(msg.get('display_content', msg['content']))
            
            # Display image if available
            if msg.get('image'):
                display_image(msg['image'])
                if msg.get('image_bytes'):
                    st.download_button(
                        "💾 Download Image", 
                        data=msg['image_bytes'],
                        file_name=f"design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                        mime="image/png"
                    )
    
    # User input
    chat_input_key = f"chat_input_{st.session_state.current_chat_id}"
    if prompt := st.chat_input("Describe your design or changes...", key=chat_input_key):
        # Prepare user input with image S3 URI if available
        if st.session_state.current_uploaded_image_uri:
            # Format the prompt with S3 URI (not HTTPS URL) for the agent
            user_input = f"IMAGE_S3_URI: {st.session_state.current_uploaded_image_uri}\nUSER_REQUEST: {prompt}"
            
            # Determine operation type based on user request
            prompt_lower = prompt.lower()
            if any(word in prompt_lower for word in ["model", "try on", "wear", "fit", "on model", "on person", "virtual try"]):
                user_input = f"Create a virtual try-on image using this garment: {st.session_state.current_uploaded_image_uri}\nDescription: {prompt}\nModel type: female"
            elif any(word in prompt_lower for word in ["edit", "modify", "change", "adjust", "recolor", "alter", "make it", "turn it"]):
                user_input = f"Edit this image: {st.session_state.current_uploaded_image_uri}\nChanges: {prompt}"
            else:
                # No image uploaded - use text-to-image
                user_input = prompt
        else:
            # No image uploaded - use text-to-image
            user_input = prompt
            user_input += "\nOPERATION_HINT: t2i (text-to-image)"
        
        # Add user message to chat
        add_message('user', prompt)
        
        # Display user message
        with st.chat_message("user"):
            display_text = prompt
            if st.session_state.current_uploaded_image_uri:
                display_text += " 🖼️"
            st.markdown(display_text)
        
        # Process with agent
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            poll_placeholder = st.empty()
            
            # Check for pending generation from previous messages
            pending_id = current_chat.get('pending_request_id')
            
            # If user confirms a pending generation
            if pending_id and any(word in prompt.lower() for word in ["ok", "generate", "yes", "show", "ready", "go", "continue"]):
                response_placeholder.markdown("🔄 Continuing with your image generation...")
                
                # Poll for the image
                image, img_bytes = poll_for_image(pending_id, poll_placeholder)
                
                if image:
                    # Success - display image
                    response_placeholder.success("✨ Your design is ready!")
                    display_image(image)
                    st.download_button(
                        "💾 Download Image", 
                        data=img_bytes,
                        file_name=f"design_{pending_id}.png", 
                        mime="image/png"
                    )
                    add_message('assistant', f"Here's your generated design with ID: {pending_id}", image, img_bytes)
                    
                    # Clear pending ID
                    current_chat['pending_request_id'] = None
                    
                    # Clear uploaded image if used
                    if st.session_state.current_uploaded_image_uri:
                        st.session_state.current_uploaded_image_uri = None
                        st.session_state.current_uploaded_image_url = None
                        current_chat['uploaded_image_uri'] = None
                        current_chat['uploaded_image_url'] = None
                else:
                    # Timeout or error
                    response_placeholder.error(f"⏰ Generation timed out or failed.\n\n**Track this ID:** `{pending_id}`")
                    add_message('assistant', f"Image generation timed out. Track ID: {pending_id}")
                
                return
            
            # Normal agent invocation
            with st.spinner("🤖 Processing your request..."):
                agent_response = invoke_bedrock_agent(user_input, current_chat['session_id'])
            
            # Debug: Show agent response
            # st.write(f"**Debug - Agent Response:** {agent_response}")
            
            if not agent_response:
                response_placeholder.error("No response from the agent. Please try again.")
                add_message('assistant', "Sorry, I couldn't process your request.")
                return
            
            # Check if generation is in progress
            if is_generation_in_progress(agent_response):
                # Extract request ID
                request_id = extract_request_id(agent_response)
                
                if request_id:
                    # Store pending request ID
                    current_chat['pending_request_id'] = request_id
                    
                    # Clear initial spinner
                    response_placeholder.empty()
                    
                    # Start polling for image
                    image, img_bytes = poll_for_image(request_id, poll_placeholder)
                    
                    if image:
                        # Success - display image
                        response_placeholder.success("✨ Your design is ready!")
                        display_image(image)
                        st.download_button(
                            "💾 Download Image", 
                            data=img_bytes,
                            file_name=f"design_{request_id}.png", 
                            mime="image/png"
                        )
                        add_message('assistant', f"Here's your generated design!", image, img_bytes)
                        
                        # Clear pending ID
                        current_chat['pending_request_id'] = None
                        
                        # Clear uploaded image if used
                        if st.session_state.current_uploaded_image_uri:
                            st.session_state.current_uploaded_image_uri = None
                            st.session_state.current_uploaded_image_url = None
                            current_chat['uploaded_image_uri'] = None
                            current_chat['uploaded_image_url'] = None
                    else:
                        # Ask user to confirm when ready
                        response_placeholder.markdown(
                            f"🔄 Your design is being generated!\n\n"
                            f"**Track ID:** `{request_id}`\n\n"
                            f"*When you're ready to see it, type 'ok', 'generate', or 'show me'*"
                        )
                        add_message('assistant', 
                                   f"Your design is being generated with ID: {request_id}. "
                                   f"Say 'ok' when you want to see it.")
                else:
                    # Generation started but no ID extracted
                    response_placeholder.markdown(
                        f"{agent_response}\n\n"
                        f"*Your image is being generated. Please wait...*"
                    )
                    add_message('assistant', agent_response)
            else:
                # Just a text response, no image generation
                response_placeholder.markdown(agent_response)
                add_message('assistant', agent_response)

if __name__ == "__main__":
    main()