import streamlit as st
import requests
from PIL import Image
import io

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
        /* Fixed Header */
        .fixed-header {
            position: fixed;
            top: 56px;
            left: 0;
            width: 100%;
            background: linear-gradient(to right, #6a11cb, #2575fc);
        
            text-align: center;
            color: white;
            font-weight: bold;
            font-size: 24px;
            z-index: 999;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        }

        /* Adjust content padding so it doesn't overlap with the header */
        .stApp {
            padding-top: 80px;
        }

        /* Chat message container */
        .chat-container {
            display: flex;
            flex-direction: column;
        }

        /* User message (aligned right) */
        .user-message {
            align-self: flex-end;
            background-color: #0078ff;
            color: white;
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 70%;
            margin: 5px;
        }

        /* Assistant message (aligned left) */
        .assistant-message {
            align-self: flex-start;
            background-color: #f1f3f6;
            color: black;
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 70%;
            margin: 5px;
        }

        /* Image Styling (small size, left-aligned) */
        .assistant-image {
            align-self: flex-start;
            max-width: 40%;
            margin: 5px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
        }

    </style>
""", unsafe_allow_html=True)

# --- Fixed Title at the Top ---
st.markdown('<div class="fixed-header">🚢 Titanic Chatbot<br><span style="font-size: 14px; font-weight: normal;">💬 Ask me anything related to the Titanic dataset!</span></div>', unsafe_allow_html=True)

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chat History Section (Scrollable) ---
chat_container = st.container()

with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
        elif isinstance(msg["content"], Image.Image):  # If message is an image
            st.image(msg["content"], use_column_width=False, width=200)  # Smaller image
        else:
            st.markdown(f'<div class="assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- Input Box (Fixed at Bottom) ---
question = st.chat_input("📝 Type your message and press Enter...")

# --- Process User Input ---
if question:
    # Store & Display User Message
    st.session_state.messages.append({"role": "user", "content": question})

    with chat_container:
        st.markdown(f'<div class="user-message">{question}</div>', unsafe_allow_html=True)

    with st.spinner("🔎 Thinking..."):
        try:
            response = requests.post("http://127.0.0.1:8000/query/", json={"question": question})

            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")

                # Handle Image Response
                if content_type.startswith("image"):
                    img = Image.open(io.BytesIO(response.content))
                    st.session_state.messages.append({"role": "assistant", "content": img})

                    with chat_container:
                        st.image(img, caption="📊 Generated Visualization", use_container_width=False, width=200)

                # Handle Text Response
                else:
                    data = response.json()
                    answer = data.get("answer", "No answer found.")
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    with chat_container:
                        st.markdown(f'<div class="assistant-message">{answer}</div>', unsafe_allow_html=True)

            else:
                error_msg = f"❌ Error {response.status_code}: Unable to process request."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

                with chat_container:
                    st.markdown(f'<div class="assistant-message">{error_msg}</div>', unsafe_allow_html=True)

        except requests.exceptions.RequestException as e:
            error_msg = f"⚠️ Network Error: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

            with chat_container:
                st.markdown(f'<div class="assistant-message">{error_msg}</div>', unsafe_allow_html=True)
