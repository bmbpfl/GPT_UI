import streamlit as st
import ollama
import os
import psutil
import subprocess

st.title("💬 Ollama ChatGPT-like UI | Streamed Responses")

# --- 🔍 Detect available memory and select a model automatically ---


def get_installed_models():
    
    try:
        result = subprocess.check_output(["ollama", "list"]).decode()
        models = [line.split()[0] for line in result.splitlines()[1:] if line.strip()]
        return models if models else ["llama3.2:latest"]
    except Exception:
        return ["llama3.2:latest"]

AVAILABLE_MODELS = get_installed_models()


default_model = st.sidebar.selectbox("Choose Model", AVAILABLE_MODELS)

model_name = st.sidebar.text_input("Model name:", default_model)

use_cpu = st.sidebar.checkbox("Force CPU mode (slower, but safer)", value=False)

if use_cpu:
    os.environ["OLLAMA_USE_CPU"] = "1"
else:
    os.environ.pop("OLLAMA_USE_CPU", None)

st.sidebar.write(f"🧩 Using model: `{model_name}`")

# --- 💬 Initialize chat history ---

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Display conversation history (skip system message)

for msg in st.session_state.messages[1:]:
    st.chat_message("user" if msg["role"] == "user" else "assistant").write(msg["content"])

# --- 🧠 Chat input ---

st.set_page_config(layout="wide")

user_input = st.chat_input("Type your message...")

if user_input:

    # Add user message to chat history

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Create a placeholder for streaming text

    response_placeholder = st.chat_message("assistant").empty()
    full_response = ""

    try:

        # Stream the model's response in real time

        with st.spinner("Generating response..."):
            stream = ollama.chat(
                model=model_name,
                messages=st.session_state.messages,
                stream=True,  # Enable token streaming
            )

            for chunk in stream:
                token = chunk["message"]["content"]
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)

        # Save the final message

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    except ollama._types.ResponseError:
        st.error("❌ Model too large for available memory. Try a smaller or quantized model.")
        st.stop()

    except Exception as e:
        st.error(f"⚠️ Unexpected error: {str(e)}")
