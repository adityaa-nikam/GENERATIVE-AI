import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# -----------------------------
# Model
# -----------------------------
model = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq"
)

# -----------------------------
# Page UI
# -----------------------------
st.set_page_config(
    page_title="Batman AI",
    page_icon="🦇",
    layout="centered"
)

st.title("🦇 Batman AI")
st.caption("Your funny AI Agent")

# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content="You are a funny Ai Agent named Batman"
        )
    ]

# -----------------------------
# Display previous messages
# -----------------------------
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# -----------------------------
# User Input
# -----------------------------
prompt = st.chat_input("Talk to Batman...")

if prompt:

    # Same functionality as your terminal version
    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    # Display user message immediately
    with st.chat_message("user"):
        st.write(prompt)

    # Get response from model
    response = model.invoke(
        st.session_state.messages
    )

    # Store AI response
    st.session_state.messages.append(
        AIMessage(content=response.content)
    )

    # Display AI response
    with st.chat_message("assistant"):
        st.write(response.content)