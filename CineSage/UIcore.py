from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
import streamlit as st

load_dotenv()

# -----------------------------
# Model
# -----------------------------
model = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq"
)

# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a professional Movie Information Extraction Assistant.

Your task:
Extract useful and relevant information from the given movie paragraph and present it in a clean format.

Rules:
- Do NOT add explanations.
- Do NOT add extra commentary.
- Follow the exact format.
- If information is missing, write NULL.
- Keep the summary short (2-3 lines max).
- Do NOT guess unknown facts.
- Extract only information mentioned in the paragraph.

Output Format:

Movie Title:
Release Year:
Genre:
Director:
Main Cast:
Setting/Location:
Plot:
Themes:
Ratings:
Notable Features:

Short Summary:
"""
    ),
    (
        "human",
        """
Extract information from this paragraph:

{paragraph}
"""
    )
])


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="CineSage",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 CineSage")
st.subheader("Movie Information Extractor")

st.write(
    "Enter a movie paragraph and extract the useful information from it."
)

# -----------------------------
# Input
# -----------------------------
para = st.text_area(
    "Enter the movie paragraph:",
    height=200,
    placeholder="Paste your movie paragraph here..."
)

# -----------------------------
# Extract Information
# -----------------------------
if st.button("Extract Information"):

    if para:
        final_prompt = prompt.invoke(
            {"paragraph": para}
        )

        response = model.invoke(final_prompt)

        st.subheader("Extracted Information")

        st.write(response.content)

    else:
        st.warning("Please enter a movie paragraph.")