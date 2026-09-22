from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

from langchain.chat_models import init_chat_model
model = init_chat_model("openai/gpt-oss-120b",model_provider="groq")

prompt = ChatPromptTemplate.from_messages([
  ("system",
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

("human",
"""
Extract information from this paragraph:

{paragraph}
""")
])

para = input("Enter the movie paragraph: ")

final_prompt = prompt.invoke(
    {"paragraph": para})


response = model.invoke(final_prompt)
print(response.content)