import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

#os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_..."

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    temperature=0.7,
    max_new_tokens=1024,
)
model = ChatHuggingFace(llm=llm)

response = model.invoke("TELL ME ABOUT INDIA")

print(response.content)