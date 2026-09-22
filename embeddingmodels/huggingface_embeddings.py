from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2" 
    #uv pip install sentence-transformers
)
texts = [
    "Hello this is Aditya Nikam",
    "Hello your name is YouTube",
    "And you all are very beautiful"
]
vector = embedding.embed_documents(texts)
print(vector)