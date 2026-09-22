# query embeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-large',
    dimensions=64
)

vector = embeddings.embed_query("You are going to learn Gen AI")

print(vector)




# multiple document embeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-large',
    dimensions=64
)

texts = [
    "You are going to learn Gen AI",
    "You are going to learn LangChain",
    "You are going to learn LLMs"
]

vectors = embeddings.embed_documents(texts)

print(vector)