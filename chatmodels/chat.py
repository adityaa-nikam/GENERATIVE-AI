from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model("openai/gpt-oss-120b",model_provider="groq")

response = model.invoke("tell me about myself")

print(response.content)


# temperature 0  to  1 -> MORE CREATIVE TASK THEN HIGH TEMPERATURE,
#  MORE LOGICAL TASK THEN LOW TEMPERATURE

# max_tokens = 20 -> generate a response with a maximum of 20 tokens

