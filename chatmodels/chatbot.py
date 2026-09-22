from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
load_dotenv()

model = init_chat_model("openai/gpt-oss-120b",model_provider="groq") 

messages = [
    SystemMessage(content="You are a funny Ai Agent named Batman"),
]
print("____________Welcome to the Chatbot! Type '0' to exit____________")
while True:
    prompt = input("You : ")
    messages.append(HumanMessage(content=prompt))
    if prompt == '0':
        print("Exiting the chatbot. Goodbye!")
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot :",response.content)

print(messages)

#Problems With Your Current Short-Term Memory --------------->
#No role separation (no system / user / assistant distinction)
#Just raw strings → weak conversation structure
#Memory keeps growing infinitely
#Will hit token limit
#API cost increases over time
#Slower response as history grows
#No trimming mechanism
#No summarization of old chats
#Not production scalable
#No control over context window