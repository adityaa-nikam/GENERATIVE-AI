from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()
from langchain.chat_models import init_chat_model
model = init_chat_model("openai/gpt-oss-120b",model_provider="groq")

class MovieInfo(BaseModel):
    title: str
    release_year: Optional[str]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    themes: Optional[List[str]]
    ratings: Optional[float]
    notable_features: Optional[List[str]]
    short_summary: str

parser = PydanticOutputParser(pydantic_object=MovieInfo)

prompt = ChatPromptTemplate.from_messages([
    ("system",
"""
Extract movie information from the given paragraph {format_instructions}
"""),
    ("human", "{paragraph}"),
])

para = input("Enter the movie paragraph: ")

final_prompt = prompt.invoke(
 {"paragraph": para,
 'format_instructions': parser.get_format_instructions()
 }
 )

response = model.invoke(final_prompt)
movie_data = parser.parse(response.content)
print(movie_data)