from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List


import os
import json5

# pylint: disable=pointless-string-statement

# 输出数据格式化显示

cfg = json5.load(fp=open('../env/cfg.josn5', encoding="utf-8"))

os.environ["OPENAI_API_KEY"]=cfg.get('OPENAI_API_KEY')

model_name = 'text-davinci-003'
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)
def jokeFun():
    print('\n***jokeFun***')

    # Define your desired data structure.
    class Joke(BaseModel):
        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")
        
        # You can add custom validation logic easily with Pydantic.
        @validator('setup')
        def question_ends_with_question_mark(cls, field):
            if field[-1] != '?':
                raise ValueError("Badly formed question!")
            return field

    # And a query intented to prompt a language model to populate the data structure.
    joke_query = "Tell me a joke."

    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=Joke)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    _input = prompt.format_prompt(query=joke_query)

    output = model(_input.to_string())

    print(parser.parse(output))


def actorFun():
    print('\n***actorFun***')
    # Here's another example, but with a compound typed field.
    class Actor(BaseModel):
        name: str = Field(description="name of an actor")
        film_names: List[str] = Field(description="list of names of films they starred in")
            
    actor_query = "Generate the filmography for a random actor."

    parser = PydanticOutputParser(pydantic_object=Actor)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    _input = prompt.format_prompt(query=actor_query)

    output = model(_input.to_string())

    print("_input.to_string()",_input.to_string())
    print(parser.parse(output))

jokeFun()
actorFun()
