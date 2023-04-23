from  langchain.output_parsers import StructuredOutputParser,ResponseSchema
from langchain.prompts import PromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

import os
import json5

# pylint: disable=pointless-string-statement

cfg = json5.load(fp=open('../env/cfg.josn5', encoding="utf-8"))

os.environ["OPENAI_API_KEY"]=cfg.get('OPENAI_API_KEY')

# pylint pointless-string-statement


response_schemas = [
    ResponseSchema(name='answer',description="answer to the user's quesion"),
    ResponseSchema(name='source',description="source used to answer the user's question, should be a website.")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
def demo1Fun():
  global output_parser,format_instructions
  prompt = PromptTemplate(
      template="answer the users question as best as possibe.\n{format_instructions}\n{question}",
      input_variables=["question"],
      partial_variables={"format_instructions": format_instructions}
  )

  model = OpenAI(temperature=0)

  _input = prompt.format_prompt(question = "what's the capital of france")
  print('*_input', _input.to_string())
  '''
  answer the users question as best as possibe.The output should be a markdown code snippet formatted in the following schema:

  ```json
  {
    "answer": string  // answer to the user's quesion        
    "source": string  // source used to answer the user's question, should be a website.
  }
  ```
  what's the capital of france
  '''
  output = model(_input.to_string())
  print('*output', output)
  '''
  ```json
  {
          "answer": "Paris",
          "source": "https://en.wikipedia.org/wiki/France"
  }
  ```
  '''
  print(output_parser.parse(output))
  '''
  {'answer': 'Paris', 'source': 'https://en.wikipedia.org/wiki/France'}
  '''

def demo2Fun():
  global output_parser,format_instructions
  chat_model = ChatOpenAI(temperature=0)
  prompt = ChatPromptTemplate(
      messages=[
          HumanMessagePromptTemplate.from_template("answer the users question as best as possible.\n{format_instructions}\n{question}")  
      ],
      input_variables=["question"],
      partial_variables={"format_instructions": format_instructions}
  )
  _input = prompt.format_prompt(question="what's the capital of france")
  print("*_input", _input.to_string())
  output = chat_model(_input.to_messages())
  print("*output",output)
  output_parser.parse(output.content)

  
# demo1Fun()
demo2Fun()
