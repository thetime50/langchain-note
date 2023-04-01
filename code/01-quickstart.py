import os
import json5

cfg = json5.load(fp=open('../env/cfg.josn5', encoding="utf-8"))
os.environ["OPENAI_API_KEY"]=cfg.get('OPENAI_API_KEY')
# https://serpapi.com/manage-api-key
os.environ["SERPAPI_API_KEY"]=cfg.get('SERPAPI_API_KEY') 

'''
构建语言模型应用
'''

def llmFun():
    from langchain.llms import OpenAI
    llm = OpenAI(temperature=0.9) # temperature大更加随机
    text = "What would be a good company name for a company that makes colorful socks?"
    print(llm(text))

def promptTemplateFun():
    from langchain.prompts import PromptTemplate
    # 就是一个文本替换的模板
    prompt = PromptTemplate(
        input_variables=['product'],
        template="What is a good name for a company that makes {product}?",
    )
    print(prompt.format(product = 'colorful socks'))

def chainsFun():
    # 把prompt template 和llm 添加到工具链中 用run方法运行
    from langchain.prompts import PromptTemplate
    from langchain.llms import OpenAI
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    from langchain.chains import LLMChain
    chain = LLMChain(llm = llm, prompt=prompt)
    print(chain.run('colorful socks'))

def agentFun():
    from langchain.agents import load_tools
    from langchain.agents import initialize_agent
    from langchain.llms import OpenAI

    llm = OpenAI(temperature = 0.9)
    tools = load_tools(['serpapi','llm-math'], llm = llm) # 'llm-match工具要用到 llm 所以需要传入
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    print( agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?"))

def memoryFun():
    from langchain import OpenAI,ConversationChain
    llm = OpenAI(temperature = 0.9)
    conversation = ConversationChain(llm = llm, verbose=True)
    print(conversation.predict(input = 'Hi there!'))
    print(conversation.predict(input="I'm doing well! Just having a conversation with an AI."))

# llmFun()
# promptTemplateFun()
# chainsFun()
# agentFun()
memoryFun()

'''
构建聊天模型应用
'''

# 从聊天模型获取消息
# LangChain 目前支持的消息类型有AIMessage, HumanMessage, SystemMessage, 和ChatMessage
# ChatMessage接受任意角色参数。大多数时候，您只会处理HumanMessage、AIMessage和SystemMessage。

# AIMessage ai返回的信息
# HumanMessage 用户对话信息
# SystemMessage 应用后台提示定制信息
# ChatMessage ?
