import os
import json5

cfg = json5.load(fp=open('../env/cfg.josn5', encoding="utf-8"))

os.environ["OPENAI_API_KEY"]=cfg.get('OPENAI_API_KEY')
# # https://serpapi.com/manage-api-key
os.environ["SERPAPI_API_KEY"]=cfg.get('SERPAPI_API_KEY') 
# https://cse.google.com/cse/create/new
os.environ["GOOGLE_API_KEY"]=cfg.get('GOOGLE_API_KEY') 
# https://developers.google.com/custom-search/v1/overview?hl=zh-cn
os.environ["GOOGLE_CSE_ID"]=cfg.get('GOOGLE_CSE_ID') 

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
    from langchain.llms import OpenAI,OpenAIChat

    # llm = OpenAI(temperature = 0.9) 
    # llm = OpenAI(temperature=0,model_name='gpt-3.5-turbo') # Could not parse LLM output: 调用不同的模型 一些格式和规则可能会不一样，会触发报错 
    llm = OpenAIChat(temperature=0.5,model_name='gpt-3.5-turbo') # 会出现概率性的报错
    # tools = load_tools(['serpapi','llm-math'], llm = llm) # 'llm-match工具要用到 llm 所以需要传入
    tools = load_tools(['google-search-results-json','llm-math'], llm = llm) # 'llm-match工具要用到 llm 所以需要传入
    
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    # print( agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?"))

    def run_test(text,state):
        res = agent.run(text)
        state = state + [(text,res)]
        return state,state
    import gradio as gr
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot", label="rpa ChatGPT")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(
                    container=False)
        txt.submit(run_test, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        demo.launch(server_name="127.0.0.1", server_port=7869)

def memoryFun():
    from langchain import OpenAI,ConversationChain
    llm = OpenAI(temperature = 0.9)
    conversation = ConversationChain(llm = llm, verbose=True)
    print(conversation.predict(input = 'Hi there!'))
    print(conversation.predict(input="I'm doing well! Just having a conversation with an AI."))

# llmFun()
# promptTemplateFun()
# chainsFun()
agentFun()
# memoryFun()

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


def messateCompletions():
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )
    chat = ChatOpenAI(temperature = 0.9)
    print(chat([
        HumanMessage(content="Translate this sentence from English to French. I love programming.")
    ]) )
    print('\n**********************\n')
    print(chat([
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="Translate this sentence from English to French. I love programming."),
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="Translate this sentence from English to French. I love programming.")
    ]))

# messateCompletions()
