from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.prebuilt import create_react_agent
from tools import tools

API_KEY = "77ff713fcbaf89e4dc2e623868e7e829.IUDHIW1kf1UdMB0f"
API_URL = "https://open.bigmodel.cn/api/paas/v4"
AI_NAME = "小智"

prompt_template = ChatPromptTemplate.from_messages([
    ("system", f"你是一个AI小助手，你叫{AI_NAME}，你可以帮助用户处理表格。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}"),
])

chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=API_URL,
    model="glm-4-flash",
    max_tokens=1024,
    temperature=0.7,
)

parser_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=API_URL,
    model="glm-3-turbo",
    max_tokens=1024,
    temperature=0.3,
)
def output_parser(output):
    parser_prompt = """
    请将输入的文本进行改写，文本表达尽可能的更自然。
    这是你需要改写的文本：{text}。只回复改写后的文本即可！
    """
    # call_tools返回dict{list[class]}
    text = output['messages'][-1].content
    return parser_model.invoke(parser_prompt.format(text=text))

# 定义一个基础链 = 固定提示词 + 语言大模型 + 输出解析器
model_with_tools = create_react_agent(chat_model, tools)
base_chain = prompt_template | model_with_tools | output_parser
# 定义一个记忆库
chat_history = ChatMessageHistory()
# 定义一个带记忆的链
chain_with_memory = RunnableWithMessageHistory(
    base_chain,
    lambda x: chat_history,
    input_messages_key="query",
    history_messages_key="chat_history",
)
def summary_memory(chain_input):
    memory = chat_history.messages
    if len(memory) >= 50:
        summary_prompt_template = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "请将上述聊天信息提炼为一条摘要信息，尽可能多地包含具体细节。"),
        ])
        summary_chain = summary_prompt_template | chat_model
        new_memory = summary_chain.invoke({"chat_history": memory})
        chat_history.clear() # 清空聊天记录
        chat_history.add_message(new_memory)
    return chain_input

# 定义一个可以总结记忆的链
chain_with_summary = summary_memory | chain_with_memory

# 封装
def chat_chain(query):
    return chain_with_summary.invoke(
        input={"query": query},
        config={"configurable": {"session_id": "unused"}},
    ).content

# 测试
# while True:
#     query = input("Q: ")
#     print("A:", chat_chain(query))
#     if query == "退出":
#         break