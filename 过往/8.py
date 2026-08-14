import os

from dotenv import load_dotenv
from openai import OpenAI
from typing import TypedDict
from langgraph.graph import StateGraph, START, END



load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("DEEPSEEK_API_KEY is not set in the environment variables.")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

SYSTEM_PROMPT = """
你是一名亚马逊电商运营助手。

你的主要工作：
1. 编写和优化亚马逊商品标题、五点描述和商品详情。
2. 根据商品资料提炼卖点和关键词。
3. 输出内容时结构清楚、语言自然。
4. 如果商品信息不足，先向用户询问，不要随意编造。
5. 默认使用中文解释；需要生成英文商品文案时，提供英文文案和中文说明。
"""

MAX_ROUNDS = 3

class AgentState(TypedDict):
    messages: list
    task_type: str
    answer: str
#这是langgraph的工作状态图创建

#任务分类器的函数，变量是会话文本
def detect_task(conversation):
    recent = conversation[-(MAX_ROUNDS*2):]
#最大轮次乘以2的段落现在输入的，  文本切片
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
                "role":"system",
                "content":"""
你是一个任务分类器

请结合上下文，判断最后一条用户消息属于哪种任务。

判断时重点关注最后一条用户消息，并利用前面的对话理解“它”“继续修改”“短一点”等指代。
不要根据助手回答中偶然出现的词语判断任务。

任务类型：
title:编写，修改或优化商品标题
bullet_points:编写，修改或优化商品五点描述
selling_points:提炼卖点和关键词
general:其他问题

如果最后一条用户消息是在继续修改上一轮内容，必须保持上一轮的任务类型。

只输出以下四种类型之一：
title
bullet_points
selling_points
general
"""
            }
        ]
        +recent,
        #上五轮切入的文本
        extra_body={"thinking": {"type": "disabled"}},
    )

    task_type = response.choices[0].message.content.strip()
#如果是第一轮那应该没有上文切片，任务类型正常判断
    valid_types = [
        "title",
        "bullet_points",
        "selling_points",
        "general"
    ]
#有效类型的列表
    if task_type not in valid_types:
        task_type = "general"

    return task_type






def call_model(conversation,task_type):
    task_prompt = {
        "title": "本次任务是生成一个优化后的亚马逊商品标题。",
        "bullet_points": "本次任务是生成五点描述。",
        "selling_points": "本次任务是提炼卖点和关键词。",
        "general": "本次任务是根据用户问题判断需求然后提供帮助。"
    }

    current_prompt = (
        SYSTEM_PROMPT + "\n" + task_prompt[task_type]
    )#此时的提示词

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "system", "content": current_prompt}] + conversation,
        extra_body={"thinking": {"type": "disabled"}},
    )#文本里已经加了完整系统提示词，系统提示词就是上面的系统词加分类词加变量文本

    answer = response.choices[0].message.content

    return answer



def detect_task_node(state: AgentState):

    print(f"\n进入 detect_task_node")
    print(f"分类前的state{state}")


    task_type = detect_task(state["messages"])

    print(f"识别到的任务类型:{task_type}")

    return {
        "task_type": task_type
    }#节点函数创建，这个是分类任务的节点

def generate_answer_node(state:AgentState):

    print(f"\n into generate_answer_node")
    print(f"生成回答前的state{state}")
    answer =call_model(
        state["messages"],
        state["task_type"]
    )#答案生成的节点

    return{
        "answer":answer
    }

workflow = StateGraph(AgentState)#工作流搭建

workflow.add_node(
    "detect_task",
    detect_task_node
)#节点名称，对应住后面的节点函数

workflow.add_node(

    "generate_answer",
    generate_answer_node
)

workflow.add_edge(
    START,
    "detect_task"
)#在画布上连线了，开始 连到 任务分类 ，有点像工作流

workflow.add_edge(
    "detect_task",
    "generate_answer"
)

workflow.add_edge(
    "generate_answer",
    END
)

agent = workflow.compile()#编译？

messages = []

while True:

    user_question = input("\n请输入你的问题(输入 '退出' 以结束)：")
    
    if user_question.strip() == "退出":
            print("程序已结束。")
            break   

    messages.append({
        "role":"user",
        "content":user_question
    })

    initial_state: AgentState = {
        "messages":messages,
        "task_type":"",
        "answer":""
    }

    print(f"\n传给invoke的初始状态：{initial_state}")

    result = agent.invoke(initial_state)

    print(f"\n工作流最终返回的result{result}")

    assistant_answer = result["answer"]

    print(f"\n电商助手：{assistant_answer}")

    messages.append({
        "role":"assistant",
        "content":assistant_answer
    })
    

    

    if len(messages) > MAX_ROUNDS * 2:
        messages = messages[-(MAX_ROUNDS * 2):]#文本最大读取上下文，3轮6段 

    