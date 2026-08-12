import os

from dotenv import load_dotenv
from openai import OpenAI

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


def detect_task(user_question):
    classify_promt = f"""
判断下面的用户问题属于哪一类任务：

任务类型：
title: 生成或优化亚马逊商品标题
bullet_points: 生成五点描述
selling_points: 提炼卖点和关键词
general: 其他问题

用户问题：{user_question}

只输出任务类型，不解释。"""

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
            "role": "system", "content": "你只是个任务分类器，只负责判断任务类型。"
            },
            {
            "role": "user", "content": classify_promt,

            }],
        extra_body={"thinking": {"type": "disabled"}},
    )

    task_type = response.choices[0].message.content.strip()

    valid_type =[
        "title",
        "bullet_points",
        "selling_points",
        "general"
    ]

    if task_type not in valid_type:
        task_type = "general"

    return task_type
#回复只有一个单词的 任务类型




def call_model(conversation,task_type):
    task_prompt = {
        "title": "本次任务是生成一个优化后的亚马逊商品标题。",
        "bullet_points": "本次任务是生成五点描述。",
        "selling_points": "本次任务是提炼卖点和关键词。",
        "general": "本次任务是根据用户问题判断需求然后提供帮助。"
    }

    current_prompt = (
        SYSTEM_PROMPT + "\n" + task_prompt[task_type]
    )#任务类型是变量 ai输出到哪个，就从任务提示词这个字典里取键值

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "system", "content": current_prompt}] + conversation,
        extra_body={"thinking": {"type": "disabled"}},
    )

    answer = response.choices[0].message.content

    return answer
#返回系统提示词加任务词 在加上会话这个变量后ai输出的结果

messages = []

while True:
    user_question = input("\n请输入你的问题(输入 '退出' 以结束)：")

    if user_question == "退出" or "tc" in user_question:
        print("程序已结束。")
        break   
    
    task_type = detect_task(user_question)

    print(f"检测到的任务类型为: {task_type}")

    messages.append(
        {
            "role": "user", 
            "content": user_question
        }
    )#文本里追加用户发言
    assistant_answer = call_model(messages, task_type)

    messages.append(
        {
            "role": "assistant", 
            "content": assistant_answer
        }
    )#文本里追加了助手发言

    if len(messages) >= MAX_ROUNDS * 2:
        messages = messages[-(MAX_ROUNDS * 2):]

    print("\n电商助手：")
    print(assistant_answer)
    print(f"{task_type}")