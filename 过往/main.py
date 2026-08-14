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
    if "标题" in user_question:
        return "title"
    elif"五点" in user_question:
        return "bullet_points"
    elif "卖点" in user_question:
        return "selling_points"
    else:
        return "general"


def call_model(conversation,task_type):
    task_prompt = {
        "title": "本次任务是生成一个优化后的亚马逊商品标题。",
        "bullet_points": "本次任务是生成五点描述。",
        "selling_points": "本次任务是提炼卖点和关键词。",
        "general": "本次任务是根据用户问题判断需求然后提供帮助。"
    }

    current_prompt = (
        SYSTEM_PROMPT + "\n" + task_prompt[task_type]
    )

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "system", "content": current_prompt}] + conversation,
        extra_body={"thinking": {"type": "disabled"}},
    )

    answer = response.choices[0].message.content

    return answer

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
    )
    assistant_answer = call_model(messages, task_type)

    messages.append(
        {
            "role": "assistant", 
            "content": assistant_answer
        }
    )

    if len(messages) >= MAX_ROUNDS * 2:
        messages = messages[-(MAX_ROUNDS * 2):]

    print("\n电商助手：")
    print(assistant_answer)