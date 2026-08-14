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


def call_model(conversation):
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
        extra_body={"thinking": {"type": "disabled"}},
    )
    answer = response.choices[0].message.content

    return answer



messages = []

while True:
    user_question = input("\n请输入你的问题(输入 '退出' 以结束)：")

    if user_question == "退出":
        print("程序已结束。")
        break

    messages.append(
        {
            "role": "user", 
            "content": user_question
            }
    )

    assistant_answer = call_model(messages)

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