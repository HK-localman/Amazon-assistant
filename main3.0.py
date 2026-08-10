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

message = []

while True:
    user_question = input("\n请输入你的问题(输入 '退出' 以结束)：")

    if user_question == "退出":
        print("程序已结束。")
        break

    message.append(
        {
            "role": "user", 
            "content": user_question
            }
    )

    print("\n本次投喂的message内容为：")
    for msg in message:
        print(f"角色: {msg['role']}, 内容: {msg['content']}")

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=message,
        extra_body={"thinking": {"type": "disabled"}},
    )

    assistant_answer = response.choices[0].message.content

    message.append(
        {
            "role": "assistant", 
            "content": assistant_answer
            }
    )

    print("\nDeepSeek 的回答：")
    print(assistant_answer)

