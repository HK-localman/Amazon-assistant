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
while True:
    user_question = input("\n请输入你的问题(输入 '退出' 以结束)：")

    if  user_question== "退出":
        print("程序已结束。")
        break   


    response = client.chat.completions.create(
         model="deepseek-v4-flash",
         messages=[
        {     
             "role": "user", 
             "content": user_question},
    ],
         extra_body={"thinking": {"type": "disabled"}},
)

    print("\nDeepSeek 的回答：")
    print(response.choices[0].message.content)