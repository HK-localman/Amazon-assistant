import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("没找到deepseek API密钥，请在.env文件中设置DEEPSEEK_API_KEY环境变量。")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "user",
         "content":"请你用一句话介绍你自己",
        }
    ],
    extra_body={"thinking":{"type":"disabled"}}
)


print(response.choices[0].message.content)