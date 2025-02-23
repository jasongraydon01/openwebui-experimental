# NOTE: remember to run: pip install openai
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-14B-Instruct-1M",
    messages=[
        {"role": "user", "content": "Tell me in one sentence what Mexico is famous for"},
    ]
)
print(chat_response.choices[0].message.content)