import openai

openai.api_base = "http://localhost:8000/v1"  # Replace with your vLLM server URL
openai.api_key = "EMPTY" # Not needed for local vLLM

completion = openai.chat.completions.create(
    model="your_model_name",  # Replace with your model name
    messages=[
        {"role": "user", "content": "Write a short story about a robot."}
    ],
)

print(completion.choices[0].message.content)