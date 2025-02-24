from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# Initialize FastAPI app
app = FastAPI()

# Connect FastAPI to vLLM running in a separate container
client = OpenAI(
    api_key="EMPTY",
    base_url="http://vllm-container:8000/v1",  # vLLM is running in a separate container
)

# Request model
class ChatRequest(BaseModel):
    model: str
    messages: list

# API Endpoint for chat completion
@app.post("/chat")
async def chat_completion(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=request.messages,
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "running"}