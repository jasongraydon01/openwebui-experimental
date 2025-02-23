from typing import Optional, Callable, Awaitable
from pydantic import BaseModel, Field
import requests
import time


class Pipe:
    class Valves(BaseModel):
        api_url: str = Field(
            default="http://host.docker.internal:5001/query",
            description="RAG API endpoint URL",
        )
        emit_interval: float = Field(
            default=2.0, description="Interval in seconds between status updates"
        )
        enable_status_indicator: bool = Field(
            default=True, description="Enable or disable status indicator updates"
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "rag_pipe"
        self.name = "RAG API Pipe"
        self.valves = self.Valves()
        self.last_emit_time = 0

    async def emit_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        level: str,
        message: str,
        done: bool,
    ):
        """Sends status updates to OpenWebUI."""
        if (
            __event_emitter__
            and self.valves.enable_status_indicator
            and (time.time() - self.last_emit_time >= self.valves.emit_interval or done)
        ):
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )
            self.last_emit_time = time.time()

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
    ) -> dict:
        """Processes user input and retrieves responses from the RAG API."""
        await self.emit_status(__event_emitter__, "info", "Calling RAG API...", False)

        messages = body.get("messages", [])
        if not messages:
            return {"error": "No messages found in the request body"}

        try:
            response = requests.post(
                self.valves.api_url,
                json={"query": messages[-1]["content"]},
                headers={"Content-Type": "application/json"},
            )
            if response.status_code == 200:
                rag_response = response.json().get("response", "No response received")
            else:
                rag_response = f"API Error: {response.status_code} - {response.text}"

        except Exception as e:
            rag_response = f"Error: {str(e)}"

        await self.emit_status(__event_emitter__, "info", "Complete", True)

        # Return the agent's response as expected by OpenWebUI
        return rag_response