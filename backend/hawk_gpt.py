from typing import Optional, Callable, Awaitable
from pydantic import BaseModel, Field
import time
import requests


class Pipe:
    class Valves(BaseModel):
        rag_api_url: str = Field(
            default="http://host.docker.internal:5001/query"
        )  # Adjusted for Docker
        emit_interval: float = Field(
            default=2.0, description="Interval in seconds between status emissions"
        )
        enable_status_indicator: bool = Field(
            default=True, description="Enable or disable status indicator emissions"
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
        """Emit status updates to OpenWebUI."""
        current_time = time.time()
        if (
            __event_emitter__
            and self.valves.enable_status_indicator
            and (
                current_time - self.last_emit_time >= self.valves.emit_interval or done
            )
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
            self.last_emit_time = current_time

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
    ) -> Optional[dict]:
        """Handles user queries and forwards them to the RAG API."""
        await self.emit_status(__event_emitter__, "info", "Processing query...", False)

        messages = body.get("messages", [])
        if not messages:
            await self.emit_status(
                __event_emitter__, "error", "No messages found", True
            )
            return {"error": "No messages found in request"}

        query = messages[-1]["content"]  # Get the latest user message

        try:
            # Log the request
            print(f"🔹 Sending query to RAG API: {self.valves.rag_api_url}")
            print(f"🔹 Query: {query}")

            # Send the query to RAG API
            response = requests.post(
                self.valves.rag_api_url,
                json={"query": query},
                headers={"Content-Type": "application/json"},
            )

            print(f"🔹 API Response Status: {response.status_code}")
            print(f"🔹 API Response Content: {response.text}")

            if response.status_code == 200:
                response_json = response.json()
                body["messages"].append(
                    {
                        "role": "assistant",
                        "content": response_json.get("response", "No response"),
                    }
                )
            else:
                body["messages"].append(
                    {
                        "role": "assistant",
                        "content": f"Error: {response.status_code} - {response.text}",
                    }
                )

        except Exception as e:
            await self.emit_status(
                __event_emitter__, "error", f"Request failed: {str(e)}", True
            )
            body["messages"].append(
                {"role": "assistant", "content": f"Request failed: {str(e)}"}
            )

        await self.emit_status(__event_emitter__, "info", "Query processed", True)
        return body