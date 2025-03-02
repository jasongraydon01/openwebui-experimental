"""
title: Semantic RAG Search Tool
author: OpenWebUI User
description: Simple document retrieval using embeddings and Pinecone for semantic search
required_open_webui_version: 0.4.0
requirements: pinecone-client, requests
version: 1.0.0
licence: Apache 2.0
"""

import os
import json
from typing import List, Dict, Any, Optional, Callable, Awaitable
import requests
import pinecone
import time
from pydantic import BaseModel, Field


class Tools:
    """
    Simple Semantic RAG Tool for OpenWebUI

    This class provides document retrieval capabilities through Pinecone
    """

    def __init__(self):
        """Initialize the RAG tool."""
        self.valves = self.Valves()
        self.last_emit_time = 0

        # HTTP headers
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "OpenWebUI-RAG-Tool/1.0",
        }

        # Initialize Pinecone client
        self.pinecone_client = None
        print("RAG Tool initialized")

    class Valves(BaseModel):
        embedding_url: str = Field(
            default="http://localhost:8001/v1/embeddings",
            description="URL for the embedding service",
        )
        embedding_model: str = Field(
            default="Alibaba-NLP/gte-Qwen2-7B-instruct",
            description="Model to use for embeddings",
        )
        pinecone_api_key: str = Field(default="", description="Pinecone API key")
        pinecone_index: str = Field(
            default="openwebui-rag", description="Pinecone index name"
        )
        pinecone_namespace: str = Field(
            default="default", description="Pinecone namespace"
        )
        top_k: int = Field(default=5, description="Number of results to return")
        similarity_threshold: float = Field(
            default=0.7, description="Minimum similarity score to include results"
        )
        emit_interval: float = Field(
            default=2.0, description="Interval in seconds between status updates"
        )
        enable_status_indicator: bool = Field(
            default=True, description="Enable or disable status indicator updates"
        )
        debug: bool = Field(
            default=True,
            description="Enable debug mode with additional print statements",
        )

    async def emit_status(
        self,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]],
        level: str,
        message: str,
        done: bool,
    ):
        """Sends status updates to OpenWebUI."""
        if self.valves.debug:
            print(f"STATUS [{level}]: {message}")

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

    async def emit_message(
        self,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]],
        content: str,
    ):
        """Sends a message to OpenWebUI chat."""
        if self.valves.debug:
            print(f"MESSAGE: {content}")

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": content},
                }
            )

    def _setup_pinecone(self) -> bool:
        """Initialize the Pinecone client if not already initialized."""
        try:
            # Initialize Pinecone client
            pc = pinecone.Pinecone(api_key=self.valves.pinecone_api_key)

            # Create index if it doesn't exist
            if not pc.has_index(self.valves.pinecone_index):
                pc.create_index(
                    name=self.valves.pinecone_index,
                    dimension=3584,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws", region=self.valves.pinecone_environment
                    ),
                )
                print(f"Created new Pinecone index: {self.valves.pinecone_index}")

            # Wait for index to be ready
            while not pc.describe_index(self.valves.pinecone_index).status["ready"]:
                print("Waiting for Pinecone index to be ready...")
                time.sleep(1)

            # Connect to the index
            self.pinecone_client = pc.Index(self.valves.pinecone_index)
            print(f"Connected to Pinecone index: {self.valves.pinecone_index}")

            return True
        except Exception as e:
            print(f"ERROR: Failed to initialize Pinecone: {str(e)}")
            return False

    def _get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the input text using the embedding API."""
        if not text or not isinstance(text, str):
            print("ERROR: Empty or invalid text provided for embedding")
            raise ValueError("Empty or invalid text provided for embedding")

        payload = {
            "input": text,
            "model": self.valves.embedding_model,
        }

        print(
            f"Getting embeddings for text: '{text[:50]}...' using model: {self.valves.embedding_model}"
        )
        print(f"Embedding URL: {self.valves.embedding_url}")

        response = requests.post(
            self.valves.embedding_url, json=payload, headers=self.headers, timeout=30
        )
        response.raise_for_status()
        data = response.json()

        if "data" in data and len(data["data"]) > 0:
            embedding = data["data"][0]["embedding"]
            print(f"Successfully generated embedding with {len(embedding)} dimensions")
            return embedding
        else:
            print("ERROR: No embedding found in response")
            print(f"Response data: {data}")
            raise ValueError("No embedding found in response")

    async def rag_search(
        self,
        query: str,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        Perform a semantic search using embeddings.

        :param query: The user query to search for
        :param __event_emitter__: Optional event emitter for status updates
        :return: Dictionary with response and sources
        """
        try:
            print(f"========== RAG SEARCH STARTING FOR: '{query}' ==========")
            await self.emit_status(
                __event_emitter__, "info", "Initializing RAG search...", False
            )

            # Setup Pinecone
            await self.emit_status(
                __event_emitter__, "info", "Connecting to Pinecone...", False
            )
            if not self._setup_pinecone():
                error_msg = "Could not connect to Pinecone database. Please check your configuration."
                await self.emit_status(__event_emitter__, "error", error_msg, True)
                return {"response": error_msg, "sources": []}

            # Get embedding for the query
            await self.emit_status(
                __event_emitter__, "info", "Generating embeddings...", False
            )
            embedding = self._get_embeddings(query)

            # Query Pinecone
            await self.emit_status(
                __event_emitter__, "info", "Searching for relevant documents...", False
            )
            print(
                f"Querying Pinecone with top_k={self.valves.top_k} in namespace={self.valves.pinecone_namespace}"
            )

            query_response = self.pinecone_client.query(
                namespace=self.valves.pinecone_namespace,
                vector=embedding,
                top_k=self.valves.top_k,
                include_metadata=True,
            )

            print(f"Received {len(query_response['matches'])} matches from Pinecone")

            # Process results
            results = []
            sources = []

            for i, match in enumerate(query_response["matches"]):
                print(f"Match {i+1}: ID={match['id']}, Score={match['score']}")

                # Skip if below similarity threshold
                if match["score"] < self.valves.similarity_threshold:
                    print(
                        f"  Skipping match {i+1} with score {match['score']} (below threshold)"
                    )
                    continue

                metadata = match["metadata"]
                print(f"  Metadata: {json.dumps(metadata)[:200]}...")

                # Extract document information
                doc_type = metadata.get("doc_type", "Document")
                file_name = metadata.get("file_name", "Unknown")
                section_id = metadata.get(
                    "section_id",
                    metadata.get("page_number", metadata.get("slide_number", "?")),
                )
                content = metadata.get("content_preview", "No content available")
                source_link = metadata.get(
                    "source_link", metadata.get("one_drive_link", "#")
                )

                # Format source reference
                source_ref = f"{doc_type} - {file_name} (Section {section_id})"
                sources.append(source_ref)
                print(f"  Adding source: {source_ref}")

                # Add to results with score and metadata
                results.append(
                    {
                        "id": match["id"],
                        "score": match["score"],
                        "content": content,
                        "source": source_ref,
                        "link": source_link,
                    }
                )

            # Prepare the context from results
            if results:
                context_parts = []

                for r in results:
                    context_parts.append(f"From {r['source']}:\n{r['content']}")
                    # Add citation event if using that feature
                    if __event_emitter__:
                        print(f"  Emitting citation for source: {r['source']}")
                        await __event_emitter__(
                            {
                                "type": "citation",
                                "data": {
                                    "document": [r["content"]],
                                    "metadata": [
                                        {
                                            "date_accessed": time.strftime(
                                                "%Y-%m-%d %H:%M:%S"
                                            ),
                                            "source": r["source"],
                                        }
                                    ],
                                    "source": {"name": r["source"], "url": r["link"]},
                                },
                            }
                        )

                context = "\n\n".join(context_parts)
                print(
                    f"Generated context with {len(context)} characters from {len(results)} sources"
                )

                response = {"response": context, "sources": sources}
            else:
                print("No matching documents found")
                response = {
                    "response": "No relevant documents found for your query.",
                    "sources": [],
                }

            await self.emit_status(
                __event_emitter__, "info", "RAG search complete", True
            )
            print("========== RAG SEARCH COMPLETED ==========")
            return response

        except Exception as e:
            error_message = f"RAG search failed: {str(e)}"
            print(f"ERROR: {error_message}")

            # Print exception traceback for debugging
            import traceback

            print(traceback.format_exc())

            await self.emit_status(__event_emitter__, "error", error_message, True)
            print("========== RAG SEARCH FAILED ==========")
            return {"response": error_message, "sources": []}

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[dict], Awaitable[dict]]] = None,
    ):
        """Process user query and return RAG results in the format expected by OpenWebUI."""
        try:
            print("========== RAG PIPE STARTING ==========")
            print(f"Received request body: {json.dumps(body)[:500]}...")

            # Extract the query from the request body - handle both formats
            query = None

            # Direct query parameter
            if "query" in body:
                query = body["query"]
                print(f"Found query directly in request body: {query}")

            # Messages array
            elif "messages" in body and body["messages"]:
                messages = body["messages"]
                if messages:
                    query = messages[-1].get("content", "")
                    print(f"Extracted query from messages: {query}")

            # No query found
            if not query:
                error_msg = "No query found in the request body"
                print(f"ERROR: {error_msg}")
                await self.emit_status(__event_emitter__, "error", error_msg, True)
                return {"response": error_msg, "sources": []}

            # Perform the RAG search
            result = await self.rag_search(query, __event_emitter__)
            print("========== RAG PIPE COMPLETED ==========")
            return result

        except Exception as e:
            error_message = f"RAG pipe failed: {str(e)}"
            print(f"ERROR: {error_message}")

            # Print exception traceback for debugging
            import traceback

            print(traceback.format_exc())

            if __event_emitter__:
                await self.emit_status(__event_emitter__, "error", error_message, True)

            print("========== RAG PIPE FAILED ==========")
            return {"response": error_message, "sources": []}