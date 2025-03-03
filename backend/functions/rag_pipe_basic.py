"""
title: Smart Semantic RAG Search Pipe
author: OpenWebUI User
description: Intelligent document retrieval that determines if RAG is needed for a query
required_open_webui_version: 0.4.0
requirements: pinecone, requests
version: 1.0.0
licence: Apache 2.0
"""

import time
import json
from typing import List, Dict, Any, Optional, Callable, Awaitable
import requests
import pinecone
from pydantic import BaseModel, Field


class Pipe:
    """
    Smart Semantic RAG Pipe for OpenWebUI that first determines if a query needs
    document retrieval before proceeding with the RAG process
    """

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
        llm_api_url: str = Field(
            default="http://vllm-model1:8000/v1/chat/completions",
            description="URL for the OpenAI compatible API (vLLM)",
        )
        llm_model: str = Field(default="", description="LLM model to use")
        temperature: float = Field(
            default=0.3, description="Temperature for generation"
        )
        max_tokens: int = Field(default=1024, description="Maximum tokens to generate")
        show_source_documents: bool = Field(
            default=False, description="Whether to show source documents in chat"
        )
        always_use_rag: bool = Field(
            default=False,
            description="If True, always use RAG for all queries. If False, let the LLM decide.",
        )

    def __init__(self):
        """Initialize the RAG pipe."""
        self.valves = self.Valves()

    def _setup_pinecone(self) -> bool:
        """Initialize the Pinecone client."""
        try:
            # Initialize Pinecone client
            pc = pinecone.Pinecone(api_key=self.valves.pinecone_api_key)

            # Connect to the index
            self.pinecone_client = pc.Index(self.valves.pinecone_index)
            return True
        except Exception as e:
            print(f"Error connecting to Pinecone: {str(e)}")
            return False

    def _get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the input text."""
        payload = {
            "input": text,
            "model": self.valves.embedding_model,
        }

        response = requests.post(
            self.valves.embedding_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if "data" in data and len(data["data"]) > 0:
            return data["data"][0]["embedding"]
        else:
            raise ValueError("No embedding found in response")

    async def _determine_query_type(self, query: str) -> bool:
        """
        Determine if the query requires document retrieval or not.

        Returns True if the query needs RAG, False otherwise.
        """
        # If always_use_rag is set, skip the decision and always use RAG
        if self.valves.always_use_rag:
            return True

        # Prepare a classification prompt for the LLM
        classification_messages = [
            {
                "role": "system",
                "content": """You are an AI assistant that determines if a query requires document retrieval.
                
Determine whether the given query is seeking content that is likely stored in an internal knowledge base (e.g., proprietary documents, clinical guidelines, company reports) or if it pertains to general, external knowledge (e.g., publicly available medical literature, general scientific concepts). Consider factors such as specificity, reference to proprietary terminology, and whether the query implies the need for document retrieval. If the query suggests retrieving specific internal information, use the RAG path; otherwise, answer directly using the model's general knowledge.

Return ONLY a JSON object with the following schema:
{
  "needs_retrieval": true/false,
  "reason": "brief explanation"
}

Set "needs_retrieval" to TRUE if the query:
- Asks about specific documents, data, or files
- Requests analysis or information about specific topics that would benefit from up-to-date or domain-specific information
- Refers to specific products, reports, or entities in a knowledge base
- Asks about specific statistics, numbers, or facts that would be in documents
- Uses phrases like "in the documents", "in the knowledge base", etc.

Set "needs_retrieval" to FALSE if the query:
- Is a general question answerable with common knowledge
- Is asking for creative content (stories, code, etc.)
- Is conversational in nature without seeking specific information
- Is asking about hypothetical scenarios
- Is asking for your capabilities or how you work
- Is a simple greeting or chitchat""",
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nDetermine if this query requires document retrieval. Return only the JSON object.",
            },
        ]

        # Call the LLM to classify the query
        classification_payload = {
            "model": self.valves.llm_model,
            "messages": classification_messages,
            "temperature": 0.1,  # Use low temperature for deterministic classification
            "max_tokens": 150,  # Classification response should be short
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self.valves.llm_api_url, json=classification_payload, headers=headers
        )
        response.raise_for_status()

        result = response.json()
        if "choices" not in result or not result["choices"]:
            # In case of an error, default to using RAG
            return True

        # Extract the classification response
        classification_text = result["choices"][0]["message"]["content"]

        # Extract JSON from the response (handle potential text around the JSON)
        try:
            # Try to find JSON object in the text
            json_start = classification_text.find("{")
            json_end = classification_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = classification_text[json_start:json_end]
                classification = json.loads(json_str)

                # Get the classification decision
                return classification.get("needs_retrieval", True)
            else:
                # If no JSON found, default to using RAG
                return True
        except json.JSONDecodeError:
            # In case of parsing error, default to using RAG
            return True

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ):
        """
        Process user query with intelligence to determine if RAG is needed.

        :param body: The request body containing the user's message
        :param __user__: User information
        :param __event_emitter__: Event emitter for adding messages to the chat
        :return: LLM response with or without RAG enhancement
        """
        try:
            # Extract user query from the body
            messages = body.get("messages", [])
            if not messages:
                return "Error: No messages found in request"

            # Get the last user message as the query
            user_messages = [m for m in messages if m.get("role") == "user"]
            if not user_messages:
                return "Error: No user messages found"

            query = user_messages[-1].get("content", "")
            if not query:
                return "Error: Empty query"

            # Determine if this query needs document retrieval
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Analyzing query...",
                            "done": False,
                        },
                    }
                )

            needs_retrieval = await self._determine_query_type(query)

            # If the query doesn't need retrieval, proceed with direct LLM answer
            if not needs_retrieval:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Answering directly without document retrieval",
                                "done": True,
                            },
                        }
                    )

                # Create direct response payload - use original messages
                direct_payload = {
                    "model": self.valves.llm_model,
                    "messages": messages,
                    "temperature": self.valves.temperature,
                    "max_tokens": self.valves.max_tokens,
                    "stream": body.get("stream", False),
                }

                # Call the LLM directly
                headers = {"Content-Type": "application/json"}
                response = requests.post(
                    self.valves.llm_api_url,
                    json=direct_payload,
                    headers=headers,
                    stream=True,  # Enable streaming
                )
                response.raise_for_status()

                # Handle streaming if enabled
                if body.get("stream", False):

                    def stream_response():
                        for line in response.iter_lines():
                            if line:
                                yield line.decode("utf-8")

                    return stream_response()
                else:
                    result = response.json()
                    # Extract content from OpenAI API response format
                    if "choices" in result and result["choices"]:
                        return result["choices"][0]["message"]["content"]
                    else:
                        return "Error: Unexpected response format from LLM API"

            # For queries that need retrieval, proceed with the RAG process
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Searching knowledge base...",
                            "done": False,
                        },
                    }
                )

            # Setup Pinecone
            if not self._setup_pinecone():
                error_msg = "Could not connect to Pinecone database. Please check your configuration."
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": error_msg, "done": True},
                        }
                    )
                return error_msg

            # Get embedding for the query
            embedding = self._get_embeddings(query)

            # Query Pinecone
            query_response = self.pinecone_client.query(
                namespace=self.valves.pinecone_namespace,
                vector=embedding,
                top_k=self.valves.top_k,
                include_metadata=True,
            )

            # Process results
            matches = query_response["matches"]
            relevant_matches = [
                m for m in matches if m["score"] >= self.valves.similarity_threshold
            ]

            # Create a consolidated context from the retrieved documents
            consolidated_context = ""
            document_summaries = []

            if not relevant_matches:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "No relevant documents found in knowledge base",
                                "done": True,
                            },
                        }
                    )

                # If no relevant documents found but retrieval was needed, proceed with direct answer
                direct_payload = {
                    "model": self.valves.llm_model,
                    "messages": messages,
                    "temperature": self.valves.temperature,
                    "max_tokens": self.valves.max_tokens,
                    "stream": body.get("stream", False),
                }

                # Call the LLM directly (for no relevant documents case)
                headers = {"Content-Type": "application/json"}
                response = requests.post(
                    self.valves.llm_api_url,
                    json=direct_payload,
                    headers=headers,
                    stream=True,  # Enable streaming
                )
                response.raise_for_status()

                # Handle streaming if enabled
                if body.get("stream", False):

                    def stream_response():
                        for line in response.iter_lines():
                            if line:
                                yield line.decode("utf-8")

                    return stream_response()
                else:
                    result = response.json()
                    # Extract content from OpenAI API response format
                    if "choices" in result and result["choices"]:
                        return result["choices"][0]["message"]["content"]
                    else:
                        return "Error: Unexpected response format from LLM API"
            else:
                # Process each document and build the context without showing it in chat
                for i, match in enumerate(relevant_matches, 1):
                    metadata = match["metadata"]
                    score = match["score"]

                    doc_type = metadata.get("doc_type", "Document")
                    file_name = metadata.get("file_name", "Unknown")
                    slide_number = metadata.get("slide_number", "?")
                    content = metadata.get("content_preview", "No content available")
                    one_drive_link = metadata.get("one_drive_link", "#")

                    # Build document summary (removed "DOCUMENT {i}:" prefix)
                    doc_summary = f"DOCUMENT: {file_name} (Slide {slide_number})\nRelevance: {score * 100:.1f}%\nContent: {content}\n\n"
                    document_summaries.append(doc_summary)

                    # Consolidated context (include the one_drive_link for the LLM to use)
                    consolidated_context += f"### {file_name} (Slide {slide_number})\nLink: {one_drive_link}\n{content}\n\n"

                    # Only emit visible messages if explicitly configured
                    if self.valves.show_source_documents and __event_emitter__:
                        # Removed {doc_type}: from the heading to avoid confusion
                        doc_message = f"""### {file_name} (Slide {slide_number})
**Relevance Score:** {score * 100:.1f}%

{content}

---
"""
                        await __event_emitter__(
                            {"type": "message", "data": {"content": doc_message}}
                        )

                    # Emit as a citation
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "citation",
                                "data": {
                                    "document": [content],
                                    "metadata": [
                                        {
                                            "date_accessed": time.strftime(
                                                "%Y-%m-%d %H:%M:%S"
                                            ),
                                            "source": f"{file_name} (Slide {slide_number})",
                                        }
                                    ],
                                    "source": {
                                        "name": file_name,
                                        "url": one_drive_link,
                                    },
                                },
                            }
                        )

                # Update status with number of documents found but don't display them
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Found {len(relevant_matches)} relevant documents in knowledge base",
                                "done": True,
                            },
                        }
                    )

                # Create a more effective prompt with the consolidated context
                query_context = f"""
QUERY: {query}

RETRIEVED DOCUMENTS:
{consolidated_context}

Based on the above documents and ONLY the information contained within them, please provide a comprehensive answer to the query. 
Cite specific sources, statistics, and findings from the documents. 
Do not introduce external information or suggest research methodologies.
If the documents don't contain sufficient information to answer the query, state this explicitly.
"""

                # Create instruction for LLM in the prompt - simplified and focused on citation format
                system_prompt = """You are answering user queries using retrieved documents. MOST IMPORTANTLY: Cite every claim directly using inline clickable links.

CITATION FORMAT: "[Claim text](one_drive_link)" 
Example: "After anticipated market events, Leqvio is projected to capture approximately 3 out of 10 Cardiovascular Disease (CVD) patients [according to Slide 10 of the Leqvio Demand Study Findings_March 31 2023.pptx](one_drive_link)".

CRITICAL INSTRUCTIONS:
1. EVERY fact or claim MUST have its own clickable citation using the exact format above
2. Use ONLY information from the retrieved documents - no external knowledge
3. Make links clickable by using the exact markdown format: [text](one_drive_link)
4. Begin with a summary of key findings relevant to the query
5. If documents lack sufficient information, say so explicitly
6. When referencing slides, include both the file name and slide number in your citation text
7. Use direct quotes when helpful and always cite their source

REMEMBER: Your primary goal is to answer the query with properly cited information that clearly indicates which document and slide each fact comes from."""

                # Replace or add system message
                new_messages = []
                found_system = False

                for message in messages:
                    if message.get("role") == "system":
                        # Update system message
                        message["content"] = system_prompt
                        found_system = True
                    new_messages.append(message)

                # Add system message if none exists
                if not found_system:
                    new_messages.insert(0, {"role": "system", "content": system_prompt})

                # Add a user message with the consolidated context before the final query
                # This ensures the model has the retrieved information directly in its context
                context_message = {"role": "user", "content": query_context}

                # Find the position of the last user message
                last_user_index = len(new_messages) - 1
                for i in range(len(new_messages) - 1, -1, -1):
                    if new_messages[i].get("role") == "user":
                        last_user_index = i
                        break

                # Replace the last user message with the context message
                new_messages[last_user_index] = context_message

                # Create OpenAI API compatible request payload with instruction-tuned parameters
                openai_payload = {
                    "model": self.valves.llm_model,
                    "messages": new_messages,
                    "temperature": self.valves.temperature,
                    "max_tokens": self.valves.max_tokens,
                    "stream": body.get("stream", False),
                }

                # Call the model with the OpenAI API format
                headers = {"Content-Type": "application/json"}
                response = requests.post(
                    self.valves.llm_api_url,
                    json=openai_payload,
                    headers=headers,
                    stream=True,
                )
                response.raise_for_status()

                # Handle streaming if enabled
                if body.get("stream", False):

                    def stream_response():
                        for line in response.iter_lines():
                            if line:
                                yield line.decode("utf-8")

                    return stream_response()
                else:
                    result = response.json()
                    # Extract content from OpenAI API response format
                    if "choices" in result and result["choices"]:
                        return result["choices"][0]["message"]["content"]
                    else:
                        return "Error: Unexpected response format from LLM API"

        except Exception as e:
            error_message = f"Error in RAG process: {str(e)}"

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_message, "done": True},
                    }
                )

            return f"Error: {error_message}"
