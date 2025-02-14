import os
import time
import pinecone
import ollama
import sqlite3
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API Key from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "openwebui-setup"
DATABASE_PATH = "file_log.db"  # SQLite database path

# Validate API Key
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key is missing. Please set PINECONE_API_KEY in your .env file.")

# Initialize Pinecone Client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Wait for index readiness
while not pc.describe_index(PINECONE_INDEX_NAME).status.get("ready", False):
    print("Waiting for Pinecone index to be ready...")
    time.sleep(1)

# Connect to Pinecone Index
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize Flask App
app = Flask(__name__)

# ✅ Ensure Embeddings are Flattened
def generate_embedding(text):
    """Generates a flattened embedding for a given text using Ollama."""
    response = ollama.embed(model="nomic-embed-text", input=text)
    embeddings = response.get("embeddings", [])
    
    # Handle potential nested lists
    if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list):
        return [item for sublist in embeddings for item in sublist]  # Flatten
    
    return embeddings

# ✅ Improve Pinecone Query Handling
def retrieve_relevant_slides(query, top_k=3):
    """Retrieves the top-k most relevant slides from Pinecone."""
    try:
        query_embedding = generate_embedding(query)
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace="ns1")

        if not results or "matches" not in results:
            return []

        slides = []
        for match in results["matches"]:
            slide_text = match["metadata"].get("summary", "Unknown content")
            slide_number = match["metadata"].get("slide_number", "Unknown slide")
            slide_source = match["metadata"].get("one_drive_link", "Unknown source")
            score = match.get("score", 0)

            slides.append({
                "text": slide_text,
                "slide_number": slide_number,
                "source": slide_source,
                "score": score
            })

        return slides
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

# ✅ Add Normal Chat Handling (No RAG)
def chat_normal(query):
    """Generates a normal response from the LLM without retrieval augmentation."""
    response = ollama.chat(model="mistral:7b", messages=[{"role": "user", "content": query}])
    return response.get("message", {}).get("content", "No response generated.")

# ✅ Add RAG-Based Chat with Context
def chat_with_rag(query, retrieved_slides=None):
    """Generates a response using retrieved slides as context."""
    if retrieved_slides is None:
        retrieved_slides = retrieve_relevant_slides(query)

    if not retrieved_slides:
        return chat_normal(query), []

    context = "\n".join([f"Slide {s['slide_number']}: {s['text']}" for s in retrieved_slides])
    prompt = f"Using the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {query}"

    response = ollama.chat(model="mistral:7b", messages=[{"role": "user", "content": prompt}])
    return response.get("message", {}).get("content", "No response generated."), retrieved_slides

# ✅ Improve Decision Logic for Using RAG
def chat_auto(query, score_threshold=0.7):
    """Decides dynamically whether to use RAG or normal chat."""
    retrieved_slides = retrieve_relevant_slides(query)
    
    if not retrieved_slides:
        return chat_normal(query), []

    max_score = max(s.get("score", 0) for s in retrieved_slides)
    if max_score >= score_threshold:
        return chat_with_rag(query, retrieved_slides)
    else:
        return chat_normal(query), []

# ✅ API Route: Query with RAG
@app.route("/query", methods=["POST"])
def query_rag():
    """API endpoint to query the RAG system."""
    data = request.json
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    response, sources = chat_auto(query)

    return jsonify({
        "response": response,
        "sources": sources
    })

# ✅ API Route: Retrieve Relevant Slides
@app.route("/retrieve", methods=["POST"])
def retrieve_slides():
    """API endpoint to retrieve relevant slides without generating a response."""
    data = request.json
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    slides = retrieve_relevant_slides(query)
    return jsonify({"slides": slides})

# ✅ Run the Flask App (Production-Ready)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)  # ❌ Disable Debug Mode in Production