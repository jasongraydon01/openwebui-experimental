import os
import pinecone
import ollama
from dotenv import load_dotenv
import time
# Load environment variables from .env file
load_dotenv()

# Retrieve API Key from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "openwebui-setup"

# Ensure API key is set
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key is missing. Please set PINECONE_API_KEY in your .env file.")

# Initialize Pinecone Client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Wait for index readiness
while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
    print("Waiting for Pinecone index to be ready...")
    time.sleep(1)

# Connect to the Pinecone Index
index = pc.Index(PINECONE_INDEX_NAME)
print(f"Connected to Pinecone Index: {PINECONE_INDEX_NAME}")

# Function to Generate Embeddings Using Ollama
def generate_embedding(text):
    """Generates an embedding for a given text using Ollama."""
    response = ollama.embed(model="nomic-embed-text", input=text)
    if "embeddings" not in response:
        raise ValueError("Failed to generate embedding from Ollama.")
    return response['embeddings']

# Function to Retrieve Relevant Slides from Pinecone
def retrieve_relevant_slides(query, top_k=3):
    """Retrieves the top-k most relevant slides from Pinecone."""
    query_embedding = generate_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace="ns1")

    if not results or "matches" not in results:
        return []

    slides = []
    for match in results["matches"]:
        slide_text = match["metadata"].get("summary", "Unknown content")
        slide_number = match["metadata"].get("slide_number", "Unknown slide")
        slide_source = match["metadata"].get("one_drive_link", "Unknown source")
        slides.append({"text": slide_text, "slide_number": slide_number, "source": slide_source})

    return slides

# Normal Chat (No RAG)
def chat_normal(query):
    """Generates a normal response from the LLM without retrieval augmentation."""
    response = ollama.chat(model="mistral:7b", messages=[{"role": "user", "content": query}])
    if "message" not in response:
        raise ValueError("Ollama chat response failed.")
    return response["message"]

# RAG Chat
def chat_with_rag(query):
    """Generates a response using retrieved slides as context."""
    retrieved_slides = retrieve_relevant_slides(query)

    if not retrieved_slides:
        return "No relevant information found in the database.", []

    context = "\n".join([f"Slide {s['slide_number']}: {s['text']}" for s in retrieved_slides])
    prompt = f"Using the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {query}"

    response = ollama.chat(model="mistral:7b", messages=[{"role": "user", "content": prompt}])
    if "message" not in response:
        raise ValueError("Ollama chat response failed.")

    return response["message"], retrieved_slides

# Interactive Query Loop
if __name__ == "__main__":
    print("\nType normal questions, or prefix with 'rag:' for retrieval-augmented Q&A.\n")
    while True:
        user_query = input("Your input (type 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            break

        # Check if user wants RAG mode (prefix: rag:)
        if user_query.lower().startswith("rag:"):
            query_text = user_query[4:].strip()
            answer, sources = chat_with_rag(query_text)
            print(f"\nAI (RAG) Response:\n{answer}\n")
            print("Sources:")
            if sources:
                for src in sources:
                    print(f" - Slide {src['slide_number']}: {src['text'][:100]}...")
            else:
                print("No relevant slides found.")
        else:
            # Normal Chat
            answer = chat_normal(user_query)
            print(f"\nAI (Normal) Response:\n{answer}\n")