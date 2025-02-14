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
    """Retrieves the top-k most relevant slides from Pinecone.
       Returns each slide with its text, slide number, source, and similarity score.
    """
    query_embedding = generate_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace="ns1")

    if not results or "matches" not in results:
        return []

    slides = []
    for match in results["matches"]:
        slide_text = match["metadata"].get("summary", "Unknown content")
        slide_number = match["metadata"].get("slide_number", "Unknown slide")
        slide_source = match["metadata"].get("one_drive_link", "Unknown source")
        score = match.get("score", 0)  # capture similarity score
        slides.append({
            "text": slide_text,
            "slide_number": slide_number,
            "source": slide_source,
            "score": score
        })

    return slides

# Normal Chat (No RAG)
def chat_normal(query):
    """Generates a normal response from the LLM without retrieval augmentation."""
    response = ollama.chat(model="mistral:7b", messages=[{"role": "user", "content": query}])
    if "message" not in response:
        raise ValueError("Ollama chat response failed.")
    return response["message"]

# RAG Chat
def chat_with_rag(query, retrieved_slides=None):
    """Generates a response using retrieved slides as context.
       Optionally uses pre-retrieved slides to avoid making a redundant call.
    """
    if retrieved_slides is None:
        retrieved_slides = retrieve_relevant_slides(query)

    if not retrieved_slides:
        return chat_normal(query), []

    context = "\n".join([f"Slide {s['slide_number']}: {s['text']}" for s in retrieved_slides])
    prompt = (
        f"Using the following context, answer the question:\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )

    response = ollama.chat(model="mistral:7b", messages=[{"role": "user", "content": prompt}])
    if "message" not in response:
        raise ValueError("Ollama chat response failed.")
    return response["message"], retrieved_slides

# Dynamic Chat: Decides whether to use RAG or normal chat
def chat_auto(query, score_threshold=0.7):
    """
    Decides dynamically whether to augment the chat with retrieval.
    It first retrieves potential relevant slides and if the top score meets the threshold,
    it uses RAG. Otherwise, it falls back to a normal chat.
    """
    retrieved_slides = retrieve_relevant_slides(query)
    # If no slides are found, or if none of the slides meet the relevance threshold, use normal chat.
    if not retrieved_slides:
        return chat_normal(query), []

    # Use the maximum score among the retrieved slides for the decision.
    max_score = max(s.get("score", 0) for s in retrieved_slides)
    if max_score >= score_threshold:
        return chat_with_rag(query, retrieved_slides)
    else:
        return chat_normal(query), []

# Interactive Query Loop
if __name__ == "__main__":
    print("\nType your query (or type 'exit' to quit).\n")
    while True:
        user_query = input("Your input (type 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            break

        # Use dynamic decision-making to choose the appropriate chatting mode.
        answer, sources = chat_auto(user_query)
        print(f"\nAI Response:\n{answer}\n")
        if sources:
            print("Sources:")
            for src in sources:
                # Print a shortened version of the slide text for clarity.
                print(f" - Slide {src['slide_number']}: {src['text'][:100]}...")
        else:
            print("No relevant slides found.")