import os
import time
import pinecone
import pptx
import requests
import sqlite3
from dotenv import load_dotenv
from pinecone import ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Retrieve API Keys & URLs
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
DATABASE_PATH = os.getenv("DATABASE_PATH")
FOLDER_PATH = os.getenv("FOLDER_PATH")

# vLLM API URLs
VLLM_CHAT_URL = "http://vllm-container:8000/v1/chat/completions"
VLLM_EMBED_URL = "http://vllm-container:8001/v1/embeddings"

# Initialize Pinecone Client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

if not pc.has_index(PINECONE_INDEX_NAME):
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # Adjust based on embedding model output
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the Pinecone Index
index = pc.Index(PINECONE_INDEX_NAME)
print(f"Connected to Pinecone Index: {PINECONE_INDEX_NAME}")

# Function to Generate Embeddings Using vLLM
def generate_embedding(text):
    payload = {
        "input": text,
        "model": "Alibaba-NLP/gte-Qwen2-7B-instruct"
    }
    response = requests.post(VLLM_EMBED_URL, json=payload)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

# Function to Summarize Slide Content using vLLM
def summarize_slide(content):
    prompt = f"""
    You are summarizing a PowerPoint slide for future retrieval. Extract 3-5 key insights.

    Slide Title: {content['title']}
    Slide Text: {content['text']}

    Table Data (if present): {content['tables']}

    Summary:
    """
    payload = {
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(VLLM_CHAT_URL, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Function to Process PowerPoint Files & Store in Pinecone
def process_pptx_files():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    vectors = []

    for pptx_file in os.listdir(FOLDER_PATH):
        if not pptx_file.endswith(".pptx"):
            continue

        pptx_path = os.path.join(FOLDER_PATH, pptx_file)
        pptx_modified_time = os.path.getmtime(pptx_path)

        # Check if the file has been processed before
        c.execute("SELECT last_modified FROM file_log WHERE file_name=?", (pptx_file,))
        result = c.fetchone()

        if result and result[0] == pptx_modified_time:
            print(f"Skipping {pptx_file}, no changes detected.")
            continue

        print(f"Processing: {pptx_file}")
        slides_data = extract_pptx_content(pptx_path)

        for slide in slides_data:
            slide_summary = summarize_slide(slide)
            embedding = generate_embedding(slide_summary)

            metadata = {
                "file_name": pptx_file,
                "slide_number": slide["slide_number"],
                "one_drive_link": f"https://onedrive.com/{pptx_file}",
                "summary": slide_summary
            }

            vectors.append({
                "id": f"{pptx_file}_{slide['slide_number']}",
                "values": embedding,
                "metadata": metadata
            })

        c.execute("REPLACE INTO file_log (file_name, last_modified, last_processed) VALUES (?, ?, ?)",
                  (pptx_file, pptx_modified_time, time.time()))

        print(f"Processed: {pptx_file}")

    conn.commit()
    conn.close()

    if vectors:
        index.upsert(vectors=vectors, namespace="ns1")
        print(f"{len(vectors)} slides uploaded to Pinecone.")

# Run Processing
if __name__ == "__main__":
    process_pptx_files()
