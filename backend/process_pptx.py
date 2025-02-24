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

# Function to Check for Removed Files and Clean Up
def check_for_removed_files():
    current_files = set(os.listdir(FOLDER_PATH))
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    
    c.execute("SELECT file_name FROM file_log")
    files_in_db = set(row[0] for row in c.fetchall())
    
    removed_files = files_in_db - current_files
    if removed_files:
        print(f"Removed files detected: {removed_files}")
        for file in removed_files:
            c.execute("DELETE FROM file_log WHERE file_name=?", (file,))
            slide_ids = [f"{file}_{i}" for i in range(1, 101)]
            index.delete(ids=slide_ids, namespace="ns1")
            print(f"Removed all slide embeddings for {file} from Pinecone.")
    
    conn.commit()
    conn.close()

def extract_pptx_content(pptx_path):
    prs = pptx.Presentation(pptx_path)
    slides_data = []
    
    for i, slide in enumerate(prs.slides):
        slide_text = []
        slide_title = slide.shapes.title.text if slide.shapes.title else ""
        
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                slide_text.append(shape.text)
        
        tables = []
        for shape in slide.shapes:
            if shape.has_table:
                table_data = [[cell.text for cell in row.cells] for row in shape.table.rows]
                tables.append(table_data)

        slides_data.append({
            "slide_number": i + 1,
            "title": slide_title,
            "text": " ".join(slide_text),
            "tables": tables
        })
    
    return slides_data

def generate_embedding(text):
    try:
        payload = {
            "input": text,
            "model": "Alibaba-NLP/gte-Qwen2-7B-instruct"
        }
        response = requests.post(VLLM_EMBED_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except requests.exceptions.RequestException as e:
        print(f"Error generating embedding: {e}")
        raise

def summarize_slide(content):
    try:
        prompt = f"""
        You are summarizing a PowerPoint slide for future retrieval. Extract 3-5 key insights.

        Slide Title: {content['title']}
        Slide Text: {content['text']}

        Table Data (if present): {content['tables']}

        Summary:
        """
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 300
        }
        response = requests.post(VLLM_CHAT_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error generating summary: {e}")
        raise

def process_pptx_files():
    check_for_removed_files()  # Add check for removed files
    
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    vectors = []

    try:
        for pptx_file in os.listdir(FOLDER_PATH):
            if not pptx_file.endswith(".pptx"):
                continue

            pptx_path = os.path.join(FOLDER_PATH, pptx_file)
            pptx_modified_time = os.path.getmtime(pptx_path)

            c.execute("SELECT last_modified FROM file_log WHERE file_name=?", (pptx_file,))
            result = c.fetchone()

            if result and result[0] == pptx_modified_time:
                print(f"Skipping {pptx_file}, no changes detected.")
                continue

            # If file was previously processed, remove old embeddings
            if result:
                slide_ids = [f"{pptx_file}_{i}" for i in range(1, 101)]
                index.delete(ids=slide_ids, namespace="ns1")
                print(f"Removing old embeddings for {pptx_file}")

            print(f"Processing: {pptx_file}")
            slides_data = extract_pptx_content(pptx_path)

            for slide in slides_data:
                try:
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
                except Exception as e:
                    print(f"Error processing slide {slide['slide_number']} in {pptx_file}: {e}")
                    continue

            c.execute("REPLACE INTO file_log (file_name, last_modified, last_processed) VALUES (?, ?, ?)",
                    (pptx_file, pptx_modified_time, time.time()))
            print(f"Processed: {pptx_file}")

        if vectors:
            # Upload in smaller batches to avoid timeout issues
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch, namespace="ns1")
                print(f"Uploaded batch of {len(batch)} vectors to Pinecone")

    except Exception as e:
        print(f"Error during processing: {e}")
        raise
    finally:
        conn.commit()
        conn.close()

# Run Processing
if __name__ == "__main__":
    process_pptx_files()
