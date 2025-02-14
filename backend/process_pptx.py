import os
import time
import pinecone
import pptx
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import ollama
import sqlite3

# Load environment variables from .env file
load_dotenv()

# Retrieve API Key from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "openwebui-setup"
DATABASE_PATH = "file_log.db"  # SQLite database path for tracking files

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

# Initialize SQLite Database for File Tracking
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS file_log (
                    file_name TEXT PRIMARY KEY,
                    last_modified TIMESTAMP,
                    last_processed TIMESTAMP
                 )''')
    conn.commit()
    conn.close()

# Folder Path (Local Directory for PowerPoints)
FOLDER_PATH = "test-rag"

# Function to Check for Removed Files and Clean Up Corresponding Data
def check_for_removed_files():
    current_files = set(os.listdir(FOLDER_PATH))  # Current files in the folder
    
    # Connect to the SQLite database
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    # Get the list of tracked files in the database
    c.execute("SELECT file_name FROM file_log")
    files_in_db = set(row[0] for row in c.fetchall())

    # Determine which files have been removed
    removed_files = files_in_db - current_files
    if removed_files:
        print(f"Removed files detected: {removed_files}")
        for file in removed_files:
            # Remove the file record from the database
            c.execute("DELETE FROM file_log WHERE file_name=?", (file,))

            # Remove all slides associated with this file from Pinecone
            slide_ids = [f"{file}_{i}" for i in range(1, 101)]  # Assuming up to 100 slides per deck
            index.delete(ids=slide_ids, namespace="ns1")
            print(f"Removed all slide embeddings for {file} from Pinecone.")

    conn.commit()
    conn.close()

# Function to Extract Text & Tables from PowerPoint
def extract_pptx_content(pptx_path):
    prs = pptx.Presentation(pptx_path)
    slides_data = []
    
    for i, slide in enumerate(prs.slides):
        slide_text = []
        slide_title = slide.shapes.title.text if slide.shapes.title else ""
        
        # Extract text from slide
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                slide_text.append(shape.text)
        
        # Extract tables
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

# Function to Generate Embeddings Using Ollama
def generate_embedding(text):
    response = ollama.embed(model="nomic-embed-text", input=text)
    embedding = response['embeddings']
    return [item for sublist in embedding for item in sublist]  # Flatten the list

# Function to Summarize Slide Content
def summarize_slide(content):
    prompt = f"""
    You are summarizing a PowerPoint slide for future retrieval. Extract 3-5 key insights.

    Slide Title: {content['title']}
    Slide Text: {content['text']}

    Table Data (if present): {content['tables']}

    Summary:
    """
    response = ollama.chat(model="mistral:7b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Function to Process PowerPoint Files & Store in Pinecone
def process_pptx_files():
    init_db()  # Initialize database if not already done
    check_for_removed_files()  # Ensure deleted files are cleaned up

    vectors = []  # Collect vectors to upload in batches
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    for pptx_file in os.listdir(FOLDER_PATH):
        if not pptx_file.endswith(".pptx"):
            continue

        pptx_path = os.path.join(FOLDER_PATH, pptx_file)
        pptx_modified_time = os.path.getmtime(pptx_path)  # Get last modified time

        # Check if the file is in the log and if it has been modified
        c.execute("SELECT last_modified FROM file_log WHERE file_name=?", (pptx_file,))
        result = c.fetchone()

        # Skip if the file has not changed
        if result and result[0] == pptx_modified_time:
            print(f"Skipping {pptx_file}, no changes detected.")
            continue

        # If modified, remove old slide embeddings from Pinecone
        if result:
            slide_ids = [f"{pptx_file}_{i}" for i in range(1, 101)]  # Assuming up to 100 slides
            index.delete(ids=slide_ids, namespace="ns1")
            print(f"Updated: {pptx_file}. Removing old embeddings before reprocessing.")

        print(f"Processing: {pptx_file}")
        slides_data = extract_pptx_content(pptx_path)

        for slide in slides_data:
            slide_summary = summarize_slide(slide)
            embedding = generate_embedding(slide_summary)

            # Prepare metadata
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

        # Update the database with the latest modified and processed times
        c.execute("REPLACE INTO file_log (file_name, last_modified, last_processed) VALUES (?, ?, ?)",
                  (pptx_file, pptx_modified_time, time.time()))

        print(f"Processed: {pptx_file}")

    conn.commit()
    conn.close()

    # Upload all vectors in batches to Pinecone
    if vectors:
        index.upsert(vectors=vectors, namespace="ns1")
        print(f"{len(vectors)} slides uploaded to Pinecone.")

# Run Processing
if __name__ == "__main__":
    process_pptx_files()