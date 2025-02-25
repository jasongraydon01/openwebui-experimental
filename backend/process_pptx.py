import os
import time
import hashlib
import re
from collections import Counter
from typing import List, Dict, Any
import json

# External dependencies
import pinecone
from pinecone import ServerlessSpec
import pptx
import docling
import requests
import sqlite3
from dotenv import load_dotenv

# -------------------------------
# Configuration and Initialization
# -------------------------------

# Load environment variables
load_dotenv()

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
DATABASE_PATH = os.getenv("DATABASE_PATH")
VLLM_EMBED_URL = os.getenv("VLLM_EMBED_URL")
FOLDER_PATH = os.getenv("FOLDER_PATH")

# Default context window size
DEFAULT_CONTEXT_WINDOW = 1

# Validate critical environment variables
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key is missing. Please set PINECONE_API_KEY in your .env file.")
if not PINECONE_INDEX_NAME:
    raise ValueError("Pinecone index name is missing. Please set PINECONE_INDEX_NAME in your .env file.")
if not DATABASE_PATH:
    raise ValueError("Database path is missing. Please set DATABASE_PATH in your .env file.")
if not FOLDER_PATH:
    raise ValueError("Folder path is missing. Please set FOLDER_PATH in your .env file.")
if not VLLM_EMBED_URL:
    raise ValueError("vLLM embed URL is missing. Please set VLLM_EMBED_URL in your .env file.")

# Initialize Pinecone client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if not pc.has_index(PINECONE_INDEX_NAME):
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=3584,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Created new Pinecone index: {PINECONE_INDEX_NAME}")

# Wait for index to be ready
while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
    print("Waiting for Pinecone index to be ready...")
    time.sleep(1)

# Connect to the index
index = pc.Index(PINECONE_INDEX_NAME)
print(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")

# -------------------------------
# Database Functions
# -------------------------------

def init_database():
    """Initialize the SQLite database with necessary tables."""
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    
    # Create file_log table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS file_log (
        file_name TEXT PRIMARY KEY,
        file_hash TEXT,
        last_modified REAL,
        last_processed REAL,
        slide_count INTEGER
    )
    ''')
    
    # Create slides table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS slides (
        vector_id TEXT PRIMARY KEY,
        file_name TEXT,
        slide_number INTEGER,
        content TEXT,
        keywords TEXT,
        FOREIGN KEY (file_name) REFERENCES file_log (file_name) ON DELETE CASCADE
    )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully")

def remove_file_data(file_name):
    """Remove all data associated with a file from the database and Pinecone."""
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    
    # Get all vector IDs for this file
    c.execute("SELECT vector_id FROM slides WHERE file_name=?", (file_name,))
    vector_ids = [row[0] for row in c.fetchall()]
    
    # Remove the file record and slides from the database
    c.execute("DELETE FROM file_log WHERE file_name=?", (file_name,))
    c.execute("DELETE FROM slides WHERE file_name=?", (file_name,))
    conn.commit()
    conn.close()
    
    # Remove all vectors associated with this file from Pinecone
    if vector_ids:
        # Process in batches of 100 to avoid overwhelming the API
        for i in range(0, len(vector_ids), 100):
            batch = vector_ids[i:i+100]
            index.delete(ids=batch, namespace="ns1")
        print(f"Removed {len(vector_ids)} vectors for {file_name} from Pinecone")
    
    return len(vector_ids) if vector_ids else 0

def check_for_removed_files():
    """Check for files that have been removed and clean up their data."""
    current_files = set(os.listdir(FOLDER_PATH))
    
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("SELECT file_name FROM file_log")
    files_in_db = set(row[0] for row in c.fetchall())
    conn.close()
    
    # Determine which files have been removed
    removed_files = files_in_db - current_files
    
    if removed_files:
        print(f"Detected {len(removed_files)} removed files")
        for file in removed_files:
            vectors_removed = remove_file_data(file)
            print(f"Cleaned up {file}: removed {vectors_removed} vectors")
    
    return removed_files

# -------------------------------
# File Processing Functions
# -------------------------------

def get_file_hash(file_path):
    """Calculate MD5 hash of a file for change detection."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def extract_pptx_content(pptx_path):
    """Extract content from PowerPoint file using both Docling and python-pptx."""
    # Use Docling for consistent document parsing
    doc = docling.load(pptx_path)
    
    # Use python-pptx for additional metadata and table extraction
    prs = pptx.Presentation(pptx_path)
    slides_data = []
    
    for i, (slide, docling_slide) in enumerate(zip(prs.slides, doc.chunks())):
        slide_title = slide.shapes.title.text if slide.shapes.title else ""
        slide_text = docling_slide.text
        
        # Extract tables
        tables = []
        for shape in slide.shapes:
            if shape.has_table:
                table_data = [[cell.text for cell in row.cells] for row in shape.table.rows]
                tables.append(table_data)

        slides_data.append({
            "slide_number": i + 1,
            "title": slide_title,
            "text": slide_text,
            "tables": tables
        })
    
    return slides_data

def extract_keywords(text, tables, max_keywords=15):
    """Extract keywords from text using a statistical approach similar to TF-IDF."""
    # Combine all text
    table_text = ""
    for table in tables:
        table_text += " ".join([" ".join(row) for row in table])
    
    all_text = f"{text} {table_text}"
    
    # Clean and normalize the text
    all_text = all_text.lower()
    all_text = re.sub(r'[^\w\s]', ' ', all_text)
    
    # Tokenize
    words = all_text.split()
    
    # Common stopwords to filter out
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
                'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such',
                'when', 'who', 'how', 'where', 'why', 'is', 'are', 'was', 'were', 'be', 'been',
                'have', 'has', 'had', 'do', 'does', 'did', 'but', 'at', 'by', 'with', 'from',
                'here', 'there', 'to', 'of', 'for', 'in', 'on', 'about', 'into', 'over', 'after'}
    
    # Filter words
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Get word frequencies
    word_counts = Counter(filtered_words)
    
    # Get top single keywords
    keywords = [word for word, count in word_counts.most_common(max_keywords)]
    
    # Extract important bigrams (2-word phrases)
    bigrams = []
    for i in range(len(words) - 1):
        if (words[i] not in stopwords and words[i+1] not in stopwords and 
            len(words[i]) > 2 and len(words[i+1]) > 2):
            bigrams.append(f"{words[i]} {words[i+1]}")
    
    bigram_counts = Counter(bigrams)
    top_bigrams = [bigram for bigram, count in bigram_counts.most_common(5)]
    
    # Combine keywords and bigrams, limiting to max_keywords total
    all_keywords = keywords[:10] + top_bigrams
    
    return ", ".join(all_keywords[:max_keywords])

def generate_embedding(text):
    """Generate embedding vector using vLLM API."""
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

def process_slide_with_context(slides_data, current_index, file_name, context_window=1):
    """Process a slide with context from adjacent slides.
    
    Args:
        slides_data: List of all slides in the presentation
        current_index: Index of the current slide to process
        file_name: Name of the PowerPoint file
        context_window: Number of slides before/after to include as context
    """
    current_slide = slides_data[current_index]
    slide_number = current_slide["slide_number"]
    slide_text = current_slide["text"]
    slide_title = current_slide["title"] or f"Slide {slide_number}"
    table_content = current_slide["tables"]
    
    # Convert current slide's table content to text
    table_text = ""
    for table in table_content:
        table_text += " ".join([" ".join(row) for row in table])
    
    # Start building the content with the main slide
    content_parts = []
    
    # Add current slide as main content with clear marking
    main_content = f"Title: {slide_title}\n\nContent: {slide_text}"
    if table_text:
        main_content += f"\n\nTables: {table_text}"
    
    content_parts.append(f"--- MAIN SLIDE {slide_number} ---\n{main_content}")
    
    # Add previous slides context
    for i in range(max(0, current_index - context_window), current_index):
        prev_slide = slides_data[i]
        prev_number = prev_slide["slide_number"]
        prev_title = prev_slide["title"] or f"Slide {prev_number}"
        # Include a preview of previous slide content
        prev_content = f"Title: {prev_title}\n\nPreview: {prev_slide['text'][:150]}..."
        content_parts.insert(0, f"--- PREVIOUS SLIDE {prev_number} ---\n{prev_content}")
    
    # Add next slides context
    for i in range(current_index + 1, min(len(slides_data), current_index + context_window + 1)):
        next_slide = slides_data[i]
        next_number = next_slide["slide_number"]
        next_title = next_slide["title"] or f"Slide {next_number}"
        # Include a preview of next slide content
        next_content = f"Title: {next_title}\n\nPreview: {next_slide['text'][:150]}..."
        content_parts.append(f"--- NEXT SLIDE {next_number} ---\n{next_content}")
    
    # Join all parts to create the final content with context
    full_content_with_context = "\n\n".join(content_parts)
    
    # Extract keywords from current slide only (not from context)
    keywords = extract_keywords(slide_text, table_content)
    
    # Create vector ID
    vector_id = f"{file_name}_{slide_number}"
    
    # Track context slides for metadata
    context_slides = []
    for i in range(max(0, current_index - context_window), current_index):
        context_slides.append(slides_data[i]["slide_number"])
    for i in range(current_index + 1, min(len(slides_data), current_index + context_window + 1)):
        context_slides.append(slides_data[i]["slide_number"])
    
    return {
        "vector_id": vector_id,
        "slide_number": slide_number,
        "content": full_content_with_context,
        "keywords": keywords,
        "context_slides": context_slides
    }

# -------------------------------
# Main Processing Function
# -------------------------------

def process_pptx_files(context_window_size=1):
    """Main function to process PowerPoint files and update the vector database.
    
    Args:
        context_window_size: Number of slides before/after to include as context
    """
    # Initialize database and check for removed files
    init_database()
    check_for_removed_files()
    
    # Track vectors to upload
    vectors_to_upload = []
    
    # Connect to database
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    
    # Update slides table to include context information if needed
    c.execute("PRAGMA table_info(slides)")
    columns = [column[1] for column in c.fetchall()]
    if "context_slides" not in columns:
        c.execute("ALTER TABLE slides ADD COLUMN context_slides TEXT")
        print("Added context_slides column to slides table")
    
    # Process each PowerPoint file in the folder
    for pptx_file in os.listdir(FOLDER_PATH):
        if not pptx_file.endswith(".pptx"):
            continue
            
        pptx_path = os.path.join(FOLDER_PATH, pptx_file)
        file_hash = get_file_hash(pptx_path)
        last_modified = os.path.getmtime(pptx_path)
        
        # Check if file has changed
        c.execute("SELECT file_hash FROM file_log WHERE file_name=?", (pptx_file,))
        result = c.fetchone()
        
        if result and result[0] == file_hash:
            print(f"Skipping {pptx_file}: No changes detected")
            continue
            
        # If file has changed or is new, process it
        if result:
            # Remove existing data first
            vectors_removed = remove_file_data(pptx_file)
            print(f"Updating {pptx_file}: Removed {vectors_removed} previous vectors")
        else:
            print(f"Processing new file: {pptx_file}")
            
        # Extract content from PowerPoint
        slides_data = extract_pptx_content(pptx_path)
        print(f"Extracted {len(slides_data)} slides from {pptx_file}")
        
        # Process each slide with context from adjacent slides
        for i, _ in enumerate(slides_data):
            processed_slide = process_slide_with_context(slides_data, i, pptx_file, context_window_size)
            
            # Generate embedding for the content with context
            embedding = generate_embedding(processed_slide["content"])
            
            # Convert context slides list to JSON string for storage
            context_slides_json = json.dumps(processed_slide.get("context_slides", []))
            
            # Prepare metadata
            metadata = {
                "file_name": pptx_file,
                "slide_number": processed_slide["slide_number"],
                "one_drive_link": f"https://onedrive.com/{pptx_file}",
                "content_preview": processed_slide["content"][:1000],
                "keywords": processed_slide["keywords"],
                "last_modified": last_modified,
                "context_slides": processed_slide.get("context_slides", [])
            }
            
            # Add to vectors batch
            vectors_to_upload.append({
                "id": processed_slide["vector_id"],
                "values": embedding,
                "metadata": metadata
            })
            
            # Store in database
            c.execute(
                "INSERT INTO slides (vector_id, file_name, slide_number, content, keywords, context_slides) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    processed_slide["vector_id"],
                    pptx_file,
                    processed_slide["slide_number"],
                    processed_slide["content"],
                    processed_slide["keywords"],
                    context_slides_json
                )
            )
        
        # Update file log
        c.execute(
            "REPLACE INTO file_log (file_name, file_hash, last_modified, last_processed, slide_count) VALUES (?, ?, ?, ?, ?)",
            (pptx_file, file_hash, last_modified, time.time(), len(slides_data))
        )
        
        print(f"Processed {pptx_file}: {len(slides_data)} slides with context window size {context_window_size}")
    
    # Commit database changes
    conn.commit()
    conn.close()
    
    # Upload vectors to Pinecone in batches
    if vectors_to_upload:
        total_vectors = len(vectors_to_upload)
        batch_size = 50  # Optimal batch size for Pinecone
        
        for i in range(0, total_vectors, batch_size):
            batch = vectors_to_upload[i:i+batch_size]
            index.upsert(vectors=batch, namespace="ns1")
            print(f"Uploaded batch {i//batch_size + 1}/{(total_vectors-1)//batch_size + 1}: {len(batch)} vectors")
        
        print(f"Successfully uploaded {total_vectors} vectors to Pinecone")
    else:
        print("No new or updated files to process")

# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":
    print("Starting PowerPoint processing pipeline")
    process_pptx_files()
    print("Processing complete")