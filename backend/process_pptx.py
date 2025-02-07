import os
import time
import pinecone
import pptx
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import ollama

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
    """Generates an embedding for a given text using Ollama's nomic-embed-text."""
    response = ollama.embed(model="nomic-embed-text", input=text)
    embedding = response['embeddings']  # Ensure it's a list of floats
    # Flatten the embedding if it's a list of lists
    flat_embedding = [item for sublist in embedding for item in sublist]  # Flatten the list
    
    return flat_embedding  # Directly return the flattened embedding as a list of floats

# Function to Summarize Using Ollama (Mistral 7B)
def summarize_slide(content):
    """Summarizes a PowerPoint slide using Ollama's Mistral 7B model."""
    prompt = f"""
    You are summarizing a PowerPoint slide for future retrieval. Extract 3-5 key insights.

    Slide Title: {content['title']}
    Slide Text: {content['text']}

    Table Data (if present): {content['tables']}
    Chart Data (if present, OCR applied): {content.get('chart_text', 'No chart text available')}

    Summary:
    """
    response = ollama.chat(model="mistral:7b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]  # Extracts the AI-generated summary

# Folder Path (Local Directory for PowerPoints)
FOLDER_PATH = "test-rag"

# Function: Extract Text & Tables from PowerPoint
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
                table_data = []
                for row in shape.table.rows:
                    table_data.append([cell.text for cell in row.cells])
                tables.append(table_data)

        # Extract charts as images & apply OCR
        chart_text = None
        chart_path = f"./temp_chart_{i}.png"
        for shape in slide.shapes:
            if "chart" in shape.__class__.__name__.lower():
                slide.export(chart_path, format="PNG")
                if os.path.exists(chart_path):
                    chart_text = extract_text_from_image(chart_path)

        slides_data.append({
            "slide_number": i + 1,
            "title": slide_title,
            "text": " ".join(slide_text),
            "tables": tables,
            "chart_text": chart_text,
            "chart_path": chart_path if os.path.exists(chart_path) else None
        })
    
    return slides_data

# Function: Extract Text from Image (OCR for Charts)
def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"OCR Failed for {image_path}: {e}")
        return None

# Function to Process PowerPoint Files & Store in Pinecone
def process_pptx_files():
    vectors = []  # Collect vectors to upload in batches
    for pptx_file in os.listdir(FOLDER_PATH):
        if pptx_file.endswith(".pptx"):
            pptx_path = os.path.join(FOLDER_PATH, pptx_file)
            slides_data = extract_pptx_content(pptx_path)
            
            for slide in slides_data:
                slide_summary = summarize_slide(slide)
                
                # Generate embedding for the slide summary
                embedding = generate_embedding(slide_summary)
                
                # Prepare metadata for Pinecone
                metadata = {
                    "file_name": pptx_file,
                    "slide_number": slide["slide_number"],
                    "one_drive_link": f"https://onedrive.com/{pptx_file}",  # Replace with actual link logic
                    "summary": slide_summary
                }

                # Add to vectors list
                vectors.append({
                    "id": f"{pptx_file}_{slide['slide_number']}",
                    "values": embedding,  # Embedding is a list of floats
                    "metadata": metadata
                })

            print(f"Processed: {pptx_file}")
            # Print the first 5 vectors in the list
    
    # Upload all vectors in batches to Pinecone
    if vectors:
        index.upsert(vectors=vectors, namespace="ns1")
        print(f"{len(vectors)} vectors uploaded to Pinecone.")

# Run Processing
if __name__ == "__main__":
    process_pptx_files()