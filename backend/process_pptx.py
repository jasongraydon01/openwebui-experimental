import os
import json
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="YOUR_API_KEY")
import pptx
import torch
import pytesseract
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
# from pdf2image import convert_from_path  # If we later convert PPT to PDF

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#  need to work on this
index = pc.Index("your-index-name") 

# Load Local Embedding Model
EMBEDDING_MODEL = "nomic-embed-text"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
embedding_model = AutoModelForCausalLM.from_pretrained(EMBEDDING_MODEL)

# Load Local LLM (for Summarization)
SUMMARIZATION_MODEL = "deepseek-7b"  # Replace with your chosen model
llm_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
llm_model = AutoModelForCausalLM.from_pretrained(SUMMARIZATION_MODEL)

# Folder Path (Local Directory for PowerPoints)
FOLDER_PATH = "./onedrive_pptx"

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
        print(f"❌ OCR Failed for {image_path}: {e}")
        return None

# Function: Summarize Slide Content using Local LLM
def summarize_slide(content):
    prompt = f"""
    You are summarizing a PowerPoint slide for future retrieval. Extract 3-5 key insights.

    Slide Title: {content['title']}
    Slide Text: {content['text']}

    Table Data (if present): {content['tables']}
    Chart Data (if present, OCR applied): {content.get('chart_text', 'No chart text available')}

    Summary:
    """
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = llm_model.generate(**inputs, max_length=250)
    return llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function: Generate Embedding
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embedding = embedding_model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
    return embedding

# Function: Process PowerPoint Files & Store in Pinecone
def process_pptx_files():
    for pptx_file in os.listdir(FOLDER_PATH):
        if pptx_file.endswith(".pptx"):
            pptx_path = os.path.join(FOLDER_PATH, pptx_file)
            slides_data = extract_pptx_content(pptx_path)
            
            for slide in slides_data:
                slide_summary = summarize_slide(slide)
                
                embedding = generate_embedding(slide_summary)
                
                metadata = {
                    "file_name": pptx_file,
                    "slide_number": slide["slide_number"],
                    "one_drive_link": f"https://onedrive.com/{pptx_file}",  # Replace with actual link logic
                    "summary": slide_summary
                }

                index.upsert(vectors=[{
                    "id": f"{pptx_file}_{slide['slide_number']}",
                    "values": embedding,
                    "metadata": metadata
                }])

            print(f"✅ Processed: {pptx_file}")

# Run Processing
if __name__ == "__main__":
    process_pptx_files()