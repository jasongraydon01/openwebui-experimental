# 📌 OneDrive → Vector Database (RAG Pipeline)

## **Overview**
This pipeline is designed to **ingest PowerPoint files from OneDrive**, generate **summarized embeddings**, and store them in a **vector database (Pinecone)** for efficient retrieval.  

Instead of embedding entire documents, we use a **hybrid approach** that generates:  
- **A summary** of each PowerPoint file.  
- **Key insights from tables and charts** on slides.  
- **Metadata** to improve searchability.

---

## **🚀 Pipeline Steps**
### **1️⃣ Ensure OneDrive Files Sync Locally**
- Ensure OneDrive is **fully synced** on the local machine or GPU instance.
- **All PowerPoint files should be accessible locally** for processing.
- Make sure the OneDrive folder is **set to always keep files available offline** (avoid "Files on Demand" mode).

---

### **2️⃣ Define the Root Folder for Ingestion**
- The script/API takes a **root folder path** that contains all synced **OneDrive PowerPoint files**.
- **For V1:** Focus only on **a single folder** (e.g., `/OneDrive/Documents/AI_PPTs/`).
- Future versions can support **recursive folder scanning** or **additional file types**.

---

### **3️⃣ Implement a Sync Mechanism (Modified-Date Based)**
- The pipeline **only processes files that have changed** to minimize redundancy.
- Steps:
  1. Maintain a **log or database** (`processed_files.json`) that tracks each file’s **last modified date**.
  2. If a file’s modified date **has not changed**, **skip processing**.
  3. If a file **has changed**, reprocess and update the log.
  4. If a file **was deleted**, remove its embeddings from the vector database.

---

### **4️⃣ Traverse OneDrive Folder & Extract PowerPoint Content**
- The script **scans all `.pptx` files** in the target folder.
- **For V1:** Ignore all other file types.
- Extract **text, tables, and charts** from each slide using `python-pptx`:
  - **Text:** Slide titles & bullet points.
  - **Tables:** Extracted & summarized by an LLM.
  - **Charts/Graphs:** Converted to images, analyzed via OCR/LLM for insights.

---

### **5️⃣ Summarization & Embedding**
- Each slide is **processed by an LLM** to generate:
  - **A structured summary of the PowerPoint file.**
  - **Key insights from tables (if present).**
  - **Key takeaways from charts/graphs (if present).**
- The LLM **does not store raw table/chart data**—only summarized insights.
- Convert:
  - **Summary → 1 embedding**
  - **Key insights → Separate embeddings**
- Metadata stored includes:
  - **File Name**
  - **Slide Number**
  - **OneDrive Link**
  - **Last Modified Date**
  - **Embedding Type (summary/key passage)**

---

### **6️⃣ Store Embeddings in Vector Database (Pinecone)**
- Insert embeddings **in batches** to improve performance.
- Attach **metadata** to embeddings to support advanced filtering.
- If a file is **deleted from OneDrive**, remove its corresponding vectors.

---

### **7️⃣ Query Processing & AI Retrieval**
- When a user queries the system:
  1. Convert the **query into an embedding**.
  2. Search Pinecone for **matching summaries & key insights**.
  3. Return:
     - **Relevant file names**
     - **Slide numbers**
     - **OneDrive links**
     - **A confidence score**
  4. If the AI **is not confident**, suggest refining the query or show related results.

---

## **🎯 Goals for V1**
✅ **Minimize redundant processing** → Only update embeddings when a file changes.  
✅ **Prioritize useful retrieval** → Use **summary + key insights** instead of storing full documents.  
✅ **Ensure metadata-driven searchability** → Improve retrieval with **file names, slide numbers, and links**.  
✅ **Deploy a scalable approach** → Start simple with **PowerPoint-only ingestion**, then expand to other file types.  

---

## **🔮 Future Enhancements**
- **Multi-folder support**: Allow scanning multiple OneDrive directories.
- **Additional file types**: Expand to PDFs, Word docs, and Excel sheets.
- **Hybrid search**: Combine vector search with **keyword-based retrieval**.
- **Automated reprocessing**: Implement a scheduled job to **run daily** and update embeddings.

---

## **💡 Why This Approach?**
This pipeline **balances retrieval accuracy, performance, and cost** by:
- **Avoiding full-document chunking** (which would be expensive to store and retrieve).  
- **Prioritizing AI-generated summaries** to capture key insights.  
- **Keeping metadata intact** for easy navigation and reference.  

This ensures an **efficient and practical RAG implementation** that provides **high-quality search results** without unnecessary storage overhead.

---

## **📌 Next Steps**
Would you like to implement this pipeline as a **Python script**, or do you prefer an **API-based solution** for greater flexibility? 🚀
ß