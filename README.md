# OpenWebUI-Setup: Local AI & RAG System

## Overview
This repository contains the implementation of a **local AI-powered system** using **Open WebUI** as the interface. The system is designed to be fully modular and containerized, allowing for deployment on both a local machine and a **GPU cloud instance** for up to **100 consultants**. 

## Project Goals
- **Build a Retrieval-Augmented Generation (RAG) system** that integrates with OneDrive for enterprise knowledge retrieval.
- **Develop AI agent workflows** to handle multi-step queries beyond basic RAG.
- **Customize Open WebUI** for structured AI interactions, folder organization, and metadata displays.
- **Ensure portability** by packaging everything into a Docker container that can run on both local and cloud GPU instances.

---

## Roadmap & Execution Plan

### **Phase 1: Core RAG Pipeline**
#### **1. OneDrive Data Ingestion**
- Authenticate and connect to **Microsoft OneDrive**.
- Extract and preprocess files (**PowerPoint, PDFs, Word, Excel**).
- Define metadata structures (file names, slide numbers, etc.).

#### **2. Embedding & Vector Storage**
- Convert text into embeddings using **nomic-embed-text**.
- Store embeddings in **Pinecone** for efficient retrieval.
- Define retrieval strategies (**chunking, metadata filtering**).

#### **3. Query Processing & Retrieval**
- Implement **search logic** for retrieving relevant documents.
- Return references (slide numbers, file links) **inside Open WebUI**.

---

### **Phase 2: AI Agent Workflows**
#### **1. Request Classification System**
- Build a lightweight backend service to:
  - Determine if a request requires **RAG retrieval or general AI processing**.
  - Route the request accordingly.
- Store logic in **Python or TypeScript** (without external frameworks initially).

#### **2. Agent Execution**
- Implement AI workflows beyond simple retrieval:
  - **Multi-step queries**
  - **Summarization & key insight extraction**

---

### **Phase 3: Open WebUI Enhancements**
#### **1. UI & Structural Improvements**
- Modify **Open WebUI** to:
  - Support **folder organization** for structured data access.
  - Display metadata (file names, sources, etc.).
  - Improve prompt handling for different workflows.

#### **2. Backend Integration**
- Modify Open WebUI to **send queries to the request classification system**.
- Ensure proper routing between Open WebUI and the backend AI pipeline.

---

### **Phase 4: Deployment & Containerization**
#### **1. Containerize the System**
- Build a **Docker image** that includes:
  - Open WebUI
  - Backend for workflow routing
  - Vector DB connection (**Pinecone**)
- Ensure **persistent storage** (e.g., mounted volumes for state retention).

#### **2. Test Portability**
- Deploy the container on a **cloud GPU instance (RunPod)** with minimal configuration changes.
- Validate **model performance and system behavior** in a near-production environment.

---

## Final Repository Structure
```plaintext
openwebui-setup/
│── backend/                     # Backend logic (request classification, AI workflows)
│   ├── __init__.py
│   ├── main.py                   # Main API entry point
│   ├── routes.py                  # API routes (handles RAG vs. general AI processing)
│   ├── agent_workflows.py         # AI agent execution logic
│   ├── rag_pipeline.py            # RAG retrieval functions
│   ├── vector_db.py               # Pinecone vector database interactions
│   ├── onedrive_ingestion.py      # OneDrive data extraction & processing
│
│── openwebui/                     # Open WebUI customizations
│   ├── docker-compose.yml         # Docker Compose for WebUI + backend
│   ├── webui-mods/                # UI modifications
│   │   ├── components/            # UI components (e.g., folder structure UI)
│   │   ├── config/                # WebUI-specific settings
│   │   ├── api.js                 # Modify how WebUI sends requests to backend
│   │   ├── routes.js              # Custom UI routing
│
│── models/                        # Model-related setup
│   ├── model_config.yaml          # Defines which models are used
│   ├── download_models.sh         # Script to pull models locally
│
│── scripts/                       # Utility scripts for setup and maintenance
│   ├── init_env.sh                # Environment setup script
│   ├── deploy_container.sh        # Automates deployment to GPU instance
│
│── config/                        # Configuration files
│   ├── .env                       # API keys, secrets, etc.
│   ├── settings.yaml               # System-wide configurations
│
│── Dockerfile                     # Main Dockerfile for full system containerization
│── docker-compose.yml              # Docker Compose for local & cloud deployment
│── requirements.txt                # Python dependencies
│── README.md                       # Documentation
│── .gitignore                      # Ignore logs, credentials, etc.
```

---

## Current Implementation
Currently, this repository only contains a **basic Open WebUI setup** with the following `docker-compose.yml`:
```yaml
version: '3'
services:
  openwebui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    volumes:
      - open-webui:/app/backend/data
volumes:
  open-webui:
```

---

## Next Steps
- Implement **OneDrive ingestion & file processing**.
- Build the **embedding pipeline & integrate with Pinecone**.
- Implement **retrieval & test querying** inside Open WebUI.
- Develop the **request classification system** for routing RAG vs. general AI processing.
- Modify **Open WebUI UI** for better **folder structure + metadata display**.
- **Containerize the entire system** and validate portability on a **RunPod GPU instance**.

This document will be updated as the project progresses. 🚀
