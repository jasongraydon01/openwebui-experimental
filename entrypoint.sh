#!/bin/bash

set -e  # Exit immediately if any command fails

echo "Installing LibreOffice..."
apt update && apt install -y libreoffice && apt clean

echo "Initializing the database..."
python /app/backend/init_db.py

echo "Initializing Pinecone index..."
python /app/backend/pinecone_index.py

# Process PowerPoint files after models are downloaded
echo "Processing PowerPoint files..."
python /app/backend/process_pptx.py