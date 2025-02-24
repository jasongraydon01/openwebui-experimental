# Use a CUDA 12.1 base image with Ubuntu 22.04
FROM python:3.10

# Set working directory
WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI server port
EXPOSE 5000

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]