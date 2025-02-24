# Use a CUDA 12.1 base image with Ubuntu 22.04
ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS base

ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update -y && apt-get install -y \
    software-properties-common git curl sudo \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

# Install PyTorch with CUDA 12 support
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
RUN pip install --no-cache-dir vllm

# Set work directory
WORKDIR /workspace

# Expose API server port
EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", \
            "--host", "0.0.0.0", "--port", "8000", \
            "--model", "Qwen/Qwen2.5-14B-Instruct", \
            "--gpu-memory-utilization", "0.95"]