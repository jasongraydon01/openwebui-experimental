# vLLM Dockerfile for M3 Mac (ARM64) - CPU Optimized

FROM python:3.12-slim-bookworm AS base

ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster pip installs
RUN pip install uv

WORKDIR /workspace

# Copy and install common requirements
COPY requirements-common.txt requirements-common.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements-common.txt

# Install specific ARM64 packages (no CUDA)
# Note: For CUDA, you would need nvidia/cuda base image and install CUDA-related packages here.
# For M3, we rely on CPU acceleration and libraries that support it.
COPY requirements-arm64.txt requirements-arm64.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements-arm64.txt

# Copy vLLM code
COPY . .

# Install vLLM itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system .

# Expose the API port
EXPOSE 8000

# Run the vLLM OpenAI API server
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server"]

# Add requirements-arm64.txt
# This file will contain packages specific to the arm64 architecture, and cpu only.
# example contents of requirements-arm64.txt:
# torch
# transformers

# Add requirements-common.txt
# This file will contain packages common accross all architectures.
# example contents of requirements-common.txt:
# fastapi
# uvicorn
# pydantic
# tokenizers