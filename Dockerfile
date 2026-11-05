# Production Dockerfile for AI Fall Detection & Activity Monitoring
# Optimized for Headless Cloud Containers (Render, Railway, AWS EC2, GCP, Azure)

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system libraries needed by OpenCV, MediaPipe, and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project code and model files
COPY . .

# Ensure data directories exist
RUN mkdir -p fall_logs templates

# Expose default web server port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/api/telemetry || exit 1

# Start FastAPI web server on port 8000
CMD ["python", "web_server.py", "--host", "0.0.0.0", "--port", "8000", "--source", "fall01.mp4"]
