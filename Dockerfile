FROM python:3.11-slim

# Install system dependencies for OpenCV and Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (Render automatically assigns one, but standard is usually needed for local docker)
EXPOSE 10000

# Start Gunicorn server (Render automatically injects the PORT env variable)
CMD gunicorn app:app -b 0.0.0.0:${PORT:-10000}
