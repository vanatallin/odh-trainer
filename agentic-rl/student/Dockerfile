FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY train.py .
COPY agent.py .
COPY environment.py .

# Create directories
RUN mkdir -p /checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the training script
CMD ["python", "train.py"]
