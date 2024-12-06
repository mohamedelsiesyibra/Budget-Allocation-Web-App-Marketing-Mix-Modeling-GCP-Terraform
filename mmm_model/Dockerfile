# Use the official Python 3.10 slim image as base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install pip and upgrade it
RUN pip install --no-cache-dir --upgrade pip

# Install JAX and JAXLIB separately due to special installation requirements
RUN pip install --no-cache-dir --upgrade "jax[cpu]==0.4.18" -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Install other dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script into the container
COPY model.py .

# Set the entrypoint for the training script
ENTRYPOINT ["python", "model.py"]
