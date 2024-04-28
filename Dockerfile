# Use the official Python base image with Python 3.9
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords

# Copy the necessary files
COPY myflow.py .
COPY data_processing.py .
COPY open_ave_data.xlsx .

# Set environment variables
ENV USERNAME=sample
ENV OMP_NUM_THREADS=1

# Run the Metaflow pipeline when the container starts
CMD ["python", "myflow.py", "run"]
