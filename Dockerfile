# Base image with Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    imagemagick \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set ImageMagick policy to allow editing
RUN echo 'policy.xml patch' \
    && sed -i 's/none/read|write/' /etc/ImageMagick-6/policy.xml

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code
COPY . .

# Set environment variable for Flask
ENV FLASK_APP=app.py
EXPOSE 5000

# Start Flask app
CMD ["python", "main.py"]
