FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    imagemagick \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Fix ImageMagick policy issue (optional)
RUN echo "<policy domain=\"coder\" rights=\"read | write\" pattern=\"PDF\" />" >> /etc/ImageMagick-6/policy.xml

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your FastAPI app
COPY . .

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
