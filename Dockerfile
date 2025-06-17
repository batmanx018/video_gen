FROM python:3.10-slim

WORKDIR /app

# Install FFmpeg, ImageMagick, and dependencies for TextClip fonts
RUN apt-get update && apt-get install -y \
    ffmpeg \
    imagemagick \
    libx11-dev \
    libxext6 \
    libsm6 \
    libxrender-dev \
    fonts-dejavu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Avoid security policy block in ImageMagick for text
RUN echo "policy.xml workaround" && \
    sed -i 's/none/read|write/' /etc/ImageMagick-6/policy.xml || true

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
