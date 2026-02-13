FROM python:3.10-slim

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (heavy ones first for caching)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    demucs \
    sounddevice>=0.4.6 \
    soundfile>=0.12.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    librosa>=0.10.0 \
    pyyaml>=6.0 \
    matplotlib>=3.7.0

# Copy application code
COPY audio-reactive/tools/ /app/audio-reactive/tools/
COPY audio-reactive/effects/ /app/audio-reactive/effects/
COPY audio-reactive/research/separation/ /app/audio-reactive/research/separation/

# Create audio directory (mount point for user's music)
RUN mkdir -p /app/audio-reactive/research/audio-segments

# Expose port
EXPOSE 8080

# Run the viewer
WORKDIR /app/audio-reactive/tools
CMD ["python", "segment.py", "web", "--port", "8080", "--host", "0.0.0.0"]
