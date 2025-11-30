# VoIP Avatar Service - GPU-backed LiveKit + MuseTalk avatar
#
# Build: docker build -t voipavatar .
# Run:   docker run --gpus all -e LIVEKIT_URL=... -e LIVEKIT_ROOM=... voipavatar

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    python3.10 \
    python3.10-venv \
    python3-pip \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Clone MuseTalk repository
WORKDIR /opt
RUN git clone https://github.com/TMElyralab/MuseTalk.git

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install MMLab stack (order matters for dependencies)
RUN pip install --no-cache-dir \
    mmengine==0.10.3 \
    mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

RUN pip install --no-cache-dir \
    mmdet==3.3.0 \
    mmpose==1.3.1

# Install MuseTalk requirements
WORKDIR /opt/MuseTalk
RUN pip install --no-cache-dir -r requirements.txt || true

# Install additional MuseTalk dependencies that might be missing
RUN pip install --no-cache-dir \
    diffusers>=0.24.0 \
    transformers>=4.35.0 \
    accelerate>=0.25.0 \
    soundfile>=0.12.0 \
    librosa>=0.10.0 \
    numba>=0.58.0

# Install service dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH to include MuseTalk
ENV PYTHONPATH="/opt/MuseTalk:${PYTHONPATH}"

# Copy service code
COPY service/ /app/service/

# Create directory for model downloads (can be mounted as volume)
RUN mkdir -p /app/models

# Health check port
EXPOSE 8080

# Default environment variables
ENV PERSONA_NAME=default
ENV LIVEKIT_IDENTITY=avatar-bot
ENV HEALTH_PORT=8080
ENV LOG_LEVEL=INFO

# Entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
