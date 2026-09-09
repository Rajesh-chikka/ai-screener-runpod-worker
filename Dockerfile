FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .

RUN python3 -m pip install --break-system-packages --no-cache-dir \
    -r requirements.txt

COPY handler.py .
COPY model_manager.py .
COPY config.py .
COPY schemas.py .

CMD ["python3", "-u", "handler.py"]