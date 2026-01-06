FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential git ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && ln -s /usr/bin/python3 /usr/bin/python || true \
    && ln -s /usr/bin/pip3 /usr/bin/pip || true \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt

RUN pip install --upgrade pip setuptools wheel

# Pin numpy first to avoid downstream conflicts
RUN pip --no-cache-dir install "numpy==1.26.4"

# Install PyTorch (cu121) and torchvision
RUN pip --no-cache-dir install torch==2.2.0+cu121 torchvision==0.17.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# Optional: CuPy for CUDA-accelerated ops (install if available)
RUN pip --no-cache-dir install cupy-cuda12x || true

# ONNX Runtime GPU (1.18) and other pip deps
RUN pip --no-cache-dir install onnxruntime-gpu==1.18.0

# Install remaining requirements (should not re-install conflicting runtime packages)
RUN pip --no-cache-dir install -r /workspace/requirements.txt

VOLUME ["/workspace/output"]

WORKDIR /workspace

CMD ["bash"]
