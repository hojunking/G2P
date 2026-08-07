# G2P training environment.
#
# Build (from segmentation/):
#   docker build -t g2p:latest .
#
# Run:
#   docker run -it --gpus '"device=0"' --shm-size 32G \
#     -v $(pwd):/workdir -v /path/to/data:/data:ro -w /workdir \
#     g2p:latest /bin/bash
#
# TORCH_CUDA_ARCH_LIST is pinned so flash-attn compiles for one architecture.
# Adjust it for your GPU (8.6 = RTX 3090 / A6000, 8.9 = RTX 4090, 9.0 = H100).

FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.6" \
    FORCE_CUDA=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ninja-build libsparsehash-dev \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        addict open3d plyfile pyyaml regex scipy \
        tensorboard tensorboardx termcolor timm tqdm yapf

RUN pip install --no-cache-dir \
        torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html \
    && pip install --no-cache-dir spconv-cu124

RUN pip install --no-cache-dir --no-build-isolation flash-attn

# Bundled CUDA extension. Copied on its own so the layer is cached
# independently of the rest of the source tree.
COPY libs/pointops /tmp/pointops
RUN cd /tmp/pointops && python setup.py install && rm -rf /tmp/pointops

WORKDIR /workdir
