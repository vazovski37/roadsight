#!/usr/bin/env bash
# One-time setup on a RunPod pod (image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404).
# torch/torchvision come from the image's system Python, so install next to them (PEP 668 flag).
# The dataset goes to the container's local disk: /workspace is network storage and thousands of
# small random reads per second from it would starve the data loader.
set -euo pipefail
cd "$(dirname "$0")/.."
pip install --break-system-packages -q opencv-python-headless pycocotools pyarrow scipy pyyaml \
    mlflow-skinny sqlalchemy alembic
python -c "import torch, torchvision; print('torch', torch.__version__, 'tv', torchvision.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python tools/prepare_data.py --raw /root/raw --out /root/bdd10k --download --workers 16
mkdir -p data runs results logs
ln -sfn /root/bdd10k data/bdd10k
