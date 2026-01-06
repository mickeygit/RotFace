#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
	echo "Usage: $0 <trained_model_path> [network: mobile0.25|resnet50] [use_gpu: 0|1]"
	exit 1
fi

TRAINED_MODEL=$1
NETWORK=${2:-mobile0.25}
USE_GPU=${3:-0}

export USE_GPU

echo "Building container (USE_GPU=${USE_GPU})..."
docker compose build --no-cache --build-arg USE_GPU=${USE_GPU}

echo "Starting container..."
docker compose up -d

SERVICE=rotface

# Ensure original weights are stored under ./weights/original on host for reproducibility
mkdir -p ./weights/original
BASE_NAME=$(basename "${TRAINED_MODEL}")
if [ -f "${TRAINED_MODEL}" ] && [[ "${TRAINED_MODEL}" != ./weights/* ]]; then
	cp -n "${TRAINED_MODEL}" "./weights/original/${BASE_NAME}"
	TRAINED_MODEL_HOST="./weights/original/${BASE_NAME}"
elif [ -f "${TRAINED_MODEL}" ]; then
	TRAINED_MODEL_HOST="${TRAINED_MODEL}"
else
	echo "Trained model '${TRAINED_MODEL}' not found on host." >&2
	exit 2
fi

echo "Running export inside container..."
# When USE_GPU=0, add --cpu flag to avoid CUDA errors on CPU-only PyTorch
if [ "$USE_GPU" = "0" ]; then
	docker compose exec -T ${SERVICE} bash -lc "python export_onnx.py --trained_model '/workspace/${TRAINED_MODEL_HOST#./}' --network ${NETWORK} --output /workspace/output/retinaface_${NETWORK}.onnx --cpu"
else
	docker compose exec -T ${SERVICE} bash -lc "python export_onnx.py --trained_model '/workspace/${TRAINED_MODEL_HOST#./}' --network ${NETWORK} --output /workspace/output/retinaface_${NETWORK}.onnx"
fi

echo "Export finished. ONNX file available in $(pwd)/output"
