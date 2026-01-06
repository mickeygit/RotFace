#!/usr/bin/env bash
set -euo pipefail

# run_detect.sh
# Usage: ./scripts/preprocessing/run_detect.sh [--video input_videos/video.mp4] [--model weights/original/Resnet50_Final.pth] [--output data] [--frame-skip 5] [--network resnet50] [--no-docker] [--auto-version]

DEFAULT_VIDEO="input_videos/video.mp4"
DEFAULT_MODEL="weights/original/Resnet50_Final.pth"
DEFAULT_OUTPUT="data"
DEFAULT_FRAME_SKIP=5
DEFAULT_NETWORK="resnet50"
USE_DOCKER=1
AUTO_VERSION=0

print_usage(){
  cat <<EOF
Usage: $0 [options]

Options:
  --video PATH        input video path (relative to project root)    [${DEFAULT_VIDEO}]
  --model PATH        model path (relative to project root)          [${DEFAULT_MODEL}]
  --output PATH       output dir (relative to project root)          [${DEFAULT_OUTPUT}]
  --frame-skip N      frame skip count                               [${DEFAULT_FRAME_SKIP}]
  --network NAME      backbone: mobile0.25|resnet50                   [${DEFAULT_NETWORK}]
  --no-docker         run locally with python3 instead of docker run
  --auto-version      enable --auto-version-naming in script
  -h, --help          show this help

Examples:
  # Docker run (recommended)
  $0 --video input_videos/video.mp4 --model weights/original/Resnet50_Final.pth --output data --frame-skip 5 --network resnet50

  # Local run (requires dependencies installed)
  $0 --no-docker --video input_videos/video.mp4 --model weights/original/Resnet50_Final.pth
EOF
}

# parse args
VIDEO="$DEFAULT_VIDEO"
MODEL="$DEFAULT_MODEL"
OUTPUT="$DEFAULT_OUTPUT"
FRAME_SKIP=$DEFAULT_FRAME_SKIP
NETWORK="$DEFAULT_NETWORK"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)
      VIDEO="$2"; shift 2;;
    --model)
      MODEL="$2"; shift 2;;
    --output)
      OUTPUT="$2"; shift 2;;
    --frame-skip)
      FRAME_SKIP="$2"; shift 2;;
    --network)
      NETWORK="$2"; shift 2;;
    --no-docker)
      USE_DOCKER=0; shift 1;;
    --auto-version)
      AUTO_VERSION=1; shift 1;;
    -h|--help)
      print_usage; exit 0;;
    *)
      echo "Unknown option: $1"; print_usage; exit 1;;
  esac
done

# Ensure paths are relative to repository root
ROOT_DIR="$(pwd)"
VIDEO_ABS="$ROOT_DIR/$VIDEO"
MODEL_ABS="$ROOT_DIR/$MODEL"
OUTPUT_ABS="$ROOT_DIR/$OUTPUT"

if [[ $USE_DOCKER -eq 1 ]]; then
  echo "Running detection inside Docker (image: rotface:latest)"
  DOCKER_CMD=(docker run --rm --gpus all -v "${ROOT_DIR}:/workspace" rotface:latest python "/workspace/scripts/preprocessing/detect_faces_from_mp4.py")
  DOCKER_CMD+=(--video-path "/workspace/${VIDEO}" --model-path "/workspace/${MODEL}" --output-dir "/workspace/${OUTPUT}" --frame-skip ${FRAME_SKIP} --network ${NETWORK})
  if [[ $AUTO_VERSION -eq 1 ]]; then
    DOCKER_CMD+=(--auto-version-naming True)
  fi
  echo "+ ${DOCKER_CMD[*]}"
  "${DOCKER_CMD[@]}"
else
  echo "Running detection locally (python3). Ensure dependencies are installed in this environment."
  PY_CMD=(python3 scripts/preprocessing/detect_faces_from_mp4.py --video-path "${VIDEO}" --model-path "${MODEL}" --output-dir "${OUTPUT}" --frame-skip ${FRAME_SKIP} --network ${NETWORK})
  if [[ $AUTO_VERSION -eq 1 ]]; then
    PY_CMD+=(--auto-version-naming True)
  fi
  echo "+ ${PY_CMD[*]}"
  "${PY_CMD[@]}"
fi
