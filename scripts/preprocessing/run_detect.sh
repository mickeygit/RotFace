#!/usr/bin/env bash
set -euo pipefail

# run_detect.sh
# Usage: ./scripts/preprocessing/run_detect.sh [--video input_videos/video.mp4] [--model weights/original/Resnet50_Final.pth] [--output data] [--frame-skip 5] [--network resnet50] [--no-docker] [--auto-version]

DEFAULT_VIDEO="/mnt/c/Users/mickey/Downloads/dfltest/REBD-827.hvec.mp4"
DEFAULT_MODEL="weights/original/Resnet50_Final.pth"
DEFAULT_OUTPUT="data"
DEFAULT_FRAME_SKIP=5
DEFAULT_NETWORK="resnet50"
USE_DOCKER=1
AUTO_VERSION=0
DEFAULT_MIN_CONFIDENCE=0.9
DEFAULT_MIN_FACE_SIZE=10

print_usage() {
	cat <<EOF
Usage: $0 [options]

Options:
  --video PATH        input video path (relative to project root)    [${DEFAULT_VIDEO}]
  --model PATH        model path (relative to project root)          [${DEFAULT_MODEL}]
  --output PATH       output dir (relative to project root)          [${DEFAULT_OUTPUT}]
  --frame-skip N      frame skip count                               [${DEFAULT_FRAME_SKIP}]
	--min-confidence N  min confidence threshold                        [${DEFAULT_MIN_CONFIDENCE}]
	--min-face-size N   min face size (px)                             [${DEFAULT_MIN_FACE_SIZE}]
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
MIN_CONFIDENCE=$DEFAULT_MIN_CONFIDENCE
MIN_FACE_SIZE=$DEFAULT_MIN_FACE_SIZE

while [[ $# -gt 0 ]]; do
	case "$1" in
	--video)
		VIDEO="$2"
		shift 2
		;;
	--model)
		MODEL="$2"
		shift 2
		;;
	--output)
		OUTPUT="$2"
		shift 2
		;;
	--frame-skip)
		FRAME_SKIP="$2"
		shift 2
		;;
	--min-confidence)
		MIN_CONFIDENCE="$2"
		shift 2
		;;
	--min-face-size)
		MIN_FACE_SIZE="$2"
		shift 2
		;;
	--network)
		NETWORK="$2"
		shift 2
		;;
	--no-docker)
		USE_DOCKER=0
		shift 1
		;;
	--auto-version)
		AUTO_VERSION=1
		shift 1
		;;
	-h | --help)
		print_usage
		exit 0
		;;
	*)
		echo "Unknown option: $1"
		print_usage
		exit 1
		;;
	esac
done

# Ensure paths are relative to repository root
ROOT_DIR="$(pwd)"
VIDEO_ABS="$ROOT_DIR/$VIDEO"
MODEL_ABS="$ROOT_DIR/$MODEL"
OUTPUT_ABS="$ROOT_DIR/$OUTPUT"

if [[ $USE_DOCKER -eq 1 ]]; then
	echo "Running detection inside Docker (image: rotface:latest)"
	# Handle absolute host video paths by mounting their parent directory into the container
	EXTRA_VOLUMES=()
	if [[ "${VIDEO}" = /* ]]; then
		VIDEO_HOST_DIR=$(dirname "${VIDEO}")
		VIDEO_BASENAME=$(basename "${VIDEO}")
		EXTRA_VOLUMES+=(-v "${VIDEO_HOST_DIR}:/host_videos")
		CONTAINER_VIDEO_PATH="/host_videos/${VIDEO_BASENAME}"
	else
		CONTAINER_VIDEO_PATH="/workspace/${VIDEO}"
	fi

	DOCKER_CMD=(docker run --rm --gpus all -v "${ROOT_DIR}:/workspace" ${EXTRA_VOLUMES[@]} -w /workspace -e PYTHONPATH=/workspace rotface:latest python "scripts/preprocessing/detect_faces_from_mp4.py")
	DOCKER_CMD+=(--video-path "${CONTAINER_VIDEO_PATH}" --model-path "/workspace/${MODEL}" --output-dir "/workspace/${OUTPUT}" --frame-skip ${FRAME_SKIP} --min-confidence ${MIN_CONFIDENCE} --min-face-size ${MIN_FACE_SIZE} --network ${NETWORK})
	if [[ $AUTO_VERSION -eq 1 ]]; then
		DOCKER_CMD+=(--auto-version-naming True)
	fi
	echo "+ ${DOCKER_CMD[*]}"
	"${DOCKER_CMD[@]}"
else
	echo "Running detection locally (python3). Ensure dependencies are installed in this environment."
	PY_CMD=(python3 scripts/preprocessing/detect_faces_from_mp4.py --video-path "${VIDEO}" --model-path "${MODEL}" --output-dir "${OUTPUT}" --frame-skip ${FRAME_SKIP} --min-confidence ${MIN_CONFIDENCE} --min-face-size ${MIN_FACE_SIZE} --network ${NETWORK})
	if [[ $AUTO_VERSION -eq 1 ]]; then
		PY_CMD+=(--auto-version-naming True)
	fi
	echo "+ ${PY_CMD[*]}"
	"${PY_CMD[@]}"
fi
