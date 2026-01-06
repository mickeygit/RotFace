# RetinaFace ONNX Export and Inference

This repository helps to convert retinface with `mobilenet` or `resnet50` backbones to `onnx`.

## 1. Install dependencies

```sh
pip3 install -r requirements.txt
```

## 2. Download weights

```sh
cd weights
./download-weights.sh
```

## 3. Export to onnx

```sh
# mobilenet
python3 scripts/export/export_to_onnx.py --model-path weights/mobilenet0.25_Final.pth --network mobile0.25 --output-dir output/onnx

# resnet50
python3 scripts/export/export_to_onnx.py --model-path weights/Resnet50_Final.pth --network resnet50 --output-dir output/onnx
```

## 4. Run inference

```sh
python3 inference_onnx.py --model-path output/onnx/retinaface_resnet50.onnx
```

## Quick detection (Docker)

```bash
docker run --rm --gpus all \
	-v "$(pwd):/workspace" -w /workspace -e PYTHONPATH=/workspace rotface:latest \
	python scripts/preprocessing/detect_faces_from_mp4.py \
		--video-path /workspace/input_videos/video.mp4 \
		--model-path /workspace/weights/original/Resnet50_Final.pth \
		--output-dir /workspace/data/detected_faces \
		--frame-skip 5 \
		--min-confidence 0.9 \
		--min-face-size 10 \
		--network resnet50
```