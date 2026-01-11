# RetinaFace ONNX Export and Inference

This repository helps to convert retinface with `mobilenet` or `resnet50` backbones to `onnx`.

> **Note**: Looking for SCRFD models? See [weights/SCRFD_MODELS.md](weights/SCRFD_MODELS.md) for information on downloading SCRFD models including `scrfd_34g_bnkps.pth`.
>
> **日本語**: SCRFD モデル（`scrfd_34g_bnkps.pth`）をお探しの方は [weights/SCRFD_README_ja.md](weights/SCRFD_README_ja.md) をご覧ください。

## 1. Install dependencies

```sh
pip3 install -r requirements.txt
```

## 2. Download weights

### RetinaFace Models (Current Support)

```sh
cd weights
./download-weights.sh
```

This will download the RetinaFace models (mobilenet0.25 and resnet50).

### SCRFD Models (Alternative)

For SCRFD models including `scrfd_34g_bnkps.pth`, see the detailed guide in [weights/SCRFD_MODELS.md](weights/SCRFD_MODELS.md).

Quick download using InsightFace:

```sh
pip3 install insightface
cd weights
python3 -c "from insightface.model_zoo import get_model; get_model('scrfd_34g_v2.0', download=True, root='.')"
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