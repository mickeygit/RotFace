# Quickstart — 実行例（Docker / ホスト）

以下はコピーして使えるコマンド例です。ホストで実行してください。

## 1) MP4 → 検知（GPU 回転、QA用画像生成）

```bash
docker run --rm --gpus all \
  -v "$(pwd)/input_videos:/workspace/input_videos" \
  -v "$(pwd)/weights/original:/workspace/weights/original" \
  -v "$(pwd)/data:/workspace/data" \
  rotface:latest python scripts/preprocessing/detect_faces_from_mp4.py \
  --video-path /workspace/input_videos/video.mp4 \
  --model-path /workspace/weights/original/Resnet50_Final.pth \
  --output-dir /workspace/data \
  --frame-skip 5 \
  --auto-version-naming True
```

- 出力: `data/detected_faces_vXXX/{original,rotated_90,...}/landmarks_qa/` に 128×128 の QA 画像

## 2) QA 確認（ホスト）
- ファイルエクスプローラで `data/detected_faces_vXXX/*/landmarks_qa/` を開き、誤検知を削除

## 3) QA 反映（自動同期）

```bash
python tools/qa_cleanup.py --detected-dir data/detected_faces_v001 --remove-orphans True
```

## 4) データセット生成（v001 + v002 を混合する例）

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/workspace/data" \
  rotface:latest python scripts/preprocessing/create_dataset.py \
  --detected-versions v001,v002 \
  --output-version v002 \
  --data-dir /workspace/data \
  --train-ratio 0.8 \
  --combine-previous True
```

- 出力: `data/train_val_split_v002/` と `dataset_manifest.json`

## 5) 学習実行（例）

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data/train_val_split_v002:/workspace/data/train_val_split" \
  -v "$(pwd)/weights/original:/workspace/weights/original" \
  -v "$(pwd)/experiments:/workspace/experiments" \
  rotface:latest python scripts/training/train.py \
  --data_dir /workspace/data/train_val_split \
  --pretrained /workspace/weights/original/Resnet50_Final.pth \
  --session_id session_002 \
  --dataset_version v002 \
  --epochs 50 \
  --lr 0.0001 \
  --checkpoint_dir /workspace/experiments/session_002
```

## 6) ONNX 変換

```bash
docker run --rm --gpus all \
  -v "$(pwd)/experiments/session_002/checkpoints:/workspace/models" \
  -v "$(pwd)/output/onnx:/workspace/output" \
  rotface:latest python scripts/export/export_to_onnx.py \
  --model_path /workspace/models/best.pth \
  --output_path /workspace/output/retinaface_final_v002.onnx
```

---

必要ならこのファイルを `README.md` に差し込みます。どちらを希望しますか？
