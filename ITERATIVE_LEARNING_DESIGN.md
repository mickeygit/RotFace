# 繰り返し学習パイプライン — データセット版管理とフロー設計

## 現在の問題点

1. **データ上書きリスク**  
   新しい MP4 を処理するたびに `detected_faces/` が上書きされる可能性
   
2. **データソース追跡困難**  
   どのバージョンのデータセットでどのモデルを学習したか不明確

3. **段階的データ混合が不便**  
   初回: 元データ + 回転検知データ v1  
   2回目: 元データ + v1 + 新しい v2 データ  
   のような混合が煩雑

4. **チェックポイント復元時の課題**  
   `session_001` を復元するとき、当時のデータセット v1 が必要だが管理がない

---

## 推奨フロー（改定版）

### 概念図
```
[初回]
input_videos/v1.mp4 
    ↓
detected_faces_v001/ (QA確認) 
    ↓
train_val_split_v001/ (QA承認済みのみ)
    ↓
session_001 学習
    ↓
model_v001.pth

[2回目追加学習]
input_videos/v2.mp4 (新動画)
    ↓
detected_faces_v002/ (QA確認)
    ↓
train_val_split_v002/ (v001 + v002 を混合)
    ↓
session_002 学習（v001 モデルから再開）
    ↓
model_v002.pth

[3回目追加学習]
input_videos/v3.mp4
    ↓
detected_faces_v003/ 
    ↓
train_val_split_v003/ (v001 + v002 + v003 を混合)
    ↓
session_003 学習（v002 モデルから再開）
    ↓
model_v003.pth
```

---

## 改定されたフォルダ構成

```
data/
├── detected_faces_v001/           # 2024-01-06 処理分
│   ├── original/
│   │   ├── images/
│   │   ├── landmarks_qa/
│   │   ├── metadata.json
│   │   └── qa_approved_ids.txt
│   ├── rotated_90/
│   ├── rotated_180/
│   └── rotated_270/
│
├── detected_faces_v002/           # 2024-01-07 処理分
│   └── [同じ構成]
│
├── detected_faces_v003/
│   └── [同じ構成]
│
├── train_val_split_v001/          # 検知 v001 のみで学習
│   ├── train/
│   │   ├── images/
│   │   └── annotations.json
│   ├── val/
│   │   ├── images/
│   │   └── annotations.json
│   └── dataset_manifest.json       ← ★ データソース記録
│       {
│         "version": "v001",
│         "created_date": "2024-01-06",
│         "detected_faces_versions": ["v001"],
│         "total_images": 5000,
│         "rotations": {"original": 2000, "rotated_90": 1000, ...}
│       }
│
├── train_val_split_v002/          # 検知 v001 + v002 で学習
│   ├── train/
│   ├── val/
│   └── dataset_manifest.json
│       {
│         "version": "v002",
│         "created_date": "2024-01-07",
│         "detected_faces_versions": ["v001", "v002"],
│         "total_images": 8500,
│         "rotations": {...}
│       }
│
└── train_val_split_v003/          # 検知 v001 + v002 + v003
    └── [同じ構成]

experiments/
├── session_001/
│   ├── checkpoints/
│   │   ├── epoch_001.pth
│   │   ├── epoch_010.pth
│   │   ├── best.pth
│   │   ├── latest.pth
│   │   └── model_history.json
│   │
│   ├── logs/                      # TensorBoard ログ
│   │   └── events.out.tfevents...
│   │
│   └── training_manifest.json     ← ★ 学習設定・データセット記録
│       {
│         "session_id": "session_001",
│         "dataset_version": "v001",
│         "pretrained_model": "weights/original/Resnet50_Final.pth",
│         "parent_model": null,
│         "epochs": 100,
│         "lr": 0.001,
│         "batch_size": 32,
│         "created_date": "2024-01-06",
│         "best_loss": 0.045
│       }
│
├── session_002/
│   ├── checkpoints/
│   ├── logs/
│   └── training_manifest.json
│       {
│         "session_id": "session_002",
│         "dataset_version": "v002",
│         "pretrained_model": "weights/original/Resnet50_Final.pth",
│         "parent_model": "experiments/session_001/checkpoints/best.pth",
│         "parent_session": "session_001",
│         "epochs": 50,                  # ← 追加学習は少ないエポック
│         "lr": 0.0001,
│         "created_date": "2024-01-07",
│         "best_loss": 0.038
│       }
│
└── session_003/
    └── [同様]
```

---

## ワークフロー（詳細版）

### ステップ 1: 新規 MP4 処理 → 検知データセット生成

```bash
# 日付ベースのバージョンを自動生成（例: 2024-01-06 → v001, 2024-01-07 → v002）
docker run --rm --gpus all \
  -v "$(pwd)/input_videos:/workspace/input_videos" \
  -v "$(pwd)/weights/original:/workspace/weights/original" \
  -v "$(pwd)/data:/workspace/data" \
  rotface:latest python scripts/preprocessing/detect_faces_from_mp4.py \
  --video_path /workspace/input_videos/video.mp4 \
  --model_path /workspace/weights/original/Resnet50_Final.pth \
  --output_dir /workspace/data \
  --auto_version_naming True        # ← v001, v002, ... を自動生成
  --frame_skip 5

# 出力: data/detected_faces_v001/
```

### ステップ 2: QA 確認

ホスト側で `data/detected_faces_v001/*/landmarks_qa/` の画像を確認し、不正なファイルを削除

### ステップ 3: QA データ同期

```bash
python tools/qa_cleanup.py \
  --detected_dir data/detected_faces_v001 \
  --remove_orphans True
```

### ステップ 4: データセット生成（初回 or 追加学習）

**初回学習**:
```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/workspace/data" \
  rotface:latest python scripts/preprocessing/create_dataset.py \
  --detected_versions v001 \
  --output_version v001 \
  --train_ratio 0.8 \
  --output_dir /workspace/data/train_val_split_v001
```

**2 回目追加学習** — 新データ v002 を v001 に追加:
```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/workspace/data" \
  rotface:latest python scripts/preprocessing/create_dataset.py \
  --detected_versions v001,v002 \
  --output_version v002 \
  --combine_previous True          # ← 前回のデータを混合
  --train_ratio 0.8 \
  --output_dir /workspace/data/train_val_split_v002
```

**出力**: `data/train_val_split_v002/` + `dataset_manifest.json`

### ステップ 5: 学習実行

**初回学習**:
```bash
docker run --rm --gpus all \
  -v "$(pwd)/data/train_val_split_v001:/workspace/data/train_val_split" \
  -v "$(pwd)/weights/original:/workspace/weights/original" \
  -v "$(pwd)/experiments:/workspace/experiments" \
  rotface:latest python scripts/training/train.py \
  --data_dir /workspace/data/train_val_split \
  --pretrained /workspace/weights/original/Resnet50_Final.pth \
  --session_id session_001 \
  --dataset_version v001 \
  --epochs 100 \
  --lr 0.001 \
  --checkpoint_dir /workspace/experiments/session_001
```

**追加学習** — session_001 のモデルから再開:
```bash
docker run --rm --gpus all \
  -v "$(pwd)/data/train_val_split_v002:/workspace/data/train_val_split" \
  -v "$(pwd)/experiments:/workspace/experiments" \
  rotface:latest python scripts/training/train.py \
  --data_dir /workspace/data/train_val_split \
  --resume_from /workspace/experiments/session_001/checkpoints/best.pth \
  --session_id session_002 \
  --dataset_version v002 \
  --epochs 50 \
  --lr 0.0001 \
  --parent_session session_001 \
  --checkpoint_dir /workspace/experiments/session_002
```

### ステップ 6: 最終 ONNX 変換

```bash
docker run --rm --gpus all \
  -v "$(pwd)/experiments/session_002/checkpoints:/workspace/models" \
  -v "$(pwd)/output/onnx:/workspace/output" \
  rotface:latest python scripts/export/export_to_onnx.py \
  --model_path /workspace/models/best.pth \
  --output_path /workspace/output/retinaface_final_v002.onnx
```

---

## 実装パッチ（detect_faces_from_mp4.py に追加）

```python
def auto_generate_version_name(data_dir: str) -> str:
    """
    既存のバージョンフォルダから次のバージョン番号を自動生成
    
    Returns:
        'v001', 'v002', ... など
    """
    import re
    from pathlib import Path
    
    existing = [d.name for d in Path(data_dir).glob('detected_faces_v*')]
    
    if not existing:
        return 'v001'
    
    versions = []
    for name in existing:
        match = re.match(r'detected_faces_v(\d+)', name)
        if match:
            versions.append(int(match.group(1)))
    
    if versions:
        next_version = max(versions) + 1
        return f'v{next_version:03d}'
    
    return 'v001'

# 使用時
if args.auto_version_naming:
    output_subdir = auto_generate_version_name(args.output_dir)
    output_dir = os.path.join(args.output_dir, f'detected_faces_{output_subdir}')
```

---

## チェックポイント復元フロー

**session_001 時点に戻したい場合**:
```bash
# training_manifest.json から dataset_version を取得 → v001
# train_val_split_v001/ を使用して学習を再開
cat experiments/session_001/training_manifest.json | jq '.dataset_version'
# → "v001"

docker run --rm --gpus all \
  -v "$(pwd)/data/train_val_split_v001:/workspace/data/train_val_split" \
  -v "$(pwd)/experiments:/workspace/experiments" \
  rotface:latest python scripts/training/train.py \
  --data_dir /workspace/data/train_val_split \
  --session_id session_001_restored \
  --dataset_version v001 \
  --epochs 100 \
  --checkpoint_dir /workspace/experiments/session_001_restored
```

---

## 利点

✅ **データトレーサビリティ**: どのデータセットで学習したかが明確  
✅ **段階的追加学習**: 新規データを簡単に既存データに混合可能  
✅ **チェックポイント復元**: 過去のセッションに戻す際の再現性確保  
✅ **実験管理**: session + dataset のペア記録で実験再現可能  
✅ **ディスク効率**: 不要な過去バージョンだけ削除可能  

