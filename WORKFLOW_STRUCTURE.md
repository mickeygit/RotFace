# RotFace 再学習パイプライン — フォルダ構成ガイド

## 全体ワークフロー

```
[1. データ作成]             [2. QA確認・修正]        [3. 再学習]          [4. ONNX変換]
MP4 (host)                   人の確認・削除          学習実行             最終出力
    ↓                             ↓                     ↓                    ↓
input_videos/         →   detected_faces/    →   train_val_split/  →   experiments/   →   output/
                         + metadata.json         (確認済み安全)        (tensorboard)       (onnx + pth)
```

## 推奨フォルダ構成

```
RotFace/
├── input_videos/                        ★ INPUT: MP4 ファイル保管場
│   └── .gitkeep
│
├── data/
│   ├── detected_faces/                 # RetinaFace で検知した顔データ
│   │   ├── original/                  # 回転なし（既存データ維持用）
│   │   │   ├── images/               # トリミング顔画像
│   │   │   ├── landmarks_qa/         # ★QA用：5点ポイント描画済み(256x256 PNG)
│   │   │   ├── bboxes.json           # バウンディングボックス
│   │   │   ├── metadata.json         # スコア・ランドマーク
│   │   │   └── qa_approved_ids.txt   # ★QA承認済み ID リスト
│   │   ├── rotated_90/               # 90° 回転検知用
│   │   │   ├── images/
│   │   │   ├── landmarks_qa/         # ★QA用：5点ポイント描画済み(256x256 PNG)
│   │   │   ├── bboxes.json
│   │   │   ├── metadata.json
│   │   │   └── qa_approved_ids.txt
│   │   ├── rotated_180/
│   │   │   ├── images/
│   │   │   ├── landmarks_qa/         # ★QA用：5点ポイント描画済み(256x256 PNG)
│   │   │   ├── bboxes.json
│   │   │   ├── metadata.json
│   │   │   └── qa_approved_ids.txt
│   │   ├── rotated_270/
│   │   │   ├── images/
│   │   │   ├── landmarks_qa/         # ★QA用：5点ポイント描画済み(256x256 PNG)
│   │   │   ├── bboxes.json
│   │   │   ├── metadata.json
│   │   │   └── qa_approved_ids.txt
│   │
│   └── train_val_split/                # マニュアル確認済み・最終学習データ
│       ├── train/
│       │   ├── images/               # 学習用顔画像
│       │   └── annotations.json      # 対応する bbox + 回転フラグ
│       ├── val/
│       │   ├── images/
│       │   └── annotations.json
│       └── dataset_info.json         # 全体統計（画像数・回転比率等）
│
├── experiments/                        ★ 学習・検証実行結果
│   ├── checkpoints/                  # チェックポイント（再開用）
│   │   └── .gitkeep
│   │
│   ├── logs/                         # TensorBoard ログ（tensorboard --logdir logs）
│   │   └── .gitkeep
│   │
│   └── training_config.yaml          # 学習設定ファイル（epoch, lr, aug等）
│
├── output/                            ★ OUTPUT: ホスト側最終成果物
│   ├── onnx/                        # ONNX 変換結果
│   │   ├── retinaface_resnet50.onnx
│   │   └── .gitkeep
│   │
│   └── trained_models/              # 学習済み PyTorch モデル
│       ├── retinaface_epoch_100.pth
│       └── .gitkeep
│
├── scripts/
│   ├── preprocessing/
│   │   ├── detect_faces_from_mp4.py        # MP4 → GPU回転 → 顔検知 + QA画像生成
│   │   ├── create_dataset.py               # 検知結果 → 学習データセット化
│   │   └── __init__.py
│   │
│   ├── training/
│   │   ├── train.py                       # 再学習メインスクリプト（TensorBoard 統合）
│   │   ├── config.py                      # 学習設定クラス
│   │   ├── callbacks.py                   # TensorBoard・チェックポイント保存
│   │   └── __init__.py
│   │
│   └── export/
│       └── export_to_onnx.py              # 学習済みモデル → ONNX 変換ラッパー
│
├── tools/
│   ├── qa_cleanup.py                  # ★QA後のテストデータ同期ツール
│   └── .gitkeep
│
├── weights/
│   ├── original/                    # 元の学習済みモデル（.gitignore 対象）
│   └── .gitkeep
│
├── notebooks/
│   ├── exploratory.ipynb            # 実験用
│   └── .gitkeep
│
├── Dockerfile, docker-compose.yml, docker_up.sh
├── requirements.txt
├── PROJECT_STRUCTURE.md
└── README.md
```

## 各フェーズでの操作（ホスト側）

### フェーズ 1: データ作成（Preprocessing）

**MP4 → GPU回転 → 顔検知 + QA用画像生成**

```bash
docker run --rm --gpus all \
   -v "$(pwd)/input_videos:/workspace/input_videos" \
   -v "$(pwd)/weights/original:/workspace/weights/original" \
   -v "$(pwd)/data/detected_faces:/workspace/data/detected_faces" \
   rotface:latest python scripts/preprocessing/detect_faces_from_mp4.py \
   --video-path /workspace/input_videos/video.mp4 \
   --model-path /workspace/weights/original/Resnet50_Final.pth \
   --output-dir /workspace/data/detected_faces \
   --frame-skip 5 \
   --min-confidence 0.9 \
   --min-face-size 10 \
   --network resnet50
```

**出力**:
- `data/detected_faces/{original,rotated_90,rotated_180,rotated_270}/images/` — 顔トリミング画像
- `data/detected_faces/{original,rotated_90,rotated_180,rotated_270}/landmarks_qa/` — **5点ポイント描画済み画像（256x256 PNG）** ← QA作業用
- `data/detected_faces/{rotation}/metadata.json` — スコア・ランドマーク情報
 - `data/detected_faces/{rotation}/frame_vis/` — フレーム全体可視化画像（JPG、bbox+ランドマーク）
 - `data/detected_faces/{rotation}/metadata.json` — スコア・ランドマーク情報

### フェーズ 2: QA チェック（マニュアル確認・削除）

**手順**:
1. ホストのファイルエクスプローラで `data/detected_faces/original/landmarks_qa/` を開く
2. 5点ポイント描画済みの 256x256 サムネイル画像で確認
3. ✅ 正常な顔 → そのまま
4. ❌ 誤検知・ノイズ → ファイルを削除

**QA後の自動同期**:
```bash
# landmarks_qa/ に存在しないもの（人が削除したもの）を images/ と metadata.json からも削除
python tools/qa_cleanup.py --detected-dir data/detected_faces --remove-orphans True
```

**結果**:
- `data/detected_faces/{rotation}/qa_approved_ids.txt` を自動生成
- 削除されたファイルに対応するテストデータも自動削除
- 確認済みデータのみが残る

### フェーズ 3: 学習用データセット作成

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data/detected_faces:/workspace/data/detected_faces" \
  -v "$(pwd)/data/train_val_split:/workspace/data/train_val_split" \
  rotface:latest python scripts/preprocessing/create_dataset.py \
  --detected_dir /workspace/data/detected_faces \
  --output_dir /workspace/data/train_val_split \
  --train_ratio 0.8
```

**入力**: `data/detected_faces/{rotation}/qa_approved_ids.txt` — QA確認済みデータのみを参照
**出力**: `data/train_val_split/{train,val}/` — 確認済みデータのセット

### フェーズ 4: 再学習（TensorBoard 監視可能）

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data/train_val_split:/workspace/data/train_val_split" \
  -v "$(pwd)/weights/original:/workspace/weights/original" \
  -v "$(pwd)/experiments:/workspace/experiments" \
  rotface:latest python scripts/training/train.py \
  --data_dir /workspace/data/train_val_split \
  --pretrained /workspace/weights/original/Resnet50_Final.pth \
  --config /workspace/experiments/training_config.yaml \
  --log_dir /workspace/experiments/logs \
  --checkpoint_dir /workspace/experiments/checkpoints

# TensorBoard （ホスト側ブラウザ）
tensorboard --logdir experiments/logs --host 0.0.0.0 --port 6006
# → http://localhost:6006
```

### フェーズ 5: ONNX 変換（最終出力）

```bash
docker run --rm --gpus all \
  -v "$(pwd)/experiments/checkpoints:/workspace/experiments/checkpoints" \
  -v "$(pwd)/output/onnx:/workspace/output/onnx" \
  rotface:latest python scripts/export/export_to_onnx.py \
  --model_path /workspace/experiments/checkpoints/best_model.pth \
  --output_dir /workspace/output/onnx \
  --network resnet50
```

## ファイル流・データフロー

```
input_videos/
    ├─→ [extract_frames_from_mp4.py]
    └─→ data/raw_frames/
        ├─→ [detect_faces.py]  (RetinaFace + 4方向回転)
        └─→ data/detected_faces/{original, rotated_90, ...}/
            + metadata.json
            ├─ [人が QA確認・不正削除]
            ├─→ qa_status.json に反映
            ├─→ [create_dataset.py]
            └─→ data/train_val_split/{train, val}/
                + annotations.json
                ├─→ [train.py]  (TensorBoard ログ出力)
                └─→ experiments/logs/  ← tensorboard --logdir logs
                    experiments/checkpoints/best_model.pth
                    ├─→ [export_to_onnx.py]
                    └─→ output/onnx/retinaface_resnet50.onnx
                        output/trained_models/best_model.pth
```

## 重要な設計原則

1. **全入出力をホスト側で管理**
   - `input_videos/`, `data/`, `experiments/`, `output/` はすべてホスト `/root/docker_env/RotFace/` 配下
   - コンテナは `-v` マウント経由で処理

2. **検知結果の可視化**
   - `detected_faces/` を回転パターン別に分け、人が確認しやすく
   - `metadata.json` で検知スコア・信頼度を保存、フィルタリング可能に

3. **段階的実験トラッキング**
   - 各学習実行は `experiments/logs/{timestamp}/` に個別保存（オプション）
   - TensorBoard で複数実験を比較可能

4. **再現性・再開性**
   - `training_config.yaml` で学習設定を保存
   - チェックポイントはエポック単位で保存、中断・再開可能

5. **.gitignore 対象**
   - `data/`, `input_videos/`, `experiments/checkpoints/`, `output/` は Git 追跡外
   - 重みやログはホスト側に蓄積

## 次の実装ステップ

1. このフォルダ構成を RotFace に適用する
2. 各 `scripts/preprocessing/` スクリプトを実装
3. `scripts/training/train.py` を実装（TensorBoard 統合）
4. `scripts/export/export_to_onnx.py` ラッパーを実装
5. `tools/qa_viewer.py` を実装（オプション）
