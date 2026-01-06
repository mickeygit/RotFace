# QA補助ツール対応のフォルダ構成（修正版）

## 改定: QA作業フロー対応設計

```
data/detected_faces/
├── original/
│   ├── images/                     # 顔トリミング画像（原寸）
│   ├── landmarks_qa/               # ★QA作業用：5点ポイント描画済み(256x256 PNG)
│   ├── bboxes.json                 # バウンディングボックス
│   ├── metadata.json               # スコア・信頼度
│   └── qa_approved_ids.txt         # QA確認済みID一覧（テスト）
│
├── rotated_90/
│   ├── images/
│   ├── landmarks_qa/               # ★ 5点ポイント描画済み(256x256 PNG)
│   ├── bboxes.json
│   ├── metadata.json
│   └── qa_approved_ids.txt
│
├── rotated_180/
│   ├── images/
│   ├── landmarks_qa/
│   ├── bboxes.json
│   ├── metadata.json
│   └── qa_approved_ids.txt
│
└── rotated_270/
    ├── images/
    ├── landmarks_qa/
    ├── bboxes.json
    ├── metadata.json
    └── qa_approved_ids.txt
```

## QA作業フロー（改定版）

### フェーズ 2-1: 自動生成（detect_faces_from_mp4.py の出力）
```bash
docker run ... python scripts/preprocessing/detect_faces_from_mp4.py \
    --video-path /workspace/input_videos/video.mp4 \
    --model-path /workspace/weights/original/Resnet50_Final.pth \
    --output-dir /workspace/data/detected_faces \
    --frame-skip 5 \
    --min-confidence 0.9 \
    --min-face-size 10 \
    --network resnet50        # ← 5点ポイント描画済み画像も生成
```

**出力内容：**
- `images/{id}.jpg` — 元サイズの顔画像
- `landmarks_qa/{id}_marked.png` — 5点ポイント描画済み(256x256 PNG) ← QA作業用

### フェーズ 2-2: 人による確認・削除
ホスト側で `data/detected_faces/original/landmarks_qa/` を開き、
- ✅ 正常な顔 → そのまま
- ❌ 誤検知・ノイズ → ファイルを削除

### フェーズ 2-3: テストデータの自動同期（補助ツール）
```bash
python tools/qa_cleanup.py \
  --detected_dir data/detected_faces \
  --remove_orphans True
```

**処理内容：**
1. `landmarks_qa/` に存在する ID を確認
2. `images/` と `metadata.json` に存在するが `landmarks_qa/` にないファイルを削除
3. `qa_approved_ids.txt` を更新
4. 最終的にテスト用に `qa_approved_ids.txt` をリストアップ

## 補助ツール：qa_cleanup.py

```python
def qa_cleanup(detected_dir, remove_orphans=True):
    """
    QA 作業後にテストデータを同期・クリーンアップ。
    landmarks_qa/ に存在しないもの（人が削除したもの）を
    images/ と metadata.json からも削除する。
    """
    for rotation in ['original', 'rotated_90', 'rotated_180', 'rotated_270']:
        rot_dir = os.path.join(detected_dir, rotation)
        
        # landmarks_qa/ に存在する ID を確認
        approved_ids = set()
        qa_dir = os.path.join(rot_dir, 'landmarks_qa')
        if os.path.exists(qa_dir):
            approved_ids = {f.replace('_marked.png', '').replace('.png', '')
                           for f in os.listdir(qa_dir)}
        
        # images/ と metadata.json を同期
        images_dir = os.path.join(rot_dir, 'images')
        if os.path.exists(images_dir):
            for img_file in os.listdir(images_dir):
                img_id = os.path.splitext(img_file)[0]
                if img_id not in approved_ids and remove_orphans:
                    os.remove(os.path.join(images_dir, img_file))
                    print(f"削除: {rotation}/images/{img_file}")
        
        # metadata.json をフィルタ
        metadata_file = os.path.join(rot_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            filtered = {k: v for k, v in metadata.items()
                       if k in approved_ids}
            
            with open(metadata_file, 'w') as f:
                json.dump(filtered, f, indent=2)
        
        # qa_approved_ids.txt を生成
        with open(os.path.join(rot_dir, 'qa_approved_ids.txt'), 'w') as f:
            for id_ in sorted(approved_ids):
                f.write(f"{id_}\n")
        
        print(f"{rotation}: {len(approved_ids)} 件承認")
```

## 利用フロー（修正）

```
[1. MP4 読み込み＋検知]
    ↓
[2. 5点ポイント描画済み画像(256x256)を landmarks_qa/ に出力]
    ↓
[3. 人が landmarks_qa/ で確認 → 不正なファイルを削除]
    ↓
[4. qa_cleanup.py を実行 → images/ と metadata.json を同期]
    ↓
[5. 確認済みのデータで train_val_split を生成]
    ↓
[6. 学習実行]
```

## 次の実装ステップ

1. **detect_faces_from_mp4.py** を実装
   - 5点ポイントの推定
    - landmarks_qa/ に 256x256 マッピング画像を出力
   - images/ に元画像を保存

2. **qa_cleanup.py** を実装
   - 削除追跡・自動同期
   - qa_approved_ids.txt 生成

3. **train.py** を実装
   - チェックポイント・再開機能

4. **create_dataset.py** を実装
   - qa_approved_ids.txt を参照しながら train/val を生成
