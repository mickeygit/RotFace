提案されたプロジェクト構成

目的: 再学習パイプライン（mp4→frames→回転検出→逆回転→フィルタ→学習）とONNX変換をわかりやすく管理する。

推奨ディレクトリ構成:

- data/
  - raw_frames/         # 生フレーム（mp4 から抽出）
  - processed_frames/   # 回転・反転・フィルタ適用後の学習用フレーム
  - FDDB/               # 既存ディレクトリは移動せず併置

- weights/
  - original/           # 元のダウンロード済み学習済みモデル（gitignore に追加済み）
  - trained/            # 再学習後の重みを保存

- scripts/
  - preprocessing/      # mp4→frames, 回転検出, bbox 逆回転 などのスクリプトを置く
  - training/           # 学習ランナー、TensorBoard 起動スクリプト
  - export/             # ONNX 変換用スクリプトラッパー

- experiments/
  - checkpoints/        # 学習チェックポイント
  - logs/               # 学習ログ（TensorBoard 用）

- output/               # ONNX 出力や推論結果（既に.gitignore 対象）
- runs/                 # TensorBoard ログ（既に.gitignore 対象）
- docker/               # Docker 関連ファイル（オプションでここに移動可能）
- notebooks/            # 実験用ノートブック
- models/               # モデル定義（既存）
- layers/, modules/, utils/ etc. (既存) 保持

作業方針（提案）:
1. まずは上記ディレクトリを作成（実施済み: `scripts/`, `data/raw_frames/`, `data/processed_frames/`, `experiments/checkpoints/`, `docker/`, `notebooks/`）。
2. `scripts/preprocessing` と `scripts/training` に実装を分割して配置。
3. 既存の `weights/original` に元モデルを集約（既にサポート済み）。
4. 将来的に `docker/` に Dockerfile 等を移動する場合は `docker/` 下で `docker compose -f docker/docker-compose.yml` を使う想定。

注意:
- 現在 Docker 関連ファイルはリポジトリルートにあり、既にコミット済みです。移動する場合はパス調整と起動方法の変更が必要です。
- `.gitignore` は出力と重みの保存先を除外するよう更新済みです。

次のステップ: 
- (A) `scripts/preprocessing` に mp4→frame 抽出スクリプトを作りますか？
- (B) `scripts/export` に ONNX 変換ラッパー（既存 `export_onnx.py` の配置と引数ラップ）を作りますか？

どちらから進めますか？
