# SCRFD モデルのダウンロード方法

`scrfd_34g_bnkps.pth` を探している方へ

## 簡単なダウンロード方法

### 方法1: 提供されているスクリプトを使用（推奨）

```bash
# insightface をインストール
pip3 install insightface onnxruntime

# SCRFD-34G モデルをダウンロード
cd weights
python3 download_scrfd.py --model scrfd_34g --output-dir .
```

### 方法2: Python で直接ダウンロード

```bash
pip3 install insightface onnxruntime
cd weights
python3 -c "from insightface.model_zoo import get_model; get_model('scrfd_34g_v2.0', download=True, root='.')"
```

## 利用可能なモデル

- **scrfd_500m**: 超軽量（500M FLOPs）
- **scrfd_1g**: 軽量（1G FLOPs）
- **scrfd_2.5g**: バランス型（2.5G FLOPs）
- **scrfd_10g**: 高精度（10G FLOPs）
- **scrfd_34g**: 最高精度（34G FLOPs）← これが `scrfd_34g_bnkps.pth` です

## 詳細情報

詳しい情報は以下のファイルをご覧ください：
- [SCRFD_MODELS.md](SCRFD_MODELS.md) - 英語の詳細ガイド

## 参考リンク

- InsightFace GitHub: https://github.com/deepinsight/insightface
- SCRFD 論文: https://arxiv.org/abs/2105.04714
- モデル Zoo: https://github.com/deepinsight/insightface/tree/master/model_zoo

## 注意事項

このリポジトリは現在 **RetinaFace** モデルをサポートしています。SCRFD モデルを使用するには、InsightFace の推論ユーティリティを使用するか、顔検出パイプラインを SCRFD 用に適応させる必要があります。
