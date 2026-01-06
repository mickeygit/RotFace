#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
検知結果からの学習用データセット生成

複数の detected_faces_vXXX を組み合わせて、
train_val_split_vXXX を生成します。

使用方法:
    # 初回: v001 のみで作成
    python scripts/preprocessing/create_dataset.py \
      --detected_versions v001 \
      --output_version v001 \
      --data_dir data \
      --train_ratio 0.8
    
    # 2回目: v001 + v002 を組み合わせ
    python scripts/preprocessing/create_dataset.py \
      --detected_versions v001,v002 \
      --output_version v002 \
      --data_dir data \
      --train_ratio 0.8 \
      --combine_previous True
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime
import random

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetCreator:
    """複数検知バージョンから学習データセット生成"""
    
    def __init__(
        self,
        data_dir: str,
        train_ratio: float = 0.8,
        seed: int = 42
    ):
        """
        初期化
        
        Args:
            data_dir: data/ ディレクトリパス
            train_ratio: 訓練データの割合
            seed: ランダムシード
        """
        self.data_dir = Path(data_dir)
        self.train_ratio = train_ratio
        self.seed = seed
        random.seed(seed)
    
    def create_dataset(
        self,
        detected_versions: List[str],
        output_version: str,
        combine_previous: bool = False
    ) -> Dict[str, any]:
        """
        データセット作成
        
        Args:
            detected_versions: 使用する検知バージョンリスト (['v001'], ['v001', 'v002'])
            output_version: 出力バージョン名 ('v001', 'v002', ...)
            combine_previous: 前のバージョンとのデータを組み合わせるか
        
        Returns:
            データセット統計
        """
        logger.info(f"データセット作成開始")
        logger.info(f"検知バージョン: {detected_versions}")
        logger.info(f"出力バージョン: {output_version}")
        
        # 出力ディレクトリ作成
        output_dir = self.data_dir / f'train_val_split_{output_version}'
        train_dir = output_dir / 'train'
        val_dir = output_dir / 'val'
        
        train_images_dir = train_dir / 'images'
        train_images_dir.mkdir(parents=True, exist_ok=True)
        val_images_dir = val_dir / 'images'
        val_images_dir.mkdir(parents=True, exist_ok=True)
        
        # QA承認済みのファイルを集約
        approved_faces = {}  # {face_id: (image_path, metadata)}
        
        # 前回バージョンのデータを取り込む
        if combine_previous:
            prev_version = self._get_previous_version(output_version)
            if prev_version:
                logger.info(f"前回バージョンからデータを取り込み: {prev_version}")
                prev_approved = self._load_dataset_annotations(
                    self.data_dir / f'train_val_split_{prev_version}'
                )
                approved_faces.update(prev_approved)
        
        # 新規検知バージョンからデータを取り込み
        for version in detected_versions:
            logger.info(f"検知バージョン {version} からデータ取り込み中...")
            
            version_faces = self._collect_approved_faces(version)
            logger.info(f"  → {len(version_faces)} 件検知")
            
            approved_faces.update(version_faces)
        
        logger.info(f"合計 {len(approved_faces)} 件のデータを処理")
        
        # Train/Val に分割
        face_ids = list(approved_faces.keys())
        random.shuffle(face_ids)
        
        split_idx = int(len(face_ids) * self.train_ratio)
        train_ids = face_ids[:split_idx]
        val_ids = face_ids[split_idx:]
        
        # Train データセット保存
        train_annotations = {}
        for face_id in train_ids:
            img_path, metadata = approved_faces[face_id]
            # 実際の実装では、イメージをコピー
            # shutil.copy(img_path, train_images_dir / f"{face_id}.jpg")
            train_annotations[face_id] = metadata
        
        # Val データセット保存
        val_annotations = {}
        for face_id in val_ids:
            img_path, metadata = approved_faces[face_id]
            # shutil.copy(img_path, val_images_dir / f"{face_id}.jpg")
            val_annotations[face_id] = metadata
        
        # Annotations を保存
        with open(train_dir / 'annotations.json', 'w', encoding='utf-8') as f:
            json.dump(train_annotations, f, indent=2, ensure_ascii=False)
        
        with open(val_dir / 'annotations.json', 'w', encoding='utf-8') as f:
            json.dump(val_annotations, f, indent=2, ensure_ascii=False)
        
        # Dataset マニフェストを生成
        manifest = {
            'version': output_version,
            'created_date': datetime.now().isoformat(),
            'detected_faces_versions': detected_versions,
            'combined_with_previous': combine_previous,
            'previous_version': self._get_previous_version(output_version) if combine_previous else None,
            'total_images': len(approved_faces),
            'train_images': len(train_ids),
            'val_images': len(val_ids),
            'train_ratio': self.train_ratio,
            'rotation_distribution': self._compute_rotation_distribution(approved_faces)
        }
        
        with open(output_dir / 'dataset_manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"データセット作成完了: {output_dir}")
        logger.info(f"  Train: {len(train_ids)} 件")
        logger.info(f"  Val: {len(val_ids)} 件")
        
        return manifest
    
    def _collect_approved_faces(self, version: str) -> Dict[str, Tuple[Path, Dict]]:
        """
        QA 承認済みの顔ファイルを収集
        
        Args:
            version: 検知バージョン (e.g., 'v001')
        
        Returns:
            {face_id: (image_path, metadata), ...}
        """
        detected_dir = self.data_dir / f'detected_faces_{version}'
        approved_faces = {}
        
        if not detected_dir.exists():
            logger.warning(f"ディレクトリが見つかりません: {detected_dir}")
            return approved_faces
        
        # 各回転ごとに QA 承認済みデータを処理
        for rotation in ['original', 'rotated_90', 'rotated_180', 'rotated_270']:
            rotation_dir = detected_dir / rotation
            qa_approved_file = rotation_dir / 'qa_approved_ids.txt'
            
            if not qa_approved_file.exists():
                logger.warning(f"qa_approved_ids.txt が見つかりません: {qa_approved_file}")
                continue
            
            # 承認済み ID を読込
            with open(qa_approved_file, 'r', encoding='utf-8') as f:
                approved_ids = [line.strip() for line in f if line.strip()]
            
            # Metadata を読込
            metadata_file = rotation_dir / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    all_metadata = json.load(f)
            else:
                all_metadata = {}
            
            # 承認済みのファイルをマッピング
            images_dir = rotation_dir / 'images'
            for face_id in approved_ids:
                img_path = images_dir / f"{face_id}.jpg"
                if img_path.exists():
                    metadata = all_metadata.get(face_id, {})
                    metadata['rotation'] = rotation
                    metadata['version'] = version
                    
                    approved_faces[face_id] = (img_path, metadata)
        
        return approved_faces
    
    def _load_dataset_annotations(self, dataset_dir: Path) -> Dict[str, Tuple[Path, Dict]]:
        """
        既存データセットの annotations を読み込む
        
        Args:
            dataset_dir: train_val_split_vXXX ディレクトリ
        
        Returns:
            {face_id: (image_path, metadata), ...}
        """
        annotations_data = {}
        
        # Train と Val の両方から読込
        for split in ['train', 'val']:
            split_dir = dataset_dir / split
            annotations_file = split_dir / 'annotations.json'
            images_dir = split_dir / 'images'
            
            if annotations_file.exists():
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                
                for face_id, metadata in annotations.items():
                    img_path = images_dir / f"{face_id}.jpg"
                    annotations_data[face_id] = (img_path, metadata)
        
        return annotations_data
    
    def _get_previous_version(self, current_version: str) -> str:
        """
        前のバージョン名を取得
        
        Args:
            current_version: 現在のバージョン (e.g., 'v002')
        
        Returns:
            前のバージョン (e.g., 'v001') または None
        """
        # バージョン番号を抽出
        try:
            version_num = int(current_version[1:])  # v002 → 2
            if version_num > 1:
                return f'v{version_num - 1:03d}'
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _compute_rotation_distribution(self, approved_faces: Dict) -> Dict[str, int]:
        """
        回転タイプの分布を計算
        
        Args:
            approved_faces: {face_id: (image_path, metadata), ...}
        
        Returns:
            {'original': count, 'rotated_90': count, ...}
        """
        distribution = {
            'original': 0,
            'rotated_90': 0,
            'rotated_180': 0,
            'rotated_270': 0
        }
        
        for face_id, (_, metadata) in approved_faces.items():
            rotation = metadata.get('rotation', 'original')
            if rotation in distribution:
                distribution[rotation] += 1
        
        return distribution


def main():
    parser = argparse.ArgumentParser(
        description='学習用データセット生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例：
  # 初回
  python scripts/preprocessing/create_dataset.py \\
    --detected_versions v001 \\
    --output_version v001
  
  # 2回目（追加学習）
  python scripts/preprocessing/create_dataset.py \\
    --detected_versions v001,v002 \\
    --output_version v002 \\
    --combine_previous True
        """
    )
    
    parser.add_argument(
        '--detected-versions',
        type=str,
        required=True,
        help='カンマ区切りの検知バージョンリスト (e.g., v001 または v001,v002)'
    )
    parser.add_argument(
        '--output-version',
        type=str,
        required=True,
        help='出力バージョン名 (e.g., v001, v002, ...)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='data/ ディレクトリパス（デフォルト: data）'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='訓練データの割合（デフォルト: 0.8）'
    )
    parser.add_argument(
        '--combine-previous',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=False,
        help='前回バージョンのデータと組み合わせる（デフォルト: False）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='ランダムシード（デフォルト: 42）'
    )
    
    args = parser.parse_args()
    
    try:
        # バージョンリストをパース
        versions = [v.strip() for v in args.detected_versions.split(',')]
        
        creator = DatasetCreator(
            data_dir=args.data_dir,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
        
        manifest = creator.create_dataset(
            detected_versions=versions,
            output_version=args.output_version,
            combine_previous=args.combine_previous
        )
        
        print("\n" + "="*60)
        print("データセット生成完了")
        print("="*60)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        print("="*60 + "\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
