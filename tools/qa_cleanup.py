#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QA作業後のテストデータ同期ツール

人が landmarks_qa/ で確認・削除したファイルに対応して、
images/ と metadata.json を自動的に同期・クリーンアップします。

使用方法:
    python tools/qa_cleanup.py --detected-dir data/detected_faces --remove-orphans True
"""

import os
import json
import argparse
from pathlib import Path
from typing import Set, Dict, Any
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QACleanupTool:
    """QA確認済みデータの同期ツール"""
    
    ROTATIONS = ['original', 'rotated_90', 'rotated_180', 'rotated_270']
    
    def __init__(self, detected_dir: str, remove_orphans: bool = True):
        """
        初期化
        
        Args:
            detected_dir: detected_faces ルートディレクトリ
            remove_orphans: landmarks_qa に無いファイルを削除するか
        """
        self.detected_dir = Path(detected_dir)
        self.remove_orphans = remove_orphans
        
        if not self.detected_dir.exists():
            raise FileNotFoundError(f"ディレクトリが見つかりません: {detected_dir}")
    
    def _get_approved_ids(self, rotation: str) -> Set[str]:
        """
        landmarks_qa/ から承認済み ID を抽出
        
        Args:
            rotation: 回転名 ('original', 'rotated_90', etc.)
        
        Returns:
            承認済み ID のセット（ファイル拡張子除去）
        """
        qa_dir = self.detected_dir / rotation / 'landmarks_qa'
        approved_ids = set()
        
        if not qa_dir.exists():
            logger.warning(f"qa ディレクトリが見つかりません: {qa_dir}")
            return approved_ids
        
        for qa_file in qa_dir.glob('*'):
            if qa_file.is_file() and qa_file.suffix in ['.png', '.jpg', '.jpeg']:
                # ファイル名から ID を抽出（_marked.png を削除）
                file_id = qa_file.stem
                if file_id.endswith('_marked'):
                    file_id = file_id[:-7]
                approved_ids.add(file_id)
        
        return approved_ids
    
    def _cleanup_images(self, rotation: str, approved_ids: Set[str]) -> int:
        """
        images/ ディレクトリをクリーンアップ
        
        Args:
            rotation: 回転名
            approved_ids: 承認済み ID
        
        Returns:
            削除ファイル数
        """
        images_dir = self.detected_dir / rotation / 'images'
        deleted_count = 0
        
        if not images_dir.exists():
            logger.warning(f"images ディレクトリが見つかりません: {images_dir}")
            return deleted_count
        
        for img_file in images_dir.glob('*'):
            if not img_file.is_file():
                continue
            
            img_id = img_file.stem
            
            if img_id not in approved_ids:
                if self.remove_orphans:
                    img_file.unlink()
                    logger.info(f"削除: {rotation}/images/{img_file.name}")
                    deleted_count += 1
                else:
                    logger.info(f"削除対象（未削除）: {rotation}/images/{img_file.name}")
        
        return deleted_count
    
    def _cleanup_metadata(self, rotation: str, approved_ids: Set[str]) -> int:
        """
        metadata.json をフィルタ・更新
        
        Args:
            rotation: 回転名
            approved_ids: 承認済み ID
        
        Returns:
            削除エントリ数
        """
        metadata_file = self.detected_dir / rotation / 'metadata.json'
        deleted_count = 0
        
        if not metadata_file.exists():
            logger.warning(f"metadata.json が見つかりません: {metadata_file}")
            return deleted_count
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON デコードエラー: {metadata_file} - {e}")
            return deleted_count
        
        original_count = len(metadata)
        
        # 承認済み ID のみフィルタ
        filtered_metadata = {
            k: v for k, v in metadata.items()
            if k in approved_ids
        }
        
        deleted_count = original_count - len(filtered_metadata)
        
        if deleted_count > 0 or self.remove_orphans:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"{rotation}: metadata.json を更新 "
                       f"({original_count} → {len(filtered_metadata)} エントリ)")
        
        return deleted_count
    
    def _save_approved_ids_list(self, rotation: str, approved_ids: Set[str]):
        """
        qa_approved_ids.txt を生成・保存
        
        Args:
            rotation: 回転名
            approved_ids: 承認済み ID
        """
        approved_list_file = self.detected_dir / rotation / 'qa_approved_ids.txt'
        
        with open(approved_list_file, 'w', encoding='utf-8') as f:
            for id_ in sorted(approved_ids):
                f.write(f"{id_}\n")
        
        logger.info(f"生成: {rotation}/qa_approved_ids.txt ({len(approved_ids)} 件)")
    
    def cleanup_all(self) -> Dict[str, Dict[str, Any]]:
        """
        すべての回転データをクリーンアップ
        
        Returns:
            各回転のクリーンアップ結果
        """
        results = {}
        
        for rotation in self.ROTATIONS:
            logger.info(f"\n--- {rotation} の処理開始 ---")
            
            approved_ids = self._get_approved_ids(rotation)
            
            if not approved_ids:
                logger.warning(f"{rotation}: 承認済みデータなし（スキップ）")
                results[rotation] = {
                    'approved_count': 0,
                    'deleted_images': 0,
                    'deleted_metadata': 0
                }
                continue
            
            # images クリーンアップ
            deleted_images = self._cleanup_images(rotation, approved_ids)
            
            # metadata クリーンアップ
            deleted_metadata = self._cleanup_metadata(rotation, approved_ids)
            
            # qa_approved_ids.txt 生成
            self._save_approved_ids_list(rotation, approved_ids)
            
            results[rotation] = {
                'approved_count': len(approved_ids),
                'deleted_images': deleted_images,
                'deleted_metadata': deleted_metadata
            }
            
            logger.info(f"{rotation}: {len(approved_ids)} 件承認、"
                       f"{deleted_images} 画像削除、{deleted_metadata} metadata エントリ削除")
        
        return results
    
    def print_summary(self, results: Dict[str, Dict[str, Any]]):
        """
        クリーンアップ結果のサマリーを表示
        
        Args:
            results: cleanup_all() の戻り値
        """
        print("\n" + "="*60)
        print("QA クリーンアップ サマリー")
        print("="*60)
        
        total_approved = 0
        total_deleted_images = 0
        total_deleted_metadata = 0
        
        for rotation, stats in results.items():
            print(f"\n{rotation}:")
            print(f"  承認済み: {stats['approved_count']} 件")
            print(f"  削除（画像）: {stats['deleted_images']} 件")
            print(f"  削除（メタデータ）: {stats['deleted_metadata']} 件")
            
            total_approved += stats['approved_count']
            total_deleted_images += stats['deleted_images']
            total_deleted_metadata += stats['deleted_metadata']
        
        print(f"\n{'---'*20}")
        print(f"合計承認済み: {total_approved} 件")
        print(f"合計削除（画像）: {total_deleted_images} 件")
        print(f"合計削除（メタデータ）: {total_deleted_metadata} 件")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='QA作業後のテストデータ同期ツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例：
  python tools/qa_cleanup.py --detected-dir data/detected_faces
  python tools/qa_cleanup.py --detected-dir data/detected_faces --remove-orphans False
        """
    )
    
    parser.add_argument(
        '--detected-dir',
        type=str,
        default='data/detected_faces',
        help='detected_faces ルートディレクトリ（デフォルト: data/detected_faces）'
    )
    
    parser.add_argument(
        '--remove-orphans',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help='landmarks_qa に無いファイルを削除するか（デフォルト: True）'
    )
    
    args = parser.parse_args()
    
    try:
        tool = QACleanupTool(args.detected_dir, args.remove_orphans)
        results = tool.cleanup_all()
        tool.print_summary(results)
        
        logger.info("QA クリーンアップ完了 ✓")
        return 0
    
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
