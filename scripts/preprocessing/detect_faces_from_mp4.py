#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MP4 からの顔検検知と 5点ポイントマッピング画像の生成

- MP4 をフレームスキップしながら読み込み
- GPU メモリ内で回転 (0°, 90°, 180°, 270°)
- RetinaFace で顔検知
- 5点ポイント（目・鼻・口）を描画した QA 作業用画像（128x128）を出力
- 検知結果（bbox, landmarks, metadata）を各回転ディレクトリに保存

使用方法:
    python scripts/preprocessing/detect_faces_from_mp4.py \
      --video_path input_videos/video.mp4 \
      --model_path weights/original/Resnet50_Final.pth \
      --output_dir data/detected_faces \
      --frame_skip 5 \
      --min_confidence 0.9 \
      --min_face_size 10
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import uuid

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LandmarkMapper:
    """5点ポイント描画ユーティリティ"""
    
    # RetinaFace の 5点出力順序: 左目, 右目, 鼻, 左口端, 右口端
    LANDMARK_NAMES = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']
    
    @staticmethod
    def draw_landmarks_on_image(
        image: np.ndarray,
        landmarks: np.ndarray,
        face_box: np.ndarray,
        output_size: int = 128
    ) -> Image.Image:
        """
        顔画像と 5点ポイントを QA 作業用画像（128x128）に描画
        
        Args:
            image: 元画像（BGR, numpy array）
            landmarks: 5点座標 [5, 2] - 元画像座標
            face_box: バウンディングボックス [x1, y1, x2, y2]
            output_size: 出力画像サイズ（デフォルト: 128）
        
        Returns:
            PIL.Image (RGB, 128x128)
        """
        # 顔領域をトリミング
        x1, y1, x2, y2 = face_box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        face_crop = image[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            # フェイルセーフ：顔領域が取得できない場合は黒塗り
            face_crop = np.zeros((output_size, output_size, 3), dtype=np.uint8)
        
        # 顔領域をリサイズ（アスペクト比を保ったパディング）
        face_h, face_w = face_crop.shape[:2]
        scale = output_size / max(face_h, face_w)
        new_h, new_w = int(face_h * scale), int(face_w * scale)
        
        face_resized = cv2.resize(face_crop, (new_w, new_h))
        
        # パディング（中央揃え）
        pad_top = (output_size - new_h) // 2
        pad_left = (output_size - new_w) // 2
        face_padded = np.full((output_size, output_size, 3), 128, dtype=np.uint8)
        face_padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = face_resized
        
        # BGR → RGB
        face_rgb = cv2.cvtColor(face_padded, cv2.COLOR_BGR2RGB)
        
        # PIL Image に変換
        pil_image = Image.fromarray(face_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # ポイント座標を出力画像座標にマッピング
        # 元画像座標 → 顔トリミング座標 → リサイズ後座標
        for i, landmark in enumerate(landmarks):
            lm_x, lm_y = landmark
            
            # トリミング座標に変換
            crop_x = lm_x - x1
            crop_y = lm_y - y1
            
            # スケール適用
            scaled_x = crop_x * scale
            scaled_y = crop_y * scale
            
            # パディング後の座標
            final_x = scaled_x + pad_left
            final_y = scaled_y + pad_top
            
            # 画像内に収まっているか確認
            if 0 <= final_x < output_size and 0 <= final_y < output_size:
                # ポイント描画（赤い円 + ID）
                radius = 4
                draw.ellipse(
                    [(final_x - radius, final_y - radius),
                     (final_x + radius, final_y + radius)],
                    fill='red', outline='white'
                )
                # ポイント ID を描画（1-indexed）
                draw.text(
                    (final_x + radius + 2, final_y - radius),
                    str(i + 1),
                    fill='white'
                )
        
        return pil_image
    
    @staticmethod
    def landmarks_to_dict(landmarks: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        ランドマーク配列を辞書に変換
        
        Args:
            landmarks: [5, 2] 配列
        
        Returns:
            {'left_eye': (x, y), ...}
        """
        return {
            LandmarkMapper.LANDMARK_NAMES[i]: tuple(lm)
            for i, lm in enumerate(landmarks)
        }


class MPGPURotator:
    """GPU メモリ内での動画フレーム回転"""
    
    @staticmethod
    def rotate_frame_gpu(
        frame: torch.Tensor,
        angle_degrees: int,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        フレームを GPU で回転
        
        Args:
            frame: [H, W, 3] torch tensor (BGR)
            angle_degrees: 回転角度 (0, 90, 180, 270)
            device: 'cuda' or 'cpu'
        
        Returns:
            回転後の tensor
        """
        if angle_degrees == 0:
            return frame
        elif angle_degrees == 90:
            return torch.rot90(frame, k=1, dims=[0, 1])  # 反時計回り
        elif angle_degrees == 180:
            return torch.rot90(frame, k=2, dims=[0, 1])
        elif angle_degrees == 270:
            return torch.rot90(frame, k=3, dims=[0, 1])
        else:
            raise ValueError(f"Unsupported angle: {angle_degrees}")


class FaceDetectionProcessor:
    """顔検知処理のメインクラス"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        min_confidence: float = 0.9,
        min_face_size: int = 10
    ):
        """
        初期化
        
        Args:
            model_path: RetinaFace モデルパス
            device: 'cuda' or 'cpu'
            min_confidence: 信頼度の最小値
            min_face_size: 顔サイズの最小値（ピクセル）
        """
        self.device = device
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        
        # モデル読み込み（簡略版 - 実装はプロジェクトの model/retinaface.py を使用）
        self.model = None  # 後で initialize() で読み込む
        self.model_path = model_path
    
    def initialize(self):
        """モデルを初期化"""
        # TODO: プロジェクトの RetinaFace モデルをインポート
        # from models.retinaface import RetinaFace
        # self.model = RetinaFace(...)
        # self.model.load_state_dict(torch.load(self.model_path))
        logger.info(f"モデル読み込み: {self.model_path}")
    
    def detect_and_process_video(
        self,
        video_path: str,
        output_dir: str,
        frame_skip: int = 5
    ) -> Dict[str, Any]:
        """
        動画から顔検知と QA 画像生成
        
        Args:
            video_path: 入力 MP4 パス
            output_dir: 出力ディレクトリ
            frame_skip: フレームスキップ数
        
        Returns:
            処理結果の統計
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"動画を開けません: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"動画情報: {total_frames} フレーム, {fps:.2f} FPS")
        
        # 出力ディレクトリの初期化
        result_dirs = {
            'original': os.path.join(output_dir, 'original'),
            'rotated_90': os.path.join(output_dir, 'rotated_90'),
            'rotated_180': os.path.join(output_dir, 'rotated_180'),
            'rotated_270': os.path.join(output_dir, 'rotated_270'),
        }
        
        for key, path in result_dirs.items():
            os.makedirs(os.path.join(path, 'images'), exist_ok=True)
            os.makedirs(os.path.join(path, 'landmarks_qa'), exist_ok=True)
        
        # 処理結果
        results = {rotation: {'total': 0, 'detected': 0, 'metadata': {}}
                  for rotation in result_dirs.keys()}
        
        frame_count = 0
        processed_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # フレームスキップ
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                processed_count += 1
                logger.info(f"処理中: フレーム {frame_count}/{total_frames} "
                           f"(処理数: {processed_count})")
                
                # フレームを GPU tensor に変換
                frame_gpu = torch.from_numpy(frame).to(self.device)
                
                # 複数角度で検知
                for angle in [0, 90, 180, 270]:
                    rotation_key = f'rotated_{angle}' if angle > 0 else 'original'
                    
                    # GPU で回転
                    rotated_frame_gpu = MPGPURotator.rotate_frame_gpu(
                        frame_gpu, angle, self.device
                    )
                    
                    # CPU に戻して numpy に変換（検知用）
                    rotated_frame = rotated_frame_gpu.cpu().numpy().astype(np.uint8)
                    
                    # 顔検知（※実装は別途 RetinaFace の detect() 使用）
                    # dets, landmarks = self.model.detect(rotated_frame)
                    
                    # 仮: ダミー検知（実装時に削除）
                    dets = np.array([[100, 100, 200, 200, 0.95]])  # dummy
                    landmarks_list = [np.random.rand(5, 2) * 100 + 100]  # dummy
                    
                    results[rotation_key]['total'] += 1
                    
                    # 検知結果を保存
                    self._save_detection_results(
                        rotated_frame,
                        dets,
                        landmarks_list,
                        result_dirs[rotation_key],
                        frame_count,
                        angle,
                        results[rotation_key]
                    )
                
                frame_count += 1
        
        finally:
            cap.release()
        
        logger.info(f"処理完了: {processed_count} フレーム処理")
        
        return results
    
    def _save_detection_results(
        self,
        frame: np.ndarray,
        detections: np.ndarray,
        landmarks_list: List[np.ndarray],
        output_dir: str,
        frame_id: int,
        angle: int,
        results: Dict[str, Any]
    ):
        """
        検知結果を保存
        
        Args:
            frame: 画像フレーム
            detections: [N, 5] bbox + confidence
            landmarks_list: List of [5, 2] landmarks
            output_dir: 出力ディレクトリ
            frame_id: フレーム ID
            angle: 回転角度
            results: 累積結果辞書（更新用）
        """
        # 信頼度フィルタリング
        confident_detections = detections[detections[:, 4] > self.min_confidence]
        
        if len(confident_detections) == 0:
            return
        
        images_dir = os.path.join(output_dir, 'images')
        qa_dir = os.path.join(output_dir, 'landmarks_qa')
        
        metadata_list = []
        
        for det_idx, (bbox, confidence) in enumerate(
            zip(confident_detections[:, :4], confident_detections[:, 4])
        ):
            # ユニーク ID 生成
            face_id = f"{frame_id:06d}_{angle:03d}_{det_idx:02d}"
            
            # 顔サイズチェック
            x1, y1, x2, y2 = bbox
            face_width = x2 - x1
            face_height = y2 - y1
            
            if min(face_width, face_height) < self.min_face_size:
                continue
            
            # 顔画像を保存
            face_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            face_path = os.path.join(images_dir, f"{face_id}.jpg")
            cv2.imwrite(face_path, face_crop)
            
            # ランドマーク用の QA 画像を生成・保存
            if det_idx < len(landmarks_list):
                landmarks = landmarks_list[det_idx]
                qa_image = LandmarkMapper.draw_landmarks_on_image(
                    frame, landmarks, bbox, output_size=128
                )
                qa_path = os.path.join(qa_dir, f"{face_id}_marked.png")
                qa_image.save(qa_path)
            
            # メタデータ
            metadata_list.append({
                'face_id': face_id,
                'frame_id': frame_id,
                'angle': angle,
                'bbox': bbox.tolist(),
                'confidence': float(confidence),
                'landmarks': LandmarkMapper.landmarks_to_dict(
                    landmarks if det_idx < len(landmarks_list) else np.zeros((5, 2))
                )
            })
            
            results['detected'] += 1
        
        # metadata.json に累積保存
        metadata_file = os.path.join(output_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        else:
            existing = {}
        
        for meta in metadata_list:
            existing[meta['face_id']] = meta
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description='MP4 からの顔検知と QA 画像生成'
    )
    
    parser.add_argument(
        '--video-path',
        type=str,
        required=True,
        help='入力 MP4 ファイルパス'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='RetinaFace モデルパス'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/detected_faces',
        help='出力ディレクトリ'
    )
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=5,
        help='フレームスキップ数（デフォルト: 5）'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.9,
        help='信頼度の最小値（デフォルト: 0.9）'
    )
    parser.add_argument(
        '--min-face-size',
        type=int,
        default=10,
        help='顔サイズの最小値ピクセル（デフォルト: 10）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='実行デバイス'
    )
    
    args = parser.parse_args()
    
    try:
        processor = FaceDetectionProcessor(
            model_path=args.model_path,
            device=args.device,
            min_confidence=args.min_confidence,
            min_face_size=args.min_face_size
        )
        processor.initialize()
        
        results = processor.detect_and_process_video(
            video_path=args.video_path,
            output_dir=args.output_dir,
            frame_skip=args.frame_skip
        )
        
        print("\n" + "="*60)
        print("検知結果サマリー")
        print("="*60)
        for rotation, stats in results.items():
            print(f"{rotation}: "
                  f"検知数 {stats['detected']}/{stats['total']}")
        print("="*60 + "\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
