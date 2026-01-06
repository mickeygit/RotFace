#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MP4 からの顔検検知と 5点ポイントマッピング画像の生成

- MP4 をフレームスキップしながら読み込み
- GPU メモリ内で回転 (0°, 90°, 180°, 270°)
- RetinaFace で顔検知
- 5点ポイント（目・鼻・口）を描画した QA 作業用画像（256x256 PNG）を出力
- 検知結果（bbox, landmarks, metadata）を各回転ディレクトリに保存

使用方法:
        python scripts/preprocessing/detect_faces_from_mp4.py \
            --video-path input_videos/video.mp4 \
            --model-path weights/original/Resnet50_Final.pth \
            --output-dir data/detected_faces \
            --frame-skip 5 \
            --min-confidence 0.9 \
            --min-face-size 10
"""

import os
import json
import argparse
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import uuid
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# RetinaFace utilities
from data.config import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm
from models.retinaface import RetinaFace

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def auto_generate_version_name(data_dir: str) -> str:
    """
    既存のバージョンフォルダから次のバージョン番号を自動生成
    
    例: detected_faces_v001, detected_faces_v002, ... → 次は v003
    
    Args:
        data_dir: data/ ディレクトリパス
    
    Returns:
        'v001', 'v002', ... など
    """
    existing = list(Path(data_dir).glob('detected_faces_v*'))
    
    if not existing:
        return 'v001'
    
    versions = []
    for folder in existing:
        match = re.match(r'detected_faces_v(\d+)', folder.name)
        if match:
            versions.append(int(match.group(1)))
    
    if versions:
        next_version = max(versions) + 1
        return f'v{next_version:03d}'
    
    return 'v001'


class LandmarkMapper:
    """5点ポイント描画ユーティリティ"""
    
    # RetinaFace の 5点出力順序: 左目, 右目, 鼻, 左口端, 右口端
    LANDMARK_NAMES = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']
    
    @staticmethod
    def draw_landmarks_on_image(
        image: np.ndarray,
        landmarks: np.ndarray,
        face_box: np.ndarray,
        output_size: int = 256
        , annotations: Optional[List[Dict[str, Any]]] = None,
        face_confidence: Optional[float] = None
    ) -> Image.Image:
        """
        顔画像と 5点ポイントを QA 作業用画像に描画
        
        Args:
            image: 元画像（BGR, numpy array）
            landmarks: 5点座標 [5, 2] - 元画像座標
            face_box: バウンディングボックス [x1, y1, x2, y2]
            output_size: 出力画像サイズ（デフォルト: 256） - より大きく見やすく
        
        Returns:
            PIL.Image (RGB, 256x256)
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
        # 色マップ（PIL）: 目・鼻・口を判別しやすい色にする
        # ユーザ指定: 目=赤, 鼻=緑, 口=紺
        color_map = {
            'left_eye': 'red',
            'right_eye': 'red',
            'nose': 'green',
            'left_mouth': 'navy',
            'right_mouth': 'navy'
        }

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
                # ポイント描画（色はランドマーク種別ごとに変更）
                radius = 6
                lm_name = LandmarkMapper.LANDMARK_NAMES[i]
                fill_col = color_map.get(lm_name, 'red')
                draw.ellipse(
                    [(final_x - radius, final_y - radius),
                     (final_x + radius, final_y + radius)],
                    fill=fill_col, outline='white', width=2
                )
                # ポイント ID を描画（1-indexed）
                draw.text(
                    (final_x + radius + 3, final_y - radius),
                    str(i + 1),
                    fill='white'
                )
        
                # 注釈が指定されていれば、ポイント近傍にテキストで描画
                if annotations is not None and i < len(annotations):
                    ann = annotations[i]
                    # 期待されるキー: 'orig' -> (x,y), 'rel' -> (rx,ry), 'eval' -> value
                    orig = ann.get('orig')
                    rel = ann.get('rel')
                    ev = ann.get('eval')

                    txt_lines = []
                    if orig is not None:
                        txt_lines.append(f"O:{int(orig[0])},{int(orig[1])}")
                    if rel is not None:
                        txt_lines.append(f"R:{rel[0]:.2f},{rel[1]:.2f}")
                    if ev is not None:
                        txt_lines.append(f"E:{ev}")

                    # 描画位置を微調整して複数行を重ねる
                    if txt_lines:
                        txt_x = final_x + radius + 3
                        txt_y = final_y + radius + 3
                        for j, line in enumerate(txt_lines):
                            draw.text((txt_x, txt_y + j * 12), line, fill='yellow')
        # 顔全体の検出スコアを描画
        if face_confidence is not None:
            try:
                draw.text((6, 14), f"conf:{face_confidence:.2f}", fill='yellow')
            except Exception:
                pass

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
    
    @staticmethod
    def draw_landmarks_on_frame(
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        output_path: str,
        angle: int = 0
    ) -> None:
        """
        元のフレーム全体に bbox と 5点ランドマークを描画して保存
        
        Args:
            frame: 元フレーム (BGR, numpy array) - 回転前のオリジナル
            detections: 検知結果リスト [{'bbox': [...], 'landmarks': [...], 'face_id': '...'}, ...]
            output_path: 出力ファイルパス
            angle: 回転角度 (0, 90, 180, 270)
        """
        frame_vis = frame.copy()
        h, w = frame.shape[:2]
        
        for det in detections:
            bbox = np.array(det['bbox'], dtype=np.float32)
            landmarks = det['landmarks']
            face_id = det.get('face_id', '')
            
            # 回転フレーム座標から元フレーム座標に逆変換
            if angle == 0:
                # 変換なし
                bbox_orig = bbox
                landmarks_orig = landmarks
            elif angle == 90:
                # 時計回り90度回転されたので、反時計回りに戻す
                # rotated: (x, y) -> original: (h - y, x)
                x1, y1, x2, y2 = bbox
                landmarks_orig = {}
                for name, (lm_x, lm_y) in landmarks.items():
                    orig_x = h - lm_y
                    orig_y = lm_x
                    landmarks_orig[name] = (orig_x, orig_y)
                bbox_orig = np.array([
                    h - y2, x1, h - y1, x2
                ], dtype=np.float32)
            elif angle == 180:
                # 180度回転されたので、180度戻す
                # rotated: (x, y) -> original: (w - x, h - y)
                x1, y1, x2, y2 = bbox
                landmarks_orig = {}
                for name, (lm_x, lm_y) in landmarks.items():
                    orig_x = w - lm_x
                    orig_y = h - lm_y
                    landmarks_orig[name] = (orig_x, orig_y)
                bbox_orig = np.array([
                    w - x2, h - y2, w - x1, h - y1
                ], dtype=np.float32)
            elif angle == 270:
                # 反時計回り90度回転されたので、時計回りに戻す
                # rotated: (x, y) -> original: (y, w - x)
                x1, y1, x2, y2 = bbox
                landmarks_orig = {}
                for name, (lm_x, lm_y) in landmarks.items():
                    orig_x = lm_y
                    orig_y = w - lm_x
                    landmarks_orig[name] = (orig_x, orig_y)
                bbox_orig = np.array([
                    y1, w - x2, y2, w - x1
                ], dtype=np.float32)
            else:
                bbox_orig = bbox
                landmarks_orig = landmarks
            
            # bbox を描画
            x1, y1, x2, y2 = [int(v) for v in bbox_orig]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_vis.shape[1], x2)
            y2 = min(frame_vis.shape[0], y2)
            
            # 青い矩形で bbox を描画
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # 顔 ID ラベルを描画
            # 顔 ID と信頼度ラベルを描画
            conf = det.get('confidence') if isinstance(det, dict) else None
            if conf is not None:
                label = f"{face_id} {conf:.2f}"
            else:
                label = face_id
            cv2.putText(
                frame_vis, label,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 1
            )
            
            # 5点ランドマークを描画（色分け）
            # ユーザ指定: 目=赤, 鼻=緑, 口=紺（BGR）
            cv_color_map = {
                'left_eye': (0, 0, 255),   # red (B,G,R)
                'right_eye': (0, 0, 255),
                'nose': (0, 255, 0),       # green
                'left_mouth': (128, 0, 0), # navy (dark blue) in BGR
                'right_mouth': (128, 0, 0)
            }

            for i, (name, coords) in enumerate(landmarks_orig.items()):
                lm_x, lm_y = coords
                lm_x, lm_y = int(lm_x), int(lm_y)

                # ランドマークが画像内に収まっているか確認
                if 0 <= lm_x < frame_vis.shape[1] and 0 <= lm_y < frame_vis.shape[0]:
                    # 色分けして円でランドマークを描画
                    col = cv_color_map.get(name, (0, 0, 255))
                    cv2.circle(frame_vis, (lm_x, lm_y), 5, col, -1)
                    # 白い枠線
                    cv2.circle(frame_vis, (lm_x, lm_y), 5, (255, 255, 255), 1)
                    # ポイント番号を描画 (1-indexed)
                    cv2.putText(
                        frame_vis, str(i + 1),
                        (lm_x + 7, lm_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1
                    )
        
        # 保存（JPG形式、品質90）
        cv2.imwrite(output_path, frame_vis, [cv2.IMWRITE_JPEG_QUALITY, 90])


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
        
        # モデル読み込み
        self.model = None  # 後で initialize() で読み込む
        self.model_path = model_path
        self.network = 'resnet50'
        self.cfg = None
    
    def initialize(self):
        """モデルを初期化"""
        # network に応じた cfg を設定
        if self.network == 'mobile0.25' or self.network == 'mobile0.25' :
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50
        # Ensure torchvision/torch hub downloads are cached under the repo to
        # avoid repeated downloads. This directory is mounted into the container
        # so cached files persist on the host (e.g. resnet50-0676ba61.pth).
        try:
            repo_root = os.getcwd()
            cache_dir = os.path.join(repo_root, 'weights', 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            os.environ.setdefault('TORCH_HOME', cache_dir)
            try:
                torch.hub.set_dir(cache_dir)
            except Exception:
                pass
        except Exception:
            pass

        logger.info(f"モデル読み込み: {self.model_path} (network={self.network})")
        # net を作成
        net = RetinaFace(cfg=self.cfg, phase='test')

        # load weights (CPU/CUDA 両対応)
        def remove_prefix(state_dict, prefix):
            f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
            return {f(key): value for key, value in state_dict.items()}

        def load_model(model, pretrained_path):
            logger.info(f"Loading pretrained model from {pretrained_path}")
            if not torch.cuda.is_available() or self.device == 'cpu':
                pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
            else:
                device = torch.cuda.current_device()
                pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

            if 'state_dict' in pretrained_dict.keys():
                pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
            else:
                pretrained_dict = remove_prefix(pretrained_dict, 'module.')

            net.load_state_dict(pretrained_dict, strict=False)
            return net

        self.model = load_model(net, self.model_path)
        self.model.eval()

        # device
        if self.device == 'cuda' and torch.cuda.is_available():
            self.device_torch = torch.device('cuda')
            logger.info('Using CUDA device')
        else:
            self.device_torch = torch.device('cpu')
            logger.info('Using CPU device')

        self.model = self.model.to(self.device_torch)
    
    def detect_and_process_video(
        self,
        video_path: str,
        output_dir: str,
        frame_skip: int = 5,
        auto_version_naming: bool = False,
        save_frame_vis: bool = False
    ) -> Dict[str, Any]:
        """
        動画から顔検知と QA 画像生成
        
        Args:
            video_path: 入力 MP4 パス
            output_dir: 出力ディレクトリ（またはその親ディレクトリ）
            frame_skip: フレームスキップ数
            auto_version_naming: バージョン名を自動生成するか
        
        Returns:
            処理結果の統計
        """
        # バージョン名を自動生成
        if auto_version_naming:
            version_name = auto_generate_version_name(output_dir)
            output_subdir = f'detected_faces_{version_name}'
            final_output_dir = os.path.join(output_dir, output_subdir)
            logger.info(f"自動バージョン生成: {version_name} → {output_subdir}/")
        else:
            final_output_dir = output_dir
            version_name = None
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"動画を開けません: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"動画情報: {total_frames} フレーム, {fps:.2f} FPS")
        
        # 出力ディレクトリの初期化
        result_dirs = {
            'original': os.path.join(final_output_dir, 'original'),
            'rotated_90': os.path.join(final_output_dir, 'rotated_90'),
            'rotated_180': os.path.join(final_output_dir, 'rotated_180'),
            'rotated_270': os.path.join(final_output_dir, 'rotated_270'),
        }
        
        for key, path in result_dirs.items():
            # 既存ディレクトリをクリア
            if os.path.exists(path):
                shutil.rmtree(path)
            # 新規作成
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
                    
                    # CPU に戻して numpy に変換（推論前処理用）
                    rotated_frame = rotated_frame_gpu.cpu().numpy().astype(np.uint8)
                    
                    # 前処理: RGB 変換・リサイズ・平均差し引き
                    frame_rgb = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB)
                    img = np.float32(frame_rgb)
                    im_height, im_width, _ = img.shape

                    # model input size
                    input_size = self.cfg.get('image_size', 640)
                    resized = cv2.resize(img, (input_size, input_size))
                    resized -= (104, 117, 123)
                    resized = resized.transpose(2, 0, 1)
                    resized = np.expand_dims(resized, 0)

                    with torch.no_grad():
                        x = torch.from_numpy(resized).to(self.device_torch)
                        if x.dtype != torch.float32:
                            x = x.float()
                        loc, conf, landms = self.model(x)

                    # numpy 化
                    loc = loc.data.cpu().numpy()
                    conf = conf.data.cpu().numpy()
                    landms = landms.data.cpu().numpy()

                    # priorbox + decode
                    priorbox = PriorBox(self.cfg, image_size=(input_size, input_size), format="numpy")
                    priors = priorbox.forward()
                    boxes = decode(np.squeeze(loc, axis=0), priors, self.cfg['variance'])
                    
                    # スケール: model input (640x640) から実際のフレームサイズへ
                    # 重要: rotated_frame の実際のサイズでスケーリングする必要がある
                    scale = np.array([im_width, im_height, im_width, im_height])
                    boxes = boxes * scale / 1
                    
                    scores = np.squeeze(conf, axis=0)[:, 1]
                    landms_dec = decode_landm(np.squeeze(landms, axis=0), priors, self.cfg['variance'])
                    scale1 = np.array([im_width, im_height] * 5)
                    landms_dec = landms_dec * scale1 / 1

                    # filter by confidence
                    inds = np.where(scores > self.min_confidence)[0]
                    if inds.shape[0] == 0:
                        continue
                    boxes = boxes[inds]
                    landms_sel = landms_dec[inds]
                    scores = scores[inds]

                    order = scores.argsort()[::-1][:5000]
                    boxes = boxes[order]
                    landms_sel = landms_sel[order]
                    scores = scores[order]

                    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                    keep = py_cpu_nms(dets, 0.4)
                    dets = dets[keep, :]
                    landms_sel = landms_sel[keep]

                    # limit
                    dets = dets[:750, :]
                    landms_sel = landms_sel[:750]

                    # prepare landmarks list
                    landmarks_list = [lm.reshape(5, 2) for lm in landms_sel]

                    results[rotation_key]['total'] += 1

                    # save
                    self._save_detection_results(
                        rotated_frame,
                        frame,  # 元フレーム（angle 0用）
                        dets,
                        landmarks_list,
                        result_dirs[rotation_key],
                        frame_count,
                        angle,
                        results[rotation_key],
                        save_frame_vis=save_frame_vis
                    )
                
                frame_count += 1
        
        finally:
            cap.release()
        
        logger.info(f"処理完了: {processed_count} フレーム処理")
        
        # メタデータを保存
        self._save_processing_manifest(
            final_output_dir,
            version_name,
            video_path,
            results
        )
        
        return results
    
    def _save_detection_results(
        self,
        rotated_frame: np.ndarray,
        original_frame: np.ndarray,
        detections: np.ndarray,
        landmarks_list: List[np.ndarray],
        output_dir: str,
        frame_id: int,
        angle: int,
        results: Dict[str, Any]
        , save_frame_vis: bool = False
    ):
        """
        検知結果を保存
        
        Args:
            rotated_frame: 回転後のフレーム
            original_frame: 元のフレーム（frame_vis用）
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
        # Ensure directories exist (extra safety in case caller didn't create them)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(qa_dir, exist_ok=True)
        
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
            
            # 顔画像を保存（座標クリッピング & 空配列チェック）
            h, w = rotated_frame.shape[:2]
            
            # bbox はフレーム全体（0～frame_width, 0～frame_height）に対する座標なので直接使用
            # ちょうど抽出幅を生成するように clip
            x1i = int(np.clip(np.floor(x1), 0, w))
            y1i = int(np.clip(np.floor(y1), 0, h))
            x2i = int(np.clip(np.ceil(x2), 0, w))
            y2i = int(np.clip(np.ceil(y2), 0, h))

            if x2i <= x1i or y2i <= y1i:
                logger.warning(
                    f"無効な bbox により顔切り出しをスキップします: face_id={face_id} bbox={bbox} frame_shape=(h={h},w={w})"
                )
                continue

            # rotated_frame[y:y+h, x:x+w] でクロップ （行・列の順）
            face_crop = rotated_frame[y1i:y2i, x1i:x2i]
            if face_crop is None or face_crop.size == 0:
                logger.warning(f"空の face_crop を検出してスキップします: face_id={face_id} dims={face_crop.shape if face_crop is not None else 'None'}")
                continue

            # 画像が非常に小さい場合もスキップ
            if face_crop.shape[0] < 2 or face_crop.shape[1] < 2:
                logger.warning(f"face_crop が小さすぎます: face_id={face_id} shape={face_crop.shape}")
                continue

            face_path = os.path.join(images_dir, f"{face_id}.jpg")
            try:
                ok = cv2.imwrite(face_path, face_crop)
                if not ok:
                    logger.warning(f"cv2.imwrite が失敗しました: {face_path}")
                    continue
            except Exception as e:
                logger.warning(f"cv2.imwrite で例外: {e} path={face_path}")
                continue
            
            # ランドマーク用の QA 画像を生成・保存
            landmarks = np.zeros((5, 2))
            if det_idx < len(landmarks_list):
                landmarks = landmarks_list[det_idx]

            # 相対座標・評価値を計算して注釈データを作成
            lm_orig_dict = LandmarkMapper.landmarks_to_dict(landmarks)
            face_w = max(1.0, float(face_width))
            face_h = max(1.0, float(face_height))
            lm_rel = {}
            lm_evals = {}
            annotations = []
            for i, name in enumerate(LandmarkMapper.LANDMARK_NAMES):
                lm_x, lm_y = float(landmarks[i, 0]), float(landmarks[i, 1])
                rel_x = (lm_x - x1) / face_w
                rel_y = (lm_y - y1) / face_h
                # 初期状態ではランドマーク単体の評価値は存在しないため None を入れる
                eval_val = None
                lm_rel[name] = (rel_x, rel_y)
                lm_evals[name] = eval_val
                annotations.append({'orig': (lm_x, lm_y), 'rel': (rel_x, rel_y), 'eval': eval_val})

                qa_image = LandmarkMapper.draw_landmarks_on_image(
                    rotated_frame, landmarks, bbox, output_size=256, annotations=annotations, face_confidence=float(confidence)
                )
            qa_path = os.path.join(qa_dir, f"{face_id}_marked.jpg")
            try:
                qa_image.save(qa_path, format='JPEG', quality=90)
            except Exception:
                # fallback
                qa_image.save(qa_path)
            
            # メタデータ
            # bbox とランドマークは回転フレーム座標のままメタデータに保存
            # (フレーム描画時に逆変換する）
            # メタデータ: 元座標・相対座標・評価値を属性として保存
            metadata_list.append({
                'face_id': face_id,
                'frame_id': frame_id,
                'angle': angle,
                'bbox': bbox.tolist(),
                'confidence': float(confidence),
                'landmarks': LandmarkMapper.landmarks_to_dict(landmarks),
                'landmarks_original': lm_orig_dict,
                'landmarks_relative': lm_rel,
                'landmark_evals': lm_evals,
                'face_eval': float(confidence)
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
            fid = meta['face_id']
            if fid in existing:
                # 既存エントリがある場合は、新規キーを追加する（既存値は上書きしない）
                for k, v in meta.items():
                    if k not in existing[fid]:
                        existing[fid][k] = v
            else:
                existing[fid] = meta
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        
        # フレーム全体にランドマークを描画・保存（オプション）
        if len(metadata_list) > 0 and save_frame_vis:
            detections_for_frame = []
            for meta in metadata_list:
                detections_for_frame.append({
                    'face_id': meta['face_id'],
                    'bbox': meta['bbox'],
                    'landmarks': meta['landmarks'],
                    'confidence': meta.get('confidence')
                })

            frame_vis_dir = os.path.join(output_dir, 'frame_vis')
            os.makedirs(frame_vis_dir, exist_ok=True)
            frame_vis_path = os.path.join(frame_vis_dir, f"{frame_id:06d}_{angle:03d}_landmarks.jpg")

            LandmarkMapper.draw_landmarks_on_frame(
                original_frame, detections_for_frame, frame_vis_path, angle=angle
            )
        

    def _save_processing_manifest(
        self,
        output_dir: str,
        version_name: str,
        video_path: str,
        results: Dict[str, Any]
    ):
        """
        処理マニフェストを保存
        
        Args:
            output_dir: 出力ディレクトリ
            version_name: バージョン名（自動生成時）
            video_path: 入力動画パス
            results: 処理結果
        """
        manifest = {
            'version': version_name,
            'video_path': video_path,
            'processing_date': datetime.now().isoformat(),
            'device': self.device,
            'results': {}
        }
        
        # 各回転の統計
        total_detected = 0
        for rotation, stats in results.items():
            manifest['results'][rotation] = {
                'total_frames': stats['total'],
                'detected_faces': stats['detected']
            }
            total_detected += stats['detected']
        
        manifest['total_detected_faces'] = total_detected
        
        manifest_path = os.path.join(output_dir, 'processing_manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"処理マニフェスト保存: {manifest_path}")


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
    parser.add_argument(
        '--network',
        type=str,
        default='resnet50',
        choices=['mobile0.25', 'resnet50'],
        help='バックボーンネットワーク'
    )
    parser.add_argument(
        '--auto-version-naming',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=False,
        help='バージョン名を自動生成（detected_faces_v001, v002, ...）'
    )
    parser.add_argument(
        '--save-frame-vis',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=False,
        help='フレーム全体の可視化画像(frame_vis)を保存するか（デフォルト: False）'
    )
    
    args = parser.parse_args()
    
    try:
        import time
        start_time = time.time()
        
        processor = FaceDetectionProcessor(
            model_path=args.model_path,
            device=args.device,
            min_confidence=args.min_confidence,
            min_face_size=args.min_face_size
        )
        processor.network = args.network
        processor.initialize()
        
        results = processor.detect_and_process_video(
            video_path=args.video_path,
            output_dir=args.output_dir,
            frame_skip=args.frame_skip,
            auto_version_naming=args.auto_version_naming,
            save_frame_vis=args.save_frame_vis
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 処理されたフレーム数を計算（originalの'total'フィールドが実処理フレーム数）
        processed_frames = results['original']['total']
        
        # フレームレート計算
        fps = processed_frames / total_time if total_time > 0 else 0
        
        print("\n" + "="*60)
        print("検知結果サマリー")
        print("="*60)
        for rotation, stats in results.items():
            print(f"{rotation}: "
                  f"検知数 {stats['detected']}/{stats['total']}")
        print("="*60)
        print(f"処理時間: {total_time:.1f}秒")
        print(f"フレームレート: {fps:.2f} FPS ({processed_frames}フレーム)")
        print("="*60 + "\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
