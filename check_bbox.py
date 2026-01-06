#!/usr/bin/env python3
import json
import cv2
import numpy as np

# メタデータを読み込み
metadata_file = 'data/detected_coords_fixed/original/metadata.json'
with open(metadata_file) as f:
    metadata = json.load(f)

# フレームを読み込み（Docker コンテナ内でも実行可能なようにパスを調整）
cap = cv2.VideoCapture('/host_videos/REBD-827.hvec.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, frame = cap.read()
cap.release()

if not ret:
    # ホストの別のパスを試す
    cap = cv2.VideoCapture('/mnt/c/Users/mickey/Downloads/dfltest/REBD-827.hvec.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    cap.release()

if not ret:
    print("Failed to read frame")
    exit(1)

h, w = frame.shape[:2]
print(f"Frame size: {w}x{h}")

# 最初のフェイスIDでテスト
face_id = '000030_000_00'
if face_id in metadata:
    bbox = metadata[face_id]['bbox']
    print(f"\n{face_id}:")
    print(f"BBox from metadata: {bbox}")
    
    # 顔領域を抽出
    x1, y1, x2, y2 = [int(v) for v in bbox]
    print(f"BBox int: ({x1}, {y1}) to ({x2}, {y2})")
    
    # 元フレームにbboxを描画
    frame_vis = frame.copy()
    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # 保存
    output_path = '/root/docker_env/RotFace/test_bbox_verify.jpg'
    cv2.imwrite(output_path, frame_vis)
    print(f"Saved visualization to {output_path}")
    
    # 顔領域を確認
    face_crop = frame[y1:y2, x1:x2]
    print(f"Face crop shape: {face_crop.shape}")
    
    if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
        print("Face crop extracted successfully")
    else:
        print("ERROR: Invalid crop dimensions!")
else:
    print(f"Face ID {face_id} not found in metadata")
