#!/usr/bin/env python3
import json
import cv2
import numpy as np

metadata_file = '/workspace/data/detected_final/original/metadata.json'
with open(metadata_file) as f:
    metadata = json.load(f)

face_id = '000030_000_00'
info = metadata[face_id]
bbox = info['bbox']
landmarks_dict = info['landmarks']

landmarks = np.array([
    landmarks_dict['left_eye'],
    landmarks_dict['right_eye'],
    landmarks_dict['nose'],
    landmarks_dict['left_mouth'],
    landmarks_dict['right_mouth']
], dtype=np.float32)

print(f"Face ID: {face_id}")
print(f"BBox: {bbox}")

cap = cv2.VideoCapture('/host_videos/REBD-827.hvec.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, frame = cap.read()
cap.release()

if ret:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    face_crop = frame[y1:y2, x1:x2]
    face_h, face_w = face_crop.shape[:2]
    output_size = 128
    scale = output_size / max(face_h, face_w)
    
    print(f"Frame: {h}x{w}, Face crop: {face_h}x{face_w}, Scale: {scale:.4f}")
    
    pad_top = (output_size - int(face_h * scale)) // 2
    pad_left = (output_size - int(face_w * scale)) // 2
    
    for i, (name, landmark) in enumerate(zip(['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth'], landmarks)):
        lm_x, lm_y = landmark
        crop_x = lm_x - x1
        crop_y = lm_y - y1
        scaled_x = crop_x * scale
        scaled_y = crop_y * scale
        final_x = scaled_x + pad_left
        final_y = scaled_y + pad_top
        in_range = 0 <= final_x < output_size and 0 <= final_y < output_size
        print(f"{name}: final=({final_x:.1f}, {final_y:.1f}) in_range={in_range}")
