#!/usr/bin/env python3
import cv2
from pathlib import Path
import sys
src_root=Path('data/detected_skip10')
dst_root=Path('debug_samples/skip10')
(dst_root/'images').mkdir(parents=True,exist_ok=True)
(dst_root/'landmarks_qa').mkdir(parents=True,exist_ok=True)
orig_images=sorted((src_root/'original'/'images').glob('*.jpg'))[:6]
print('found', len(orig_images), 'orig images')
for p in orig_images:
    print('processing', p)
    img=cv2.imread(str(p))
    if img is None:
        print('failed read',p)
        continue
    h,w=img.shape[:2]
    scale=256/max(h,w)
    new=(max(1,int(w*scale)), max(1,int(h*scale)))
    thumb=cv2.resize(img, new)
    op=str(dst_root/'images'/p.name)
    cv2.imwrite(op, thumb)
    print('wrote',op)
    qa_name_jpg = p.stem + '_marked.jpg'
    qa_name_png = p.stem + '_marked.png'
    qa_src = src_root/'original'/'landmarks_qa'/qa_name_jpg
    if not qa_src.exists():
        qa_src = src_root/'original'/'landmarks_qa'/qa_name_png
    if qa_src.exists():
        qa=cv2.imread(str(qa_src))
        if qa is not None:
            h,w=qa.shape[:2]
            scale=256/max(h,w)
            new=(max(1,int(w*scale)), max(1,int(h*scale)))
            thumbs=cv2.resize(qa,new)
            qaop=str(dst_root/'landmarks_qa'/qa_src.name)
            cv2.imwrite(qaop, thumbs)
            print('wrote',qaop)
print('done')
