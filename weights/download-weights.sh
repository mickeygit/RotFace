#!/bin/bash
# Download RetinaFace model weights

pip3 install gdown

# RetinaFace models (existing)
gdown https://drive.google.com/uc?id=1Z-xJW8jK60vfTlmH4BYKRmL52tLPErJY
gdown https://drive.google.com/uc?id=1XLgPuX3VW-xMI1ujFCnW9K62MlJ4Fskp
gdown https://drive.google.com/uc?id=14hBGOuUHDLn5Ur5HUmOf2FlSvaEumvg1

# SCRFD models (optional)
# For SCRFD model download instructions, see SCRFD_MODELS.md
# To download SCRFD-34GF (scrfd_34g_bnkps.pth), use:
# pip3 install insightface
# python3 -c "from insightface.model_zoo import get_model; get_model('scrfd_34g_v2.0', download=True, root='.')"