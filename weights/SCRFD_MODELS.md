# SCRFD Models

SCRFD (Sample and Computation Redistribution for Efficient Face Detection) is an efficient face detection model from InsightFace.

## Quick Download (Recommended)

Use the provided download script:

```bash
# Download SCRFD-34G model (scrfd_34g_bnkps.pth)
cd weights
python3 download_scrfd.py --model scrfd_34g --output-dir .

# Download SCRFD-10G model
python3 download_scrfd.py --model scrfd_10g --output-dir .

# Download all SCRFD models
python3 download_scrfd.py --model all --output-dir scrfd
```

**Note**: You need to install insightface first:
```bash
pip3 install insightface onnxruntime
```

## Available Models

### SCRFD-34GF (scrfd_34g_bnkps.pth)

This is the SCRFD model with 34G FLOPs, which provides a good balance between accuracy and speed.

## Download Links

SCRFD models are available from the InsightFace model zoo:

### Option 1: Direct Download from InsightFace

Visit the InsightFace model zoo:
- GitHub: https://github.com/deepinsight/insightface/tree/master/detection/scrfd
- Model Zoo: https://github.com/deepinsight/insightface/tree/master/model_zoo

### Option 2: Using InsightFace Python Package

```bash
pip install insightface
```

Then download programmatically:

```python
import insightface
from insightface.model_zoo import get_model

# Download SCRFD 34GF model
model = get_model('scrfd_34g_v2.0', download=True, root='./weights')
```

### Option 3: Direct Download Links

The SCRFD models are hosted on various platforms:

1. **Hugging Face** (Recommended):
   - Link: https://huggingface.co/SCRFD/SCRFD-34GF
   - File: `scrfd_34g_bnkps.pth` or `scrfd_34g_v2.0.onnx`

2. **Google Drive** (InsightFace official):
   - Check the InsightFace repository's README for the latest Google Drive links

3. **OneDrive** (Alternative):
   - Links provided in InsightFace documentation

## Model Variants

- **SCRFD-500M**: Ultra-lightweight, fastest inference
- **SCRFD-1G**: Lightweight, good for mobile
- **SCRFD-2.5G**: Balanced model
- **SCRFD-10G**: Higher accuracy
- **SCRFD-34G**: Best accuracy (this is scrfd_34g_bnkps.pth)

## Usage Notes

Currently, this repository supports **RetinaFace** models. To use SCRFD models, you would need to:

1. Download the SCRFD model file
2. Implement SCRFD inference code (or use InsightFace's inference utilities)
3. Adapt the face detection pipeline to use SCRFD instead of RetinaFace

## File Location

Once downloaded, place the model file in:
```
weights/scrfd_34g_bnkps.pth
```

Or create a subdirectory:
```
weights/scrfd/scrfd_34g_bnkps.pth
```

## References

- InsightFace GitHub: https://github.com/deepinsight/insightface
- SCRFD Paper: https://arxiv.org/abs/2105.04714
- Model Zoo: https://github.com/deepinsight/insightface/tree/master/model_zoo
