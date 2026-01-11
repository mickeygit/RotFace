#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRFD Model Downloader

Download SCRFD face detection models from InsightFace model zoo.

Usage:
    python3 download_scrfd.py --model scrfd_34g --output-dir weights/scrfd
    python3 download_scrfd.py --model scrfd_10g --output-dir weights/scrfd
    python3 download_scrfd.py --model all --output-dir weights/scrfd
"""

import argparse
import os
import sys
from pathlib import Path

def download_scrfd_model(model_name: str, output_dir: str) -> bool:
    """
    Download SCRFD model using InsightFace
    
    Args:
        model_name: Model name (e.g., 'scrfd_34g', 'scrfd_10g', 'scrfd_2.5g', 'scrfd_1g', 'scrfd_500m')
        output_dir: Output directory path
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from insightface.model_zoo import get_model
    except ImportError:
        print("Error: insightface package not found.")
        print("Please install it using: pip3 install insightface")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Map model names to InsightFace model zoo names
    # Note: InsightFace uses different naming conventions for different model versions
    model_map = {
        'scrfd_500m': 'scrfd_500m_bnkps',
        'scrfd_1g': 'scrfd_1g_bnkps',
        'scrfd_2.5g': 'scrfd_2.5g_bnkps',
        'scrfd_10g': 'scrfd_10g_bnkps',
        'scrfd_34g': 'scrfd_34g_v2.0',  # v2.0 is the latest version of 34G model
    }
    
    if model_name not in model_map:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {', '.join(model_map.keys())}")
        return False
    
    zoo_name = model_map[model_name]
    
    print(f"Downloading {model_name} ({zoo_name})...")
    print(f"Output directory: {output_dir}")
    
    try:
        model = get_model(zoo_name, download=True, root=output_dir)
        print(f"✓ Successfully downloaded {model_name}")
        
        # Try to find the downloaded file
        # InsightFace may download with different naming conventions
        output_path = Path(output_dir)
        pth_files = list(output_path.glob("*.pth"))
        onnx_files = list(output_path.glob("*.onnx"))
        
        # Filter for files that might be related to this model
        relevant_pth = [f for f in pth_files if 'scrfd' in f.name.lower()]
        relevant_onnx = [f for f in onnx_files if 'scrfd' in f.name.lower()]
        
        if relevant_pth:
            print(f"  → PyTorch model files found:")
            for f in relevant_pth:
                print(f"     {f.name}")
        if relevant_onnx:
            print(f"  → ONNX model files found:")
            for f in relevant_onnx:
                print(f"     {f.name}")
        
        if not relevant_pth and not relevant_onnx:
            print(f"  Note: Model downloaded successfully but files may be in a different format or location.")
            print(f"  Check {output_dir} for downloaded files.")
        
        return True
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download SCRFD face detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download SCRFD-34G model (scrfd_34g_bnkps.pth)
  python3 download_scrfd.py --model scrfd_34g --output-dir weights/scrfd
  
  # Download SCRFD-10G model
  python3 download_scrfd.py --model scrfd_10g --output-dir weights/scrfd
  
  # Download all SCRFD models
  python3 download_scrfd.py --model all --output-dir weights/scrfd

Available models:
  - scrfd_500m: Ultra-lightweight (500M FLOPs)
  - scrfd_1g:   Lightweight (1G FLOPs)
  - scrfd_2.5g: Balanced (2.5G FLOPs)
  - scrfd_10g:  High accuracy (10G FLOPs)
  - scrfd_34g:  Best accuracy (34G FLOPs) - this is scrfd_34g_bnkps.pth
  - all:        Download all models
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['scrfd_500m', 'scrfd_1g', 'scrfd_2.5g', 'scrfd_10g', 'scrfd_34g', 'all'],
        help='SCRFD model to download'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='weights/scrfd',
        help='Output directory for downloaded models (default: weights/scrfd)'
    )
    
    args = parser.parse_args()
    
    # Check if insightface is installed
    try:
        import insightface
    except ImportError:
        print("=" * 60)
        print("ERROR: insightface package is not installed")
        print("=" * 60)
        print("Please install it using:")
        print("  pip3 install insightface")
        print("")
        print("Additional dependencies may be needed:")
        print("  pip3 install onnxruntime")
        print("=" * 60)
        sys.exit(1)
    
    print("=" * 60)
    print("SCRFD Model Downloader")
    print("=" * 60)
    
    if args.model == 'all':
        models = ['scrfd_500m', 'scrfd_1g', 'scrfd_2.5g', 'scrfd_10g', 'scrfd_34g']
        success_count = 0
        for model in models:
            if download_scrfd_model(model, args.output_dir):
                success_count += 1
            print("")
        
        print("=" * 60)
        print(f"Downloaded {success_count}/{len(models)} models successfully")
        print("=" * 60)
    else:
        success = download_scrfd_model(args.model, args.output_dir)
        print("=" * 60)
        if success:
            print("Download completed successfully!")
        else:
            print("Download failed!")
            sys.exit(1)
        print("=" * 60)


if __name__ == '__main__':
    main()
