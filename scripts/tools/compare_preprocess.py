import torch
import cv2
import numpy as np
import math

def main():
    np.random.seed(42)
    img_np = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

    # OpenCV pipeline: BGR->RGB, float32, resize, mean subtraction
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_float = np.float32(img_rgb)
    resized_cv = cv2.resize(img_float, (640, 640), interpolation=cv2.INTER_LINEAR)
    resized_cv_sub = resized_cv - np.array([104.0, 117.0, 123.0], dtype=np.float32)

    # Torch variants (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_torch = torch.from_numpy(img_np.transpose(2,0,1)).float().to(device)  # [3,H,W] BGR
    img_torch_rgb = img_torch[[2,1,0],:,:]  # BGR->RGB
    img_torch_rgb_b = img_torch_rgb.unsqueeze(0)  # [1,3,H,W]

    mean = torch.tensor([104.0,117.0,123.0], device=device).view(1,3,1,1)

    bilinear_false = torch.nn.functional.interpolate(img_torch_rgb_b, size=(640,640), mode='bilinear', align_corners=False)
    bilinear_false_sub = bilinear_false - mean

    bilinear_true = torch.nn.functional.interpolate(img_torch_rgb_b, size=(640,640), mode='bilinear', align_corners=True)
    bilinear_true_sub = bilinear_true - mean

    area = torch.nn.functional.interpolate(img_torch_rgb_b, size=(640,640), mode='area')
    area_sub = area - mean

    cv_flat = resized_cv_sub.flatten()
    # convert Torch outputs from CHW to HWC before flattening to match OpenCV layout
    torch_false_np = bilinear_false_sub.squeeze().detach().cpu().numpy().transpose(1,2,0)
    torch_true_np = bilinear_true_sub.squeeze().detach().cpu().numpy().transpose(1,2,0)
    area_np = area_sub.squeeze().detach().cpu().numpy().transpose(1,2,0)

    torch_false_flat = torch_false_np.flatten()
    torch_true_flat = torch_true_np.flatten()
    area_flat = area_np.flatten()

    def stats(name, arr):
        print(f"{name}: min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}, std={arr.std():.2f}")

    def compare(name, arr):
        cc = np.corrcoef(cv_flat, arr)[0,1]
        rmse = math.sqrt(np.mean((cv_flat - arr)**2))
        md = np.max(np.abs(cv_flat - arr))
        print(f"{name}: corr={cc:.6f}, rmse={rmse:.3f}, maxdiff={md:.3f}")

    print("="*60)
    print("OpenCV vs Torch 前処理差分")
    print("="*60)
    stats("OpenCV", cv_flat)
    stats("Torch bilinear ac=False", torch_false_flat)
    stats("Torch bilinear ac=True", torch_true_flat)
    stats("Torch area", area_flat)
    print("-"*60)
    compare("Torch bilinear ac=False", torch_false_flat)
    compare("Torch bilinear ac=True", torch_true_flat)
    compare("Torch area", area_flat)
    print("="*60)

if __name__ == '__main__':
    main()
