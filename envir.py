#配置环境
import platform
import psutil
import torch
import sys
import torchvision
import numpy as np
import cv2
import PIL

# 操作系统信息
print(f"操作系统: {platform.system()} {platform.release()}")
print(f"CPU核心数: {psutil.cpu_count(logical=True)}")

# GPU信息（基于PyTorch）
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    print("CUDA不可用")

print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"torchvision版本: {torchvision.__version__}")
print(f"NumPy版本: {np.__version__}")
print(f"OpenCV版本: {cv2.__version__}")
print(f"Pillow版本: {PIL.__version__}")