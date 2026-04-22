import torch

print("PyTorch 版本：", torch.__version__)
print("CUDA 是否可用：", torch.cuda.is_available())
print("你的显卡型号：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无可用GPU")