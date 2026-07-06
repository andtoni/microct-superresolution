import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("built for CUDA:", torch.version.cuda)