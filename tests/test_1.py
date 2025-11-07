import torch
print("PyTorch Version:", torch.__version__)
print("PyTorch CUDA:", torch.cuda.is_available())
print("PyTorch Device Count:", torch.cuda.device_count())
