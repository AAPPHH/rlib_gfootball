import torch

def get_device_name(index: int = 0) -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(index)
    return "CPU"

if __name__ == "__main__":
    print(get_device_name())