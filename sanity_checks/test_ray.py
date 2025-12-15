import os
import ray
import torch

# GPU-Test mit PyTorch
print("CUDA verfügbar?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Gerät:", torch.cuda.get_device_name(0))

# Wichtig: GPU-Autodetect deaktivieren (falls deine Ray-Version es nutzt)
os.environ["RAY_DISABLE_GPU_AUTODETECT"] = "1"

# HIER der entscheidende Unterschied:
# -> KEIN resources={"GPU": 1} mehr
# -> Stattdessen num_gpus explizit setzen
ray.init(
    include_dashboard=False,
    ignore_reinit_error=True,
    num_gpus=1,         # Ray bekommt 1 GPU-Ressource
)

@ray.remote(num_gpus=1)
def gpu_task():
    import torch
    x = torch.randn(1000, 1000, device="cuda")
    return float(x.mean())

result = ray.get(gpu_task.remote())
print("GPU Task Ergebnis:", result)

ray.shutdown()
