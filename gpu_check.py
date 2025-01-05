import torch

def GPUcheck():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA device is available: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device is available (Apple Silicon).")
    else:
        device = torch.device("cpu")
        print("Neither CUDA nor MPS found, using CPU.")
        print(f"MPS built: {torch.backends.mps.is_built()}")
        
