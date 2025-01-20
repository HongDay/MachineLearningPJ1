import torch

def GPUcheck():
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x, ": device is available")
    else:
        print ("MPS device not found.")

    print(torch.backends.mps.is_built(),'\n')