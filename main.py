import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from load_data import dataloader
from model_train import train
from net import Baseline
from gpu_check import GPUcheck


if __name__ == '__main__':

    # Check avaiable GPU device
    GPUcheck()
    
    # DataLoading part
    train_dir = './train_images'    # folder containing training images
    test_dir = './test_images'    # folder containing test images
    train_loader, valid_loader, test_loader, classes = dataloader(train_dir, test_dir)
    
    # Training part
    device = torch.device("mps")
    n_epochs = 1
    model = Baseline()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss() #or nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train(train_loader, valid_loader, device, n_epochs, model, criterion, optimizer)

