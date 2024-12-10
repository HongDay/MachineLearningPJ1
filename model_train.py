import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from test import eval_accuracy


def train(train_loader, valid_loader, device, n_epochs, model, criterion, optimizer):

    # loop over epochs: one epoch = one pass through the whole training dataset
    for epoch in range(1, n_epochs+1):  
        #   loop over iterations: one iteration = 1 batch of examples
        i = 0
        for data, target in train_loader:
            model.train()
            i += 1
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if i % (int(len(train_loader)/10)+1) == 0:
                model.eval()
                eval_accuracy(valid_loader,model,device)
                print("-------------------------------------",np.round(i/len(train_loader)*100,2),"% complete")
            
        print(f"============================= epoch {epoch} done out of {n_epochs} !!")
        model.eval()
        eval_accuracy(valid_loader,model,device)
        print("\n")
