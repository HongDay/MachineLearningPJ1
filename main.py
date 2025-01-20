import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from load_data import dataloader
from model_train import train
from net import Baseline
from gpu_check import GPUcheck

from test import eval_accuracy
from test import plot_confusion_matrix
from test import plot_training_curves

if __name__ == '__main__':

    # Check avaiable GPU device
    GPUcheck()
    
    # DataLoading part
    train_dir = './train_images'    # folder containing training images
    test_dir = './test_images'    # folder containing test images
    train_loader, valid_loader, test_loader, classes = dataloader(train_dir, test_dir)

    # Training part
    device = torch.device("cpu")
    n_epochs = 5
    model = Baseline()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss() #or nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_loss, valid_loss, train_acc, valid_acc = train(train_loader, valid_loader, device, n_epochs, model, criterion, optimizer)
  
    plot_confusion_matrix(valid_loader, model, device, classes)
    plot_confusion_matrix(test_loader, model, device, classes)

    plot_training_curves(train_loss, valid_loss, train_acc, valid_acc, n_epochs)

    eval_accuracy(valid_loader,model, device, criterion, "Validation")
    eval_accuracy(test_loader, model, device, criterion, "Test")