import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import warnings


# Basic NN model which was in skeleton code provided by professor
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 6 * 6, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        print(size)
        for s in size:
            num_features *= s
            print(s)
        return num_features



## We should imporve the accuracy by modify this basic model 
'''
class Newmodel_ver1(nn.Module):
    def __init__(self):
        super(Newmodel_ver1, self).__init__()
        # construct the layers

    def forward(self, x):
        # forward through the layers
        return x
'''


# prepare pre-trained vgg16 model

warnings.filterwarnings("ignore", category=UserWarning)

def get_vgg16():
    # Load pre-trained ResNet-50
    model = models.vgg16(pretrained=True)

    # Replace the first convolutional layer to accept 1 channel
    model.features[0] = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
    
    # Replace the final fully connected layer for your specific task
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2)
    
    return model