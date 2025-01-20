import torch.nn as nn
import torch.nn.functional as F


# Basic NN model which was in skeleton code provided by professor
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(16 * 6 * 6, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(F.relu(x))

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(F.relu(x))

        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.dropout(x)
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