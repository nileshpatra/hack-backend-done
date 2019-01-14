import torch
import torchvision
import pickle
from torch import optim 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,datasets
import apple_call
class neural_net(nn.Module):
    def __init__(self):
        super().__init__()
        #64*64*3
        self.conv1 = nn.Conv2d(3 , 32 , 3 , padding = 1)
        #32*32*32
        self.conv2 = nn.Conv2d(32 , 64 , 3 , padding = 1)
        #16*16*64
        self.conv3 = nn.Conv2d(64 , 32 , 3 , padding = 1)
        #8*8*32
        self.dropout = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(8*8*32 , 100)
        self.fc2 = nn.Linear(100 , 64)
        self.fc3 = nn.Linear(64 , 32)
        self.fc4 = nn.Linear(32 , 4)
        
        
    def forward(self , x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1 , 8*8*32)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x) , dim = 1)
        return x