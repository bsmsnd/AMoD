import torch
import torch.nn as nn
import torch.nn.functional as F
from ReplayMemory import *
import math
import random

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


class DuelingDQN(nn.Module):

    def __init__(self, n_features, n_actions):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 128)
        self.feature = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            # Stage 1
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2, stride=2),
            # Stage 2
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU()
            # nn.MaxPool2d(2, stride=2)
        )
        self.fc4 = nn.Linear(16*3*3, 9)
        self.advantage = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        

    def forward(self, x, y, z):
        # x: open requests batch_size x 9
        # y: available vehicles batch_size x 9
        # z: history requests batch_size x 4 x 3 x 3
        out1 = self.conv(z)
        out2 = self.fc4(out1.view(x.size(0), -1))
        out3 = torch.cat((x, y, out2), 1)
        out4 = self.relu(self.fc1(out3))
        out5 = self.relu(self.fc2(out4))
        out6 = self.relu(self.feature(out5))
        advantage = self.advantage(out6)
        value = self.value(out6)
        
        return value + advantage - advantage.mean()


def saveweight(model, path):
    # model: the network model
    # path: the path where the weights save
    torch.save(model.state_dict(), path)


def loadweight(model, path):
    # model: the new defined network model
    # path: the path where the weights save
    model.load_state_dict(torch.load(path))
    return model




