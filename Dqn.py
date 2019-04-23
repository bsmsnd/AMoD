import torch
import torch.nn as nn
import torch.nn.functional as F
from ReplayMemory import *
import math
import random

class DQN(nn.Module):

    def __init__(self, n_features, n_actions):
        super(DQN, self).__init__()
        self.fc2 = nn.Linear(n_features, 128)
        self.drop1 = nn.Dropout(p=0.4)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, n_actions)
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
        self.fc1 = nn.Linear(16*3*3, 9)

    def forward(self, x, y, z):
        # x: open requests batch_size x 9
        # y: available vehicles batch_size x 9
        # z: history requests batch_size x 4 x 3 x 3
        output = self.conv(z)
        output = self.fc1(output.view(x.size(0), -1))
        output = torch.cat((x, y, output), 1)
        output = self.relu(self.fc2(output))
        output = self.drop1(output)
        output = self.relu(self.fc3(output))
        output = self.relu(self.fc4(output))
        output = self.drop2(output)
        output = self.relu(self.fc5(output))
        output = self.relu(self.fc6(output))
        output = self.fc7(output)
        return output


def saveweight(model, path):
    # model: the network model
    # path: the path where the weights save
    torch.save(model.state_dict(), path)


def loadweight(model, path):
    # model: the new defined network model
    # path: the path where the weights save
    model.load_state_dict(torch.load(path))
    return model




