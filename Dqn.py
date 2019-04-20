import torch
import torch.nn as nn
import torch.nn.functional as F
from ReplayMemory import *
import math
import random

class DQN(nn.Module):

    def __init__(self, n_features, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_actions)
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
        self.fc5 = nn.Linear(4*4, 4)

    def forward(self, x, y, z, x_global, y_global, z_global):
        # x: open requests batch_size x 9
        # y: available vehicles batch_size x 9
        # z: history requests batch_size x 4 x 3 x 3
        # x_global: batch_size X 4
        # y_global: batch_size X 4
        # z_global: batch_size X 4 x 4
        output = self.conv(z) # batch_size x channel x 3 x 3
        output = self.fc4(output.view(x.size(0), -1))  # batch_size * 9

        # reshape x_global, y_global and z_global into 3D(the first dimension is batch_size)

        z_global = self.fc5(z_global.contiguous().view(x.size()[0], -1))
        #z_global = torch.ones([output.size()[0], z_global.size()[0], z_global.size()[1]]) * z_global

        #z_global = z_global.view(z_global.size()[0], -1)

        # concat x, y, z to x_global, y_global and z_global
        xx = torch.cat((x, x_global), 1)
        yy = torch.cat((y, y_global), 1)
        zz = torch.cat((output, z_global), 1)

        output = torch.cat((xx, yy, zz), 1)
        output = self.relu(self.fc1(output))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)
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




