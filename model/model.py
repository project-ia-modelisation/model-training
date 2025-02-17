import torch
import torch.nn as nn

class Simple3DGenerator(nn.Module):
    def __init__(self, num_vertices=1000):
        super(Simple3DGenerator, self).__init__()
        self.num_vertices = num_vertices
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, num_vertices * 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x.view(-1, self.num_vertices, 3)  # Reshape to the correct number of vertices

class Simple2DGenerator(nn.Module):
    def __init__(self):
        super(Simple2DGenerator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 64 * 64)  # Assuming 64x64 image size

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x.view(-1, 64, 64)  # Reshape to 2D image
