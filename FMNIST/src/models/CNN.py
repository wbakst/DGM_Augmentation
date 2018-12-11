import torch
from torch import nn
from torch.nn import functional as F

class BasicCNN(nn.Module):
    def __init__(self, name='basic_cnn'):
        super().__init__()
        self.name = name
        self.conv1 = nn.Conv2d(1, 5, 5)
        self.conv2 = nn.Conv2d(5, 10, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(10 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def loss(self, x, y):
        return nn.CrossEntropyLoss()(x, y)

    def classify(self, outputs):
        _, preds = torch.max(outputs, 1)
        return preds

class AdvancedCNN(nn.Module):
    def __init__(self, name='advanced_cnn'):
        super().__init__()
        self.name = name
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def loss(self, x, y):
        return nn.CrossEntropyLoss()(x, y)

    def classify(self, outputs):
        _, preds = torch.max(outputs, 1)
        return preds
