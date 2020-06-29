import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import os
import torch.nn.functional as f


print("print something")

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
DEVICE = torch.device('0' if torch.cuda.is_available() else "cpu")

data_home = 'F:\\work'
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root=os.path.join(data_home, 'dataset/CIFAR10'), train=True, download=True, transform=train_transform)
test_set = torchvision.datasets.CIFAR10(root=os.path.join(data_home, 'dataset/CIFAR10'), train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=1)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, 2)
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def run():
    model = LeNet()
    model = model.to(DEVICE)
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    # for epoch in range(50):
    #     for images, targets in train_loader:
    #         images, targets = images.to(DEVICE), targets.to(DEVICE)

    #         output = model(images)
    #         optimizer.zero_grad()
    #         loss = criterion(output, targets)
    #         loss.backward()
    #         optimizer.step()
    for i in range(10):
        its = iter(train_loader)
    

if __name__ == "__main__":
    run()