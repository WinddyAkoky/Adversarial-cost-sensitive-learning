import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import sys
import torch.optim as optim
import os
import argparse
from utils import *

# 参数
parser = argparse.ArgumentParser(description="Mnist Training")
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=10, type=int, help='training epoches')
parser.add_argument('--c', default=5, type=int, help='constant')
parser.add_argument('--NORMALIZE', default=True, type=bool, help='data is normial')
parser.add_argument('--cuda_num', default='cpu', type=str, help='cuda number')
parser.add_argument('--epsilon',default=0.3, type=float, help='FGSM eposilon')
parser.add_argument('--gamma', default=0.95, type=float, help='lr_scheduel')
parser.add_argument('--batch_size', default=64, type=int, help='input batch size')
parser.add_argument('--model_path_home', default='../model', type=str, help='save path')

args = parser.parse_args()
lr = args.lr
epochs = args.epochs
c = args.c
NORMALIZE = args.NORMALIZE
cuda_num = args.cuda_num
epsilon = args.epsilon
gamma = args.gamma
batch_size = args.batch_size
model_path_home = args.model_path_home

DEVICE = torch.device(cuda_num if torch.cuda.is_available() else "cpu")

data_home = 'D:\\'


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = LeNet()
# 记录结果
results_infos = {}


def run_training():

    if NORMALIZE:
        train_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
        test_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    else:
        train_transform = transforms.Compose([transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.MNIST(root=os.path.join(data_home, 'dataset/MNIST'), train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.MNIST(root=os.path.join(data_home, 'dataset/MNIST'), train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    schedule = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)

    criterion = nn.CrossEntropyLoss().to(DEVICE)

    for epoch in range(epochs):
        correct = 0
        loss_sum = 0
        count = 0
        model.train()
        print('lr:', schedule.get_lr())
        for bat_nums, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += data.shape[0]

            print('\r bat_nums:{}, loss:{}'.format(bat_nums, loss_sum), end='')
        schedule.step()

        # test
        model.eval()
        correct = 0
        for bat_num, (data, target) in enumerate(test_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print('epoch:{}, Acc:{}'.format(epoch, correct/len(test_loader.dataset)))


def save_model():
    if NORMALIZE:
        path_model = os.path.join(model_path_home, 'LeNet_MNIST.pt')
    else:
        path_model = os.path.join(model_path_home, 'LeNet_MNIST_unnormalized.pt')
    torch.save(model.state_dict(), path_model)
    print('save model: {}'.format(path_model))
    

def test_on_adv():
    if NORMALIZE:
        test_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    else:
        test_transform = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.MNIST(root=os.path.join(data_home, 'dataset/MNIST'), train=False, download=True, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    model = LeNet()
    if NORMALIZE:
        path_model = os.path.join(model_path_home, 'LeNet_MNIST.pt')
    else:
        path_model = os.path.join(model_path_home, 'LeNet_MNIST_unnormalized.pt')
    model.load_state_dict(torch.load(path_model))
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    for bat_num, (data, target) in enumerate(test_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        data, sign = my_fgsm(data, target, model, criterion, epsilon, DEVICE)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('Acc on adv:{}'.format(correct/len(test_loader.dataset)))



if __name__ == "__main__":
    print('lr: {}'.format(lr))
    print('epochs:{}'.format(epochs))
    print('c:{}'.format(c))
    print('normalize:{}'.format(NORMALIZE))
    print('epsilon:{}'.format(epsilon))
    print('gamma:{}'.format(gamma))
    print('batch_size:{}'.format(batch_size))
    run_training()
    save_model()
    test_on_adv()








