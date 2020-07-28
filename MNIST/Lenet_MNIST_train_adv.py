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
import logging

# 参数
parser = argparse.ArgumentParser(description="MNIST Training")
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=30, type=int, help='training epoches')
parser.add_argument('--c', default=10, type=int, help='constant')
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


# 记录日志
path_log = '../log/Lenet_MNIST_adv_train.log'
logger = Logger(path_log, level='debug')


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
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


model = LeNet()
# path_model_u = '../model/LeNet_CIFAR_unnormalized.pt'
# model.load_state_dict(torch.load(path_model_u))
# print('load: {}'.format(path_model_u))
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
        
        logger.logger.info('lr:{}'.format(schedule.get_lr()))
        for bat_nums, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            data, sign = my_fgsm(data, target, model, criterion, epsilon, DEVICE)
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
        logger.logger.info('epoch:{}, Acc on clean:{}'.format(epoch, correct/len(test_loader.dataset)))

        # test
        model.eval()
        correct = 0
        for bat_num, (data, target) in enumerate(test_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            data, sign = my_fgsm(data, target, model, criterion, epsilon, DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        logger.logger.info('epoch:{}, Acc on adv:{}'.format(epoch, correct/len(test_loader.dataset)))


def save_model():
    if NORMALIZE:
        path_model = os.path.join(model_path_home, 'LeNet_MNIST_adv_e'+str(epsilon)+'.pt')
    else:
        path_model = os.path.join(model_path_home, 'LeNet_MNIST_unnormalized_adv_e'+str(epsilon)+'.pt')
    torch.save(model.state_dict(), path_model)
    logger.logger.info('save model: {}'.format(path_model))

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
        path_model = os.path.join(model_path_home, 'LeNet_MNIST_adv_e'+str(epsilon)+'.pt')
    else:
        path_model = os.path.join(model_path_home, 'LeNet_MNIST_unnormalized_adv_e'+str(epsilon)+'.pt')
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
    logger.logger.info('Acc on adv:{}'.format(correct/len(test_loader.dataset)))


if __name__ == "__main__":
    logger.logger.info("================================")
    logger.logger.info('lr: {}'.format(lr))
    logger.logger.info('epochs:{}'.format(epochs))
    logger.logger.info('c:{}'.format(c))
    logger.logger.info('normalize:{}'.format(NORMALIZE))
    logger.logger.info('epsilon:{}'.format(epsilon))
    logger.logger.info('gamma:{}'.format(gamma))
    logger.logger.info('batch_size:{}'.format(batch_size))

    run_training()
    save_model()
    test_on_adv()








