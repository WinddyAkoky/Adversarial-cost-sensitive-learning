import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as f
import sys
import torch.optim as optim
import os
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
from utils import *

# 参数
parser = argparse.ArgumentParser(description="CIFAR CSA Training")
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=50, type=int, help='training epoches')
parser.add_argument('--c', default=10, type=int, help='constant')
parser.add_argument('--NORMALIZE', default=True, type=bool, help='data is normial')
parser.add_argument('--cuda_num', default='0', type=str, help='cuda number')
parser.add_argument('--epsilon',default=0.3, type=float, help='FGSM eposilon')
parser.add_argument('--momentum', default=0.5, type=float, help='optim:momentum')
parser.add_argument('--gamma', default=0.95, type=float, help='lr_scheduel')
parser.add_argument('--batch_size', default=64, type=int, help='input batch size')

args = parser.parse_args()
lr = args.lr
epochs = args.epochs
c = args.c
NORMALIZE = args.NORMALIZE
cuda_num = args.cuda_num
epsilon = args.epsilon
momentum = args.momentum
gamma = args.gamma
batch_size = args.batch_size

DEVICE = torch.device(cuda_num if torch.cuda.is_available() else "cpu")

data_home = 'F:\\work'
if NORMALIZE:
    path_model_adv = '../model/' + 'LeNet_FASHION_MNIST_adv_e'+str(epsilon)+'.pt'
else:
    pass
    # path_model_adv = '../model/' + 'LeNet_CIFAR_unnormalized_adv_e'+str(epsilon)+'.pt'



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

# 记录结果
results_infos = {}

# 记录日志
path_log = '../log/Lenet_FASHION_MNIST_CSA.log'
logger = Logger(path_log, level='debug')

def run_training():
    model_adv = LeNet()
    
    logger.logger.info('load: {}'.format(path_model_adv))
    model_adv.load_state_dict(torch.load(path_model_adv))

    
    if NORMALIZE:
        train_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.2862,), (0.3204,))])
        test_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.2862,), (0.3204,))])
    else:
        train_transform = transforms.Compose([transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.FashionMNIST(root=os.path.join(data_home, 'dataset/Fashion_mnist'), train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.FashionMNIST(root=os.path.join(data_home, 'dataset/Fashion_mnist'), train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    
    criterion = Loss_cost_sensitive()

    # 循环 对每一个类分别进行保护
    for i_label in range(10):
        # 读取预训练模型
        model_cost_sensitive = LeNet()
        model_cost_sensitive.load_state_dict(torch.load(path_model_adv))
        logger.logger.info('load model for initialization: {}'.format(path_model_adv))
        model_cost_sensitive = model_cost_sensitive.to(DEVICE)
        
        optimizer = torch.optim.Adam(params=model_cost_sensitive.parameters(), lr=lr)
        schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
        
        C = get_cost_matric(i_label, c)
        C = C.to(DEVICE)
        logger.logger.info('protect label: {}'.format(i_label))
        logger.logger.info('load cost matric: ')
        logger.logger.info(C)
        
        LABEL = 'Protect Label ' + str(i_label)
        
        # 记录最好的模型
        correst_adv_best = 0

        # 开始训练
        logger.logger.info('开始训练')
        for epoch in range(epochs):
            count = 0
            model_cost_sensitive.train()
            for data, target in train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)

                optimizer.zero_grad()
                output = model_cost_sensitive(data)
                loss = criterion(output, target, C)
                loss.backward()
                optimizer.step()

                data, sign = my_fgsm(data, target, model_cost_sensitive, criterion, epsilon, DEVICE, C)
                optimizer.zero_grad()
                output = model_cost_sensitive(data)
                loss = criterion(output, target, C)
                loss.backward()
                optimizer.step()
                          
                count += len(data)
                print('\r {}|{}'.format(count, len(train_loader.dataset)), end='')
            schedule.step()

            # 测试
            correct = 0
            model_cost_sensitive.eval()
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model_cost_sensitive(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            logger.logger.info('epoch: {}, test correct on clean: {}'.format(epoch,correct/len(test_loader.dataset)))

            # 测试
            correct = 0
            model_cost_sensitive.eval()
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
    #             
                data, sign = my_fgsm(data, target, model_cost_sensitive, criterion, epsilon, DEVICE, C)
                output = model_cost_sensitive(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            logger.logger.info('epoch: {}, test correct on adv: {}'.format(epoch,correct/len(test_loader.dataset)))

            if correct/len(test_loader.dataset) > correst_adv_best:
                correst_adv_best = correct/len(test_loader.dataset)
                 # 保存模型
                if NORMALIZE:
                    path_model_save = '../model/LeNet_FASHION_MNIST_adv_cost_sensitive_' + str(i_label) + '_e' + str(epsilon) +'.pt'
                else:
                    path_model_save = '../model/LeNet_FASHION_MNIST_unnormalized_adv_cost_sensitive_' + str(i_label) + '_e' + str(epsilon) +'.pt'
                torch.save(model_cost_sensitive.state_dict(), path_model_save)
                logger.logger.info('save model:{}'.format(path_model_save))
        
        # 训练结束
    #     比较结果

        ## 对 model_cost_sensitive 的评估
        logger.logger.info('对 model_cost_sensitive 的评估')
        results_info = {}
        model_cost_sensitive.eval()
        images_targets = {}
        for special_index in range(10):
            count = 0
            correct = 0

            for data, target in test_loader:
                data = data[target==special_index]
                target = target[target==special_index]
                if len(data) == 0:
                    continue

                data, target = data.to(DEVICE), target.to(DEVICE)
                data, sign = my_fgsm(data, target, model_cost_sensitive, criterion, epsilon, DEVICE, C)
                output = model_cost_sensitive(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += len(data)
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            logger.logger.info('\n {} correct: {}'.format(special_index,correct/count))
        results_info[1] = images_targets
        
        ## 对 model_adv 的评估
        logger.logger.info('对 model_adv 的评估')
        model_adv.eval()
        images_targets = {}
        criterion_adv = nn.CrossEntropyLoss()
        for special_index in range(10):
            count = 0
            correct = 0

            for data, target in test_loader:
                data = data[target==special_index]
                target = target[target==special_index]
                if len(data) == 0:
                    continue

                data, target = data.to(DEVICE), target.to(DEVICE)
                data, sign = my_fgsm(data, target, model_adv, criterion_adv, epsilon, DEVICE)
                output = model_adv(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += len(data)
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            logger.logger.info('\n {} correct: {}'.format(special_index,correct/count))
        results_info[2] = images_targets
        
        # 记录结果
        results_infos[i_label] = results_info


if __name__ == "__main__":
    logger.logger.info('lr: {}'.format(lr))
    logger.logger.info('epochs:{}'.format(epochs))
    logger.logger.info('c:{}'.format(c))
    logger.logger.info('normalize:{}'.format(NORMALIZE))
    logger.logger.info('epsilon:{}'.format(epsilon))
    logger.logger.info('momentum:{}'.format(momentum))
    logger.logger.info('gamma:{}'.format(gamma))
    logger.logger.info('batch_size:{}'.format(batch_size))

    run_training()








