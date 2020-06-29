import torch 
import numpy as np
import os
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import copy
from torch.autograd.gradcheck import zero_gradients
from utils import *
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description="CIFAR EVALUATION")
parser.add_argument('--epsilon_model', default=0.03, type=float, help='epsilon_model')
parser.add_argument('--epsilon_attack', default=0.03, type=float, help='epsilon_attack')
parser.add_argument('--max_iter', default=3, type=int, help='deepfool: max iterative')
args = parser.parse_args()
epsilon_model = args.epsilon_model
epsilon_attack = args.epsilon_attack
max_iter = args.max_iter
data_home = 'F:\\work'
cuda_num = 'cpu'
NORMALIZE = True


# 记录日志
path_log = '../log/Lenet_CIFAR_evaluation.log'
logger = Logger(path_log, level='debug')


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


def show_result(results_infos):
    I_avg = {'CSA':[], 'CSE':[], 'ADV':[], 'NORMAL':[]}
    writer = pd.ExcelWriter(os.path.join('./output/FASHION_MNIST_'+INFO+ '_model' +str(epsilon_model)+ '_attach'+ str(epsilon_attack) +'.xlsx'))

    for i in results_infos.keys():
        tmp = results_infos[i]
        
        I_avg['CSA'].append(tmp[1][i][1])
        I_avg['CSE'].append(tmp[2][i][1])
        I_avg['ADV'].append(tmp[3][i][1])
        I_avg['NORMAL'].append(tmp[4][i][1])

        df  = pd.DataFrame(tmp)
        df.columns = ['CSA', 'CSE', 'ADV', 'NORMAL']
        df = pd.DataFrame([df[i].apply(lambda x: x[1]) for i in df.columns])
        df = df.sort_index()

        df.to_excel(writer, sheet_name=str(i))
    writer.save()

    logger.logger.info('I of CSA: {}'.format(np.mean(I_avg['CSA'])))
    logger.logger.info('I of CSE: {}'.format(np.mean(I_avg['CSE'])))
    logger.logger.info('I of ADV: {}'.format(np.mean(I_avg['ADV'])))
    logger.logger.info('I of NORMAL: {}'.format(np.mean(I_avg['NORMAL'])))  


def test_on_clean():

    DEVICE = torch.device(cuda_num if torch.cuda.is_available() else "cpu")
    batch_size = 64

    if NORMALIZE:
        train_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        train_transform = transforms.Compose([transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root=os.path.join(data_home, 'dataset/CIFAR10'), train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=os.path.join(data_home, 'dataset/CIFAR10'), train=False, download=True, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

        # 记录结果
    results_infos = {}
    criterion_CSA = Loss_cost_sensitive()

    # 循环 对每一个类分别进行保护
    for i_label in range(10):
        ################################
        # 读取模型
        model_CSA = LeNet()
        path_model_CSA = '../model/LeNet_CIFAR_adv_cost_sensitive_'+ str(i_label) +'_e'+ str(epsilon_model) +'.pt'
        model_CSA.load_state_dict(torch.load(path_model_CSA))
        print('load model for initialization: {}'.format(path_model_CSA))
        model_CSA = model_CSA.to(DEVICE)
        
        C = get_cost_matric(i_label)
        C = C.to(DEVICE)
        print('protect label: {}'.format(i_label))
        print('load cost matric: ')
        print(C)
        LABEL = 'Protect Label ' + str(i_label)

        ## 对 model_CSA 的评估
        print('对 model_CSA 的评估')
        results_info = {}
        model_CSA.eval()
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
                output = model_CSA(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += len(data)
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            print('\n {} correct: {}'.format(special_index,correct/count))
        results_info[1] = images_targets
        
        ##################################
        ## 对 model_CSE 的评估
        print('对 model_CSE 的评估')
        # 读取模型
        model_CSE = LeNet()
        path_model_CSE = '../model/LeNet_CIFAR_cost_sensitive_extension_'+str(i_label)+'.pt'
        model_CSE.load_state_dict(torch.load(path_model_CSE))
        print('load model for initialization: {}'.format(path_model_CSE))
        model_CSE = model_CSE.to(DEVICE)
        criterion_CSE = Loss_CSE(model_CSE,1,1,1)

        model_CSE.eval()
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
                output = model_CSE(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += len(data)
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            print('\n {} correct: {}'.format(special_index,correct/count))
        results_info[2] = images_targets
        
        ##################################
        ## 对 model_ADV 的评估
        print('对 model_ADV 的评估')

        model_ADV.eval()
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
                output = model_ADV(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += len(data)
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            print('\n {} correct: {}'.format(special_index,correct/count))
        results_info[3] = images_targets
        
        ##################################
        ## 对 model_NORMAL 的评估
        print('对 model_NORMAL 的评估')

        model_NORMAL.eval()
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
                output = model_NORMAL(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += len(data)
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            print('\n {} correct: {}'.format(special_index,correct/count))
        results_info[4] = images_targets 
        # 记录结果
        results_infos[i_label] = results_info
    show_result(results_infos)


def test_on_FGSM():

    DEVICE = torch.device(cuda_num if torch.cuda.is_available() else "cpu")
    batch_size = 64

    if NORMALIZE:
        train_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        train_transform = transforms.Compose([transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root=os.path.join(data_home, 'dataset/CIFAR10'), train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=os.path.join(data_home, 'dataset/CIFAR10'), train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    # 记录结果
    results_infos = {}

    # 先读取未经过对抗训练的模型
    # 在进行对抗训练
    criterion_CSA = Loss_cost_sensitive()

    # 循环 对每一个类分别进行保护
    for i_label in range(10):
        ################################
        # 读取模型
        model_CSA = LeNet()
        path_model_CSA = '../model/LeNet_CIFAR_adv_cost_sensitive_'+ str(i_label) +'_e'+ str(epsilon_model) +'.pt'
        model_CSA.load_state_dict(torch.load(path_model_CSA))
        print('load model for initialization: {}'.format(path_model_CSA))
        model_CSA = model_CSA.to(DEVICE)
        
        
        C = get_cost_matric(i_label)
        C = C.to(DEVICE)
        print('protect label: {}'.format(i_label))
        print('load cost matric: ')
        print(C)
        LABEL = 'Protect Label ' + str(i_label)

        ## 对 model_CSA 的评估
        print('对 model_CSA 的评估')
        results_info = {}
        model_CSA.eval()
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
                data, sign = my_fgsm(data, target, model_CSA, criterion_CSA, epsilon_attack, DEVICE, C)
                output = model_CSA(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += len(data)
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            print('\n {} correct: {}'.format(special_index,correct/count))
        results_info[1] = images_targets
        
    ##################################
        ## 对 model_CSE 的评估
        print('对 model_CSE 的评估')
        # 读取模型
        model_CSE = LeNet()
        path_model_CSE = '../model/LeNet_CIFAR_cost_sensitive_extension_'+str(i_label)+'.pt'
        model_CSE.load_state_dict(torch.load(path_model_CSE))
        print('load model for initialization: {}'.format(path_model_CSE))
        model_CSE = model_CSE.to(DEVICE)
        criterion_CSE = Loss_CSE(model_CSE,1,1,1)

        model_CSE.eval()
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
                data, sign = my_fgsm(data, target, model_CSE, criterion_CSE, epsilon_attack, DEVICE, C)
                output = model_CSE(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += len(data)
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            print('\n {} correct: {}'.format(special_index,correct/count))
        results_info[2] = images_targets
        
    ##################################
        ## 对 model_ADV 的评估
        print('对 model_ADV 的评估')


        model_ADV.eval()
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
                data, sign = my_fgsm(data, target, model_ADV, criterion, epsilon_attack, DEVICE)
                output = model_ADV(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += len(data)
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            print('\n {} correct: {}'.format(special_index,correct/count))
        results_info[3] = images_targets
        
    ##################################
        ## 对 model_NORMAL 的评估
        print('对 model_NORMAL 的评估')

        model_NORMAL.eval()
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
                data, sign = my_fgsm(data, target, model_NORMAL, criterion, epsilon_attack, DEVICE)
                output = model_NORMAL(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += len(data)
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            print('\n {} correct: {}'.format(special_index,correct/count))
        results_info[4] = images_targets 
        # 记录结果
        results_infos[i_label] = results_info
    show_result(results_infos)


def test_on_deepfool():
    DEVICE = torch.device(cuda_num if torch.cuda.is_available() else "cpu")
    batch_size = 1

    if NORMALIZE:
        train_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        train_transform = transforms.Compose([transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root=os.path.join(data_home, 'dataset/CIFAR10'), train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=os.path.join(data_home, 'dataset/CIFAR10'), train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    # 记录结果
    results_infos = {}

    # 先读取未经过对抗训练的模型
    # 在进行对抗训练

    # 参数
    criterion_CSA = Loss_cost_sensitive()

    # 循环 对每一个类分别进行保护
    for i_label in range(10):
        ################################
        # 读取模型
        model_CSA = LeNet()
        path_model_CSA = '../model/LeNet_CIFAR_adv_cost_sensitive_'+ str(i_label) +'_e'+ str(epsilon_model) +'.pt'
        model_CSA.load_state_dict(torch.load(path_model_CSA))
        print('load model for initialization: {}'.format(path_model_CSA))
        model_CSA = model_CSA.to(DEVICE)
        
        
        C = get_cost_matric(i_label)
        C = C.to(DEVICE)
        print('protect label: {}'.format(i_label))
        print('load cost matric: ')
        print(C)
        LABEL = 'Protect Label ' + str(i_label)

        ## 对 model_CSA 的评估
        print('对 model_CSA 的评估')
        results_info = {}
        model_CSA.eval()
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
                data = data.reshape(3,32,32)
                r, loop_i, label_orig, label_pert, pert_image = deepfool(data, model_CSA, max_iter=max_iter)
                output = model_CSA(pert_image)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += 1
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            print('\n {} correct: {}'.format(special_index,correct/count))
        results_info[1] = images_targets
        
    ##################################
        ## 对 model_CSE 的评估
        print('对 model_CSE 的评估')
        # 读取模型
        model_CSE = LeNet()
        path_model_CSE = '../model/LeNet_CIFAR_cost_sensitive_extension_'+str(i_label)+'.pt'
        model_CSE.load_state_dict(torch.load(path_model_CSE))
        print('load model for initialization: {}'.format(path_model_CSE))
        model_CSE = model_CSE.to(DEVICE)
        criterion_CSE = Loss_CSE(model_CSE,1,1,1)

        model_CSE.eval()
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
                data = data.reshape(3,32,32)
                r, loop_i, label_orig, label_pert, pert_image = deepfool(data, model_CSE, max_iter=max_iter)
                output = model_CSE(pert_image)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += 1
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            print('\n {} correct: {}'.format(special_index,correct/count))
        results_info[2] = images_targets
        
    ##################################
        ## 对 model_ADV 的评估
        print('对 model_ADV 的评估')


        model_ADV.eval()
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
                data = data.reshape(3,32,32)
                r, loop_i, label_orig, label_pert, pert_image = deepfool(data, model_ADV, max_iter=max_iter)
                output = model_ADV(pert_image)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += 1
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            print('\n {} correct: {}'.format(special_index,correct/count))
        results_info[3] = images_targets
        
    ##################################
        ## 对 model_NORMAL 的评估
        print('对 model_NORMAL 的评估')

        model_NORMAL.eval()
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
                data = data.reshape(3,32,32)
                r, loop_i, label_orig, label_pert, pert_image = deepfool(data, model_NORMAL, max_iter=max_iter)
                output = model_NORMAL(pert_image)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                count += 1
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            print('\n {} correct: {}'.format(special_index,correct/count))
        results_info[4] = images_targets 
        # 记录结果
        results_infos[i_label] = results_info
    show_result(results_infos)


def test_on_PGD():

    CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    DEVICE = torch.device(cuda_num if torch.cuda.is_available() else "cpu")
    batch_size = 64

    if NORMALIZE:
        train_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        train_transform = transforms.Compose([transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root=os.path.join(data_home, 'dataset/CIFAR10'), train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=os.path.join(data_home, 'dataset/CIFAR10'), train=False, download=True, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

        # 记录结果
    results_infos = {}

    # 先读取未经过对抗训练的模型
    # 在进行对抗训练

    # 参数
    criterion_CSA = Loss_cost_sensitive()


    # 循环 对每一个类分别进行保护
    for i_label in range(10):
        ################################
        # 读取模型
        model_CSA = LeNet()
        path_model_CSA = '../model/LeNet_CIFAR_adv_cost_sensitive_'+ str(i_label) +'_e'+ str(epsilon_model) +'.pt'
        model_CSA.load_state_dict(torch.load(path_model_CSA))
        print('load model for initialization: {}'.format(path_model_CSA))
        model_CSA = model_CSA.to(DEVICE)
        
        
        C = get_cost_matric(i_label)
        C = C.to(DEVICE)
        print('protect label: {}'.format(i_label))
        print('load cost matric: ')
        print(C)
        LABEL = 'Protect Label ' + str(i_label)

        ## 对 model_CSA 的评估
        print('对 model_CSA 的评估')
        results_info = {}
        model_CSA.eval()
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
                data = pgd_attack(model_CSA, data, target, eps=epsilon_attack, alpha=2/255, iters=40, loss=criterion_CSA, C=C)
                output = model_CSA(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += len(data)
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            print('\n {} correct: {}'.format(special_index,correct/count))
        results_info[1] = images_targets
        
    ##################################
        ## 对 model_CSE 的评估
        print('对 model_CSE 的评估')
        # 读取模型
        model_CSE = LeNet()
        path_model_CSE = '../model/LeNet_CIFAR_cost_sensitive_extension_'+str(i_label)+'.pt'
        model_CSE.load_state_dict(torch.load(path_model_CSE))
        print('load model for initialization: {}'.format(path_model_CSE))
        model_CSE = model_CSE.to(DEVICE)
        criterion_CSE = Loss_CSE(model_CSE,1,1,1)

        model_CSE.eval()
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
                data = pgd_attack(model_CSE, data, target, eps=epsilon_attack, alpha=2/255, iters=40, loss=criterion_CSE, C=C)
                output = model_CSE(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += len(data)
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            print('\n {} correct: {}'.format(special_index,correct/count))
        results_info[2] = images_targets
        
    ##################################
        ## 对 model_ADV 的评估
        print('对 model_ADV 的评估')


        model_ADV.eval()
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
                data = pgd_attack(model_ADV, data, target, eps=epsilon_attack, alpha=2/255, iters=40, loss=criterion)
                output = model_ADV(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += len(data)
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            print('\n {} correct: {}'.format(special_index,correct/count))
        results_info[3] = images_targets
        
    ##################################
        ## 对 model_NORMAL 的评估
        print('对 model_NORMAL 的评估')

        model_NORMAL.eval()
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
                data = pgd_attack(model_NORMAL, data, target, eps=epsilon_attack, alpha=2/255, iters=40, loss=criterion)
                output = model_NORMAL(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += len(data)
                print('\r {}'.format(count), end='')
            images_targets[special_index] = [count, correct/count]
            print('\n {} correct: {}'.format(special_index,correct/count))
        results_info[4] = images_targets 
        # 记录结果
        results_infos[i_label] = results_info
    show_result(results_infos)







if __name__ == "__main__":
    logger.logger.info('model epsilon: {}'.format(epsilon_model))
    logger.logger.info('attack epsilon: {}'.format(epsilon_attack))
    logger.logger.info('deepfool: max iterative: {}'.format(max_iter))

    # 一般的对抗模型模型
    model_ADV = LeNet()
    model_ADV.load_state_dict(torch.load('../model/LeNet_CIFAR_adv_e'+str(epsilon_model)+'.pt'))

    model_NORMAL = LeNet()
    model_NORMAL.load_state_dict(torch.load('../model/Lenet_CIFAR.pt'))
    criterion = nn.CrossEntropyLoss()
    # logger.logger.info('==============test on clean=====================')
    # INFO = 'clean'
    # test_on_clean()
    # logger.logger.info('==============test on FGSM=====================')
    # INFO = 'FGSM'
    # test_on_FGSM()
    # logger.logger.info('==============test on DeepFool=====================')
    # INFO = 'DeepFool'
    # test_on_deepfool()
    logger.logger.info('==============test on PGD=====================')
    INFO = 'PGD'
    test_on_PGD()



