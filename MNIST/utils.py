import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as f
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import logging
from logging import handlers
from scipy import stats


class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

class Loss_cost_sensitive(nn.Module):
    def __init__(self):
        super(Loss_cost_sensitive, self).__init__()
    def forward(self, data, target, c):
        # data: 模型输出
        # target: 标签
        # c: 代价矩阵
        l1 = F.cross_entropy(data, target, reduction='mean')
        p = F.softmax(data, 1)
        
        cost_sentive = c[:,target]
        cost_sentive = cost_sentive.T
        l2 = p.mul(cost_sentive)
        l2 = l2.sum(1).mean()
        return l1+l2
    
def my_fgsm(input, labels, model, criterion, epsilon, device, c=None):
    assert isinstance(model, torch.nn.Module), "Input parameter model is not nn.Module. Check the model"
    assert isinstance(criterion, torch.nn.Module), "Input parameter criterion is no Loss. Check the criterion"
    assert (0 <= epsilon <= 1), "episilon must be 0 <= epsilon <= 1"

    # For calculating gradient
    input_for_gradient = Variable(input, requires_grad=True).to(device)
    out = model(input_for_gradient)
    if c==None:
        loss = criterion(out, Variable(labels))
    else:
        loss = criterion(out, Variable(labels), c)

    # Calculate gradient
    loss.backward()

    # Calculate sign of gradient
    signs = torch.sign(input_for_gradient.grad.data)

    # Add
    input_for_gradient.data = input_for_gradient.data + (epsilon * signs)

    return input_for_gradient, signs

def get_cost_matric(i_label, c=10):
    # 生成保护i_label类的代价矩阵
    # i_label：受保护类
    C = torch.ones(10,10)
    C[i_label,:] = c
    C[:,i_label] = c
    C = C - torch.diag(C.diag())
    return C


class Loss_CSE(nn.Module):
    def __init__(self,model):
        super(Loss_CSE, self).__init__()
        self.model = model
        
    def forward(self, data, target, c):
        
        l1 = F.cross_entropy(data, target, reduction='mean')
        p = F.softmax(data, 1)
        
        cost_sentive = c[:,target]
        cost_sentive = cost_sentive.T
        l2 = p.mul(cost_sentive)
        l2 = l2.sum(1).mean()
        
        conv_weight = self.model.conv1.weight
        loss_x = torch.norm(conv_weight, p=1) - torch.norm(conv_weight, p=2)
        
        return l1+l2 + loss_x


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


class Loss_CSE2(nn.Module):
    def __init__(self,model, a=1.0, b=1.0):
        super(Loss_CSE2, self).__init__()
        self.a = a
        self.b = b
        self.model = model
        
    def forward(self, data, target, c):
        
        l1 = F.cross_entropy(data, target, reduction='mean')
        p = F.softmax(data, 1)
        
        cost_sentive = c[:,target]
        cost_sentive = cost_sentive.T
        l2 = p.mul(cost_sentive)
        l2 = l2.sum(1).mean()
        
        conv_weight = self.model.conv1.weight.detach().cpu().numpy()
        conv_weight = conv_weight.reshape(-1,)
        loss_x = stats.kurtosis(conv_weight)

        return l1 + self.a*l2 - self.b*loss_x



class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2):
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        self.weight_info(self.weight_list)
 
    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
 
    def forward(self, model):
        self.weight_list=self.get_weight(model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
 
    def get_weight(self,model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
 
    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")


class Loss_cost_sensitive_L2(nn.Module):
    def __init__(self, model, weight_decay=1e-2):
        super(Loss_cost_sensitive_L2, self).__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.reg_loss = Regularization(self.model, self.weight_decay)
        
    def forward(self, data, target, c):
        
        l1 = F.cross_entropy(data, target, reduction='mean')
        p = F.softmax(data, 1)
        
        cost_sentive = c[:,target]
        cost_sentive = cost_sentive.T
        l2 = p.mul(cost_sentive)
        l2 = l2.sum(1).mean()
        
        return l1+l2+self.reg_loss(self.model)
    
def pgd_attack(model, images, labels, eps=0.3, alpha=2/255, iters=40, loss=None, device=None, C=None):
    if loss==None:
        loss = nn.CrossEntropyLoss()
        
    ori_images = images.data
    
    for i in range(iters) :  
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        if C==None:
            cost = loss(outputs, labels).to(device)
        else:
            cost = loss(outputs, labels, C)
        cost.backward()
        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
#         images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        images = (ori_images + eta).detach_()
        images = torch.clamp(ori_images + eta, min=-0.4242, max=2.8214).detach_()
    
    return images