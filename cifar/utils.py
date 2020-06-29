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
import copy
from logging import handlers
from torch.autograd.gradcheck import zero_gradients


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
    def __init__(self,model, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0):
        super(Loss_CSE, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.model = model
        
    def forward(self, data, target, c):
        
        l1 = F.cross_entropy(data, target, reduction='mean')
        p = F.softmax(data, 1)
        
        cost_sentive = c[:,target]
        cost_sentive = cost_sentive.T
        l2 = p.mul(cost_sentive)
        l2 = l2.sum(1).mean()
        
        conv_weight = self.model.conv1.weight
        conv_weight2 = self.model.conv2.weight
        # loss_x = torch.norm(conv_weight, p=1)
        # loss_x = torch.norm(conv_weight, p=1) - torch.std(conv_weight)
        loss_x = torch.norm(conv_weight, p=1) - torch.norm(conv_weight, p=2)
        loss_x_2 = torch.norm(conv_weight2, p=1) - torch.norm(conv_weight2, p=2)


        # print('l1:{}, l2:{}, loss_x:{}, loss_2:{}'.format(l1, l2, loss_x, loss_x_2))
        # print('l1:{}, l2:{}, loss_x:{}, loss_2:{}'.format(l1.shape, l2.shape, loss_x.shape, loss_x_2.shape))
        # print('a:{}, b:{}, c:{}'.format(self.a, self.b, self.c))
        loss_final = l1 + self.lambda_1*l2 + self.lambda_2*loss_x + self.lambda_3*loss_x_2
        # print('final: {}'.format(loss_final.shape))
        return loss_final


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=3):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
#         print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        pass
#         print("Using CPU")


    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_image


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