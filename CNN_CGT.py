import argparse
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from collections import OrderedDict
import copy
import csv
from random import shuffle, sample
from time import perf_counter
import warnings

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

class CifarCNN(torch.nn.Module):
    def __init__(self, classes=10):
        super().__init__()

        in_channels = 3
        kernel_size = 5
        in1, in2, in3 = 512, 84, 84

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=kernel_size, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels=128, kernel_size=kernel_size, padding=1)

        self.dropout = nn.Dropout2d(0.25)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

        self.dense1 = nn.Linear(in1, in2)
        self.dense2 = nn.Linear(in3, classes)

    def forward(self, x):  # Fixed indentation - this should be at class level, not inside __init__
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 512)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x  # This return statement belongs to forward()

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.dense = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x

# basic optimizer class
class Base(Optimizer):
    def __init__(self, params, idx, w, agents, lr=0.02, num=0.5, kmult=0.007, name=None, device=None, 
                amplifier=.1, theta=np.inf, damping=.4, eps=1e-5, weight_decay=0, kappa=0.9, stratified=True):

        defaults = dict(idx=idx, lr=lr, w=w, num=num, kmult=kmult, agents=agents, name=name, device=device,
                amplifier=amplifier, theta=theta, damping=damping, eps=eps, weight_decay=weight_decay, kappa=kappa, lamb=lr, stratified=stratified)
        
        super(Base, self).__init__(params, defaults)

    # grad and param difference norms
    def compute_dif_norms(self, prev_optimizer):
        for group, prev_group in zip(self.param_groups, prev_optimizer.param_groups):
            grad_dif_norm = 0
            param_dif_norm = 0
            for p, prev_p in zip(group['params'], prev_group['params']):
                if p.grad is None or prev_p.grad is None:
                    continue
                d_p = p.grad.data
                prev_d_p = prev_p.grad.data
                grad_dif_norm += (d_p - prev_d_p).norm().item() ** 2
                param_dif_norm += (p.data - prev_p.data).norm().item() ** 2

            gr, pra = np.sqrt(grad_dif_norm), np.sqrt(param_dif_norm)
            group['grad_dif_norm'] = np.sqrt(grad_dif_norm)
            group['param_dif_norm'] = np.sqrt(param_dif_norm)
        return gr, pra

    # compute gradient differences`
    def compute_grad_dif(self, prev_optimizer):
        grad_diffs = []  # 存储所有参数的梯度差值张量

        # 遍历每个参数组
        for group, prev_group in zip(self.param_groups, prev_optimizer.param_groups):
            # 为当前参数组初始化梯度差值列表
            group_grad_diffs = []

            # 遍历组内每个参数
            for p, prev_p in zip(group['params'], prev_group['params']):
                if p.grad is None or prev_p.grad is None:
                    group_grad_diffs.append(None)  
                    continue

                grad_diff = p.grad.data - prev_p.grad.data
                group_grad_diffs.append(grad_diff.clone())  # 深拷贝避免后续修改

            # 将当前组的梯度差值存入参数组字典
            group['grad_differences'] = group_grad_diffs
            grad_diffs.extend(group_grad_diffs)

        return grad_diffs  # 返回所有参数的梯度差值列表

    #weird one
    def set_norms(self, grad_diff, param_diff):
        for group in self.param_groups:
            group['grad_dif_norm'] = grad_diff
            group['param_dif_norm'] = param_diff
    
    def collect_params(self, lr=False):
        for group in self.param_groups:
            grads = []
            vars = []
            if lr:
                return group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                vars.append(p.data.clone().detach())
                grads.append(p.grad.data.clone().detach())
        return vars, grads
    
    def step(self):
        pass

class DAdSGD(Base):
    def __init__(self, *args, **kwargs):
        super(DAdSGD, self).__init__(*args, **kwargs)
    
    def step(self, k, vars=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            idx = group['idx']
            w = group['w']
            agents = group["agents"]
            device=group["device"]
            
            eps = group['eps']
            lr = group['lr']

            if group["stratified"]:
                alpha = 1
                amplifier = 0.02
            else:
                alpha = 1/.9
                amplifier = 0.05
            
            theta = group['theta']
            grad_dif_norm = group['grad_dif_norm']
            param_dif_norm = group['param_dif_norm']
            lr_new = min(lr * np.sqrt(1 + amplifier * theta), alpha * param_dif_norm / grad_dif_norm) + eps
            theta = lr_new / lr
            group['theta'] = theta
            group['lr'] = lr_new

            lr = lr_new

            sub = 0
            for i, p in enumerate(group['params']):
                summat = torch.zeros(p.data.size()).to(device)

                if p.grad is None:
                    sub -= 1
                    continue

                for j in range(agents):
                    summat += w[idx, j] * (vars[j][i+sub].to(device))

                p.data = summat - lr * p.grad.data

        return loss


# use the optimizer for each agent
class Fedrcu(Base):
    def __init__(self, *args, **kwargs):
        super(DAdSGD, self).__init__(*args, **kwargs)
        # Initialize gradient tracking variables
        for group in self.param_groups:
            # Initialize y with zeros (will be updated in first step)
            group['y'] = [torch.zeros_like(p.data) for p in group['params']]
            # Initialize prev_y for neighbors (needs to be a list of lists)
            group['prev_y'] = {j: [torch.zeros_like(p.data) for p in group['params']] 
                              for j in range(group['agents'])}
            group['prev_y_temp'] = {j: [torch.zeros_like(p.data) for p in group['params']] 
                              for j in range(group['agents'])}
        self.initialized = False  # 新增初始化标志
    
    def step(self, vars=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            idx = group['idx']
            w = group['w']
            agents = group["agents"]
            device=group["device"]
            
            lr = group['lr']

            sub = 0
            for i, p in enumerate(group['params']):
                summat = torch.zeros(p.data.size()).to(device)

                if p.grad is None:
                    sub -= 1
                    continue

                for j in range(agents):
                    summat += w[idx, j] * (vars[j][i+sub].to(device))

                p.data = summat - lr * p.grad.data

        return loss
    def global_step(self, k, vars=None, closure=None):
        pass



class DTrainer:
    def __init__(self, 
                dataset="cifar10", 
                epochs=100, 
                batch_size=32, 
                lr=0.02, 
                workers=4, 
                agents=5,
                num=0.5, 
                kmult=0.0, 
                exp=0.7,
                w=None,
                kappa=0.9,
                fname=None,
                stratified=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_accuracy = []
        self.test_accuracy = []
        self.train_iterations = []
        self.test_iterations = []
        self.lr_logs = {}
        self.lambda_logs = {}
        self.loss_list = []

        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.workers = workers
        self.agents = agents
        self.num = num
        self.kmult = kmult
        self.exp = exp
        self.kappa = kappa
        self.fname = fname
        self.stratified = stratified
        self.load_data()
        self.w = w
        self.criterion = torch.nn.CrossEntropyLoss()
        self.agent_setup()

    def _log(self, accuracy):
        ''' Helper function to log accuracy values'''
        self.train_accuracy.append(accuracy)
        self.train_iterations.append(self.running_iteration)

    def _save(self):
        with open(self.fname, mode='a') as csv_file:
            file = csv.writer(csv_file, lineterminator = '\n')
            file.writerow([f"{self.opt_name}, {self.num}, {self.kmult}, {self.batch_size}, {self.epochs}"])
            file.writerow(self.train_iterations)
            file.writerow(self.train_accuracy)
            file.writerow(self.test_iterations)
            file.writerow(self.test_accuracy)
            file.writerow(self.loss_list)
            file.writerow(["ETA"])
            for i in range(self.agents):
                file.writerow(self.lr_logs[i])
            if self.opt_name == "DLAS":
                file.writerow(["LAMBDA"])
                for i in range(self.agents):
                    file.writerow(self.lambda_logs[i])
            file.writerow([])

    def load_data(self):
        print("==> Loading Data")
        self.train_loader = {}
        self.test_loader = {}

        if self.dataset == 'cifar10':
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            transform_test = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            self.class_num = 10
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            
        elif self.dataset == "mnist":
            transform_train = transforms.Compose([transforms.ToTensor(),])
            transform_test = transforms.Compose([transforms.ToTensor(),])

            self.class_num = 10
            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        else:
            raise ValueError(f'{self.dataset} is not supported')

        if self.stratified:
            train_len, test_len = int(len(trainset)), int(len(testset))

            temp_train = torch.utils.data.random_split(trainset, [int(train_len//self.agents)]*self.agents)
            
            for i in range(self.agents):
                self.train_loader[i] = torch.utils.data.DataLoader(temp_train[i], batch_size=self.batch_size, shuffle=True)

            self.test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
        else:
            train_len, test_len = int(len(trainset)), int(len(testset))
            idxs = {}
            for i in range(0, 10, 2):
                arr = np.array(trainset.targets, dtype=int)
                idxs[int(i/2)] = list(np.where(arr == i)[0]) + list(np.where(arr == i+1)[0])
                shuffle(idxs[int(i/2)])
            
            percent_main = 0.5
            percent_else = (1 - percent_main) / (self.agents-1)
            main_samp_num = int(percent_main * len(idxs[0]))
            sec_samp_num = int(percent_else * len(idxs[0]))

            for i in range(self.agents):
                agent_idxs = []
                for j in range(self.agents):
                    if i == j:
                        agent_idxs.extend(sample(idxs[j], main_samp_num))
                    else:
                        agent_idxs.extend(sample(idxs[j], sec_samp_num))
                    idxs[j] = list(filter(lambda x: x not in agent_idxs, idxs[j]))
                temp_train = copy.deepcopy(trainset)
                temp_train.targets = [temp_train.targets[i] for i in agent_idxs]
                temp_train.data = [temp_train.data[i] for i in agent_idxs]
                self.train_loader[i] = torch.utils.data.DataLoader(temp_train, batch_size=self.batch_size, shuffle=True)               
            self.test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    def agent_setup(self):
        for i in range(self.agents):
            self.lr_logs[i] = []
            self.lambda_logs[i] = []

        self.agent_models = {}
        self.prev_agent_models = {}
        self.agent_optimizers = {}
        self.prev_agent_optimizers = {}

        if self.dataset == 'cifar10':
            model = CifarCNN()
        
        elif self.dataset == "imagenet":
            raise ValueError("ImageNet Not Supported: Low Computing Power")

        elif self.dataset == "mnist":
            model = MnistCNN()

        for i in range(self.agents):
            if i == 0:
                if int(torch.cuda.device_count()) > 1:
                    self.agent_models[i] = torch.nn.DataParallel(model)
                else:
                    self.agent_models[i] = model

            else:
                if int(torch.cuda.device_count()) > 1:
                    self.agent_models[i] = copy.deepcopy(self.agent_models[0])
                else:
                    self.agent_models[i] = copy.deepcopy(model)

            self.agent_models[i].to(self.device)
            self.agent_models[i].train()

            if self.opt_name == "DAdSGD" or self.opt_name == "DLAS":
                self.prev_agent_models[i] = copy.deepcopy(model)
                self.prev_agent_models[i].to(self.device)
                self.prev_agent_models[i].train()
                self.prev_agent_optimizers[i] = self.opt(
                                params=self.prev_agent_models[i].parameters(),
                                idx=i,
                                w=self.w,
                                agents=self.agents,
                                lr=self.lr, 
                                num=self.num, 
                                kmult=self.kmult, 
                                name=self.opt_name,
                                device=self.device,
                                kappa=self.kappa,
                                stratified=self.stratified
                            )

            self.agent_optimizers[i] = self.opt(
                            params=self.agent_models[i].parameters(), 
                            idx=i,
                            w=self.w,
                            agents=self.agents,
                            lr=self.lr, 
                            num=self.num, 
                            kmult=self.kmult, 
                            name=self.opt_name,
                            device=self.device,
                            kappa=self.kappa,
                            stratified=self.stratified
                        )

    def eval(self, dataloader):
        total_acc, total_count = 0, 0

        with torch.no_grad():

            for i in range(self.agents):
                self.agent_models[i].eval()

                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    predicted_label = self.agent_models[i](inputs)

                    total_acc += (predicted_label.argmax(1) == labels).sum().item()
                    total_count += labels.size(0)

        self.test_iterations.append(self.running_iteration)
        self.test_accuracy.append(total_acc/total_count)

        return total_acc/total_count

    def it_logger(self, total_acc, total_count, epoch, log_interval, tot_loss, start_time):
        self._log(total_acc/total_count)
        t_acc = self.eval(self.test_loader)
        for i in range(self.agents):
            self.lr_logs[i].append(self.agent_optimizers[i].collect_params(lr=True))
            if self.opt_name == "DLAS":
                self.lambda_logs[i].append(self.agent_optimizers[i].collect_lambda())

        ss = self.lr_logs[0][-1] if self.opt_name != "DLAS" else self.lambda_logs[0][-1]
        print(
            f"Epoch: {epoch+1}, Iteration: {self.running_iteration}, "+ 
            f"Accuracy: {total_acc/total_count:.4f}, "+ 
            f"Test Accuracy: {t_acc:.4f}, " + 
            f"Loss: {tot_loss/(self.agents * log_interval):.4f}, "+
            f"ss: {ss:.5f}, "+
            f"Time taken: {perf_counter()-start_time:.4f}"
        )
                
        self.loss_list.append(tot_loss/(self.agents * log_interval))

    def trainer(self):
        if self.opt_name == "DAdSGD" or self.opt_name == "DLAS":
            print(f"==> Starting Training for {self.opt_name}, {self.epochs} epochs and {self.agents} agents on the {self.dataset} dataset, via {self.device}")
        else:
            print(f"==> Starting Training for {self.opt_name}, {self.epochs} epochs and {self.agents} agents on the {self.dataset} dataset, via {self.device}" +
                  f" for {self.num}, {self.kmult}")
        for i in range(self.agents):
            self.test_accuracy = []
            self.train_accuracy = []

        for i in range(self.epochs):
            self.epoch_iterations(i, self.train_loader)


class DAdSGDTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = DAdSGD
        self.opt_name="DAdSGD"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch, dataloader):
        start_time = perf_counter()
        if self.dataset == "cifar10":
            log_interval = int(len(dataloader[0]) - 1)
        else:
            log_interval = 25
        
        loss, prev_loss = {}, {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*dataloader.values())):
            self.running_iteration = idx + epoch * len(dataloader[0])
            vars, grads, grad_diff, param_diff = {}, {}, {}, {}
            

            for i in range(self.agents):
                self.agent_optimizers[i].zero_grad()
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()
 
                self.prev_agent_optimizers[i].zero_grad()
                prev_predicted_label = self.prev_agent_models[i](inputs)
                prev_loss[i] = self.criterion(prev_predicted_label, labels)
                prev_loss[i].backward()

                grad_diff[i], param_diff[i] = self.agent_optimizers[i].compute_dif_norms(self.prev_agent_optimizers[i])

                if torch.cuda.device_count() > 1:
                    new_mod_state_dict = OrderedDict()
                    
                    for k, v in self.agent_models[i].state_dict().items():
                        new_mod_state_dict[k[7:]] = v
                    self.prev_agent_models[i].load_state_dict(new_mod_state_dict)
                else:
                    self.prev_agent_models[i].load_state_dict(self.agent_models[i].state_dict())


                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                tot_loss += loss[i].item()
            
            for i in range(self.agents):
                self.agent_optimizers[i].set_norms(grad_diff[i], param_diff[i])
                self.agent_optimizers[i].step(self.running_iteration, vars=vars)
            
            if idx % log_interval == 0 and idx > 0 and epoch % 2 != 0:
                self.it_logger(total_acc, total_count, epoch, log_interval, tot_loss, start_time)
                total_acc, total_count, tot_loss = 0, 0, 0
                self.agent_models[i].train()
                start_time = perf_counter()
        return total_acc


class ClippedGTA(Base):
    def __init__(self, *args, c0=1.0, theta=0.1, K0=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.c0 = c0
        self.theta = theta
        self.K0 = K0
        self.k = 0  
        # Initialize gradient tracking variables
        for group in self.param_groups:
            # Initialize y with zeros (will be updated in first step)
            group['y'] = [torch.zeros_like(p.data) for p in group['params']]
            # Initialize prev_y for neighbors (needs to be a list of lists)
            group['prev_y'] = {j: [torch.zeros_like(p.data) for p in group['params']] 
                              for j in range(group['agents'])}
            group['prev_y_temp'] = {j: [torch.zeros_like(p.data) for p in group['params']] 
                              for j in range(group['agents'])}
        self.initialized = False  # 新增初始化标志

    def step(self, k, vars=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            idx = group['idx']
          
            w = group['w']
            agents = group["agents"]
            device = group["device"]
            lr = group['lr'] 
            y = group['y']
            prev_y = group['prev_y']
            prev_y_temp = group['prev_y_temp']

            prev_grad = [p.grad.data.clone() for p in group['params']]
            
            # 1. 计算带噪声的加权平均
            weighted_avgs = []  # 存储每个参数的加权平均结果
            with torch.no_grad():
                sub = 0
                for i, p in enumerate(group['params']):
                    neighbor_params = [vars[j][i+sub].to(device) for j in range(agents)]
                    weighted_avg = torch.zeros_like(p.data)
                    for j in range(agents):
                        weighted_avg += w[idx, j] * neighbor_params[j]
                    
                    if self.k > 0 and self.k % self.K0 == 0:
                        noise = self.theta * torch.randn_like(weighted_avg)
                        weighted_avg += noise
                    
                    weighted_avgs.append(weighted_avg)
            
            # 2. 更新本地优化变量（带梯度剪切）
            # for i, p in enumerate(group['params']):
            #     if p.grad is None:
            #         continue
                
            #     y_norm = torch.norm(y[i])
            #     print(y_norm)
            # clip_coef = min(1.0, self.c0 / (y_norm + 1e-8))
            # print(clip_coef)
            # p.data = weighted_avg - lr * clip_coef * y[i]
            with torch.no_grad():
                
                valid_y = []
                for i, p in enumerate(group['params']):
                    if p.grad is not None and y[i] is not None:
                        valid_y.append(y[i].view(-1))  # 只收集有效梯度
                
                if len(valid_y) > 0:
                    global_y = torch.cat(valid_y)
                    y_norm = torch.norm(global_y, p=2)
                    clip_coef = min(1.0, self.c0 / (y_norm + 1e-8))
                else:
                    clip_coef = 1.0

                # 应用参数更新
                for i, p in enumerate(group['params']):
                    if p.grad is None or y[i] is None:
                        continue
                    
                    # 使用对应参数的加权平均结果
                    p.data.copy_(weighted_avgs[i] - lr * clip_coef * y[i]) 

            
            # 3. 重新计算梯度
            if closure is not None:
                loss = closure()
            new_grad = [p.grad.data.clone() for p in group['params']]
            # if self.k % 312 == 0 and idx == 1:
                # print('k:',self.k,'grad',new_grad[1])
            # 4. 更新 y 值（混合邻居梯度 + 梯度变化）
            for i in range(len(group['params'])):
                if not self.initialized:  # 首次迭代特殊处理
                    y[i] = new_grad[i].clone()
                    # print('initalization status')
                else:
                    mixed_y = torch.zeros_like(y[i])
                    for j in range(agents):
                        mixed_y += w[idx, j] * group['prev_y'][j][i].to(device)
                    y[i] = mixed_y + (new_grad[i] - prev_grad[i])

            self.initialized = True  # 标记已初始化



           
            # 每个 agent 保存自己的当前 y 到自己的 prev_y
            for j in range(agents):
                if j == idx:  # 只更新当前agent的梯度
                    group['prev_y_temp'][j] = [yi.detach().clone() for yi in y]
            if self.k % 312 == 0:
                print('prev_y[1][1]:', group['prev_y'][1][1])

        group['prev_y'] = group['prev_y_temp']        
        self.k += 1
        return loss



class CGTATrainer(DTrainer):
    def __init__(self, *args, c0=1.0, theta=0.1, K0=10, **kwargs):
        self.opt = ClippedGTA
        self.opt_name = "CGTA"
        self.c0 = c0
        self.theta = theta
        self.K0 = K0
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def agent_setup(self):
        super().agent_setup()
        # 为每个优化器添加额外参数
        for i in range(self.agents):
            self.agent_optimizers[i].c0 = self.c0
            self.agent_optimizers[i].theta = self.theta
            self.agent_optimizers[i].K0 = self.K0

    def epoch_iterations(self, epoch, dataloader=None):
        dataloader = self.train_loader 
        start_time = perf_counter()
        if self.dataset == "cifar10":
            log_interval = int(len(dataloader[0]) - 1)
        else:
            log_interval = 25
        
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*self.train_loader.values())):
            self.running_iteration = idx + epoch * len(self.train_loader[0])
            vars, grads = {}, {}

            # 前向传播和梯度计算
            for i in range(self.agents):
                self.agent_optimizers[i].zero_grad()
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.agent_models[i](inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()
                tot_loss += loss.item()
                total_acc += (outputs.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

            # 参数更新
            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars)

            # 记录日志
            if idx % log_interval == 0 and idx > 0:
                self.it_logger(total_acc, total_count, epoch, log_interval, tot_loss, start_time)
                total_acc, total_count, tot_loss = 0, 0, 0
                start_time = perf_counter()

        return total_acc



agents = 5
w = np.array([[0.6, 0, 0, 0.4, 0],[0.2, 0.8, 0, 0, 0], [0.2, 0.1, 0.4, 0, 0.3], [0, 0, 0, 0.6, 0.4],[0, 0.1, 0.6, 0, 0.3]])

dataset = "cifar10"
epochs = 800
bs = 32
stratified = True  # Add this line to define the variable

cwd = os.getcwd()
results_path = os.path.join(cwd, "results")
if not os.path.isdir(results_path):
    os.mkdir(results_path)

fname = os.path.join(results_path,f"{dataset}_e{epochs}_hom{stratified}_0.csv")  # Changed args.test_num to 0

# DAdSGDTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified)
CGTATrainer(
    dataset=dataset,
    batch_size=bs,
    epochs=epochs,
    w=w,
    fname=fname,
    stratified=stratified,
    c0=500,
    theta=0,  
    K0=500,       
    lr=2000 
)