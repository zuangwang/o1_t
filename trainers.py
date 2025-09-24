import argparse
import os

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from collections import OrderedDict
import copy
import csv
from random import shuffle, sample
from time import perf_counter

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Subset
from datetime import datetime
from models import *
from optimizers import *

class DTrainer:
    def __init__(self, 
                dataset="cifar10", 
                epochs=100,
                communication_round = 100,
                communication_prob = 0.1,
                local_steps=None, 
                batch_size=32, 
                lr=0.02,
                dir_alpha=0.5, 
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
        self.global_rounds = []
        self.test_accuracy = []
        self.train_iterations = []
        self.test_iterations = []
        self.lr_logs = {}
        self.lambda_logs = {}
        self.loss_list = []

        self.dataset = dataset
        self.epochs = epochs
        self.communication_round = communication_round
        self.local_steps = local_steps
        self.communication_prob = 1/(1+local_steps) if communication_prob is None else communication_prob
        self.batch_size = batch_size
        self.lr = lr
        self.dir_alpha = dir_alpha
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
        self.commu_count = 0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.agent_setup()

    def _log(self, accuracy, global_round):
        ''' Helper function to log accuracy values'''
        self.train_accuracy.append(accuracy)
        self.train_iterations.append(self.running_iteration)
        self.global_rounds.append(global_round)

    def _save(self):
        # Create a directory with the same name as the csv file
        csv_dir = os.path.splitext(self.fname)[0]
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20250910_163045
        os.makedirs(csv_dir, exist_ok=True)
        
        # New path for the csv file
        new_fname = os.path.join(csv_dir, os.path.basename(self.fname))

        with open(new_fname, mode='a') as csv_file:
            file = csv.writer(csv_file, lineterminator = '\n')
            file.writerow([f"{self.opt_name}, {self.lr}, {self.batch_size}, {self.communication_round}"])
            file.writerow(self.global_rounds)
            file.writerow(self.train_accuracy)
            # file.writerow(self.test_iterations)
            file.writerow(self.test_accuracy)
            file.writerow(self.loss_list)
            # file.writerow(["ETA"])
            # for i in range(self.agents):
            #     file.writerow(self.lr_logs[i])
            # if self.opt_name == "DLAS":
            #     file.writerow(["LAMBDA"])
            #     for i in range(self.agents):
            #         file.writerow(self.lambda_logs[i])
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

        elif self.dataset == 'cifar100':
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            transform_test = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            self.class_num = 100
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

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
            # train_len, test_len = int(len(trainset)), int(len(testset))
            # idxs = {}
            # for i in range(0, 10, 2):
            #     arr = np.array(trainset.targets, dtype=int)
            #     idxs[int(i/2)] = list(np.where(arr == i)[0]) + list(np.where(arr == i+1)[0])
            #     shuffle(idxs[int(i/2)])
            
            # percent_main = 0.5
            # percent_else = (1 - percent_main) / (self.agents-1)
            # main_samp_num = int(percent_main * len(idxs[0]))
            # sec_samp_num = int(percent_else * len(idxs[0]))

            # for i in range(self.agents):
            #     agent_idxs = []
            #     for j in range(self.agents):
            #         if i == j:
            #             agent_idxs.extend(sample(idxs[j], main_samp_num))
            #         else:
            #             agent_idxs.extend(sample(idxs[j], sec_samp_num))
            #         idxs[j] = list(filter(lambda x: x not in agent_idxs, idxs[j]))
            #     temp_train = copy.deepcopy(trainset)
            #     temp_train.targets = [temp_train.targets[i] for i in agent_idxs]
            #     temp_train.data = [temp_train.data[i] for i in agent_idxs]
            #     self.train_loader[i] = torch.utils.data.DataLoader(temp_train, batch_size=self.batch_size, shuffle=True)               
            # self.test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

            # Custom non-IID splitting into pairs of classes
            # Step 1: Group classes in pairs (0-1, 2-3, ...)
            # num_classes = len(set(trainset.targets))
            # class_pairs = [(i, i+1) for i in range(0, num_classes, 2)]
            
            # # Step 2: Collect indices for each class pair
            # idxs = {}
            # arr = np.array(trainset.targets)
            # for j, (c1, c2) in enumerate(class_pairs):
            #     idxs[j] = list(np.where(arr == c1)[0]) + list(np.where(arr == c2)[0])
            #     shuffle(idxs[j])
            
            # main_fraction = 0.5
            # # Step 3: Determine sample sizes
            # percent_else = (1 - main_fraction) / (self.agents - 1)
            # main_samp_num = int(main_fraction * len(idxs[0]))
            # sec_samp_num = int(percent_else * len(idxs[0]))
            
            # # Step 4: Create agent-specific datasets
            # self.train_loader = []
            # for i in range(self.agents):
            #     agent_idxs = []
            #     for j in range(self.agents):
            #         if i == j:
            #             agent_idxs.extend(sample(idxs[j], main_samp_num))
            #         else:
            #             agent_idxs.extend(sample(idxs[j], sec_samp_num))
            #         # Remove chosen indices from the pool
            #         idxs[j] = list(filter(lambda x: x not in agent_idxs, idxs[j]))
                
            #     # Create a subset dataset for this agent
            #     agent_trainset = copy.deepcopy(trainset)
            #     agent_trainset.targets = [agent_trainset.targets[k] for k in agent_idxs]
            #     agent_trainset.data = [agent_trainset.data[k] for k in agent_idxs]
                
            #     loader = torch.utils.data.DataLoader(agent_trainset, batch_size=self.batch_size, shuffle=True)
            #     self.train_loader.append(loader)
            
            num_classes = len(trainset.classes)
            labels = np.array(trainset.targets)
            idxs = np.arange(len(trainset))

            # Initialize per-user indices
            user_idxs = {i: [] for i in range(self.agents)}

            # For each class, split its indices across users
            for c in range(num_classes):
                class_idx = idxs[labels == c]
                np.random.shuffle(class_idx)

                # Dirichlet proportions for this class
                proportions = np.random.dirichlet(np.repeat(self.dir_alpha, self.agents))
                proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]

                # Split indices according to proportions
                split_idx = np.split(class_idx, proportions)
                for u in range(self.agents):
                    user_idxs[u].extend(split_idx[u])

            # Convert to Subsets
            self.train_loader = [torch.utils.data.DataLoader(Subset(trainset, user_idxs[u]), batch_size=self.batch_size, shuffle=True) for u in range(self.agents)]
            # Step 5: Test loader (shared for all agents)
            self.test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    # assign models and optimizers to each local agents
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

        elif self.dataset == 'cifar100':
            model = CifarCNN(100)
        
        elif self.dataset == "imagenet":
            model = models.resnet152(pretrained=True)

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

            # if self.opt_name == "DAdSGD" or self.opt_name == "DLAS":
            #     self.prev_agent_models[i] = copy.deepcopy(model)
            #     self.prev_agent_models[i].to(self.device)
            #     self.prev_agent_models[i].train()
            #     self.prev_agent_optimizers[i] = self.opt(
            #                     params=self.prev_agent_models[i].parameters(),
            #                     idx=i,
            #                     w=self.w,
            #                     agents=self.agents,
            #                     lr=self.lr, 
            #                     num=self.num, 
            #                     kmult=self.kmult, 
            #                     name=self.opt_name,
            #                     device=self.device,
            #                     kappa=self.kappa,
            #                     stratified=self.stratified
            #                 )

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

                    if self.dataset == 'cifar100':
                        total_acc += (predicted_label.topk(5, 1, True, True)[1] == labels.view(-1, 1)).sum().item()
                    else:
                        total_acc += (predicted_label.argmax(1) == labels).sum().item()
                    total_count += labels.size(0)

        self.test_iterations.append(self.running_iteration)
        self.test_accuracy.append(total_acc/total_count)

        return total_acc/total_count

    def it_logger(self, total_acc, total_count, global_round, tot_loss, start_time):
        self._log(total_acc/total_count, global_round)
        t_acc = self.eval(self.test_loader)
        for i in range(self.agents):
            self.lr_logs[i].append(self.agent_optimizers[i].collect_params(lr=True))
            if self.opt_name == "DLAS":
                self.lambda_logs[i].append(self.agent_optimizers[i].collect_lambda())

        ss = self.lr_logs[0][-1] if self.opt_name != "DLAS" else self.lambda_logs[0][-1]
        print(
            f"Global Round: {global_round}, "+ 
            f"Accuracy: {total_acc/total_count:.4f}, "+ 
            f"Test Accuracy: {t_acc:.4f}, " + 
            f"Loss: {tot_loss/(self.agents):.4f}, "+
            # f"ss: {ss:.5f}, "+
            f"Time taken: {perf_counter()-start_time:.4f}"
        )
                
        self.loss_list.append(tot_loss/(self.agents))

    def trainer(self):
        if self.opt_name == "DAdSGD" or self.opt_name == "DLAS":
            print(f"==> Starting Training for {self.opt_name}, {self.communication_round} communication rounds and {self.agents} agents on the {self.dataset} dataset, via {self.device}")
        else:
            print(f"==> Starting Training for {self.opt_name}, {self.communication_round} communication rounds and {self.agents} agents on the {self.dataset} dataset, via {self.device}" +
                  f" for {self.num}, {self.kmult}")
        for i in range(self.agents):
            self.test_accuracy = []
            self.train_accuracy = []

        for i in range(self.communication_round+1):
            self.local_iterations(i, self.train_loader)

class FedrcuTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt_name = "FedRecu"
        self.opt = Fedrcu
        super(FedrcuTrainer, self).__init__(*args, **kwargs)
        self.trainer()
        self._save()
        
    def _sync_batchnorm_buffers(self):
    # gather BN layers from each model (unwrap DataParallel if present)
        bn_lists = []
        for i in range(self.agents):
            model = self.agent_models[i].module if hasattr(self.agent_models[i], "module") else self.agent_models[i]
            bn_lists.append([m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)])

        # assume same architecture across agents
        for layers in zip(*bn_lists):
            with torch.no_grad():
                mean = sum(l.running_mean for l in layers) / len(layers)
                var  = sum(l.running_var  for l in layers) / len(layers)
                nbt  = max(int(l.num_batches_tracked) for l in layers)
                for l in layers:
                    l.running_mean.copy_(mean)
                    l.running_var.copy_(var)
                    l.num_batches_tracked.fill_(nbt)

    def local_iterations(self, global_round, dataloader):
        """
        Implements Algorithm 1 (FedRecu) over one epoch:
          - Each step: all clients compute grads on their own minibatch
          - Depending on t and tau (local_epochs):
              * v-round: x <- avg_v
              * w-round: x <- 2*x - avg_w
              * local-only: x <- 2*x - x_prev - lr*(g_t - g_{t-1})
        """
        # log_interval = 100
        tau = self.local_steps if (self.local_steps is not None and self.local_steps >= 1) else 1
        
        # step count = smallest dataloader length (lockstep across clients)
        # steps_per_epoch = min(len(dataloader[i]) for i in range(self.agents))

        self.running_iteration =  global_round * tau
        tot_loss = 0.0
        total_acc, total_count = 0, 0
        start_time = perf_counter()

        for step in range(tau):
            # -------- 1) Each client computes current gradients on its minibatch --------
            iters = {i: iter(dataloader[i]) for i in range(self.agents)}
            for i in range(self.agents):
                self.agent_models[i].train()
                self.agent_optimizers[i].zero_grad()

                inputs, labels = next(iters[i])
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                logits = self.agent_models[i](inputs)
                loss_i = self.criterion(logits, labels)
                loss_i.backward()  # fills ∇f_i(x_i(t))
                clip_grad_norm_(self.agent_models[i].parameters(), max_norm=5.0)
                # tot_loss += loss_i.item()

                # with torch.no_grad():
                #     total_acc += (logits.argmax(1) == labels).sum().item()
                #     total_count += labels.size(0)

            # -------- 2) FedRecu branching by tau (Algorithm 1) --------
            t = self.running_iteration
            if ((t + 1) % tau) == 0:
                # v-round: v_i(t) -> avg -> x_i(t+1) = avg_v
                all_v = [self.agent_optimizers[i].transmit_v() for i in range(self.agents)]
                avg_v = _average_lists(all_v)
                for i in range(self.agents):
                    self.agent_optimizers[i].apply_avg_v(avg_v)
                self._sync_batchnorm_buffers()
            elif (t % tau) == 0:
                # w-round: w_i(t) -> avg -> x_i(t+1) = 2 x_i(t) - avg_w
                all_w = [self.agent_optimizers[i].transmit_w() for i in range(self.agents)]
                avg_w = _average_lists(all_w)
                for i in range(self.agents):
                    self.agent_optimizers[i].apply_avg_w(avg_w)
                self._sync_batchnorm_buffers()
            else:
                # local-only step
                for i in range(self.agents):
                    self.agent_optimizers[i].local_step()

            # -------- 3) Logging / bookkeeping --------
            self.running_iteration += 1
            if (step + 1) == tau and global_round % 10 == 0:
            # if (t % tau) == 0:
                for i in range(self.agents):
                    self.agent_models[i].train()
                    self.agent_optimizers[i].zero_grad()

                    inputs, labels = next(iters[i])
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    logits = self.agent_models[i](inputs)
                    loss_i = self.criterion(logits, labels)
                    tot_loss += loss_i.item()

                    with torch.no_grad():
                        total_acc += (logits.argmax(1) == labels).sum().item()
                        total_count += labels.size(0)
                self.it_logger(total_acc, total_count, global_round, tot_loss, start_time)
                tot_loss = 0.0
                total_acc, total_count = 0, 0
                start_time = perf_counter()


class ScaffoldTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt_name = "Scaffold"
        self.opt = Scaffold
        super(ScaffoldTrainer, self).__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def _sync_batchnorm_buffers(self):
        # gather BN layers from each model (unwrap DataParallel if present)
        bn_lists = []
        for i in range(self.agents):
            model = self.agent_models[i].module if hasattr(self.agent_models[i], "module") else self.agent_models[i]
            bn_lists.append([m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)])

        # assume same architecture across agents
        for layers in zip(*bn_lists):
            with torch.no_grad():
                mean = sum(l.running_mean for l in layers) / len(layers)
                var  = sum(l.running_var  for l in layers) / len(layers)
                nbt  = max(int(l.num_batches_tracked) for l in layers)
                for l in layers:
                    l.running_mean.copy_(mean)
                    l.running_var.copy_(var)
                    l.num_batches_tracked.fill_(nbt)

    def local_iterations(self, global_round, dataloader):
        """
        SCAFFOLD local round of length `tau = self.local_steps`:
          - per-step: compute grads on each client, clip, then optimizer.local_step()
          - at end of round (t+1 % tau == 0): FedAvg params + control variate update
        """
        tau = self.local_steps if (self.local_steps is not None and self.local_steps >= 1) else 1

        # mark round start for control-variates
        for i in range(self.agents):
            self.agent_optimizers[i].begin_round()

        self.running_iteration = global_round * tau
        tot_loss = 0.0
        total_acc, total_count = 0, 0
        start_time = perf_counter()

        for step in range(tau):
            # ---- 1) grads on current minibatch (all clients) ----
            # lockstep iterators across clients
            iters = {i: iter(dataloader[i]) for i in range(self.agents)}
            for i in range(self.agents):
                self.agent_models[i].train()
                self.agent_optimizers[i].zero_grad()

                inputs, labels = next(iters[i])
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                logits = self.agent_models[i](inputs)
                loss_i = self.criterion(logits, labels)
                loss_i.backward()
                clip_grad_norm_(self.agent_models[i].parameters(), max_norm=5.0)

                with torch.no_grad():
                    total_acc += (logits.argmax(1) == labels).sum().item()
                    total_count += labels.size(0)

            # ---- 2) one SCAFFOLD local step per client ----
            for i in range(self.agents):
                self.agent_optimizers[i].local_step()

            # ---- 3) communication at the end of the local round ----
            t = self.running_iteration
            if ((t + 1) % tau) == 0:
                # gather payloads
                all_params = [self.agent_optimizers[i].transmit_params()   for i in range(self.agents)]
                all_deltas = [self.agent_optimizers[i].transmit_c_delta()  for i in range(self.agents)]

                avg_params  = _average_lists(all_params)   # FedAvg model
                avg_delta_c = _average_lists(all_deltas)   # mean control variate delta

                # apply averaged model to all clients
                for i in range(self.agents):
                    self.agent_optimizers[i].apply_global_params(avg_params)
                self._sync_batchnorm_buffers()

                # update global and client control variates; start next round
                for i in range(self.agents):
                    self.agent_optimizers[i].apply_c_updates(avg_delta_c, all_deltas[i])
                    self.agent_optimizers[i].begin_round()

            # ---- 4) bookkeeping / optional logging ----
            self.running_iteration += 1
            # if (step + 1) == tau and global_round % 10 == 0:
            #     self.it_logger(total_acc, total_count, global_round, tot_loss, start_time)
            #     tot_loss = 0.0
            #     total_acc, total_count = 0, 0
            #     start_time = perf_counter()

        for i in range(self.agents):
            self.agent_models[i].train()
            self.agent_optimizers[i].zero_grad()

            x, y = next(iters[i])
            x, y = x.to(self.device), y.to(self.device)

            logits = self.agent_models[i](x)
            loss_i = self.criterion(logits, y)
            tot_loss += loss_i.item()

            with torch.no_grad():
                total_acc += (logits.argmax(1) == y).sum().item()
                total_count += y.size(0)

        if (step + 1) == tau and global_round % 10 == 0:
            self.it_logger(total_acc, total_count, global_round, tot_loss, start_time)
            tot_loss = 0.0
            total_acc, total_count = 0, 0
            start_time = perf_counter()


# ---- tiny helpers ----
@torch.no_grad()
def _average_lists(lists):
    K = len(lists[0])
    out = []
    for k in range(K):
        s = None
        for L in lists:
            s = L[k] if s is None else (s + L[k])
        out.append(s / len(lists))
    return out

def _unwrap(model):
    return model.module if hasattr(model, "module") else model

@torch.no_grad()
def _snapshot_params(model):
    return [p.data.clone() for p in _unwrap(model).parameters()]

@torch.no_grad()
def _load_params(model, flat):
    i = 0
    for p in _unwrap(model).parameters():
        p.data.copy_(flat[i]); i += 1

def _snapshot_grads(model):
    return [ (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
             for p in _unwrap(model).parameters() ]


class FedLinTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt_name = "FedLin"
        self.opt = FedLin
        super(FedLinTrainer, self).__init__(*args, **kwargs)
        # server vector g_t (list[tensor]); set on first call
        self.g_global = None
        self.trainer()
        self._save()

    def _sync_batchnorm_buffers(self):
        # gather BN layers from each model (unwrap DataParallel if present)
        bn_lists = []
        for i in range(self.agents):
            model = _unwrap(self.agent_models[i])
            bn_lists.append([m for m in model.modules()
                             if isinstance(m, nn.modules.batchnorm._BatchNorm)])

        # assume same architecture across agents
        for layers in zip(*bn_lists):
            with torch.no_grad():
                mean = sum(l.running_mean for l in layers) / len(layers)
                var  = sum(l.running_var  for l in layers) / len(layers)
                nbt  = max(int(l.num_batches_tracked) for l in layers)
                for l in layers:
                    l.running_mean.copy_(mean)
                    l.running_var.copy_(var)
                    l.num_batches_tracked.fill_(nbt)

    def _ensure_g_initialized(self, dataloader):
        """Compute g_1 = mean_i ∇f_i( x̄_1 ) on one minibatch per client."""
        if self.g_global is not None:
            return
        iters = {i: iter(dataloader[i]) for i in range(self.agents)}
        grads = []
        for i in range(self.agents):
            self.agent_optimizers[i].zero_grad()
            x, y = next(iters[i])
            x, y = x.to(self.device), y.to(self.device)
            logits = self.agent_models[i](x)
            loss_i = self.criterion(logits, y)
            loss_i.backward()
            grads.append(_snapshot_grads(self.agent_models[i]))
        self.g_global = _average_lists(grads)
        for i in range(self.agents):
            self.agent_optimizers[i].set_g(self.g_global)

    def _grad_at_global_point(self, client_idx, bar_params, inputs, labels):
        """Compute ∇f_i( x̄_t ) using the given (inputs, labels)."""
        model = self.agent_models[client_idx]
        cur = _snapshot_params(model)
        _load_params(model, bar_params)
        self.agent_optimizers[client_idx].zero_grad()
        logits = model(inputs)
        loss = self.criterion(logits, labels)
        loss.backward()
        gbar = _snapshot_grads(model)
        _load_params(model, cur)
        self.agent_optimizers[client_idx].zero_grad()
        return gbar

    def local_iterations(self, global_round, dataloader):
        """
        FedLin round (no compression, no EF):
          - per step: each client computes grad_local and grad at x̄_t on the same batch,
                      then does x <- x - η(grad_local - grad_bar + g_t)
          - end of round: server averages params to x̄_{t+1} and sets
                          g_{t+1} = mean_i ∇f_i( x̄_{t+1} ), then broadcasts g_{t+1}
        """
        tau = self.local_steps if (self.local_steps is not None and self.local_steps >= 1) else 1

        # init g_1 if needed
        self._ensure_g_initialized(dataloader)

        # snapshot global params at start of round (all clients assumed synced)
        bar_params = _snapshot_params(self.agent_models[0])

        # broadcast (redundant after init, cheap)
        for i in range(self.agents):
            self.agent_optimizers[i].set_g(self.g_global)

        self.running_iteration = global_round * tau
        tot_loss = 0.0
        total_acc, total_count = 0, 0
        start_time = perf_counter()

        for step in range(tau):
            # ---- 1) per-client: grads + FedLin local step ----
            # lockstep iterators across clients
            iters = {i: iter(dataloader[i]) for i in range(self.agents)}
            for i in range(self.agents):
                self.agent_models[i].train()
                self.agent_optimizers[i].zero_grad()

                inputs, labels = next(iters[i])
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # grad at current local x
                logits = self.agent_models[i](inputs)
                loss_i = self.criterion(logits, labels)
                loss_i.backward()
                clip_grad_norm_(_unwrap(self.agent_models[i]).parameters(), max_norm=5.0)
                # tot_loss += loss_i.item()

                # local FedLin step
                self.agent_optimizers[i].local_step()

            # ---- 2) bookkeeping / optional logging ----
            self.running_iteration += 1
            # if (step + 1) == tau and global_round % 10 == 0:
            #     self.it_logger(total_acc, total_count, global_round, tot_loss, start_time)
            #     tot_loss = 0.0
            #     total_acc, total_count = 0, 0
            #     start_time = perf_counter()

        # ===== end of round: average params -> x̄_{t+1} =====
        all_params = [self.agent_optimizers[i].transmit_params() for i in range(self.agents)]
        avg_params = _average_lists(all_params)
        for i in range(self.agents):
            _load_params(self.agent_models[i], avg_params)
        self._sync_batchnorm_buffers()

        # ===== compute g_{t+1} = mean_i ∇f_i( x̄_{t+1} ) and broadcast =====
        grads_next = []
        for i in range(self.agents):
            self.agent_optimizers[i].zero_grad()
            x, y = next(iters[i])
            x, y = x.to(self.device), y.to(self.device)
            logits = self.agent_models[i](x)
            loss_i = self.criterion(logits, y)
            loss_i.backward()
            clip_grad_norm_(_unwrap(self.agent_models[i]).parameters(), max_norm=5.0)
            grads_next.append(_snapshot_grads(self.agent_models[i]))
            tot_loss += loss_i.item()

            # logging
            with torch.no_grad():
                total_acc += (logits.argmax(1) == y).sum().item()
                total_count += y.size(0)
            # grad at global point x̄_t+1
            self.agent_optimizers[i].set_grad_bar(_snapshot_grads(self.agent_models[i]))
        if global_round % 10 == 0:
            self.it_logger(total_acc, total_count, global_round, tot_loss, start_time)
            tot_loss = 0.0
            total_acc, total_count = 0, 0
            start_time = perf_counter()

        self.g_global = _average_lists(grads_next)
        for i in range(self.agents):
            self.agent_optimizers[i].set_g(self.g_global)


class ScaffnewTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt_name = "Scaffnew"
        self.opt = Scaffnew
        super(ScaffnewTrainer, self).__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def _sync_batchnorm_buffers(self):
        # keep BN buffers consistent when we DO average (optional but helpful)
        bn_lists = []
        for i in range(self.agents):
            model = _unwrap(self.agent_models[i])
            bn_lists.append([m for m in model.modules()
                             if isinstance(m, nn.modules.batchnorm._BatchNorm)])
        for layers in zip(*bn_lists):
            with torch.no_grad():
                mean = sum(l.running_mean for l in layers) / len(layers)
                var  = sum(l.running_var  for l in layers) / len(layers)
                nbt  = max(int(l.num_batches_tracked) for l in layers)
                for l in layers:
                    l.running_mean.copy_(mean)
                    l.running_var.copy_(var)
                    l.num_batches_tracked.fill_(nbt)

    def local_iterations(self, global_round, dataloader):
        """
        Round with length tau=self.local_steps.
        Each inner step:
          1) grads on each client
          2) build xhat_i
          3) flip shared coin with prob p (defined HERE in trainer)
          4) if communicate: average xhat and apply; else: apply own xhat
          5) h updated inside optimizer using (p/γ)(x - xhat)
        """
        tau = self.local_steps if (self.local_steps and self.local_steps >= 1) else 1

        # define probability p HERE (single source of truth)
        p_comm = self.communication_prob
        if p_comm is None:
            p_comm = float(self.num)  # default: reuse your existing `num` arg
        p_comm = float(p_comm)

        self.running_iteration = global_round * tau

        for step in range(tau):
            tot_loss = 0.0
            total_acc, total_count = 0, 0
            start_time = perf_counter()
            # 1) grads

            iters = {i: iter(dataloader[i]) for i in range(self.agents)}
            for i in range(self.agents):
                self.agent_models[i].train()
                self.agent_optimizers[i].zero_grad()

                x, y = next(iters[i])
                x, y = x.to(self.device), y.to(self.device)

                logits = self.agent_models[i](x)
                loss_i = self.criterion(logits, y)
                loss_i.backward()
                clip_grad_norm_(self.agent_models[i].parameters(), max_norm=5.0)

                with torch.no_grad():
                    total_acc += (logits.argmax(1) == y).sum().item()
                    total_count += y.size(0)

            # 2) build xhat for all clients
            all_xhat = [self.agent_optimizers[i].transmit_xhat() for i in range(self.agents)]

            # 3) shared coin
            coin = (np.random.rand() < p_comm)

            # 4) apply & h update (optimizer needs p_comm)
            if coin:

                avg_xhat = _average_lists(all_xhat)
                for i in range(self.agents):
                    self.agent_optimizers[i].apply_from_xhat(avg_xhat, p_comm)
                self._sync_batchnorm_buffers()
                if self.commu_count == 10:
                    self.commu_count = 0

                    for i in range(self.agents):
                        self.agent_models[i].train()
                        self.agent_optimizers[i].zero_grad()

                        x, y = next(iters[i])
                        x, y = x.to(self.device), y.to(self.device)

                        logits = self.agent_models[i](x)
                        loss_i = self.criterion(logits, y)
                        tot_loss += loss_i.item()

                    with torch.no_grad():
                        total_acc += (logits.argmax(1) == y).sum().item()
                        total_count += y.size(0)

                    self.it_logger(total_acc, total_count, global_round, tot_loss, start_time)
                    tot_loss = 0.0
                    total_acc, total_count = 0, 0
                    start_time = perf_counter()
                else:
                    self.commu_count += 1
                
            else:
                for i in range(self.agents):
                    self.agent_optimizers[i].apply_from_xhat(all_xhat[i], p_comm)

            # 5) bookkeeping / optional logging
            self.running_iteration += 1
        
        
