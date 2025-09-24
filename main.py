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

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from trainers import *

agents = 10
dataset = "cifar100"
communication_round = 3000
local_steps = 10
bs = 128

iid = False  # default dir_alpha=0.5

L = 2
dir_alpha = 10

for i in range(5):
    cwd = os.getcwd()
    results_path = os.path.join(cwd, "results")
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    fname = os.path.join(results_path,f"{dataset}_e{communication_round}_hom{iid}_0_L_{L}_dir_{dir_alpha}.csv")  # Changed args.test_num to 0


    # scaffnew
    ScaffnewTrainer(
        dataset=dataset,
        batch_size=bs,
        dir_alpha=1,
        agents = agents,
        communication_round = communication_round,
        local_steps = local_steps,
        fname=fname,
        stratified=iid,
        lr=1/L
        # lr=1/L
    )
    
    # FedrcuTrainer(
    #     dataset=dataset,
    #     batch_size=bs,
    #     dir_alpha = 1,
    #     agents = agents,
    #     communication_round = communication_round,
    #     local_steps = local_steps,
    #     fname=fname,
    #     stratified=iid,       
    #     lr=8/(17 * L  * local_steps)
    # )
    FedLinTrainer(
        dataset=dataset,
        batch_size=bs,
        dir_alpha=1,
        agents = agents,
        communication_round = communication_round,
        local_steps = local_steps,
        fname=fname,
        stratified=iid,       
        lr=2/(5 * L  * local_steps - L)
    )


    ScaffoldTrainer(
        dataset=dataset,
        batch_size=bs,
        dir_alpha=1,
        agents = agents,
        communication_round = communication_round,
        local_steps = local_steps,
        fname=fname,
        stratified=iid,       
        lr=1/(81 * L  * local_steps)
    )

    FedLinTrainer(
        dataset=dataset,
        batch_size=bs,
        dir_alpha=1,
        agents = agents,
        communication_round = communication_round,
        local_steps = local_steps,
        fname=fname,
        stratified=iid,       
        lr=1/(10 * L  * local_steps)
    )





# agents = 10
# dataset = "cifar10"
# communication_round = 2000
# local_steps = 10
# bs = 128

# iid = False  # default dir_alpha=0.5

# L = 2.0
# dir_alpha = 1.5

# cwd = os.getcwd()
# results_path = os.path.join(cwd, "results")
# if not os.path.isdir(results_path):
#     os.mkdir(results_path)

# fname = os.path.join(results_path,f"{dataset}_e{communication_round}_hom{iid}_0_L_{L}_dir_{dir_alpha}.csv")  # Changed args.test_num to 0

# FedrcuTrainer(
#     dataset=dataset,
#     batch_size=bs,
#     dir_alpha = 1,
#     agents = agents,
#     communication_round = communication_round,
#     local_steps = local_steps,
#     fname=fname,
#     stratified=iid,       
#     lr=8/(17 * L  * local_steps)
# )

# # scaffnew
# ScaffnewTrainer(
#     dataset=dataset,
#     batch_size=bs,
#     dir_alpha=1,
#     agents = agents,
#     communication_round = communication_round,
#     local_steps = local_steps,
#     fname=fname,
#     stratified=iid,
#     lr=1/L
#     # lr=1/L
# )

# ScaffoldTrainer(
#     dataset=dataset,
#     batch_size=bs,
#     dir_alpha=1,
#     agents = agents,
#     communication_round = communication_round,
#     local_steps = local_steps,
#     fname=fname,
#     stratified=iid,       
#     lr=1/(24 * L  * local_steps)
# )

# FedLinTrainer(
#     dataset=dataset,
#     batch_size=bs,
#     dir_alpha=1,
#     agents = agents,
#     communication_round = communication_round,
#     local_steps = local_steps,
#     fname=fname,
#     stratified=iid,       
#     lr=1/(26 * L  * local_steps)
# )

