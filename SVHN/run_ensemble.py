#!/usr/bin/env python3

import sys
import pickle
import tarfile
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import torch

from torch import nn
import torchvision
import torchvision.transforms as transforms

from scipy.io import loadmat
from sklearn.metrics import make_scorer, accuracy_score

sys.path.append('../submodules/deep-ensembles-v2/')
from Utils import Flatten, weighted_cross_entropy, weighted_mse_loss, weighted_squared_hinge_loss, cov, weighted_cross_entropy_with_softmax, weighted_lukas_loss, Clamp, Scale
from Models import Model
from DeepDecisionTreeClassifier import DeepDecisionTreeClassifier
from BinarisedNeuralNetworks import BinaryConv2d, BinaryLinear, BinaryTanh

sys.path.append('../submodules/experiment_runner/')
from experiment_runner import run_experiments

def read_data(arg ,*args, **kwargs):
    path, is_test = arg

    if is_test:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        # transform = None
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    split = "test" if is_test else "train"
    dataset = torchvision.datasets.SVHN(root=path, split=split, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4)

    X = next(iter(loader))[0].numpy()
    Y = next(iter(loader))[1].numpy()

    return X,Y 

def split_estimator(*args, **kwargs):
    model = [
        Flatten(),
        nn.Linear(3*32*32, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 2),
        nn.Sigmoid()
    ]

    model = filter(None, model)
    return nn.Sequential(*model)

def leaf_estimator(*args, **kwargs):
    model = [
        Flatten(),
        nn.Linear(3*32*32, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    ]

    model = filter(None, model)
    return nn.Sequential(*model)

scheduler = {
    "method" : torch.optim.lr_scheduler.StepLR,
    "step_size" : 20,
    "gamma": 0.5
}

optimizer = {
    "method" : torch.optim.Adam,
    #"method" : torch.optim.SGD,
    # "method" : torch.optim.RMSprop,
    "lr" : 1e-3,
    "epochs" : 100,
    "batch_size" : 256,
    "amsgrad":True
}

basecfg = { 
    "no_runs":1,
    # "train":"train_32x32.mat",
    # "test":"test_32x32.mat",
    "train":("./", False),
    "test":("./", True),
    "data_loader":read_data,
    "scoring": {
        'accuracy': make_scorer(accuracy_score, greater_is_better=True),
    },
    "out_path":datetime.now().strftime('%d-%m-%Y-%H:%M:%S'),
    "verbose":True,
    "store_model":False,
}

cuda_devices = [1]
models = []

models.append(
    {
        "model":DeepDecisionTreeClassifier,
        "split_estimator":split_estimator,
        "leaf_estimator":leaf_estimator,
        "depth":2,
        "optimizer":optimizer,
        "scheduler":scheduler,
        "loss_function":weighted_cross_entropy_with_softmax,
    }
)

run_experiments(basecfg, models, cuda_devices = cuda_devices, n_cores=len(cuda_devices))
