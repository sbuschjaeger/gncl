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

from sklearn.metrics import make_scorer, accuracy_score

sys.path.append('../submodules/deep-ensembles-v2/')
from Utils import Flatten, weighted_cross_entropy, weighted_mse_loss, weighted_squared_hinge_loss, cov, weighted_cross_entropy_with_softmax, weighted_lukas_loss, Clamp, Scale
from Models import SKLearnModel
from BinarisedNeuralNetworks import BinaryConv2d, BinaryLinear, BinaryTanh

sys.path.append('../submodules/experiment_runner/')
from experiment_runner import run_experiments

# Constants for data normalization are taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py 
def read_data(arg, *args, **kwargs):
    path, is_test = arg

    if is_test:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        # transform = None
        transform = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    dataset = torchvision.datasets.CIFAR10(root=path, train=not is_test, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4)

    X = next(iter(loader))[0].numpy()
    Y = next(iter(loader))[1].numpy()

    return X,Y 
    
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
# VGG13
def bnn_model(model_type, *args, **kwargs):
    if "binary" in model_type:
        ConvLayer = BinaryConv2d
        LinearLayer = BinaryLinear
        Activation = BinaryTanh
    else:
        ConvLayer = nn.Conv2d
        LinearLayer = nn.Linear
        Activation = nn.ReLU

    return nn.Sequential(
        ConvLayer(3, 128, kernel_size=3, padding=1, stride = 1),
        nn.BatchNorm2d(128),
        Activation(),
        ConvLayer(128, 128, kernel_size=3, padding=1, stride = 1),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.BatchNorm2d(128),
        Activation(),

        ConvLayer(128, 256, kernel_size=3, padding=1, stride = 1),
        nn.BatchNorm2d(256),
        Activation(),
        ConvLayer(256, 256, kernel_size=3, padding=1, stride = 1),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.BatchNorm2d(256),
        Activation(),

        ConvLayer(256, 512, kernel_size=3, padding=1, stride = 1),
        nn.BatchNorm2d(512),
        Activation(),
        ConvLayer(512, 512, kernel_size=3, padding=1, stride = 1),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.BatchNorm2d(512),
        Activation(),

        # ConvLayer(128, 256, kernel_size=3, padding=0, stride = 1),
        # nn.BatchNorm2d(256),
        # Activation(),
        #nn.MaxPool2d(kernel_size=2,stride=2),

        Flatten(),
        LinearLayer(8192, 1024),
        nn.BatchNorm1d(1024),
        Activation(),
        LinearLayer(1024, 10),
        Scale()
    )

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
    "epochs" : 250,
    "batch_size" : 128,
    "amsgrad":True
}

basecfg = { 
    "no_runs":1,
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

cuda_devices = [0]
models = []

models.append(
    {
        "model":SKLearnModel,
        "base_estimator": partial(bnn_model, model_type="binary"),
        "optimizer":optimizer,
        "scheduler":scheduler,
        "loss_function":weighted_cross_entropy_with_softmax,
        "transformer":
            transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    }
)

run_experiments(basecfg, models, cuda_devices = cuda_devices, n_cores=len(cuda_devices))
