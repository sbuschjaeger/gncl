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

from deep_ensembles_v2.Utils import Flatten, weighted_cross_entropy, weighted_mse_loss, weighted_squared_hinge_loss, cov, weighted_cross_entropy_with_softmax, weighted_lukas_loss, Clamp, Scale

from deep_ensembles_v2.Models import SKLearnModel
from deep_ensembles_v2.SGDEnsembleClassifier import SGDEnsembleClassifier
from deep_ensembles_v2.BaggingClassifier import BaggingClassifier
from deep_ensembles_v2.DeepDecisionTreeClassifier import DeepDecisionTreeClassifier
from deep_ensembles_v2.BinarisedNeuralNetworks import BinaryConv2d, BinaryLinear, BinaryTanh

from experiment_runner.experiment_runner import run_experiments

# Constants for data normalization are taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py 
def read_data(arg, *args, **kwargs):
    path, is_test = arg

    if is_test:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    dataset = torchvision.datasets.CIFAR10(root=path, train=not is_test, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4)

    X = next(iter(loader))[0].numpy()
    Y = next(iter(loader))[1].numpy()

    return X,Y 
    
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
# VGG13
def vgg_model(model_type, n_channels=16, n_layers=2, width=512, *args, **kwargs):
    if "binary" in model_type:
        ConvLayer = BinaryConv2d
        LinearLayer = BinaryLinear
        Activation = BinaryTanh
    else:
        ConvLayer = nn.Conv2d
        LinearLayer = nn.Linear
        Activation = nn.ReLU

    def make_layers(level, n_channels):
        return [
            ConvLayer(3 if level == 0 else level*n_channels, (level+1)*n_channels, kernel_size=3, padding=1, stride = 1, bias=True),
            nn.BatchNorm2d((level+1)*n_channels),
            Activation(),
            ConvLayer((level+1)*n_channels, (level+1)*n_channels, kernel_size=3, padding=1, stride = 1, bias=True),
            nn.BatchNorm2d((level+1)*n_channels),
            Activation(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        ]

    model = []
    for i in range(n_layers):
        model.extend(make_layers(i, n_channels))

    # This only works for kernel_size = 3 
    if n_layers == 1:
        lin_size = 506*n_channels #
    elif n_layers == 2:
        lin_size = 128*n_channels
    elif n_layers == 3:
        lin_size = 48*n_channels
    elif n_layers == 4:
        lin_size = 16*n_channels
    else:
        lin_size = 5*n_channels

    model.extend(
        [
            Flatten(),
            LinearLayer(lin_size, width),
            nn.BatchNorm1d(width),
            Activation(),
            LinearLayer(width, 10),
            None if not "binary" in model_type else Scale()
        ]
    )

    model = filter(None, model)
    return nn.Sequential(*model)

    # return nn.Sequential(
    #     ConvLayer(3, 128, kernel_size=3, padding=1, stride = 1),
    #     nn.BatchNorm2d(128),
    #     Activation(),
    #     ConvLayer(128, 128, kernel_size=3, padding=1, stride = 1),
    #     nn.MaxPool2d(kernel_size=2,stride=2),
    #     nn.BatchNorm2d(128),
    #     Activation(),

    #     ConvLayer(128, 256, kernel_size=3, padding=1, stride = 1),
    #     nn.BatchNorm2d(256),
    #     Activation(),
    #     ConvLayer(256, 256, kernel_size=3, padding=1, stride = 1),
    #     nn.MaxPool2d(kernel_size=2,stride=2),
    #     nn.BatchNorm2d(256),
    #     Activation(),

    #     ConvLayer(256, 512, kernel_size=3, padding=1, stride = 1),
    #     nn.BatchNorm2d(512),
    #     Activation(),
    #     ConvLayer(512, 512, kernel_size=3, padding=1, stride = 1),
    #     nn.MaxPool2d(kernel_size=2,stride=2),
    #     nn.BatchNorm2d(512),
    #     Activation(),

    #     Flatten(),
    #     LinearLayer(8192, 1024),
    #     nn.BatchNorm1d(1024),
    #     Activation(),
    #     LinearLayer(1024, 10),
    #     Scale()
    # )

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
    "epochs" : 50,
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
    "local_mode":True
}

cuda_devices = [0]
models = []

models.append(
    {
        # "model":SKLearnModel,
        "model":SGDEnsembleClassifier,
        "n_estimators":5,
        "base_estimator": partial(vgg_model, model_type="float", n_layers=3, n_channels=128, width=512),
        "optimizer":optimizer,
        "scheduler":scheduler,
        "eval_test":5,
        "loss_function":weighted_cross_entropy_with_softmax,
        "transformer":
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    }
)

run_experiments(basecfg, models, cuda_devices = cuda_devices, n_cores=len(cuda_devices))
