#!/usr/bin/env python3

import sys
import pickle
import gzip
import os
from datetime import datetime
from functools import partial

from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import torch
import scipy

import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

from sklearn.metrics import make_scorer, accuracy_score

from deep_ensembles_v2.Utils import Flatten, Clamp, Scale

from deep_ensembles_v2.Models import SKLearnModel
from deep_ensembles_v2.E2EEnsembleClassifier import E2EEnsembleClassifier
from deep_ensembles_v2.BaggingClassifier import BaggingClassifier
from deep_ensembles_v2.GNCLClassifier import GNCLClassifier
from deep_ensembles_v2.StackingClassifier import StackingClassifier
from deep_ensembles_v2.DeepDecisionTreeClassifier import DeepDecisionTreeClassifier
from deep_ensembles_v2.SMCLClassifier import SMCLClassifier
from deep_ensembles_v2.BinarisedNeuralNetworks import BinaryConv2d, BinaryLinear, BinaryTanh
from deep_ensembles_v2.Utils import pytorch_total_params, apply_in_batches, TransformTensorDataset

from experiment_runner.experiment_runner import run_experiments

#sys.path.append("..")
from Metrics import avg_accurcay,diversity,avg_loss,loss
#from MobileNetV3 import MobileNetV3


def read_data(arg, *args, **kwargs):
    path, is_test = arg
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.FashionMNIST(root=path, train=not is_test, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4)

    X = next(iter(loader))[0].numpy()
    Y = next(iter(loader))[1].numpy()

    return X,Y 

def vgg_model(model_type, hidden_size = 1024, n_channels = 16, depth = 2, *args, **kwargs):
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
            ConvLayer(1 if level == 0 else level*n_channels, (level+1)*n_channels, kernel_size=3, padding=1, stride = 1, bias=True),
            nn.BatchNorm2d((level+1)*n_channels),
            Activation(),
            ConvLayer((level+1)*n_channels, (level+1)*n_channels, kernel_size=3, padding=1, stride = 1, bias=True),
            nn.BatchNorm2d((level+1)*n_channels),
            Activation(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        ]

    model = []
    for i in range(depth):
        model.extend(make_layers(i, n_channels))

    if depth == 1:
        lin_size = 253*n_channels
    elif depth == 2:
        lin_size = 98*n_channels
    elif depth == 3:
        lin_size = 27*n_channels
    else:
        lin_size = 4*n_channels

    model.extend(
        [
            Flatten(),
            LinearLayer(lin_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            Activation(),
            LinearLayer(hidden_size, 10)
        ]
    )

    if "binary" in model_type:
        model.append(Scale())

    model = filter(None, model)
    return nn.Sequential(*model)


scheduler = {
    "method" : torch.optim.lr_scheduler.StepLR,
    "step_size" : 30,
    "gamma": 0.5
}

optimizer = {
    "method" : torch.optim.Adam,
    "lr" : 1e-3,
    "epochs" : 100,
    "batch_size" : 128,
    "amsgrad":True
}

basecfg = { 
    "no_runs":1,
    "train":("/data/s1/buschjae/", False),
    "test":("/data/s1/buschjae/", True),
    "data_loader":read_data,
    "scoring": {
        # TODO Maybe add "scoring" to SKLearnModel and score it on each eval?
        'accuracy': make_scorer(accuracy_score, greater_is_better=True),
        'params': pytorch_total_params,
        'diversity': diversity,
        'avg_accurcay' : avg_accurcay,
        'avg_loss' : avg_loss,
        'loss' : loss
    },
    "out_path":datetime.now().strftime('%d-%m-%Y-%H:%M:%S'),
    "store_model":False,

    "verbose":False,
    "local_mode":False,
    "ray_head":"auto",
    "redis_password":"5241590000000000",
    "num_cpus":5,
    "num_gpus":1

    # "verbose":True,
    # "local_mode":True
}

cuda_devices = [0]
models = []

for s in [1, 2]:
    for t in ["float", "binary"]:
        models.append(
            {
                "model":SKLearnModel,
                #"base_estimator": lambda: MobileNetV3(mode='small', classes_num=10, input_size=32, width_multiplier=1.0, dropout=0.0, BN_momentum=0.1, zero_gamma=False, in_channels=1),
                "base_estimator": partial(vgg_model, hidden_size = s*128, model_type = t, depth = 4, n_channels = s*32),
                "optimizer":optimizer,
                "scheduler":scheduler,
                "eval_test":1,
                "loss_function":nn.CrossEntropyLoss(reduction="none"),
                "transformer":
                    transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.RandomCrop(28, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()
                    ])
            }
        )

        for m in [8]:
            models.append(
                {
                    "model":BaggingClassifier,
                    "n_estimators":m,
                    "train_method":"fast",
                    "base_estimator": partial(vgg_model, hidden_size = s*128, model_type = t, depth = 4, n_channels = s*32),
                    "optimizer":optimizer,
                    "scheduler":scheduler,
                    "eval_test":5,
                    "loss_function":nn.CrossEntropyLoss(reduction="none"),
                    "transformer":
                        transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomCrop(28, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()
                        ])
                }
            )

            models.append(
                {
                    "model":E2EEnsembleClassifier,
                    "n_estimators":m,
                    "base_estimator": partial(vgg_model, hidden_size = s*128, model_type = t, depth = 4, n_channels = s*32),
                    "optimizer":optimizer,
                    "scheduler":scheduler,
                    "eval_test":5,
                    "loss_function":nn.CrossEntropyLoss(reduction="none"),
                    "transformer":
                        transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomCrop(28, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()
                        ])
                }
            )

            models.append(
                {
                    "model":SMCLClassifier,
                    "n_estimators":m,
                    "combination_type":"best",
                    "base_estimator": partial(vgg_model, hidden_size = s*128, model_type = t, depth = 4, n_channels = s*32),
                    "optimizer":optimizer,
                    "scheduler":scheduler,
                    "eval_test":5,
                    "loss_function":nn.CrossEntropyLoss(reduction="none"),
                    "transformer":
                        transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomCrop(28, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()
                        ])
                }
            )

            for l_reg in [0, 0.1, 0.2, 0.3, 0.4, 0.5]: 
                models.append(
                    {
                        "model":GNCLClassifier,
                        "n_estimators":m,
                        "mode":"exact",
                        "l_reg":l_reg,
                        "combination_type":"average",
                        "base_estimator": partial(vgg_model, hidden_size = s*128, model_type = t, depth = 4, n_channels = s*32),
                        "optimizer":optimizer,
                        "scheduler":scheduler,
                        "eval_test":5,
                        "loss_function":nn.CrossEntropyLoss(reduction="none"),
                        "transformer":
                            transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.RandomCrop(28, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                            ])
                    }
                )

                models.append(
                    {
                        "model":GNCLClassifier,
                        "n_estimators":m,
                        "mode":"upper",
                        "l_reg":l_reg,
                        "combination_type":"average",
                        "base_estimator": partial(vgg_model, hidden_size = s*128, model_type = t, depth = 4, n_channels = s*32),
                        "optimizer":optimizer,
                        "scheduler":scheduler,
                        "eval_test":5,
                        "loss_function":nn.CrossEntropyLoss(reduction="none"),
                        "transformer":
                            transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.RandomCrop(28, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                            ])
                    }
                )

# base = models[0]["base_estimator"]().cuda()
# print(summary(base, (1, 28, 28)))

run_experiments(basecfg, models, cuda_devices = cuda_devices, n_cores=len(cuda_devices))
