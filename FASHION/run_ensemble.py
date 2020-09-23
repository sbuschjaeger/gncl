#!/usr/bin/env python3

import sys
import pickle
import gzip
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

from deep_ensembles_v2.Utils import Flatten, weighted_cross_entropy, weighted_mse_loss, weighted_squared_hinge_loss, cov, weighted_cross_entropy_with_softmax, weighted_lukas_loss, Clamp, Scale

from deep_ensembles_v2.Models import SKLearnModel
from deep_ensembles_v2.BaggingClassifier import BaggingClassifier
from deep_ensembles_v2.DeepDecisionTreeClassifier import DeepDecisionTreeClassifier
from deep_ensembles_v2.BinarisedNeuralNetworks import BinaryConv2d, BinaryLinear, BinaryTanh

from experiment_runner.experiment_runner import run_experiments

def read_examples(path):
    f = gzip.open(path,'r') #'train-images-idx3-ubyte.gz'
    image_size = 28

    f.read(16)
    #buf = f.read(image_size * image_size * N)
    buf = f.read()
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    N = int(len(data) / (image_size*image_size))
    data = data.reshape(N, 1, image_size, image_size)
    f.close()
    return data

def read_targets(path):
    f = gzip.open(path,'r')
    f.read(8)
    #buf = f.read(1 * N)
    buf = f.read()
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    f.close()
    return data

def read_data(arg, *args, **kwargs):
    path, is_test = arg
    if is_test:
        return read_examples(path + "/t10k-images-idx3-ubyte.gz"), read_targets(path + "/t10k-labels-idx1-ubyte.gz")
    else:
        return read_examples(path + "/train-images-idx3-ubyte.gz"), read_targets(path + "/train-labels-idx1-ubyte.gz")

# def dl_model(*args, **kwargs):
#     return nn.Sequential(
#         nn.Conv2d(1, 32, kernel_size=3, padding=1, stride = 1),
#         nn.BatchNorm2d(32),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.Conv2d(32, 32, kernel_size=3, padding=1, stride = 1),
#         nn.BatchNorm2d(32),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         Flatten(),
#         nn.Linear(7*7*32,64),
#         nn.BatchNorm1d(64),
#         nn.ReLU(),
#         nn.Linear(64,10),
#         # Clamp(min_out = -1, max_out = 1)
#     )

def cnn_model(model_type, n_channels = 16, depth = 2, *args, **kwargs):
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
        lin_size = 75*n_channels
    else:
        lin_size = 4*n_channels

    model.extend(
        [
            Flatten(),
            LinearLayer(lin_size, 1024),
            nn.BatchNorm1d(1024),
            Activation(),
            LinearLayer(1024, 10)
        ]
    )

    if "binary" in model_type:
        model.append(Scale())

    model = filter(None, model)
    return nn.Sequential(*model)

scheduler = {
    "method" : torch.optim.lr_scheduler.StepLR,
    "step_size" : 5,
    "gamma": 0.5
}

optimizer = {
    "method" : torch.optim.Adam,
    #"method" : torch.optim.SGD,
    # "method" : torch.optim.RMSprop,
    "lr" : 1e-3,
    "epochs" : 50,
    "batch_size" : 16,
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
    "verbose" : True,
    "store_model" : False,
    "local_mode" : True
}

cuda_devices = [0]
models = []

models.append(
    {
        "model":SKLearnModel,
        "base_estimator":partial(cnn_model, model_type="binary",n_channels = 32, depth = 4),
        "eval_test":1,
        "optimizer":optimizer,
        "scheduler":scheduler,
        "loss_function":weighted_cross_entropy_with_softmax,
        "x_test":None,
        "y_test":None
    }
)

# models.append(
#     {
#         "model":SKLearnModel,
#         "base_estimator":partial(cnn_model, model_type="binary",n_channels = 8, depth = 2),
#         "eval_test":1,
#         "optimizer":optimizer,
#         "scheduler":scheduler,
#         "loss_function":weighted_cross_entropy_with_softmax,
#         "x_test":None,
#         "y_test":None
#     }
# )

# models.append(
#     {
#         "model":BaggingClassifier,
#         "base_estimator":partial(cnn_model, model_type="float",n_channels = 32, depth = 4),
#         "bootstrap":False,
#         "freeze_layers":True,
#         "n_estimators":5,
#         "x_test":None,
#         "y_test":None,
#         "optimizer":optimizer,
#         "scheduler":scheduler,
#         "loss_function":weighted_cross_entropy_with_softmax
#     }
# )

# models.append(
#     {
#         "model":DeepDecisionTreeClassifier,
#         "split_estimator":split_estimator,
#         "leaf_estimator":leaf_estimator,
#         "depth":2,
#         "optimizer":optimizer,
#         "scheduler":scheduler,
#         "loss_function":weighted_cross_entropy_with_softmax,
#     }
# )

run_experiments(basecfg, models, cuda_devices = cuda_devices, n_cores=len(cuda_devices))
