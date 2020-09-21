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

sys.path.append('../submodules/deep-ensembles-v2/')
from Utils import Flatten, weighted_cross_entropy, weighted_mse_loss, weighted_squared_hinge_loss, cov, weighted_cross_entropy_with_softmax, weighted_lukas_loss, Clamp, Scale
from Models import SKLearnModel
from BaggingClassifier import BaggingClassifier
from DeepDecisionTreeClassifier import DeepDecisionTreeClassifier
from BinarisedNeuralNetworks import BinaryConv2d, BinaryLinear, BinaryTanh

sys.path.append('../submodules/experiment_runner/')
from experiment_runner import run_experiments

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

def split_estimator(*args, **kwargs):
    model = [
        nn.Conv2d(1, 32, kernel_size=3, padding=1, stride = 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 32, kernel_size=3, padding=1, stride = 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(7*7*32,64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64,1),
        nn.BatchNorm1d(1),
        nn.Sigmoid()
    ]

    model = filter(None, model)
    return nn.Sequential(*model)

def leaf_estimator(*args, **kwargs):
    model = [
        nn.Conv2d(1, 32, kernel_size=3, padding=1, stride = 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 32, kernel_size=3, padding=1, stride = 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(7*7*32,64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64,10)
    ]

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
    "epochs" : 20,
    "batch_size" : 128,
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

cuda_devices = [0]
models = []

models.append(
    {
        "model":DeepDecisionTreeClassifier,
        "split_estimator":split_estimator,
        "leaf_estimator":leaf_estimator,
        "depth":3,
        "soft":False,
        "optimizer":optimizer,
        "scheduler":scheduler,
        "loss_function":weighted_cross_entropy_with_softmax,
    }
)

run_experiments(basecfg, models, cuda_devices = cuda_devices, n_cores=len(cuda_devices))
