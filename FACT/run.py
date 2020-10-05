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
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler

import photon_stream


sys.path.append('../submodules/deep-ensembles-v2/')
from Utils import Flatten, weighted_cross_entropy, weighted_mse_loss, weighted_squared_hinge_loss, cov, weighted_cross_entropy_with_softmax, weighted_lukas_loss, Clamp, Scale
from Models import SKLearnModel
from BinarisedNeuralNetworks import BinaryConv2d, BinaryLinear, BinaryTanh
from SGDEnsembleClassifier import SGDEnsembleClassifier

sys.path.append('../submodules/experiment_runner/')
from experiment_runner import run_experiments

def read_data(arg ,*args, **kwargs):
    path, is_test = arg
    X, Y = np.load(path + "/x_train.npy", allow_pickle=True), np.load(path + "/y_train.npy", allow_pickle=True)

    if is_test:
        X, Y = np.load(path + "/x_test.npy", allow_pickle=True), np.load(path + "/y_test.npy", allow_pickle=True)
        # print("Checking for blank images")
        # num_zero_image = sum([1 if np.all(x.reshape(-1) == 0) else 0 for x in X])
        # print("Num blank images: {}".format(num_zero_image))
    else:
        X, Y = np.load(path + "/x_train.npy", allow_pickle=True), np.load(path + "/y_train.npy", allow_pickle=True)
    N = X.shape[0]
    width = X.shape[1]
    height = X.shape[2]
    # X = scaler.transform(X.reshape(X.shape[0], width*height))
    X = X.reshape(X.shape[0], 1, width, height).astype(np.float32)
    
    # not_zero = np.where(X != 0)
    # print(not_zero)
    # print(x[not_zero])
    # X[not_zero] = 2*np.sqrt(X[not_zero]+3.0/8.0)
    
    # print("AFTER NORMALISATION:", X.shape)
    # pos = sum([y == 1 for y in Y])
    # neg = sum([y == 0 for y in Y])
    # print("Class distributions for {} is {}/{}".format("testing" if is_test else "training", neg, pos))
    # print(list(X[0,:].reshape(-1)))

    return X,Y

class AnscombeNormalization1d(nn.Module):
    """
    a layernorm module in the TF style (epsilon inside the square root).
    """
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(AnscombeNormalization1d, self).__init__()
        self.register_buffer('running_mean', torch.zeros(hidden_size))
        self.register_buffer('running_var', torch.ones(hidden_size))

        self.variance_epsilon = eps

    def forward(self, x):
        if self.training:
            mu = x.mean(dim=0,keepdim=True)
            s = x.std(dim=0,keepdim=True)
            # print("MU 1: ", mu)
            # print("x: ", x)
            
            # if torch.isnan(mu).any():
            #     asdf

            self.running_mean = self.running_mean * 0.99 + 0.01 * mu.detach()
            self.running_var = self.running_var * 0.99 + 0.01 * s.detach()
            #x = (x - u) / (s + self.variance_epsilon)
        else:
            mu = self.running_mean
            s = self.running_var
            #x = (x - self.running_mean) / (self.running_var + self.variance_epsilon)

        # print("")
        # print("MU 2: ", mu)
        # print("")
        a = 1
        #t = -3.0/8.0*a - (s**2)/a + mu
        #t = torch.cat(x.shape[0]*[t])
        t = a*x+3.0/8.0*(a**2)-a*mu
        is_greater = t > 0
        # print("X: ", x.shape)
        # print("t: ", t.shape)
        # print(is_greater)
        # print("is_greater ", is_greater.shape)
        # print("SQRT: ", a*is_greater*x+3.0/8.0*(a**2)-a*mu)
        x = is_greater * 2.0/a*torch.sqrt(is_greater*(a*x+3.0/8.0*(a**2)-a*mu)+0.01)
        #print(is_greater * x + 0.01)
        #x = is_greater * 2.0 / a * torch.sqrt(is_greater * x + 0.01)
        # x[x != x] = 0.01 # EVIL HACK

        # print(x.shape)
        # print(x)
        # asdf
        # # print("ts: ", t_stacked.shape)
        # # print("mu: ", mu.shape)
        # # asdf
        # x[x > t] = 2.0/a*torch.sqrt(a*x[x>t]+3.0/8.0*(a**2)-a*mu)
        # x[x <= t] = 0
        return x    

class AnscombeNormalization2d(nn.Module):
    """
    a layernorm module in the TF style (epsilon inside the square root).
    """
    def __init__(self, n_filters, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(AnscombeNormalization2d, self).__init__()
        self.register_buffer('running_mean', torch.zeros(n_filters))
        self.register_buffer('running_var', torch.ones(n_filters))

        self.variance_epsilon = eps

    def forward(self, x):
        if self.training:
            mu = x.mean(dim=(0,2,3),keepdim=True)
            s = x.std(dim=(0,2,3),keepdim=True)

            self.running_mean = self.running_mean * 0.99 + 0.01 * mu.detach()
            self.running_var = self.running_var * 0.99 + 0.01 * s.detach()
            #x = (x - u) / (s + self.variance_epsilon)
        else:
            mu = self.running_mean
            s = self.running_var
            #x = (x - self.running_mean) / (self.running_var + self.variance_epsilon)

        a = 1
        #t = -3.0/8.0*a - (s**2)/a + mu
        #t = torch.cat(x.shape[0]*[t])
        t = a*x+3.0/8.0*(a**2)-a*mu
        is_greater = t > 0
        x = is_greater * 2.0/a*torch.sqrt(is_greater*(a*x+3.0/8.0*(a**2)-a*mu)+0.01)
        return x    

class SparseBN1d(nn.Module):
    """
    a layernorm module in the TF style (epsilon inside the square root).
    """
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(SparseBN1d, self).__init__()
        self.register_buffer('running_mean', torch.zeros(hidden_size))
        self.register_buffer('running_var', torch.ones(hidden_size))
        #self.register_buffer('running_var', torch.ones(1))

        self.variance_epsilon = eps

    def forward(self, x):
        # # Anscombe
        # not_zero = torch.where(x != 0)
        # print(not_zero)
        # print(x[not_zero])
        # x[not_zero] = 2*torch.sqrt(x[not_zero]+3.0/8.0)
        # print(x)
        # asdf
        # return x

        # ABS
        # u = torch.zeros(x.shape[1]).cuda()
        # s = x.abs().max()

        # if self.training:
        #     self.running_mean = self.running_mean * 0.99 + 0.01 * u.detach()
        #     self.running_var = self.running_var * 0.99 + 0.01 * s.detach()
        #     x = (x - u) / (s + self.variance_epsilon)
        # else:
        #     x = (x - self.running_mean) / (self.running_var + self.variance_epsilon)
        # return x    
        u = x.mean(dim=0,keepdim=True)
        s = x.std(dim=0,keepdim=True)
        # print()
        if self.training and x.requires_grad:
            self.running_mean = self.running_mean * 0.99 + 0.01 * u.detach()
            self.running_var = self.running_var * 0.99 + 0.01 * s.detach()
        else:
            print("NO GRAD!!")
        x = (x - self.running_mean) / (self.running_var + self.variance_epsilon)
        return x 

class SparseBN2d(nn.Module):
    """
    a layernorm module in the TF style (epsilon inside the square root).
    """
    def __init__(self, n_channels, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(SparseBN2d, self).__init__()
        self.register_buffer('running_var', torch.ones((1,n_channels,1,1)))

        self.variance_epsilon = eps

    def forward(self, x):
        #u = x.mean(0, keepdim=True)
        # print(u.shape)
        # asdf
        s = x.abs().max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=0, keepdim=True)[0]
        # print(s.shape)
        #s = x.abs().max()#[0]#0, keepdim=True

        if self.training:
            self.running_var = self.running_var * 0.99 + 0.01 * s.detach()
            x = x / (s + self.variance_epsilon)
        else:
            x = x / (self.running_var + self.variance_epsilon)
        return x  

def mlp_model(model_type, n_layers, l_size,*args, **kwargs):
    if "binary" in model_type:
        ConvLayer = BinaryConv2d
        LinearLayer = BinaryLinear
        Activation = BinaryTanh
    else:
        ConvLayer = nn.Conv2d
        LinearLayer = nn.Linear
        Activation = nn.ReLU

    model = [Flatten()]
    for i in range(n_layers):
        model.extend(
            [
                LinearLayer(l_size, l_size) if i > 0 else LinearLayer(45*46, l_size),
                #nn.BatchNorm1d(l_size),
                AnscombeNormalization1d(l_size),
                Activation(),
            ]
        )

    model.append(LinearLayer(l_size, 2))

    if "binary" in model_type:
        model.append(Scale())

    model = filter(None, model)
    return nn.Sequential(*model)

def cnn_model(model_type, n_channels = 16, depth = 2, use_anscombe = False, *args, **kwargs):
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
            AnscombeNormalization2d((level+1)*n_channels) if use_anscombe else nn.BatchNorm2d((level+1)*n_channels),
            Activation(),
            ConvLayer((level+1)*n_channels, (level+1)*n_channels, kernel_size=3, padding=1, stride = 1, bias=True),
            AnscombeNormalization2d((level+1)*n_channels) if use_anscombe else nn.BatchNorm2d((level+1)*n_channels),
            Activation(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        ]

    model = []
    for i in range(depth):
        model.extend(make_layers(i, n_channels))

    if depth == 1:
        lin_size = 506*n_channels
    elif depth == 2:
        lin_size = 242*n_channels
    elif depth == 3:
        lin_size = 75*n_channels
    else:
        lin_size = 1024

    model.extend(
        [
            Flatten(),
            LinearLayer(lin_size, 1024),
            AnscombeNormalization1d(1024) if use_anscombe else nn.BatchNorm1d(1024),
            Activation(),
            LinearLayer(1024, 2)
        ]
    )

    if "binary" in model_type:
        model.append(Scale())

    model = filter(None, model)
    return nn.Sequential(*model)

scheduler = {
    "method" : torch.optim.lr_scheduler.StepLR,
    "step_size" : 10,
    "gamma": 0.5
}

optimizer = {
    "method" : torch.optim.Adam,
    "lr" : 1e-3,
    "epochs" : 50,
    "batch_size" : 128,
    "amsgrad":True
}

basecfg = { 
    "no_runs":1,
    # "train":"train_32x32.mat",
    # "test":"test_32x32.mat",
    "train":("/data/s1/buschjae/FACT/", False),
    "test":("/data/s1/buschjae/FACT/", True),
    "data_loader":read_data,
    "scoring": {
        'accuracy': make_scorer(accuracy_score, greater_is_better=True),
        'roc_auc': make_scorer(roc_auc_score, greater_is_better=True)
    },
    "out_path":datetime.now().strftime('%d-%m-%Y-%H:%M:%S'),
    "verbose":True,
    "store":False,
}

cuda_devices = [0]
models = []

models.append(
    {
        "model":SKLearnModel,
        "base_estimator": partial(cnn_model, model_type="float", n_channels=16, depth=3, use_anscombe=False),
        # "base_estimator": partial(mlp_model, model_type="float", n_layers=3, l_size=1024),
        "optimizer":optimizer,
        "scheduler":scheduler,
        "loss_function":weighted_cross_entropy_with_softmax,
        "x_test":None,
        "y_test":None
    }
)

models.append(
    {
        "model":SKLearnModel,
        "base_estimator": partial(cnn_model, model_type="float", n_channels=16, depth=3, use_anscombe=True),
        # "base_estimator": partial(mlp_model, model_type="float", n_layers=3, l_size=1024),
        "optimizer":optimizer,
        "scheduler":scheduler,
        "loss_function":weighted_cross_entropy_with_softmax,
        "x_test":None,
        "y_test":None
    }
)


models.append(
    {
        "model":SKLearnModel,
        "base_estimator": partial(cnn_model, model_type="float", n_channels=64, depth=3, use_anscombe=False),
        # "base_estimator": partial(mlp_model, model_type="float", n_layers=3, l_size=1024),
        "optimizer":optimizer,
        "scheduler":scheduler,
        "loss_function":weighted_cross_entropy_with_softmax,
        "x_test":None,
        "y_test":None
    }
)

models.append(
    {
        "model":SKLearnModel,
        "base_estimator": partial(cnn_model, model_type="float", n_channels=64, depth=3, use_anscombe=True),
        # "base_estimator": partial(mlp_model, model_type="float", n_layers=3, l_size=1024),
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
#         #"base_estimator": partial(bnn_model, model_type="binary", n_channels=8, depth=2),
#         "base_estimator": partial(mlp_model, model_type="binary", n_layers=3, l_size=1024),
#         "optimizer":optimizer,
#         "scheduler":scheduler,
#         "loss_function":weighted_cross_entropy_with_softmax,
#         "x_test":None,
#         "y_test":None
#     }
# )

# for ne in [2,3,4,5]:
#     models.append(
#         {
#             "model":SGDEnsembleClassifier,
#             #"base_estimator": partial(bnn_model, model_type="binary", n_channels=8, depth=2),
#             "base_estimator": partial(mlp_model, model_type="binary", n_layers=3, l_size=1024),
#             "optimizer":optimizer,
#             "scheduler":scheduler,
#             "loss_function":weighted_cross_entropy_with_softmax,
#             "n_estimators":ne,
#             "x_test":None,
#             "y_test":None
#         }
#     )

# models.append(
#     {
#         "model":SKLearnModel,
#         "base_estimator": partial(bnn_model, model_type="binary"),
#         "optimizer":optimizer,
#         "scheduler":scheduler,
#         "loss_function":weighted_cross_entropy_with_softmax,
#     }
# )


run_experiments(basecfg, models, cuda_devices = cuda_devices, n_cores=len(cuda_devices))
