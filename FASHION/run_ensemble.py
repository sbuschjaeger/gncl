#!/usr/bin/env python3

import sys
import pickle
import gzip
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
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.int_)
    f.close()
    return data

def read_data(arg, *args, **kwargs):
    path, is_test = arg
    if is_test:
        return read_examples(path + "/t10k-images-idx3-ubyte.gz"), read_targets(path + "/t10k-labels-idx1-ubyte.gz")
    else:
        return read_examples(path + "/train-images-idx3-ubyte.gz"), read_targets(path + "/train-labels-idx1-ubyte.gz")

def vgg_model(model_type, n_channels = 16, depth = 2, *args, **kwargs):
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

def mobilenet_model(model_type, *args, **kwargs):
    if "binary" in model_type:
        ConvLayer = BinaryConv2d
        LinearLayer = BinaryLinear
        Activation = BinaryTanh
    else:
        ConvLayer = nn.Conv2d
        LinearLayer = nn.Linear
        Activation = nn.ReLU

    # https://modelzoo.co/model/pytorch-mobilenet
    def conv_bn(inp, oup, stride):
        return [
            ConvLayer(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            Activation()
        ]

    def conv_dw(inp, oup, stride):
        return [
            ConvLayer(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            Activation(),
            ConvLayer(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            Activation()
        ]
    
    model = []
    model.extend(conv_bn(  1,  32, 2))
    model.extend(conv_dw( 32,  64, 1))
    model.extend(conv_dw( 64, 128, 2))
    model.extend(conv_dw(128, 128, 1))
    model.extend(conv_dw(128, 256, 2))
    model.extend(conv_dw(256, 256, 1))
    model.extend(conv_dw(256, 512, 2))
    
    model.extend( [
        nn.AvgPool2d(2),
        Flatten(),
        None if not "binary" in model_type else nn.BatchNorm1d(512),
        LinearLayer(512, 10),
        None if not "binary" in model_type else Scale()
        #nn.Softmax()
    ] )

    return nn.Sequential(*model)
    # return nn.Sequential(
    #     conv_bn(  3,  32, 2), 
    #     conv_dw( 32,  64, 1),
    #     conv_dw( 64, 128, 2),
    #     # conv_dw(128, 128, 1),
    #     # conv_dw(128, 256, 2),
    #     # conv_dw(256, 256, 1),
    #     # conv_dw(256, 512, 2),
    #     # conv_dw(512, 512, 1),
    #     # conv_dw(512, 512, 1),
    #     # conv_dw(512, 512, 1),
    #     # conv_dw(512, 512, 1),
    #     # conv_dw(512, 512, 1),
    #     # conv_dw(512, 1024, 2),
    #     # conv_dw(1024, 1024, 1),
    #     nn.AvgPool2d(7),
    #     Flatten(),
    #     LinearLayer(1024, 100)
    # )
    #self.fc = nn.Linear(1024, 1000)

def diversity(model, x, y):
    # This is basically a copy/paste from the GNCLClasifier regularizer, which can also be used for 
    # other classifier. I tried to do it with numpy first and I think it should work but I did not 
    # really understand numpy's bmm variant, so I opted for the safe route here. 
    # Also, pytorch seems a little faster due to gpu support
    if not hasattr(model, "estimators_"):
        return 0
    model.eval()
    
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)
    dataset = TransformTensorDataset(x_tensor, y_tensor, transform=None)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = model.batch_size)
    
    diversities = []
    for batch in test_loader:
        data, target = batch[0], batch[1]
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        with torch.no_grad():
            f_bar, base_preds = model.forward_with_base(data)
        
        if isinstance(model.loss_function, nn.MSELoss): 
            n_classes = f_bar.shape[1]
            n_preds = f_bar.shape[0]

            eye_matrix = torch.eye(n_classes).repeat(n_preds, 1, 1).cuda()
            D = 2.0*eye_matrix
        elif isinstance(model.loss_function, nn.NLLLoss):
            n_classes = f_bar.shape[1]
            n_preds = f_bar.shape[0]
            D = torch.eye(n_classes).repeat(n_preds, 1, 1).cuda()
            target_one_hot = torch.nn.functional.one_hot(target, num_classes = n_classes).type(torch.cuda.FloatTensor)

            eps = 1e-7
            diag_vector = target_one_hot*(1.0/(f_bar**2+eps))
            D.diagonal(dim1=-2, dim2=-1).copy_(diag_vector)
        elif isinstance(model.loss_function, nn.CrossEntropyLoss):
            n_preds = f_bar.shape[0]
            n_classes = f_bar.shape[1]
            f_bar_softmax = nn.functional.softmax(f_bar,dim=1)
            D = -1.0*torch.bmm(f_bar_softmax.unsqueeze(2), f_bar_softmax.unsqueeze(1))
            diag_vector = f_bar_softmax*(1.0-f_bar_softmax)
            D.diagonal(dim1=-2, dim2=-1).copy_(diag_vector)
        else:
            D = torch.tensor(1.0)

        batch_diversities = []
        for pred in base_preds:
            diff = pred - f_bar 
            covar = torch.bmm(diff.unsqueeze(1), torch.bmm(D, diff.unsqueeze(2))).squeeze()
            div = 1.0/model.n_estimators * 0.5 * covar
            batch_diversities.append(div)

        diversities.append(torch.stack(batch_diversities, dim = 1))
    div = torch.cat(diversities,dim=0)
    return div.sum(dim=1).mean(dim=0).item()

def loss(model, x, y):
    model.eval()
    
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)
    dataset = TransformTensorDataset(x_tensor, y_tensor, transform=None)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = model.batch_size)
    
    losses = []
    for batch in test_loader:
        data, target = batch[0], batch[1]
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        with torch.no_grad():
            pred = model(data)
        
        losses.append(model.loss_function(pred, target).mean().item())
    
    return np.mean(losses)

def avg_loss(model, x, y):
    if not hasattr(model, "estimators_"):
        return 0
    model.eval()
    
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)
    dataset = TransformTensorDataset(x_tensor, y_tensor, transform=None)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = model.batch_size)
    
    losses = []
    for batch in test_loader:
        data, target = batch[0], batch[1]
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        with torch.no_grad():
            f_bar, base_preds = model.forward_with_base(data)
        
        ilosses = []
        for base in base_preds:
            ilosses.append(model.loss_function(base, target).mean().item())
            
        losses.append(np.mean(ilosses))

    return np.mean(losses)

def avg_accurcay(model, x, y):
    if not hasattr(model, "estimators_"):
        return 0
    model.eval()
    
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)
    dataset = TransformTensorDataset(x_tensor, y_tensor, transform=None)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = model.batch_size)
    
    accuracies = []
    for batch in test_loader:
        data, target = batch[0], batch[1]
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        with torch.no_grad():
            _, base_preds = model.forward_with_base(data)
        
        iaccuracies = []
        for base in base_preds:
            iaccuracies.append( 100.0*(base.argmax(1) == target).type(torch.cuda.FloatTensor) )
            
        accuracies.append(torch.cat(iaccuracies,dim=0).mean().item())

    return np.mean(accuracies)

scheduler = {
    "method" : torch.optim.lr_scheduler.StepLR,
    "step_size" : 30,
    "gamma": 0.5
}

optimizer = {
    "method" : torch.optim.Adam,
    #"method" : torch.optim.SGD,
    # "method" : torch.optim.RMSprop,
    "lr" : 1e-3,
    "epochs" : 80,
    "batch_size" : 256,
    "amsgrad":True
}

basecfg = { 
    "no_runs":1,
    "train":("/home/buschjae/projects/bnn/FASHION", False),
    "test":("/home/buschjae/projects/bnn/FASHION", True),
    "data_loader":read_data,
    "scoring": {
        # TODO Maybe add "scoring" to model and score it on each eval?
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

models.append(
    {
        "model":SKLearnModel,
        # "n_estimators":16,
        #"base_estimator": partial(vgg_model, model_type="float", n_layers=2, n_channels=32, width=None),
        "base_estimator": partial(mobilenet_model, model_type="binary"),
        "optimizer":optimizer,
        "scheduler":scheduler,
        "eval_test":5,
        "loss_function":nn.CrossEntropyLoss(reduction="none"),
    }
)

models.append(
    {
        "model":BaggingClassifier,
        "n_estimators":8,
        "train_method":"fast",
        #"base_estimator": partial(vgg_model, model_type="float", n_layers=2, n_channels=32, width=None),
        "base_estimator": partial(mobilenet_model, model_type="binary"),
        "optimizer":optimizer,
        "scheduler":scheduler,
        "eval_test":5,
        "loss_function":nn.CrossEntropyLoss(reduction="none"),
    }
)

models.append(
    {
        "model":E2EEnsembleClassifier,
        "n_estimators":8,
        # "l_reg":l_reg,
        # "combination_type":"softmax",
        #"base_estimator": partial(vgg_model, model_type="float", n_layers=2, n_channels=32, width=None),
        "base_estimator": partial(mobilenet_model, model_type="binary"),
        "optimizer":optimizer,
        "scheduler":scheduler,
        "eval_test":5,
        "loss_function":nn.CrossEntropyLoss(reduction="none"),
    }
)

models.append(
    {
        "model":SMCLClassifier,
        "n_estimators":8,
        # "l_reg":l_reg,
         "combination_type":"softmax",
        #"base_estimator": partial(vgg_model, model_type="float", n_layers=2, n_channels=32, width=None),
        "base_estimator": partial(mobilenet_model, model_type="binary"),
        "optimizer":optimizer,
        "scheduler":scheduler,
        "eval_test":5,
        "loss_function":nn.CrossEntropyLoss(reduction="none"),
    }
)

for l_reg in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: 
    models.append(
        {
            "model":GNCLClassifier,
            "n_estimators":8,
            "l_reg":l_reg,
            "combination_type":"average",
            #"base_estimator": partial(vgg_model, model_type="float", n_layers=2, n_channels=32, width=None),
            "base_estimator": partial(mobilenet_model, model_type="binary"),
            "optimizer":optimizer,
            "scheduler":scheduler,
            "eval_test":5,
            "loss_function":nn.CrossEntropyLoss(reduction="none"),
        }
    )

run_experiments(basecfg, models, cuda_devices = cuda_devices, n_cores=len(cuda_devices))
