#!/usr/bin/env python3

import sys
import pickle
import tarfile
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

#from ... import MobilenetV3
# sys.path.append("..")
from MobilenetV3 import mobilenetv3
from EfficientNet import EfficientNetB0
from ResNet import ResNet18, ResNet11

# Constants for data normalization are taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py 
def read_data(arg, *args, **kwargs):
    path, is_test = arg

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    dataset = torchvision.datasets.CIFAR100(root=path, train=not is_test, download=False, transform=transform)
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
            None if width is None else LinearLayer(lin_size, width),
            None if width is None else nn.BatchNorm1d(width),
            None if width is None else Activation(),
            LinearLayer(lin_size, 100) if width is None else LinearLayer(width, 100),
            None if not "binary" in model_type else Scale()
        ]
    )

    model = filter(None, model)
    return nn.Sequential(*model)

# I guess this is MobileNetV1 ? Should check the paper :D
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
    model.extend(conv_bn(  3,  32, 2))
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
        LinearLayer(512, 100),
        None if not "binary" in model_type else Scale()
        #nn.Softmax()
    ] )
    model = filter(None, model)
    
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
    
    # dsum = torch.sum(torch.cat(diversities,dim=0), dim = 0)
    # return dsum
    # base_preds = []
    # for e in model.estimators_:
    #     ypred = apply_in_batches(e, x, 128)
    #     base_preds.append(ypred)
    
    # f_bar = np.mean(base_preds, axis=0)
    # if isinstance(model.loss_function, nn.MSELoss): 
    #     n_classes = f_bar.shape[1]
    #     n_preds = f_bar.shape[0]

    #     eye_matrix = np.eye(n_classes).repeat(n_preds, 1, 1)
    #     D = 2.0*eye_matrix
    # elif isinstance(model.loss_function, nn.NLLLoss):
    #     n_classes = f_bar.shape[1]
    #     n_preds = f_bar.shape[0]
    #     D = np.eye(n_classes).repeat(n_preds, 1, 1)
    #     target_one_hot = np.zeros((y.size, n_classes))
    #     target_one_hot[np.arange(y.size),y] = 1

    #     eps = 1e-7
    #     diag_vector = target_one_hot*(1.0/(f_bar**2+eps))
    #     #D[np.diag_indices(D.shape[0])] = diag_vector
    #     for i in range(D.shape[0]):
    #         np.fill_diagonal(D[i,:], diag_vector[i,:])
    # elif isinstance(model.loss_function, nn.CrossEntropyLoss):
    #     n_preds = f_bar.shape[0]
    #     n_classes = f_bar.shape[1]
    #     f_bar_softmax = scipy.special.softmax(f_bar,axis=1)

    #     D = -1.0 * np.expand_dims(f_bar_softmax, axis=2) @ np.expand_dims(f_bar_softmax, axis=1)

    #     # D = -1.0*torch.bmm(f_bar_softmax.unsqueeze(2), f_bar_softmax.unsqueeze(1))
    #     diag_vector = f_bar_softmax*(1.0-f_bar_softmax)
    #     for i in range(D.shape[0]):
    #         np.fill_diagonal(D[i,:], diag_vector[i,:])
    # else:
    #     D = np.array([1.0])

    # diversities = []
    # for pred in base_preds:
    #     # https://stackoverflow.com/questions/63301019/dot-product-of-two-numpy-arrays-with-3d-vectors
    #     # https://stackoverflow.com/questions/51479148/how-to-perform-a-stacked-element-wise-matrix-vector-multiplication-in-numpy
    #     diff = pred - f_bar 
    #     tmp = np.sum(D * diff[:,:,None], axis=1)
    #     covar = np.sum(tmp*diff,axis=1)

    #     # covar = torch.bmm(diff.unsqueeze(1), torch.bmm(D, diff.unsqueeze(2))).squeeze()
    #     div = 1.0/model.n_estimators * 0.5 * covar
    #     diversities.append(np.mean(div))
    #return np.sum(diversities)

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
    # accuracies = torch.cat(accuracies,dim=0)
    # return accuracies.mean().item()

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
    "batch_size" : 128,
    "amsgrad":True
}

basecfg = { 
    "no_runs":1,
    "train":("/data/s1/buschjae/CIFAR100/", False),
    "test":("/data/s1/buschjae/CIFAR100/", True),
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

# models.append(
#     {
#         "model": StackingClassifier,
#         "n_estimators":16,
#         "base_estimator": partial(mobilenet_model, model_type="float"),
#         "classifier" : lambda : nn.Sequential( torch.nn.Linear(16*100,100) ),
#         "optimizer":optimizer,
#         "scheduler":scheduler,
#         "eval_test":5,
#         "loss_function":nn.CrossEntropyLoss(reduction="none"),
#         # TODO The transformer is not applied during scoring, which so that training data is not normalized during final scoring
#         # IDEA: Introduce test and train transformer so that the model applies it on its own during testing / training?! 
#         "transformer": 
#             transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.RandomCrop(32, padding=4),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#             ])
#     }
# )

# for mtype in ["float", "binary"]:
# for base in [partial(EfficientNetB0, model_type="binary"), partial(EfficientNetB0, model_type="float"), ResNet11]:
    # models.append(
    #     {
    #         "model":SKLearnModel,
    #         # "n_estimators":16,
    #         #"base_estimator": partial(vgg_model, model_type="float", n_layers=2, n_channels=32, width=None),
    #         #"base_estimator": partial(mobilenetv3, model_type=mtype, num_classes = 100),
    #         #"base_estimator": partial(EfficientNetB0, model_type=mtype),
    #         "base_estimator": base,
    #         "optimizer":optimizer,
    #         "scheduler":scheduler,
    #         "eval_test":2,
    #         "loss_function":nn.CrossEntropyLoss(reduction="none"),
    #         "transformer":
    #             transforms.Compose([
    #                 transforms.Normalize(
    #                     mean= [-m/s for m, s in zip([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])],
    #                     std= [1/s for s in [0.2023, 0.1994, 0.2010]]
    #                 ),
    #                 transforms.ToPILImage(),
    #                 transforms.RandomCrop(32, padding=4),
    #                 transforms.RandomHorizontalFlip(),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #             ])
    #     }
    # )
#ResNet11
for m in [16]:
    for base in [partial(EfficientNetB0, model_type="binary"), partial(EfficientNetB0, model_type="float")]:
        models.append(
            {
                "model":BaggingClassifier,
                "n_estimators":m,
                "train_method":"fast",
                #"base_estimator": partial(vgg_model, model_type="float", n_layers=2, n_channels=32, width=None),
                #"base_estimator": partial(mobilenet_model, model_type=mtype),
                "base_estimator": base,
                "optimizer":optimizer,
                "scheduler":scheduler,
                "eval_test":10,
                "loss_function":nn.CrossEntropyLoss(reduction="none"),
                "transformer":
                    transforms.Compose([
                        transforms.Normalize(
                            mean= [-m/s for m, s in zip([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])],
                            std= [1/s for s in [0.2023, 0.1994, 0.2010]]
                        ),
                        transforms.ToPILImage(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
            }
        )

        models.append(
            {
                "model":E2EEnsembleClassifier,
                "n_estimators":m,
                # "l_reg":l_reg,
                # "combination_type":"softmax",
                #"base_estimator": partial(vgg_model, model_type="float", n_layers=2, n_channels=32, width=None),
                "base_estimator": base,
                "optimizer":optimizer,
                "scheduler":scheduler,
                "eval_test":10,
                "loss_function":nn.CrossEntropyLoss(reduction="none"),
                "transformer":
                    transforms.Compose([
                        transforms.Normalize(
                            mean= [-m/s for m, s in zip([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])],
                            std= [1/s for s in [0.2023, 0.1994, 0.2010]]
                        ),
                        transforms.ToPILImage(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
            }
        )

        models.append(
            {
                "model":SMCLClassifier,
                "n_estimators":m,
                # "l_reg":l_reg,
                "combination_type":"softmax",
                #"base_estimator": partial(vgg_model, model_type="float", n_layers=2, n_channels=32, width=None),
                "base_estimator": base,
                "optimizer":optimizer,
                "scheduler":scheduler,
                "eval_test":10,
                "loss_function":nn.CrossEntropyLoss(reduction="none"),
                "transformer":
                    transforms.Compose([
                        transforms.Normalize(
                            mean= [-m/s for m, s in zip([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])],
                            std= [1/s for s in [0.2023, 0.1994, 0.2010]]
                        ),
                        transforms.ToPILImage(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
            }
        )

        for l_reg in [0, 0.1, 0.2, 0.3, 0.4, 0.5]: 
            models.append(
                {
                    "model":GNCLClassifier,
                    "n_estimators":m,
                    "l_reg":l_reg,
                    "combination_type":"average",
                    #"base_estimator": partial(vgg_model, model_type="float", n_layers=2, n_channels=32, width=None),
                    "base_estimator": base,
                    "optimizer":optimizer,
                    "scheduler":scheduler,
                    "eval_test":10,
                    "loss_function":nn.CrossEntropyLoss(reduction="none"),
                    "transformer":
                        transforms.Compose([
                            # After loading we normlaize the input data, which is fine.
                            # For training however, we want to transform it a bit and then normalize it. Thus, we inverse the normalization first
                            transforms.Normalize(
                                mean= [-m/s for m, s in zip([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])],
                                std= [1/s for s in [0.2023, 0.1994, 0.2010]]
                            ),
                            transforms.ToPILImage(),
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])
                }
            )

run_experiments(basecfg, models, cuda_devices = cuda_devices, n_cores=len(cuda_devices))
