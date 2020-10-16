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
from deep_ensembles_v2.models.BinarisedNeuralNetworks import BinaryConv2d, BinaryLinear, BinaryTanh
from deep_ensembles_v2.Utils import pytorch_total_params, apply_in_batches, TransformTensorDataset

from experiment_runner.experiment_runner import run_experiments

#from ... import MobilenetV3
# sys.path.append("..")
from deep_ensembles_v2.Metrics import avg_accurcay,diversity,avg_loss,loss
from deep_ensembles_v2.models.VGG import VGGNet
from deep_ensembles_v2.models.SimpleResNet import SimpleResNet
from deep_ensembles_v2.models.MobileNetV3 import MobileNetV3
from deep_ensembles_v2.models.BinarisedNeuralNetworks import BinaryModel

# from MobilenetV3 import mobilenetv3
# from EfficientNet import EfficientNetB0
# from ResNet import ResNet18, ResNet11

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

    # X = np.random.normal(size=(256,3,32,32))
    # Y = np.random.randint(2,size=(256,1))
    return X,Y 

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
    "store_model":False
}

DEBUG = False

if DEBUG:
    basecfg.update({
        "verbose":True,
        "local_mode":True
    })
else:
    basecfg.update({
        "verbose":False,
        "local_mode":False,
        "ray_head":"auto",
        "redis_password":"5241590000000000",
        "num_cpus":5,
        "num_gpus":1
    })

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

for s in ["small"]:
    for t in ["float", "binary"]:
        scheduler = {
            "method" : torch.optim.lr_scheduler.CosineAnnealingLR,
            "T_max" : 20
        }

        optimizer = {
            "method" : torch.optim.Adam if "binary" in t else torch.optim.SGD,
            "lr" : 1e-3 if "binary" in t else 0.3,
            "epochs" : 200,
            "batch_size" : 128,
        }

        def mobilenetv3(size, model_type):
            if "binary" == model_type:
                return BinaryModel(MobileNetV3(mode=size, classes_num=100, input_size=32, width_multiplier=1.0, dropout=0.2, BN_momentum=0.1, zero_gamma=False, in_channels = 3), keep_activation=True)
            else:
                return MobileNetV3(mode=size, classes_num=100, input_size=32, width_multiplier=1.0, dropout=0.2, BN_momentum=0.1, zero_gamma=False, in_channels = 3)

        models.append(
            {
                "model":SKLearnModel,
                "base_estimator": partial(mobilenetv3, size=s, model_type=t),
                "optimizer":optimizer,
                "scheduler":scheduler,
                "eval_test":5,
                "loss_function":nn.CrossEntropyLoss(reduction="none"),
                "transformer":
                    transforms.Compose([
                        # After loading we normlaize the input data, which is fine.
                        # For training however, we want to transform it a bit and _then_ normalize it. Thus, we inverse the normalization first
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

        for m in [16]:
            models.append(
                {
                    "model":BaggingClassifier,
                    "n_estimators":m,
                    "train_method":"fast",
                    "base_estimator": partial(mobilenetv3, size=s, model_type=t),
                    "optimizer":optimizer,
                    "scheduler":scheduler,
                    "eval_test":5,
                    "loss_function":nn.CrossEntropyLoss(reduction="none"),
                    "transformer":
                        transforms.Compose([
                            # After loading we normlaize the input data, which is fine.
                            # For training however, we want to transform it a bit and _then_ normalize it. Thus, we inverse the normalization first
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
                    "base_estimator": partial(mobilenetv3, size=s, model_type=t),
                    "optimizer":optimizer,
                    "scheduler":scheduler,
                    "eval_test":5,
                    "loss_function":nn.CrossEntropyLoss(reduction="none"),
                    "transformer":
                        transforms.Compose([
                            # After loading we normlaize the input data, which is fine.
                            # For training however, we want to transform it a bit and _then_ normalize it. Thus, we inverse the normalization first
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
                    "combination_type":"best",
                    "base_estimator": partial(mobilenetv3, size=s, model_type=t),
                    "optimizer":optimizer,
                    "scheduler":scheduler,
                    "eval_test":5,
                    "loss_function":nn.CrossEntropyLoss(reduction="none"),
                    "transformer":
                        transforms.Compose([
                            # After loading we normlaize the input data, which is fine.
                            # For training however, we want to transform it a bit and _then_ normalize it. Thus, we inverse the normalization first
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
                        "mode":"exact",
                        "l_reg":l_reg,
                        "combination_type":"average",
                        "base_estimator": partial(mobilenetv3, size=s, model_type=t),
                        "optimizer":optimizer,
                        "scheduler":scheduler,
                        "eval_test":5,
                        "loss_function":nn.CrossEntropyLoss(reduction="none"),
                        "transformer":
                            transforms.Compose([
                                # After loading we normlaize the input data, which is fine.
                                # For training however, we want to transform it a bit and _then_ normalize it. Thus, we inverse the normalization first
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

try:
    base = models[0]["base_estimator"]().cuda()
    print(summary(base, (3, 32, 32)))
except:
    pass

run_experiments(basecfg, models, cuda_devices = cuda_devices, n_cores=len(cuda_devices))
