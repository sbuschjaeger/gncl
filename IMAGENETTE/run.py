#!/usr/bin/env python3

import os
import sys
import pickle
import tarfile
from datetime import datetime
from functools import partial
import argparse
import glob
import PIL

import numpy as np
import pandas as pd
import torch
import scipy

import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torchsummaryX import summary

# Lets try the newest shit https://github.com/juntang-zhuang/Adabelief-Optimizer
from adabelief_pytorch import AdaBelief

from sklearn.metrics import make_scorer, accuracy_score

from pysembles.Utils import Flatten, Clamp, Scale

from pysembles.Models import Model
from pysembles.E2EEnsembleClassifier import E2EEnsembleClassifier
from pysembles.BaggingClassifier import BaggingClassifier
from pysembles.GNCLClassifier import GNCLClassifier
from pysembles.StackingClassifier import StackingClassifier
from pysembles.DeepDecisionTreeClassifier import DeepDecisionTreeClassifier
from pysembles.SMCLClassifier import SMCLClassifier
from pysembles.GradientBoostedNets import GradientBoostedNets
from pysembles.SnapshotEnsembleClassifier import SnapshotEnsembleClassifier

from pysembles.models.BinarisedNeuralNetworks import BinaryConv2d, BinaryLinear, BinaryTanh
from pysembles.Utils import pytorch_total_params, apply_in_batches, TransformTensorDataset

from experiment_runner.experiment_runner_v2 import run_experiments, get_ctor_arguments

#from ... import MobilenetV3
# sys.path.append("..")
from pysembles.Metrics import accuracy,avg_accurcay,diversity,avg_loss,loss
from pysembles.models.VGG import VGGNet
from pysembles.models.SimpleResNet import SimpleResNet
from pysembles.models.MobileNetV3 import MobileNetV3
from pysembles.models.BinarisedNeuralNetworks import BinaryModel

'''
Mean std have been computed this way 
dataset = torchvision.datasets.ImageFolder("/data/s1/buschjae/IMAGENETTE/imagenette2-160", transform = test_transformation()) #, transform = train_transformation()
loader = torch.utils.data.DataLoader(dataset,batch_size=10,num_workers=0,shuffle=False)

mean = 0.
std = 0.
for images, _ in loader:
    batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)

mean /= len(loader.dataset)
std /= len(loader.dataset)

print("MEAN: ", mean)
print("STD: ", std)
'''

def train_transformation():
    return transforms.Compose([
        transforms.RandomCrop(160, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(128, PIL.Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4660, 0.4580, 0.4296], std=[0.2376, 0.2316, 0.2393])
    ])

def test_transformation():
    return transforms.Compose([
        transforms.CenterCrop(160),
        transforms.Resize(128, PIL.Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4660, 0.4580, 0.4296], std=[0.2376, 0.2316, 0.2393])
    ])

def pre(cfg):
    model_ctor = cfg.pop("model")
    tmpcfg = cfg
    expected = {}
    for key in get_ctor_arguments(model_ctor):
        if key in tmpcfg:
            expected[key] = tmpcfg[key]
    
    model = model_ctor(**expected)
    return model

def post(cfg, model):
    scores = {}
    train_loader = torch.utils.data.DataLoader(cfg["train_data"], **cfg["loader"])
    scores["train_loss"] = loss(model, train_loader)
    scores["train_accuracy"] = accuracy(model, train_loader)
    scores["train_diversity"] = diversity(model, train_loader)
    scores["train_loss"] = loss(model, train_loader)
    scores["train_avg_loss"] = avg_loss(model, train_loader)
    scores["train_avg_accurcay"] = avg_accurcay(model, train_loader)

    test_loader = torch.utils.data.DataLoader(cfg["test_data"], **cfg["loader"])
    scores["test_loss"] = loss(model, test_loader)
    scores["test_accuracy"] = accuracy(model, test_loader)
    scores["test_diversity"] = diversity(model, test_loader)
    scores["test_loss"] = loss(model, test_loader)
    scores["test_avg_loss"] = avg_loss(model, test_loader)
    scores["test_avg_accurcay"] = avg_accurcay(model, test_loader)
    scores["params"] = pytorch_total_params(model)

    return scores

def fit(cfg, model):
    checkpoints = glob.glob(os.path.join(cfg["out_path"], "*.tar"))
    if len(checkpoints) > 0:
        print("Found some checkpoints - loading!")
        epochs = [ (int(os.path.basename(fname)[:-4].split("_")[1]), fname) for fname in checkpoints]
        
        # Per default python checks for the first argument in tuples. 
        _ , checkpoint_to_load = max(epochs)
        print("Loading {}".format(checkpoint_to_load))

        checkpoint = torch.load(checkpoint_to_load)
        model.restore_checkoint(checkpoint_to_load)
        model.epochs = cfg["optimizer"]["epochs"]

    model.fit(cfg["train_data"])
    return model

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--local", help="Run on local machine",action="store_true", default=False)
parser.add_argument("-r", "--ray", help="Run via Ray",action="store_true", default=False)
parser.add_argument("--ray_head", help="Run via Ray",action="store_true", default="auto")
parser.add_argument("--redis_password", help="Run via Ray",action="store_true", default="5241590000000000")
args = parser.parse_args()

if (args.local and args.ray) or (not args.local and not args.ray):
    print("Either you specified to use both, ray _and_ local mode or you specified to use none of both. Please choose either. Defaulting to `local` processing.")
    args.local = True


if args.local:
    basecfg = {
        "out_path":os.path.join(datetime.now().strftime('%d-%m-%Y-%H:%M:%S')),
        "pre": pre,
        "post": post,
        "fit": fit,
        "backend": "local",
        "verbose":False
    }
else:
    pass
    # basecfg = {
    #     "out_path":os.path.join("FASHION", "results", datetime.now().strftime('%d-%m-%Y-%H:%M:%S')),
    #     "pre": pre,
    #     "post": post,
    #     "fit": fit,
    #     "backend": "ray",
    #     "ray_head": args.ray_head,
    #     "redis_password": args.redis_password,
    #     "verbose":False
    # }


models = []

scheduler = {
    "method" : torch.optim.lr_scheduler.StepLR,
    "step_size" : 25,
    "gamma": 0.5
}

optimizer = {
    #"method" : AdaBelief, # torch.optim.Adam, #if "binary" in t else torch.optim.SGD,
    "method" : AdaBelief,
    "lr" : 1e-2, #1e-3, #if "binary" in t else 0.1,
    # "momentum" : 0.9,
    # "nesterov" : True,
    # "weight_decay" : 1e-4, 
    "epochs" : 150,
    "eps" : 1e-12,
    "betas" : (0.9,0.999)
}

loader = {
    "num_workers": 1, 
    "batch_size" : 256,
    "pin_memory": True
}

def simpleresnet(size, model_type):
    if "small" == size:
        n_channels = 32
        depth = 4
    else:
        n_channels = 96
        depth = 4

    if "binary" == model_type:
        return BinaryModel(SimpleResNet(in_channels = 3, n_channels = n_channels, depth = depth, num_classes=10, lin_size = 36*n_channels), keep_activation=True)
    else:
        return SimpleResNet(in_channels = 3, n_channels = n_channels, depth = depth, num_classes=10, lin_size = 36*n_channels)

def mobilenetv3(size, model_type):
    if "binary" == model_type:
        return BinaryModel(MobileNetV3(classes_num = 10, mode=size, width_multiplier=1.0, dropout=0.0, BN_momentum=0.1, zero_gamma=False, in_channels = 3), keep_activation=True)
    else:
        return MobileNetV3(classes_num = 10, mode=size, width_multiplier=1.0, dropout=0.0, BN_momentum=0.1, zero_gamma=False, in_channels = 3)

def stacking_classifier(model_type):
    classifier = torch.nn.Linear(16*10,10)

    if "binary" == model_type:
        return BinaryModel(classifier, keep_activation=True)
    else:
        return classifier

for s in ["tiny", "small"]: #"large"
    for t in ["float", "binary"]: 
        models.append(
            {
                "model":Model,
                "base_estimator": partial(mobilenetv3, size=s, model_type=t),
                "optimizer":optimizer,
                "scheduler":scheduler,
                "loader":loader,
                "eval_every":5,
                "store_every":0,
                "loss_function":nn.CrossEntropyLoss(reduction="none"),
                "use_amp":True,
                "device":"cuda",
                "train_data": torchvision.datasets.ImageFolder("/data/s1/buschjae/IMAGENETTE/imagenette2-160/train", transform = train_transformation()),
                "test_data": torchvision.datasets.ImageFolder("/data/s1/buschjae/IMAGENETTE/imagenette2-160/val", transform = test_transformation()),
                "verbose":True
            }
        )

        for m in [16]:
            models.append(
                {
                    "model":StackingClassifier,
                    "base_estimator": partial(mobilenetv3, size=s, model_type=t),
                    "classifier" : partial(stacking_classifier, model_type=t),
                    "n_estimators":m,
                    "optimizer":optimizer,
                    "scheduler":scheduler,
                    "loader":loader,
                    "eval_every":5,
                    "store_every":0,
                    "loss_function":nn.CrossEntropyLoss(reduction="none"),
                    "use_amp":True,
                    "device":"cuda",
                    "train_data": torchvision.datasets.ImageFolder("/data/s1/buschjae/IMAGENETTE/imagenette2-160/train", transform = train_transformation()),
                    "test_data": torchvision.datasets.ImageFolder("/data/s1/buschjae/IMAGENETTE/imagenette2-160/val", transform = test_transformation()),
                    "verbose":True
                }
            )

            models.append(
                {
                    "model":BaggingClassifier,
                    "n_estimators":m,
                    "train_method":"fast",
                    "base_estimator": partial(mobilenetv3, size=s, model_type=t),
                    "optimizer":optimizer,
                    "scheduler":scheduler,
                    "loader":loader,
                    "eval_every":5,
                    "store_every":0,
                    "loss_function":nn.CrossEntropyLoss(reduction="none"),
                    "use_amp":True,
                    "device":"cuda",
                    "train_data": torchvision.datasets.ImageFolder("/data/s1/buschjae/IMAGENETTE/imagenette2-160/train", transform = train_transformation()),
                    "test_data": torchvision.datasets.ImageFolder("/data/s1/buschjae/IMAGENETTE/imagenette2-160/val", transform = test_transformation()),
                    "verbose":True
                }
            )

            models.append(
                {
                    "model":GradientBoostedNets,
                    "n_estimators":m,
                    "base_estimator": partial(mobilenetv3, size=s, model_type=t),
                    "optimizer":optimizer,
                    "scheduler":scheduler,
                    "loader":loader,
                    "eval_every":5,
                    "store_every":0,
                    "loss_function":nn.CrossEntropyLoss(reduction="none"),
                    "use_amp":True,
                    "device":"cuda",
                    "train_data": torchvision.datasets.ImageFolder("/data/s1/buschjae/IMAGENETTE/imagenette2-160/train", transform = train_transformation()),
                    "test_data": torchvision.datasets.ImageFolder("/data/s1/buschjae/IMAGENETTE/imagenette2-160/val", transform = test_transformation()),
                    "verbose":True
                }
            )

            models.append(
                {
                    "model":SnapshotEnsembleClassifier,
                    "n_estimators":m,
                    "list_of_snapshots":[2,3,4,5,10,15,20,25,30,40,50,60,70,80,90],
                    "base_estimator": partial(mobilenetv3, size=s, model_type=t),
                    "optimizer":optimizer,
                    "scheduler":scheduler,
                    "loader":loader,
                    "eval_every":5,
                    "store_every":0,
                    "loss_function":nn.CrossEntropyLoss(reduction="none"),
                    "use_amp":True,
                    "device":"cuda",
                    "train_data": torchvision.datasets.ImageFolder("/data/s1/buschjae/IMAGENETTE/imagenette2-160/train", transform = train_transformation()),
                    "test_data": torchvision.datasets.ImageFolder("/data/s1/buschjae/IMAGENETTE/imagenette2-160/val", transform = test_transformation()),
                    "verbose":True
                }
            )

            models.append(
                {
                    "model":E2EEnsembleClassifier,
                    "n_estimators":m,
                    "base_estimator": partial(mobilenetv3, size=s, model_type=t),
                    "optimizer":optimizer,
                    "scheduler":scheduler,
                    "loader":loader,
                    "eval_every":5,
                    "store_every":0,
                    "loss_function":nn.CrossEntropyLoss(reduction="none"),
                    "use_amp":True,
                    "device":"cuda",
                    "train_data": torchvision.datasets.ImageFolder("/data/s1/buschjae/IMAGENETTE/imagenette2-160/train", transform = train_transformation()),
                    "test_data": torchvision.datasets.ImageFolder("/data/s1/buschjae/IMAGENETTE/imagenette2-160/val", transform = test_transformation()),
                    "verbose":True
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
                    "loader":loader,
                    "eval_every":5,
                    "store_every":0,
                    "loss_function":nn.CrossEntropyLoss(reduction="none"),
                    "use_amp":True,
                    "device":"cuda",
                    "train_data": torchvision.datasets.ImageFolder("/data/s1/buschjae/IMAGENETTE/imagenette2-160/train", transform = train_transformation()),
                    "test_data": torchvision.datasets.ImageFolder("/data/s1/buschjae/IMAGENETTE/imagenette2-160/val", transform = test_transformation()),
                    "verbose":True
                }
            )

            for l_reg in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: 
                models.append(
                    {
                        "model":GNCLClassifier,
                        "n_estimators":m,
                        "mode":"upper",
                        "l_reg":l_reg,
                        "combination_type":"average",
                        "base_estimator": partial(mobilenetv3, size=s, model_type=t),
                        "optimizer":optimizer,
                        "scheduler":scheduler,
                        "loader":loader,
                        "eval_every":5,
                        "store_every":0,
                        "loss_function":nn.CrossEntropyLoss(reduction="none"),
                        "use_amp":True,
                        "device":"cuda",
                        "train_data": torchvision.datasets.ImageFolder("/data/s1/buschjae/IMAGENETTE/imagenette2-160/train", transform = train_transformation()),
                        "test_data": torchvision.datasets.ImageFolder("/data/s1/buschjae/IMAGENETTE/imagenette2-160/val", transform = test_transformation()),
                        "verbose":True
                    }
                )

try:
    base = models[0]["base_estimator"]().cuda()
    rnd_input = torch.rand((1, 3, 128, 128)).cuda()
    print(summary(base, rnd_input))
except:
    pass

run_experiments(basecfg, models)
