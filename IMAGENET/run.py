#!/usr/bin/env python3

import sys
import os
import pickle
import tarfile
import glob
from datetime import datetime
from functools import partial

import warnings
import numpy as np
import pandas as pd
import torch
import scipy
import argparse
import PIL

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

from torchvision.models.densenet import DenseNet

#from ... import MobilenetV3
# sys.path.append("..")
from pysembles.Metrics import avg_accurcay,diversity,avg_loss,loss,accuracy
from pysembles.models.VGG import VGGNet
from pysembles.models.SimpleResNet import SimpleResNet
from pysembles.models.ResNet import ResNet18
from pysembles.models.MobileNetV3 import MobileNetV3
#from deep_ensembles_v2.
from pysembles.models.BinarisedNeuralNetworks import BinaryModel

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

class ImageNetWithProgressiveResizing(Dataset):
    def __init__(self, datasets, switch_every = 5):
        self.datasets = datasets
        self.stage = 0
        self.switch_every = switch_every
        self.epoch = 0

    def __getitem__(self, index):
        return self.datasets[self.stage].__getitem__(index)
    
    def next_stage(self):
        if self.stage < len(self.datasets) - 1:
            self.stage += 1

    def __len__(self):
        return self.datasets[self.stage].__len__()

    def end_of_epoch(self):
        self.epoch += 1
        if self.epoch % self.switch_every == 0:
            self.next_stage()

def train_transformation(image_size):
    return transforms.Compose([
        #transforms.Resize(image_size, PIL.Image.BICUBIC),
        #transforms.CenterCrop(image_size),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def test_transformation(image_size):
    return transforms.Compose([
        transforms.Resize(image_size, PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        if isinstance(cfg["train_data"], ImageNetWithProgressiveResizing):
            for i in range(model.cur_epoch):
                cfg["train_data"].end_of_epoch()
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
        "out_path":os.path.join("/data/d3/buschjae/gncl/imagenet", datetime.now().strftime('%d-%m-%Y-%H:%M:%S')),
        #"out_path":"/data/d3/buschjae/gncl/imagenet/12-11-2020-15:54:41",
        "pre": pre,
        "post": post,
        "fit": fit,
        "backend": "local",
        "verbose":False
    }
else:
    pass
    # basecfg = {
    #     "out_path":os.path.join("/data/d3/buschjae/gncl/imagenet", datetime.now().strftime('%d-%m-%Y-%H:%M:%S')),
    #     #"out_path":"/data/d3/buschjae/gncl/imagenet/12-11-2020-14:41:16",
    #     "pre": pre,
    #     "post": post,
    #     "fit": fit,
    #     "backend": "ray",
    #     "ray_head": args.ray_head,
    #     "redis_password": args.redis_password,
    #     "verbose":False
    # }

cuda_devices = [0]
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
    "epochs" : 70,
    "eps" : 1e-12,
    "betas" : (0.9,0.999)
}

loader = {
    "num_workers": 96, 
    "batch_size" : 256,
    "pin_memory": True
}

# print("CREATING TRAIN DATA")
# x_tensor = torch.rand((5000,3,224,224))
# y_tensor = torch.randint(low=0,high=1000,size=(5000,))
# devel_dataset = TransformTensorDataset(x_tensor,y_tensor)
# print("CREATING TRAIN DATA...DONE")

# There seems to be a single corrupted image in iamgenet data. This messes with our tqdm output. So ignore the warning.
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

def mobilenetv3():
    return MobileNetV3(mode="tiny", width_multiplier=1.0, dropout=0.0, BN_momentum=0.1, zero_gamma=False, in_channels = 3)

def densenet():
    return DenseNet(growth_rate = 12, block_config = (3, 6, 12, 8), num_classes = 1000, bn_size = 4)
    #return densenet121(pretrained=False,progress=False)

def stacking_classifier():
    return torch.nn.Linear(8*1000,1000)

models.append(
    {
        "model":Model,
        "base_estimator": mobilenetv3,
        "optimizer":optimizer,
        "scheduler":scheduler,
        "loader":loader,
        "eval_every":2,
        "store_every":10,
        "loss_function":nn.CrossEntropyLoss(reduction="none"),
        "use_amp":True,
        "device":"cuda",
        "train_data": 
           ImageNetWithProgressiveResizing(
                [
                    torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(64)),
                    torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(96)),
                    torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(128)),
                    torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(196)),
                    torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(224))
                ], switch_every=10
            ),
        "test_data": torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="val", transform = test_transformation(224)),
        "verbose":True
    }
)

n_estimators = 8

# models.append(
#     {
#         "model":StackingClassifier,
#         "base_estimator": mobilenetv3,
#         "classifier" : stacking_classifier,
#         "n_estimators":n_estimators,
#         "optimizer":optimizer,
#         "scheduler":scheduler,
#         "loader":loader,
#         "eval_every":2,
#         "store_every":10,
#         "loss_function":nn.CrossEntropyLoss(reduction="none"),
#         "use_amp":True,
#         "device":"cuda",
#         "train_data": 
#            ImageNetWithProgressiveResizing(
#                 [
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(64)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(96)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(128)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(196)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(224))
#                 ], switch_every=10
#             ),
#         "test_data": torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="val", transform = test_transformation(224)),
#         "verbose":True
#     }
# )

# models.append(
#     {
#         "model":GradientBoostedNets,
#         "n_estimators":n_estimators,
#         "combination_type":"average",
#         "base_estimator": mobilenetv3,
#         "optimizer":optimizer,
#         "scheduler":scheduler,
#         "loader":loader,
#         "eval_every":2,
#         "store_every":10,
#         "loss_function":nn.CrossEntropyLoss(reduction="none"),
#         "use_amp":True,
#         "device":"cuda",
#         "train_data": 
#            ImageNetWithProgressiveResizing(
#                 [
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(64)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(96)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(128)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(196)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(224))
#                 ], switch_every=10
#             ),
#         "test_data": torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="val", transform = test_transformation(224)),
#         "verbose":True,
#     }
# )

# models.append(
#     {
#         "model":SnapshotEnsembleClassifier,
#         "list_of_snapshots":[10,20,30,40,50,60,70],
#         "n_estimators":n_estimators,
#         "combination_type":"average",
#         "base_estimator": mobilenetv3,
#         "optimizer":optimizer,
#         "scheduler":scheduler,
#         "loader":loader,
#         "eval_every":2,
#         "store_every":10,
#         "loss_function":nn.CrossEntropyLoss(reduction="none"),
#         "use_amp":True,
#         "device":"cuda",
#         "train_data": 
#            ImageNetWithProgressiveResizing(
#                 [
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(64)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(96)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(128)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(196)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(224))
#                 ], switch_every=10
#             ),
#         "test_data": torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="val", transform = test_transformation(224)),
#         "verbose":True,
#     }
# )

# models.append(
#     {
#         "model":E2EEnsembleClassifier,
#         "n_estimators":n_estimators,
#         "combination_type":"average",
#         "base_estimator": mobilenetv3,
#         "optimizer":optimizer,
#         "scheduler":scheduler,
#         "loader":loader,
#         "eval_every":2,
#         "store_every":10,
#         "loss_function":nn.CrossEntropyLoss(reduction="none"),
#         "use_amp":True,
#         "device":"cuda",
#         "train_data": 
#            ImageNetWithProgressiveResizing(
#                 [
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(64)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(96)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(128)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(196)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(224))
#                 ], switch_every=10
#             ),
#         "test_data": torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="val", transform = test_transformation(224)),
#         "verbose":True,
#     }
# )

# models.append(
#     {
#         "model":SMCLClassifier,
#         "n_estimators":n_estimators,
#         "combination_type":"average",
#         "base_estimator": mobilenetv3,
#         "optimizer":optimizer,
#         "scheduler":scheduler,
#         "loader":loader,
#         "eval_every":2,
#         "store_every":10,
#         "loss_function":nn.CrossEntropyLoss(reduction="none"),
#         "use_amp":True,
#         "device":"cuda",
#         "train_data": 
#            ImageNetWithProgressiveResizing(
#                 [
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(64)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(96)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(128)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(196)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(224))
#                 ], switch_every=10
#             ),
#         "test_data": torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="val", transform = test_transformation(224)),
#         "verbose":True,
#     }
# )

# models.append(
#     {
#         "model":BaggingClassifier,
#         "n_estimators":n_estimators,
#         "combination_type":"average",
#         "base_estimator": mobilenetv3,
#         "optimizer":optimizer,
#         "scheduler":scheduler,
#         "loader":loader,
#         "eval_every":2,
#         "store_every":10,
#         "loss_function":nn.CrossEntropyLoss(reduction="none"),
#         "use_amp":True,
#         "device":"cuda",
#         "train_data": 
#            ImageNetWithProgressiveResizing(
#                 [
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(64)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(96)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(128)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(196)),
#                     torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(224))
#                 ], switch_every=10
#             ),
#         "test_data": torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="val", transform = test_transformation(224)),
#         "verbose":True,
#     }
# )

# for l_reg in [0, 0.2, 0.4, 0.6, 0.8, 1]: 
#     models.append(
#         {
#             "model":GNCLClassifier,
#             "n_estimators":n_estimators,
#             "mode":"upper",
#             "l_reg":l_reg,
#             "base_estimator": mobilenetv3,
#             "optimizer":optimizer,
#             "scheduler":scheduler,
#             "loader":loader,
#             "eval_every":2,
#             "store_every":10,
#             "loss_function":nn.CrossEntropyLoss(reduction="none"),
#             "use_amp":True,
#             "device":"cuda",
#             "train_data": 
#                 ImageNetWithProgressiveResizing(
#                     [
#                         #torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(64)),
#                         #torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(96)),
#                         torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(128)),
#                         torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(196)),
#                         torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="train", transform = train_transformation(224))
#                     ], switch_every=10
#                 ),
#             "test_data": torchvision.datasets.ImageNet("/data/d2/pfahler/imagenet", split="val", transform = test_transformation(224)),
#             "verbose":True,
#         }
#     )

base = models[0]["base_estimator"]().cuda()
rnd_input = torch.rand((1, 3, 224, 224)).cuda()
print(summary(base, rnd_input))

run_experiments(basecfg, models)
