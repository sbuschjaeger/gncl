#!/usr/bin/env python3

import os
import sys
import pickle
import tarfile
from datetime import datetime
from functools import partial
import argparse
import glob

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
In this script we train ensembles on CIFAR100 with minimal data augmentation

We configure an experiment by three common functions which are called via the experiment_runner wrapper and some additional base information. 
Each experiment is configured through a dictionary. Apart from a few reserved keywords (e.g. pre/post/fit) every other field is simply passed to 
to each function and can be used as desired. The three functions are
    - pre: Everything which needs to be done before running the experiments
    - fit: Running the actual experiments
    - post: Everything which needs to be done after the experiment
'''

# Perform data augmentation for training data
# Constants for data normalization are taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py 
def train_transformation():
    return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

# Only normalize the testing data without further augmentation 
# Constants for data normalization are taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py 
def test_transformation():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

# PyTorch's MSE does not support classification per se. For our MSE experiments we use this simple function
def one_hot_mse(output, target):
    one_hot = torch.nn.functional.one_hot(target, num_classes = 100)
    loss = (output - one_hot)**2
    return loss.sum(axis=1)

# In order to compute the ensembles diversity, we need to compute the hessian of for the chosen loss. 
# This is an implementation for the MSE which supports batched predictions. The diversity implementations for other loss
# functions (compatible with PyTorch) can be found in Metrics.py. 
#   model: The ensemble for which we want to get the diversity
#   data_loader: The data_loder which contains the data
def mse_diversity(model, data_loader):
    if not hasattr(model, "estimators_"):
        return 0

    model.eval()
    diversities = []
    for batch in data_loader:
        data, target = batch[0], batch[1]
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        with torch.no_grad():
            f_bar, base_preds = model.forward_with_base(data)
        
        n_classes = f_bar.shape[1]
        n_preds = f_bar.shape[0]

        eye_matrix = torch.eye(n_classes).repeat(n_preds, 1, 1).cuda()
        D = 2.0*eye_matrix

        batch_diversities = []
        for pred in base_preds:
            diff = pred - f_bar 
            covar = torch.bmm(diff.unsqueeze(1), torch.bmm(D, diff.unsqueeze(2))).squeeze()
            div = 1.0/model.n_estimators * 0.5 * covar
            batch_diversities.append(div)

        diversities.append(torch.stack(batch_diversities, dim = 1))

    div = torch.cat(diversities,dim=0)
    return div.sum(dim=1).mean(dim=0).item()

# This function is called by the experiment_runner to prepare an experiment.
# It expects each experiment configuration to contain a "model" field which is the c'tor of a new model. 
# For convenience, this function checks for any matching c'tor argument in the current experiment config 
# and creates the approprate model. Note however, that get_ctor_arguments sometimes does not return all 
# arguments for inheritance (??). 
def pre(cfg):
    model_ctor = cfg.pop("model")
    tmpcfg = cfg
    expected = {}
    for key in get_ctor_arguments(model_ctor):
        if key in tmpcfg:
            expected[key] = tmpcfg[key]
    
    model = model_ctor(**expected)
    return model

# This function computes statistics after the model has been fit. 
# In this case, it simply stores accuracy / loss / no. parameters as well as the diversity values
def post(cfg, model):
    scores = {}
    train_loader = torch.utils.data.DataLoader(cfg["train_data"], **cfg["loader"])
    scores["train_loss"] = loss(model, train_loader)
    scores["train_accuracy"] = accuracy(model, train_loader)
    # if scores["train_loss"] == one_hot_mse:
    if cfg["loss_function"] == one_hot_mse:
        scores["train_diversity"] = mse_diversity(model, train_loader)
    else:
        scores["train_diversity"] = diversity(model, train_loader)
    scores["train_loss"] = loss(model, train_loader)
    scores["train_avg_loss"] = avg_loss(model, train_loader)
    scores["train_avg_accurcay"] = avg_accurcay(model, train_loader)

    test_loader = torch.utils.data.DataLoader(cfg["test_data"], **cfg["loader"])
    if cfg["loss_function"] == one_hot_mse:
    #if scores["test_loss"] == one_hot_mse:
        scores["test_diversity"] = mse_diversity(model, test_loader)
    else:
        scores["test_diversity"] = diversity(model, test_loader)
    scores["test_loss"] = loss(model, test_loader)
    scores["test_accuracy"] = accuracy(model, test_loader)
    scores["test_loss"] = loss(model, test_loader)
    scores["test_avg_loss"] = avg_loss(model, test_loader)
    scores["test_avg_accurcay"] = avg_accurcay(model, test_loader)
    scores["params"] = pytorch_total_params(model)

    return scores

# This method implements the actual fitting of the model. Basic support for resuming checkpoints is also implemented.
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

# Gather some input configuration
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--local", help="Run on local machine",action="store_true", default=False)
parser.add_argument("-r", "--ray", help="Run via Ray",action="store_true", default=False)
parser.add_argument("--ray_head", help="Run via Ray",action="store_true", default="auto")
parser.add_argument("--redis_password", help="Run via Ray",action="store_true", default="5241590000000000")
args = parser.parse_args()

if (args.local and args.ray) or (not args.local and not args.ray):
    print("Either you specified to use both, ray _and_ local mode or you specified to use none of both. Please choose either. Defaulting to `local` processing.")
    args.local = True


'''
Check if we train on a single machine or via Ray and create the basic configuration for it. The basic configuration is shared across all experiments:
basecfg = {
    "out_path": Path where results should be stored
    "pre": The pre-function
    "post": The post-function
    "fit": The fit-function
    "backend": The running mode-, e.g. "local"/"ray"/"multiprocessing"
    "verbose": True / False if outputs should be printed
    "param_1": Additional parameter 1 required by the backend, e.g. the "ray_head"
    "param_2": Additional parameter 2 required by the backend, e.g. the "redis_password" etc
}
'''
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
    basecfg = {
        "out_path":os.path.join("results", datetime.now().strftime('%d-%m-%Y-%H:%M:%S')),
        "pre": pre,
        "post": post,
        "fit": fit,
        "backend": "ray",
        "ray_head": args.ray_head,
        "redis_password": args.redis_password,
        "verbose":False
    }

# This is a list of all model configurations we want to train
models = []

# This is the configuration of the learning rate scheduler used in all experiments
scheduler = {
    "method" : torch.optim.lr_scheduler.StepLR,
    "step_size" : 50,
    "gamma": 0.5
}

# This is the configuration of the optimizer used in all experiments
optimizer = {
    "method" : AdaBelief,
    "lr" : 1e-3,
    "epochs" : 200,
    "eps" : 1e-12,
    "betas" : (0.9,0.999)
}

# This is the configuration of the data loader used in all experiments
loader = {
    "num_workers": 1, 
    "batch_size" : 256,
    "pin_memory": True
}

# This is the ResNet architecture which is used as a base model in all experiments. 
# This function returns a new model depending on the size and type (float / binary) given.
def simpleresnet(size, model_type):
    if "small" == size:
        n_channels = 32
        depth = 4
    else:
        n_channels = 96
        depth = 4

    if "binary" == model_type:
        return BinaryModel(SimpleResNet(in_channels = 3, n_channels = n_channels, depth = depth, num_classes=100), keep_activation=True)
    else:
        return SimpleResNet(in_channels = 3, n_channels = n_channels, depth = depth, num_classes=100)

# This function constructs a classifier model for stacking. In this case its a simple linear layer.
def stacking_classifier(model_type):
    classifier = torch.nn.Linear(16*100,100)

    if "binary" == model_type:
        return BinaryModel(classifier, keep_activation=True)
    else:
        return classifier

# This is the loss function used in all experiments.
loss_function = one_hot_mse
#loss_function = nn.CrossEntropyLoss(reduction="none")

'''
These are the actual methods / experiments we want to run with all necessary options. Regardless of the implementation, the exact parameters should be sufficient to reproduce the results. The options are as the following:
model.append(
    {
        "model": The ensemble method, e.g, Bagging
        "n_estimators": The number of estimators, e.g. 16
        "param_1": Additional parameter 1 for "model"
        "param_2": Additional parameter 2 for "model" and so on
        "base_estimator": A function which creates a new base learner for the ensemble 
        "optimizer": A dictionary containing all configs for the optimizer
        "scheduler": A dictionary containing all configs for the scheduler
        "loader": A dictionary containing all configs for the data loader
        "eval_every": if test data is supplied, we want to evaluate our model on it every "eval_every" epochs
        "store_every": Store a checkpoint of the model every "store_every" epochs
        "loss_function": The loss function to be minimized
        "use_amp": True / False if PyTorch mixed precision should be used
        "device": The device ("cpu" / "cuda") used for training
        "train_data": The torch dataset containing the training data
        "test_data": The torch dataset containing the testing data
        "verbose":True / False if a TQDM progress bar should be shown
    }
)
'''
for s, t in [ ("small", "float"), ("large", "float"), ("small", "binary")]: 
    for m in [16]:
        models.append(
            {
                "model":BaggingClassifier,
                "n_estimators":m,
                "train_method":"fast",
                "base_estimator": partial(simpleresnet, size=s, model_type=t),
                "optimizer":optimizer,
                "scheduler":scheduler,
                "loader":loader,
                "eval_every":5,
                "store_every":0,
                "loss_function":loss_function,
                "use_amp":False,
                "device":"cuda",
                "train_data": torchvision.datasets.CIFAR100(".", train=True, transform = train_transformation()),
                "test_data": torchvision.datasets.CIFAR100(".", train=False, transform = test_transformation()),
                "verbose":True
            }
        )

        models.append(
            {
                "model":StackingClassifier,
                "base_estimator": partial(simpleresnet, size=s, model_type=t),
                "classifier" : partial(stacking_classifier, model_type=t),
                "n_estimators":m,
                "optimizer":optimizer,
                "scheduler":scheduler,
                "loader":loader,
                "eval_every":5,
                "store_every":0,
                "loss_function":loss_function,
                "use_amp":False,
                "device":"cuda",
                "train_data": torchvision.datasets.CIFAR100(".", train=True, transform = train_transformation()),
                "test_data": torchvision.datasets.CIFAR100(".", train=False, transform = test_transformation()),
                "verbose":True
            }
        )

        models.append(
            {
                "model":GradientBoostedNets,
                "n_estimators":m,
                "base_estimator": partial(simpleresnet, size=s, model_type=t),
                "optimizer":optimizer,
                "scheduler":scheduler,
                "loader":loader,
                "eval_every":5,
                "store_every":0,
                "loss_function":loss_function,
                "use_amp":False,
                "device":"cuda",
                "train_data": torchvision.datasets.CIFAR100(".", train=True, transform = train_transformation()),
                "test_data": torchvision.datasets.CIFAR100(".", train=False, transform = test_transformation()),
                "verbose":True
            }
        )

        models.append(
            {
                "model":SnapshotEnsembleClassifier,
                "n_estimators":m,
                "list_of_snapshots":[2,3,4,5,10,15,20,25,30,40,50,60,70,80,90],
                "base_estimator": partial(simpleresnet, size=s, model_type=t),
                "optimizer":optimizer,
                "scheduler":scheduler,
                "loader":loader,
                "eval_every":5,
                "store_every":0,
                "loss_function":loss_function,
                "use_amp":False,
                "device":"cuda",
                "train_data": torchvision.datasets.CIFAR100(".", train=True, transform = train_transformation()),
                "test_data": torchvision.datasets.CIFAR100(".", train=False, transform = test_transformation()),
                "verbose":True
            }
        )

        models.append(
            {
                "model":E2EEnsembleClassifier,
                "n_estimators":m,
                "base_estimator": partial(simpleresnet, size=s, model_type=t),
                "optimizer":optimizer,
                "scheduler":scheduler,
                "loader":loader,
                "eval_every":5,
                "store_every":0,
                "loss_function":loss_function,
                "use_amp":False,
                "device":"cuda",
                "train_data": torchvision.datasets.CIFAR100(".", train=True, transform = train_transformation()),
                "test_data": torchvision.datasets.CIFAR100(".", train=False, transform = test_transformation()),
                "verbose":True
            }
        )

        models.append(
            {
                "model":SMCLClassifier,
                "n_estimators":m,
                "combination_type":"best",
                "base_estimator": partial(simpleresnet, size=s, model_type=t),
                "optimizer":optimizer,
                "scheduler":scheduler,
                "loader":loader,
                "eval_every":5,
                "store_every":0,
                "loss_function":loss_function,
                "use_amp":False,
                "device":"cuda",
                "train_data": torchvision.datasets.CIFAR100(".", train=True, transform = train_transformation()),
                "test_data": torchvision.datasets.CIFAR100(".", train=False, transform = test_transformation()),
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
                    "base_estimator": partial(simpleresnet, size=s, model_type=t),
                    "optimizer":optimizer,
                    "scheduler":scheduler,
                    "loader":loader,
                    "eval_every":5,
                    "store_every":0,
                    "loss_function":loss_function,
                    "use_amp":False,
                    "device":"cuda",
                    "train_data": torchvision.datasets.CIFAR100(".", train=True, transform = train_transformation()),
                    "test_data": torchvision.datasets.CIFAR100(".", train=False, transform = test_transformation()),
                    "verbose":True
                }
            )

    models.append(
        {
            "model":Model,
            "base_estimator": partial(simpleresnet, size=s, model_type=t),
            "optimizer":optimizer,
            "scheduler":scheduler,
            "loader":loader,
            "eval_every":5,
            "store_every":0,
            "loss_function":loss_function,
            "use_amp":False,
            "device":"cuda",
            "train_data": torchvision.datasets.CIFAR100(".", train=True, transform = train_transformation(), download = True),
            "test_data": torchvision.datasets.CIFAR100(".", train=False, transform = test_transformation(), download = True),
            "verbose":True
        }
    )

# When playing around with different base-learners it is easy to miss-configure the first linear layer (after all the conv layers)
# In that case it might be helpful to 
#   1) Check the base model before running any experiments
#   2) Print additional statistics about the size of the base learners to make sure the entire ensemble fits our GPU
try:
    base = models[0]["base_estimator"]().cuda()
    rnd_input = torch.rand((1, 3, 32, 32)).cuda()
    print(summary(base, rnd_input))
except:
    pass

# Run the experiments with the given basic configuration (shared across all experiments) and the specific configuration
run_experiments(basecfg, models)