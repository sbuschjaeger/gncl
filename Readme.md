# Generalized Negative Correlation Learning for Deep Ensembling
This repoistory contains code do train Deep Learning Ensembles on CIFAR10/100, FACT, FashionMNIST and SVHN. It's a - more or less - random collection of code for various experiments. 

In general there is code for training

- Single models with an sklearn-like API supporting (a primitive version of) pipelining
- Autoencoders
- Soft Decision Trees
- Binarized Neural Networks. Many thanks to Mikail Yayla (mikail.yayla@tu-dortmund.de) for providing CUDA kernels for BNN training. He maintains a more evolved repository on BNNs - check it out at https://github.com/myay/BFITT

Each "baselearner" can be combined into ensembles via
- Bagging 
- Negative Correlation Learning
- Gradient Boosting
- Stacking
- End2End Training


# How to use this code
This repository combines code from two other repoistories as submodules, so you'll have to clone it recursivley
`git clone --recurse-submodules git@github.com:sbuschjaeger/bnn.git`

You can find a conda `environment.yml` file which contains *some* dependencies (see below). You can built it via

    source /opt/anaconda3/bin/activate 
    conda env create -f environment.yml  --force
    conda activate bnn

Note the `--force` flag which overrides any existing conda environment with the name `bnn`. The development / experiments machine I used have `cuda 10.1` available (e.g. check via `nvidia-smi` or `nvcc --version`), so I needed to install PyTorch with cuda 10.1. Usually you can do that via conda as mentioned on the PyTorch website (https://pytorch.org/get-started/locally/), but with this approach I would frequently run into some version mismatches (Not quite sure why, maybe conda caches some stuff). Thus, I manually installed PyTorch via pip after bulding the conda environment as suggested by the PyTorch-team:

`pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`

If you want to, you can also try to install it via conda. Then, just un-comment the specific lines in `environment.yml` and skip the pip installation. Last, you'll need to install the two previously mentioned submodules as packages which require PyTorch.

    cd bnn
    pip install --user -e ./submodules/experiment_runner/
    pip install --user -e ./submodules/deep_ensembles_v2/

Note, that the second call will invoke `nvcc` to compile the cuda kernel. PyTorch internally respects `CUDA_HOME` which must point to your cuda installation. On ubuntu you can also install `nvidia-cuda-toolkit` via `sudo apt-get install nvidia-cuda-toolkit` which was enough for me to make this work. If you have installed cuda manually, just set `CUDA_HOME` to the correct path and it should work.


## Random stuff
Currently base_learners of all learners are expected to be functions returning a new object of the base_learner and _not_ the baselearner itself. However, this is not true for the binarized wrapper