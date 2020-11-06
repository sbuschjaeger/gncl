# Generalized Negative Correlation Learning for Deep Ensembling

This repository contains the experiments for our paper "Generalized Negative Correlation Learning for Deep Ensembling" (https://arxiv.org/abs/2011.02952)

## How to use this code
This repository combines code from two other repositories as submodules, so you'll have to clone it recursively
`git clone --recurse-submodules git@github.com:sbuschjaeger/gncl.git`

You can find a conda `environment.yml` file which contains *some* dependencies (see below). You can build it via

    source /opt/anaconda3/bin/activate 
    conda env create -f environment.yml  --force
    conda activate gncl

Note the `--force` flag which overrides any existing conda environment with the name `gncl`. The development / experiments machine I used have `cuda 10.1` available (e.g. check via `nvidia-smi` or `nvcc --version`), so I needed to install PyTorch with CUDA 10.1. Usually you can do that via conda as mentioned on the PyTorch website (https://pytorch.org/get-started/locally/), but with this approach I would frequently run into some version mismatches (Not quite sure why, maybe conda caches some stuff). Thus, I manually installed PyTorch via pip after building the conda environment as suggested by the PyTorch-team:

`pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`

If you want to, you can also try to install it via conda. Then, just un-comment the specific lines in `environment.yml` and skip the pip installation. Last, you'll need to install the two previously mentioned submodules as packages which require PyTorch.

    cd gncl
    pip install --user -e ./submodules/experiment_runner/
    pip install --user -e ./submodules/Pysembles/

Note, that the second call will invoke `nvcc` to compile the cuda kernel. PyTorch internally respects `CUDA_HOME` which must point to your CUDA installation. On Ubuntu you can also install `nvidia-cuda-toolkit` via `sudo apt-get install nvidia-cuda-toolkit` which was enough for me to make this work. If you have installed CUDA manually, just set `CUDA_HOME` to the correct path and it should work.

## Running the experiments

For each dataset, you can find a folder with a `run.py` file. This file configures the experiments for the dataset and will run _all_ experiments for it. If you don't have the correspnding training / testing data in the same folder it will download the data as required (e.g. by using `torchvision.datasets`). To run multiple experiments distributed we use [ray](https://github.com/ray-project/) which is wrapped around a custom [experiment_runner](https://github.com/sbuschjaeger/experiment_runner) that fits our environment. Please note, that this assumes a shared folder where results are stored.
If you don't want to use Ray or only train a single model you can set `DEBUG = True` in `run.py` (default) which will schedule each experiment sequentially on your machine. Results are stored in a folder named after the current time and date in the specific data-set (e.g. `cifar100/13-10-2020-18:36:12`) and you can use the `explore_results` Jupyter notebook to view and plot the results after training. 

## Citing our Paper

    @misc{buschjäger2020generalized,
        title={Generalized Negative Correlation Learning for Deep Ensembling}, 
        author={Sebastian Buschjäger and Lukas Pfahler and Katharina Morik},
        year={2020},
        eprint={2011.02952},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }

## Acknowledgments 
Special Thanks goes to [Lukas Pfahler](https://github.com/Whadup) (lukas.pfahler@tu-dortmund.de) who probably found more bugs in my code than lines I wrote. 