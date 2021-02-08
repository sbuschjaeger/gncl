# Generalized Negative Correlation Learning for Deep Ensembling

This repository contains the experiments for our paper "Generalized Negative Correlation Learning for Deep Ensembles"

## Repository structure

This repository combines code for multiple datasets and two different repositories. It is structured as the following:

- `submodules/experiment_runner` Contains code for the `experiment_runner` package. This is a small tool which we use to run multiple experiments in our computing environment. Basically, the `experiment_runner` accepts a list of experiments and runs these either on multiple machines or in a multi-threaded environment. All results are written to a `*.jsonl`. This code is sparsely documented. For a commented example please have a look at `CIFAR100/run.py`.
- `submodules/Pysembles` Contains implementations of all the ensembling approaches we used for experiments. This package is fairly well documented. You can find docs under `submodules/Pysembles/docs`. 
- `DATASET/run.py` Contains code to run the experiments on the specific DATASET. 
- `DATASET/init.sh` Some datasets must be downloaded manually. In this case you can use the provided `init.sh` to do so.
- `DATASET/explore_data.ipynb` A simple jupyter notebook to explore the data. Available for some datasets.
- `explore_results.ipynb` A juypter notebook to plot the results
  
## How to use this code

You can find a conda `environment.yml` file which contains *some* dependencies (see below). You can build it via

    source /opt/anaconda3/bin/activate 
    conda env create -f environment.yml  --force
    conda activate gncl

Note the `--force` flag which overrides any existing conda environment with the name `gncl`. The development / experiments machine I used have `cuda 10.1` available (e.g. check via `nvidia-smi` or `nvcc --version`), so I needed to install PyTorch with CUDA 10.1. Usually you can do that via conda as mentioned on the PyTorch website (https://pytorch.org/get-started/locally/), but with this approach I would frequently run into some version mismatches (Not quite sure why, maybe conda caches some stuff). Thus, I manually installed PyTorch via pip after building the conda environment as suggested by the PyTorch-team:

`pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`

If you want to, you can also try to install it via conda. Then, just un-comment the specific lines in `environment.yml` and skip the pip installation. Last, you'll need to install the two previously mentioned submodules:

    pip install --user -e ./submodules/experiment_runner/
    pip install --user -e ./submodules/Pysembles/

Note, that the second call will invoke `nvcc` to compile some CUDA kernels. PyTorch internally respects `CUDA_HOME` which must point to your CUDA installation. On Ubuntu you can also install `nvidia-cuda-toolkit` via `sudo apt-get install nvidia-cuda-toolkit` which was enough for me to make this work. If you have installed CUDA manually, just set `CUDA_HOME` to the correct path and it should work. The `nvcc` is only required to compile some custom CUDA kernels for faster BNN training. If you do not care about the BNN part or you have problems with the `nvcc` you can

   1) Remove the CUDA part from `submodules/Pysembles/setup.py` (line 8 - 13), re-install the package and remove all BNN parts in corresponding `run.py`
   2) Remove the CUDA part from `submodules/Pysembles/setup.py` (line 8 - 13), activate the python-only BNNs in `submodules/Pysembles/pysembles/models/BinarisedNeuralNetworks.py` (comment line 14 and uncomment line 54 - 68) and re-install the package


## Running the experiments

For each dataset, you find a folder with a `run.py` file. For detailed information have a look at `CIFAR100/run.py` which explains the file structure in more detail. Each `run.py` configures the experiments for the corresponding dataset and will run _all_ experiments for it. Please make sure that the training data is available. Most `run.py` will automatically download the training / testing data once started (e.g. by using `torchvision.datasets`). Some folders contain an additional `init.sh` file which will download the training data for you if they are not available through PyTorch. 
To run multiple experiments on different machines we use [ray](https://github.com/ray-project/) which is wrapped in the `experiment_runner` script to fit our environment. Please note, that this assumes a shared folder where results are stored (e.g. by NFS). Setting up Ray can be a bit tricky, especially if versions do not match. In this case, feel free to hack around in `submodules/experiment_runner/experiment_runner_v2.py` (There is also `submodules/experiment_runner/experiment_runner.py` which is a bit dated). In any case, good luck! 
If you don't want to use Ray or only train a single model you can supply `--local` to `run.py` (default) which will schedule each experiment sequentially on your machine. Results are stored in a folder named after the current time and date in the specific data-set subfolder (e.g. `cifar100/13-10-2020-18:36:12`).

## Displaying results

You can explore the results and re-produce the plots from the paper by using the Jupyter notebook `explore_results`. The first cell in this notebook loads the desired results and the second / third cell produce the plots. For convenience, we included the `results.jsonl` file for each dataset (excluding additional meta data and model weights) so you can re-produce the plots from the paper / appendix. The notebook contains a fair amount of comments, so feel free to play around with it. Just make sure that the correct folder path is set. All plots in the paper / appendix have been created with this notebook.

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

For each dataset, you can find a folder with a `run.py` file. This file configures the experiments for the dataset and will run _all_ experiments for it. If you don't have the corresponding training / testing data in the same folder it will download the data as required (e.g. by using `torchvision.datasets`). To run multiple experiments distributed we use [ray](https://github.com/ray-project/) which is wrapped around a custom [experiment_runner](https://github.com/sbuschjaeger/experiment_runner) that fits our environment. Please note, that this assumes a shared folder where results are stored.
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