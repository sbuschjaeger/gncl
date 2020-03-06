#!/usr/bin/env python3

# Modified version of https://github.com/kbruegge/cnn_cherenkov/blob/master/convert.py

import numpy as np
import photon_stream as ps
from tqdm import tqdm
import h5py
from fact.io import initialize_h5py, append_to_h5py
from astropy.table import Table
import click
import pickle
from joblib import Parallel, delayed
from itertools import islice

@click.command()
@click.argument('in_files', nargs=-1)
@click.argument('out_path')
@click.option('--n_samples', default=200000)
@click.option('--ratio', default=0.7)
def main(in_files, out_path, n_samples, ratio):
    '''
    Reads all photon_stream files and converts them to images.
    '''
    files = sorted(in_files)

    print("Loading files")
    x_train = []
    y_train = []
    x_test = []
    y_test = [] 
    for fs in tqdm(files):
        y = 1 if "proton" in fs else 0

        images = np.load(fs, allow_pickle = True)
        np.random.shuffle(images)
        
        if n_samples > images.shape[0]:
            n_samples = images.shape[0]
        
        n_train = int(ratio*n_samples)
        x_train.extend(images[0:n_train, :])
        y_train.extend([y for i in range(n_train)]) 

        x_test.extend(images[n_train:n_samples, :])
        y_test.extend([y for i in range(n_samples-n_train)]) 

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print("x_train:", x_train.shape)        
    print("y_train:", y_train.shape)        
    print("x_test:", x_test.shape)        
    print("y_test:", y_test.shape) 

    print("Storing files")
    np.save("{}/x_train.npy".format(out_path), x_train)
    np.save("{}/y_train.npy".format(out_path), y_train)
    np.save("{}/x_test.npy".format(out_path), x_test)
    np.save("{}/y_test.npy".format(out_path), y_test)
    # X = np.array(X[:n_samples+int(0.5*n_samples),:])
    # Y = np.array(Y[:n_samples+int(0.5*n_samples)])

    # x_train = X[]

    # Y = np.array(Y)
    # print(X.shape)
    # print(Y.shape)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True, stratify=True)
    # df = pd.DataFrame()
    # df["X"] = np.vstack(X)
    # df["Y"] = np.array(Y)
    # print( df.head() )

if __name__ == '__main__':
    main()