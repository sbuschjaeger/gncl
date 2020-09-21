#!/bin/bash

for f in SVHN CIFAR10; do
        echo "STARTING $f"
        cd $f 
        pwd 
        ./run.py 
        cd ..
        pwd 
done