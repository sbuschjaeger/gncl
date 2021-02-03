#!/bin/bash

wget -r -np -R "index.html*" https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/ .
for f in *.tbz; 
do 
    echo "Extracting $f"; 
    tar -xjf $f
done
bzip2 *.bz2