#!/usr/bin/env bash

if [ "$#" -ne 1 ] 
then
    kaggle datasets download --unzip paultimothymooney/chest-xray-pneumonia
else
    kaggle datasets download -p $1 --unzip paultimothymooney/chest-xray-pneumonia  
fi
