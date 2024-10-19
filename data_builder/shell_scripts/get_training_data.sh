#!/usr/bin/env bash

#-----------------------------------------------------------------------------
# (Optional) Download Training corpora
#-----------------------------------------------------------------------------
cd data/
mkdir -p train/
cd train/
mkdir -p Disease Chemical Species Gene

wget https://zenodo.org/records/12704543/files/train.zip?download=1
unzip train.zip?download=1
rm train.zip?download=1

cd ../../