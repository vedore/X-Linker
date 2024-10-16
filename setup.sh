#!/usr/bin/env bash

yes | apt upgrade
yes | apt update
yes | apt-get install make
yes | apt install wget
yes | apt install curl
yes | apt install git
yes | apt install less
yes | apt install nano
yes | apt install unzip
yes | apt install gawk
yes | apt install libxml2-utils 
yes | apt-get install xmlstarlet
yes | apt-get install gzip
yes | apt-get install -y default-jdk && apt-get autoclean -y

# Install Pip requirements
pip install -r requirements.txt 


#--------------------------------------------------------------------------------
# Setup abbreviation detector
#--------------------------------------------------------------------------------
#Ab3P: https://github.com/ncbi-nlp/Ab3P

mkdir -p abbreviation_detector
cd abbreviation_detector/

#Get repositories
wget https://github.com/ncbi-nlp/Ab3P/archive/refs/heads/master.zip
unzip master.zip
mv Ab3P-master Ab3P
rm master.zip

wget https://github.com/ncbi-nlp/NCBITextLib/archive/refs/heads/master.zip
unzip master.zip
mv NCBITextLib-master NCBITextLib
rm master.zip

#Install 
yes | apt-get install g++

# 1. Install NCBITextLib
cd NCBITextLib/lib/
make

cd ../../

## 2. Install Ab3P
cd Ab3P
sed -i "s#\*\* location of NCBITextLib \*\*#../NCBITextLib#" Makefile
sed -i "s#\*\* location of NCBITextLib \*\*#../../NCBITextLib#" lib/Makefile
make

cd ../

cd ../