#!/usr/bin/env bash

#-----------------------------------------------------------------------------
# (Optional) Download PubTator 3 Central annotations
#-----------------------------------------------------------------------------
cd data/
mkdir -p pubtator/
cd pubtator/

##Disease
wget https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3/disease2pubtator3.gz 
gzip -d disease2pubtator3.gz

###Chemical
wget https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3/chemical2pubtator3.gz
gzip -d chemical2pubtator3.gz

###Species
wget https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3/species2pubtator3.gz
gzip -d species2pubtator3.gz

###Gene
wget https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3/gene2pubtator3.gz
gzip -d gene2pubtator3.gz

cd ../../