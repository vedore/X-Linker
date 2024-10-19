#!/usr/bin/env bash

#-----------------------------------------------------------------------------
#                                  KBs
#-----------------------------------------------------------------------------
mkdir -p kbs/
cd kbs/

# MEDIC vocabulary (version: Feb 28 2024 10:59 EST)
mkdir -p medic/
cd medic
wget https://ctdbase.org/reports/CTD_diseases.tsv.gz
gzip -d CTD_diseases.tsv.gz
cd ../

#CTD-Chemicals (version: Feb 28 2024 10:59 EST)
mkdir -p ctd_chemicals
cd ctd_chemicals
wget https://ctdbase.org/reports/CTD_chemicals.tsv.gz
gzip -d CTD_chemicals.tsv.gz
cd ../

#CTD-Gene (version: Feb 28 2024 11:00 EST)
mkdir ctd_genes
cd ctd_genes
wget https://ctdbase.org/reports/CTD_genes.tsv.gz
gzip -d CTD_genes.tsv.gz
cd ../

# NCBI taxon (version: 2024-03-28 11:27)
#cd ../

cd ../

#-----------------------------------------------------------------------------
#Generate files storing the info related to the the following KOS:
# CTD-Chemicals
# CTD-Diseases
# CTD-Genes

#Output directory is "data/kbs/"
bash src/bash/generate_kbs.sh