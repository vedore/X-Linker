#!/usr/bin/env bash

mkdir -p data/
cd data/

#-----------------------------------------------------------------------------
#                                  KBs
#-----------------------------------------------------------------------------
wget https://zenodo.org/records/13152812/files/kbs.zip?download=1
unzip 'kbs.zip?download=1'
rm 'kbs.zip?download=1'

#-----------------------------------------------------------------------------
#Generate files storing the info related to the the following KOS:
# CTD-Chemicals
# CTD-Diseases
# CTD-Genes

#Output directory is "data/kbs/"
cd ../
bash src/bash/generate_kbs.sh

#-----------------------------------------------------------------------------
#                                  DATASETS
#-----------------------------------------------------------------------------
cd data/
mkdir datasets/
cd datasets/

#------------------------------------------------------------------------------
# BC5CDR
wget https://github.com/JHnlp/BioCreative-V-CDR-Corpus/raw/master/CDR_Data.zip
unzip CDR_Data.zip
mv CDR_Data bc5cdr
rm CDR_Data.zip
rm -rf __MACOSX

#------------------------------------------------------------------------------
# NCBI-Disease
mkdir ncbi_disease
cd ncbi_disease

wget https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItrainset_corpus.zip
unzip NCBItrainset_corpus.zip
rm NCBItrainset_corpus.zip
wget https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBIdevelopset_corpus.zip
unzip NCBIdevelopset_corpus.zip
rm NCBIdevelopset_corpus.zip
wget https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItestset_corpus.zip
unzip NCBItestset_corpus.zip
rm NCBItestset_corpus.zip
cd ../

#------------------------------------------------------------------------------
# BioRED
wget https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip
unzip BIORED.zip
rm BIORED.zip
mv BioRED biored
mv biored/Test.BioC.JSON biored/test.json

#------------------------------------------------------------------------------
# NLM-Chem
wget https://ftp.ncbi.nlm.nih.gov/pub/lu/NLMChem/NLM-Chem-corpus.zip
unzip NLM-Chem-corpus.zip
rm NLM-Chem-corpus.zip
mv FINAL_v1 nlm_chem

#------------------------------------------
#The following datasets were not used in evaluation, but the respective documents
#were removed from the training data to allow a future evaluation

# MedMentions

# CRAFT corpus

# LINNAEUS

#------------------------------------------

cd ../../

#-----------------------------------------------------------------------------
# Prepare all datasets: Convert evaluation datasets into text format.

#BC5CDR
awk -F'\t' '!/\|t\|/ && !/\|a\|/ && $0 ~ /Disease/' data/datasets/bc5cdr/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt > data/datasets/bc5cdr/test_Disease.txt
awk -F'\t' '!/\|t\|/ && !/\|a\|/ && $0 ~ /Chemical/' data/datasets/bc5cdr/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt > data/datasets/bc5cdr/test_Chemical.txt
mkdir data/datasets/bc5cdr/txt/
awk -F'|' '/\|t\|/ || /\|a\|/ {print $3 > "data/datasets/bc5cdr/txt/" $1}' data/datasets/bc5cdr/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt 

#BioRED
awk -F'\t' '!/\|t\|/ && !/\|a\|/ && $0 ~ /DiseaseOrPhenotypicFeature/' data/datasets/biored/Test.PubTator > data/datasets/biored/test_Disease.txt
sed -i 's/DiseaseOrPhenotypicFeature/Disease/g' data/datasets/biored/test_Disease.txt
awk -F'\t' '!/\|t\|/ && !/\|a\|/ && $0 ~ /ChemicalEntity/' data/datasets/biored/Test.PubTator > data/datasets/biored/test_Chemical.txt
sed -i 's/ChemicalEntity/Chemical/g' data/datasets/biored/test_Chemical.txt
mkdir data/datasets/biored/txt/
awk -F'|' '/\|t\|/ || /\|a\|/ {print $3 > "data/datasets/biored/txt/" $1}' data/datasets/biored/Test.PubTator 

#NCBI-Disease**
awk -F'\t' '!/\|t\|/ && !/\|a\|/ && ($0 ~ /Disease/ || $0 ~ /Modifier/ || $0 ~ /SpecificDisease/ || $0 ~ /DiseaseClass/ || $0 ~ /CompositeMention/)' data/datasets/ncbi_disease/NCBItestset_corpus.txt > data/datasets/ncbi_disease/test_Disease.txt
sed -i 's/Modifier/Disease/g' data/datasets/ncbi_disease/test_Disease.txt
sed -i 's/SpecificDisease/Disease/g' data/datasets/ncbi_disease/test_Disease.txt
sed -i 's/DiseaseClass/Disease/g' data/datasets/ncbi_disease/test_Disease.txt
sed -i 's/CompositeMention/Disease/g' data/datasets/ncbi_disease/test_Disease.txt
mkdir data/datasets/ncbi_disease/txt/
awk -F'|' '/\|t\|/ || /\|a\|/ {print $3 > "data/datasets/ncbi_disease/txt/" $1}' data/datasets/ncbi_disease/NCBItestset_corpus.txt

#NLM-Chem***
python3 -c "from data_builder.shell_utils.convert_nlm_chem_2_pubtator import convert_nlm_chem_2_pubtator;convert_nlm_chem_2_pubtator()"
awk -F'\t' '!/\|t\|/ && !/\|a\|/ && !/\|p\|/ && !/\|f\|/ && ($0 ~ /Chemical/)' data/datasets/nlm_chem/test_pubtator.txt > data/datasets/nlm_chem/test_Chemical.txt
mkdir data/datasets/nlm_chem/txt/
awk -F'|' '/\|t\|/ || /\|a\|/ || /\|p\|/ || /\|f\|/ {print $3 > "data/datasets/nlm_chem/txt/" $1}' data/datasets/nlm_chem/test_Chemical.txt


#---------------------------------------------------------------------------
# Remove evaluation documents from training data
# To remove from the training data the documents that are also present in 
# the evaluation datasets to prevent data leakage:

python3 -c "from data_builder.shell_utils.create_datasets_pmids_list import create_datasets_pmids_list;create_datasets_pmids_list(email='add your email')"

#Note: add your personal email. It is necessary to use the Entrez utilities, since this script converts the PMCIDs present in the Linnaeus corpus in to PMIDs.
#This run generates the file 'data/datasets/ignore_pmids.txt' including 
#a list of PMIDs to exclude from training data and the file 
#'{datasets_dir}/ignore_pmids_info.txt' with detailed information about 
#the source datasets.

#---------------------------------------------------------------------------
# Get PECOS-EL models
#---------------------------------------------------------------------------
cd data

wget https://zenodo.org/records/13152812/files/models.zip?download=1
unzip 'models.zip?download=1'
rm 'models.zip?download=1'

#-----------------------------------------------------------------------------
#  Get SapBERT models
#-----------------------------------------------------------------------------
wget https://zenodo.org/records/12704543/files/sapbert.zip?download=1
unzip 'sapbert.zip?download=1'
rm 'sapbert.zip?download=1'

cd ../