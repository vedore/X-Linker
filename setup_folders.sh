#!/bin/bash

# Define the base path for the data folder
BASE_PATH="data"

# Create the main data folder and subfolders
mkdir -p "$BASE_PATH/external/biobert"
mkdir -p "$BASE_PATH/external/synonyms"
mkdir -p "$BASE_PATH/processed/embeddings"
mkdir -p "$BASE_PATH/processed/index_labels"
mkdir -p "$BASE_PATH/processed/mesh_processed"
mkdir -p "$BASE_PATH/raw/mesh_data/ctd_chemicals"
mkdir -p "$BASE_PATH/raw/mesh_data/ctd_genes"
mkdir -p "$BASE_PATH/raw/mesh_data/medic"
mkdir -p "$BASE_PATH/raw/other_raw_files"

echo "Data folder structure created successfully."
