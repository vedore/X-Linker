#!/usr/bin/env bash
set -e # Exit immediately if a command exits with a non-zero status

# Function to update and upgrade system packages
update_system() {
    apt-get update && 
    apt-get install -y build-essential git python3 python3-distutils python3-venv
}

# Function to install necessary packages
install_packages() {
    echo "Installing required packages..."
    apt install -y wget curl less nano unzip gawk libxml2-utils xmlstarlet g++
}

install_python_requirements() {
    echo "Installing python requirements from requirements.txt..."
    python3 -m venv venv
    source venv/bin/activate
    python3 -m pip install -r requirements.txt
    deactivate
}

# Function to set up the abbreviation detector
setup_abbreviation_detector() {
    echo "Setting up abbreviation detector..."

    mkdir -p abbreviation_detector
    cd abbreviation_detector/

    download_and_extract_ab3p
    download_and_extract_ncbi_text_lib

    # Install NCBITextLib
    echo "Installing NCBITextLib..."
    cd NCBITextLib/lib/
    make
    cd ../../

    # Install Ab3P
    echo "Installing Ab3P..."
    cd Ab3P
    sed -i "s#\*\* location of NCBITextLib \*\*#../NCBITextLib#" Makefile
    sed -i "s#\*\* location of NCBITextLib \*\*#../../NCBITextLib#" lib/Makefile
    make
    cd ../
    cd ../
}

# Function to download and extract Ab3P if not already done
download_and_extract_ab3p() {
    local ab3p_dir="Ab3P"
    local zip_file="master.zip"
    local ab3p_url="https://github.com/ncbi-nlp/Ab3P/archive/refs/heads/master.zip"

    if [ ! -d "$ab3p_dir" ]; then
        echo "Downloading and extracting Ab3P..."
        wget -q "$ab3p_url" -O "$zip_file"
        unzip -q "$zip_file"
        mv Ab3P-master "$ab3p_dir"
        rm "$zip_file"  # Remove the zip file after extraction
    else
        echo "Ab3P directory already exists. Skipping download and extraction."
    fi
}

# Function to download and extract NCBITextLib
download_and_extract_ncbi_text_lib() {
    local ncbi_text_lib_dir="NCBITextLib"
    local zip_file="master.zip"
    local ncbi_text_lib_url="https://github.com/ncbi-nlp/NCBITextLib/archive/refs/heads/master.zip"

    if [ ! -d "$ncbi_text_lib_dir" ]; then
        echo "Downloading and extracting NCBITextLib..."
        wget -q "$ncbi_text_lib_url" -O "$zip_file"
        unzip -q "$zip_file"
        mv NCBITextLib-master "$ncbi_text_lib_dir"
        rm "$zip_file"  # Remove the zip file after extraction
    else
        echo "NCBITextLib directory already exists. Skipping download and extraction."
    fi
}

# Main script execution
update_system
install_packages
install_python_requirements
setup_abbreviation_detector

echo "Script completed successfully."