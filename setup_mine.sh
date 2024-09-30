#!/usr/bin/env bash
set -e # Exit immediately if a command exits with a non-zero status

# Function to check for sudo access
check_sudo() {
    if ! sudo -v >/dev/null 2>&1; then
        echo "You need sudo privileges to run this script."
        exit 1
    fi
}

# Function to update and upgrade system packages
update_system() {
    echo "Updating package lists..."
    sudo apt update

    echo "Upgrading installed packages..."
    sudo apt upgrade -y
}

# Function to install necessary packages
install_packages() {
    echo "Installing required packages..."
    sudo apt install -y wget curl less nano unzip gawk libxml2-utils xmlstarlet g++
}

install_python_requirements() {
    echo "Installing python requirements from requirements.txt..."
    pip install -r requirements.txt
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
check_sudo
update_system
install_packages
# install_python_requirements
setup_abbreviation_detector

echo "Script completed successfully."