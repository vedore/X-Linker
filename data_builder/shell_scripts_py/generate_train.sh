#!/bin/bash

# Arguments to pass to the Python script
type=$1
kb=$2
pubtator=$3

# Execute the Python script with the provided arguments
python3 generate_training_data.py "$type" "$kb" "$pubtator"
