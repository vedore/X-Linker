#!/bin/bash

# for cuda users 
# docker run --gpus all -it -v $(pwd)/X_Linker:/app/x_linker ubuntu2404 /bin/bash

# for cpu users
docker run -it --name ubuntu2404 -v $(pwd):/app/x_linker ubuntu2404 /bin/bash
