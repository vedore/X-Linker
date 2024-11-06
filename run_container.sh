#!/bin/bash

# for cuda users 
# docker run --gpus all -it --name rapidcontainer -v $(pwd)/X_Linker:/app/x_linker image /bin/bash

# for cpu users
docker run -it --name rapidcontainer -v $(pwd):/app/x_linker rapids_cuda /bin/bash
