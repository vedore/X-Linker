FROM rapidsai/base:24.10-cuda12.5-py3.12

# Set enviroment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Set the working directory
WORKDIR /app


