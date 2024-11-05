# Use the official Ubuntu 24.04 image as a base
FROM ubuntu:24.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install dependencies and tools
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    software-properties-common \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.12 python3.12-dev python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

# Remove existing python3 symlink if it exists and create a new one
RUN [ -e /usr/bin/python3 ] && rm /usr/bin/python3; ln -s /usr/bin/python3.12 /usr/bin/python3

# Install pip for Python 3.12 using the package manager
RUN apt-get update && \
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Copy your project files into the container (optional)
# COPY . .

# Install Python packages if you have a requirements.txt file
# RUN pip3 install -r requirements.txt

# Specify the command to run your application
# CMD ["python3", "your_script.py"]


