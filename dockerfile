FROM python:3.12-slim-bullseye

WORKDIR /x_linker

# Copy all files from the current folder to the container's workdir
# COPY . /x_linker
# COPY requirements.txt /x_linker

# Update apt and install bash
RUN apt-get update && apt-get install -y bash python3-venv && apt-get clean

# Create a virtual environment
RUN python3 -m venv .mount_venv \
    && . .mount_venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r requirements.txt


# Always import app as an pythonpath
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Set bash as the default shell when the container is started interactively
CMD ["bash"]

