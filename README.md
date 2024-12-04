Para dar setup no python3.11.10

apt-get update && apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt-get update

apt install python3.11 -y

apt-get install -y python3.11-venv python3.11-dev

pip install --upgrade pip

NVIDIA DOCKER 

apt-get install nvidia-container-runtime

pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.10.* dask-cudf-cu12==24.10.* cuml-cu12==24.10.* \
    cugraph-cu12==24.10.* nx-cugraph-cu12==24.10.* cuspatial-cu12==24.10.* \
    cuproj-cu12==24.10.* cuxfilter-cu12==24.10.* cucim-cu12==24.10.* \
    pylibraft-cu12==24.10.* raft-dask-cu12==24.10.* cuvs-cu12==24.10.* \
    nx-cugraph-cu12==24.10.*