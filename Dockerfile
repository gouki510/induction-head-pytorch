FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

SHELL ["/bin/bash", "-c"]

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    curl \
    git \
    ffmpeg \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    net-tools \
    software-properties-common \
    swig \
    unzip \
    vim \
    wget \
    xpra \
    xserver-xorg-dev \
    zlib1g-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# python
# RUN pip install blobfile mod torcheval
# RUN conda install scipy matplotlib pandas plotly seaborn
# RUN conda install -c conda-forge lightning
# RUN conda install -c conda-forge wandb
# RUN conda install -c conda-forge einops
# RUN pip install -r emergent_in_context_learning/requirements.txt

