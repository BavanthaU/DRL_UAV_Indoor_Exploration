FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_ENV=isaacsim50
ARG ISAAC_SIM_VERSION=5.0.0
ARG ISAAC_LAB_TAG=v2.2.1

ENV CONDA_DIR=/opt/conda \
    CONDA_ENV=${CONDA_ENV} \
    PATH=/opt/conda/envs/${CONDA_ENV}/bin:/opt/conda/bin:${PATH} \
    OMNI_KIT_ACCEPT_EULA=YES \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bzip2 \
    ca-certificates \
    curl \
    git \
    build-essential \
    cmake \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libglvnd0 \
    libgtk-3-0 \
    libice6 \
    libnss3 \
    libsm6 \
    libvulkan1 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxkbcommon0 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniforge.sh \
    && conda create -y -n "${CONDA_ENV}" python=3.11 \
    && conda clean -afy

RUN source "${CONDA_DIR}/etc/profile.d/conda.sh" \
    && conda activate "${CONDA_ENV}" \
    && python -m pip install --upgrade pip \
    && python -m pip install \
        torch==2.7.0+cu128 \
        torchvision==0.22.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install \
        "isaacsim[all,extscache]==${ISAAC_SIM_VERSION}" \
        --extra-index-url https://pypi.nvidia.com

RUN git clone --depth 1 --branch "${ISAAC_LAB_TAG}" https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab \
    && source "${CONDA_DIR}/etc/profile.d/conda.sh" \
    && conda activate "${CONDA_ENV}" \
    && cd /opt/IsaacLab \
    && TERM=xterm ./isaaclab.sh --install none \
    && python -m pip install \
        psutil==5.9.8 \
        wheel==0.42.0 \
        "ipython<9"

COPY requirements.txt /tmp/drl_uav_requirements.txt
RUN source "${CONDA_DIR}/etc/profile.d/conda.sh" \
    && conda activate "${CONDA_ENV}" \
    && python -m pip install -r /tmp/drl_uav_requirements.txt pybind11 \
    && python -m pip install \
        wandb==0.16.6 \
        click==8.1.7 \
        sentry-sdk==1.43.0 \
    && python -m pip check

WORKDIR /workspace/project

CMD ["/bin/bash"]
