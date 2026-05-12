# Habitat-sim + DD-PPO + custom dh-spatial policies, runtime image for RCP H200/A100.
#
# Solves the EGL/fork/CUDA-binding issues we hit when trying to run habitat-sim
# inside minimal cuda:devel container by pre-installing all the system libs
# habitat-sim needs at OS level (apt packages), not user-level conda-forge.
#
# Build (on RCP via kaniko):
#   kaniko --dockerfile=docker/Dockerfile --context=. --destination=registry.rcp.epfl.ch/dhlab-wxu/habitat:v1
#
# Use:
#   runai-rcp-prod submit ... --image=registry.rcp.epfl.ch/dhlab-wxu/habitat:v1 ...

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------------------------
# 1. System libs (the things missing in cuda:devel that habitat-sim needs)
# ----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    git \
    wget \
    curl \
    cmake \
    pkg-config \
    # Graphics / EGL / OpenGL — the critical missing pieces
    libgl1-mesa-glx \
    libegl1-mesa \
    libegl1 \
    libgles2-mesa \
    libglu1-mesa \
    libglvnd-dev \
    libgl1 \
    libglib2.0-0 \
    libglib2.0-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    # Python build deps + general utils
    libbz2-dev \
    libssl-dev \
    libffi-dev \
    libreadline-dev \
    libsqlite3-dev \
    zlib1g-dev \
    libncurses-dev \
    # ifconfig / ip (for habitat-sim's network probe)
    iproute2 \
    net-tools \
    # tmux / vim for interactive debug if needed
    tmux \
    vim \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# 2. NVIDIA EGL ICD config (so habitat-sim finds NVIDIA's EGL via libglvnd)
# ----------------------------------------------------------------------------
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{\n\
    "file_format_version" : "1.0.0",\n\
    "ICD" : {\n\
        "library_path" : "libEGL_nvidia.so.0"\n\
    }\n\
}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# ----------------------------------------------------------------------------
# 3. Miniconda (Python 3.9)
# ----------------------------------------------------------------------------
ENV CONDA_DIR=/opt/miniconda3
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    ${CONDA_DIR}/bin/conda config --set channel_priority strict && \
    # Miniconda 24.x has `conda tos`; older versions don't. Tolerate both.
    ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true && \
    ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

ENV PATH=${CONDA_DIR}/bin:${PATH}

# ----------------------------------------------------------------------------
# 4. Habitat env: habitat-sim 0.3.3 (with bullet, headless) + torch CUDA
# ----------------------------------------------------------------------------
RUN conda create -n habitat python=3.9 cmake=3.22 -y -c conda-forge && \
    conda clean -afy

# Activate by default in subsequent RUNs
SHELL ["/bin/bash", "-c"]
RUN echo "source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate habitat" >> /etc/bash.bashrc

# habitat-sim from aihabitat (matches Izar's setup)
RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate habitat && \
    conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat -y && \
    conda clean -afy

# Torch with CUDA 12.1
RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate habitat && \
    pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 \
        --index-url https://download.pytorch.org/whl/cu121

# ----------------------------------------------------------------------------
# 5. habitat-lab + habitat-baselines (from source, with blind-policy patch)
# ----------------------------------------------------------------------------
RUN git clone --branch stable --depth 1 https://github.com/facebookresearch/habitat-lab.git /opt/habitat-lab && \
    cd /opt/habitat-lab && \
    # Apply the blind-policy patch (resnet_policy.py: guard normalize_visual_inputs with channel check)
    sed -i 's|if normalize_visual_inputs:|if normalize_visual_inputs and self._n_input_channels > 0:|' \
        habitat-baselines/habitat_baselines/rl/ddppo/policy/resnet_policy.py && \
    source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate habitat && \
    pip install --no-cache-dir -e habitat-lab && \
    pip install --no-cache-dir -e habitat-baselines

# ----------------------------------------------------------------------------
# 6. Pin protobuf 4.x (matches torch 2.1's bundled libprotobuf, avoids ABI segfault)
# ----------------------------------------------------------------------------
RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate habitat && \
    pip install --no-cache-dir 'protobuf>=4.21,<5' 'pillow==10.4.0' 'opencv-python<4.10' 'numpy<2'

# ----------------------------------------------------------------------------
# 7. EGL env vars (so habitat-sim binds to NVIDIA EGL under runai)
# ----------------------------------------------------------------------------
ENV EGL_DEVICE_ID=0
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV MAGNUM_LOG=quiet
ENV CONDA_DEFAULT_ENV=habitat

# ----------------------------------------------------------------------------
# 8. Sanity check (build fails if imports break)
# ----------------------------------------------------------------------------
RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate habitat && \
    python -c "import torch; print('torch:', torch.__version__)" && \
    python -c "import habitat_sim; print('habitat_sim:', habitat_sim.__version__)" && \
    python -c "import habitat; print('habitat:', habitat.__version__)" && \
    python -c "from habitat_baselines.rl.ddppo.algo.ddppo import DDPPO; print('DDPPO OK')"

WORKDIR /workspace

CMD ["bash"]
