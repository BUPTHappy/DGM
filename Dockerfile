FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl ca-certificates git \
    build-essential pkg-config cmake \
    python3 python3-dev python3-pip \
    libglew-dev libosmesa6-dev libgl1-mesa-glx libglfw3 libgl1-mesa-dev \
    patchelf gfortran \
 && rm -rf /var/lib/apt/lists/*

# Micromamba (conda replacement)
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
SHELL ["/bin/bash", "-lc"]
RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin bin/micromamba --strip-components=1
ENV PATH="${MAMBA_ROOT_PREFIX}/bin:${PATH}"

WORKDIR /workspace
COPY . /workspace

# Environment setup
ARG ENV_FILE=fast_policy_environment.yml
ARG ENV_NAME=fast_policy
RUN if [[ -f "${ENV_FILE}" ]]; then \
        micromamba create -y -n ${ENV_NAME} -f ${ENV_FILE}; \
    else \
        micromamba create -y -n ${ENV_NAME} python=3.10; \
        if [[ -f "requirements.txt" ]]; then \
            micromamba run -n ${ENV_NAME} python -m pip install -U pip && \
            micromamba run -n ${ENV_NAME} pip install -r requirements.txt; \
        fi \
    fi \
 && micromamba clean -a -y

# Optional: Jupyter
RUN micromamba run -n ${ENV_NAME} python -m pip install jupyter

# MuJoCo (if needed)
ENV MUJOCO_HOME=/opt/mujoco/mujoco210
RUN mkdir -p /opt/mujoco \
 && wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz \
 && tar -xf /tmp/mujoco.tar.gz -C /opt/mujoco \
 && rm /tmp/mujoco.tar.gz
ENV LD_LIBRARY_PATH=${MUJOCO_HOME}/bin:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV MUJOCO_GL=egl

CMD ["bash"]
