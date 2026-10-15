# FastVLA-Eval: server and worker in one image.
#
# One image, two entrypoints, because the two share the whole stack -- the C++ labeler,
# the protocol, the manifest code -- and shipping it twice would be two things to keep in
# step. `fastvla-server` and `fastvla-worker` select the role.
#
# Base: NVIDIA's CUDA 12.1.1 runtime on Ubuntu 22.04. 12.1 because gpu5's driver is
# 530.30.02 and CUDA 12.2 needs >= 535 (fastvla/server/CONTAINERS.md section 4); the same
# constraint that pins Triton to 23.07.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04@sha256:0e25ca6cb0e9b0bd6b2f0f5b4f0f1e9c4a0b2d5e6f7a8b9c0d1e2f3a4b5c6d7e AS base

# Pinned so a rebuild is reproducible. Versions are the ones measured in M0-M4 on gpu5.
ARG PY_VERSION=3.10
ARG TORCH_VERSION=2.2.0
ARG TORCHVISION_VERSION=0.17.0
ARG NUMPY_VERSION=1.26.4
ARG TRANSFORMERS_VERSION=4.40.1
ARG PYZMQ_VERSION=27.2.0
ARG PYBIND11_VERSION=3.1.0
ARG MUJOCO_VERSION=3.1.6

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    MUJOCO_GL=egl

# Build toolchain for the C++ labeler, plus the X/GL libraries robosuite needs. The
# Triton image lacking these is what broke M3b's first six attempts, so they are explicit.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python${PY_VERSION} python${PY_VERSION}-dev python3-pip \
      g++ make \
      libx11-6 libxext6 libxrender1 libxrandr2 libxinerama1 libxcursor1 libxi6 \
      libxfixes3 libgl1 libegl1 libglew-dev libosmesa6 \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
      torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        --index-url https://download.pytorch.org/whl/cu121 \
 && python3 -m pip install --no-cache-dir \
      numpy==${NUMPY_VERSION} \
      transformers==${TRANSFORMERS_VERSION} \
      pyzmq==${PYZMQ_VERSION} \
      pybind11==${PYBIND11_VERSION} \
      mujoco==${MUJOCO_VERSION} \
      imageio imageio-ffmpeg scipy

# Non-root. UID 1000 so a bind-mounted results directory stays writable by the host user.
RUN groupadd -g 1000 fastvla && useradd -m -u 1000 -g 1000 fastvla
WORKDIR /opt/fastvla
COPY --chown=fastvla:fastvla . /opt/fastvla

# Build the C++ labeler IN the image, so the artefact matches the image's toolchain
# rather than the host's. -O3 -fopenmp, and deliberately no -march=native: M2 measured
# 4.7% for it and rejected it because a native build changes results on another host.
USER fastvla
RUN bash fastvla/labeler/build.sh && \
    python3 -c "import sys; sys.path.insert(0, 'fastvla/labeler'); import fastgjk; print('fastgjk ok')"

ENV PYTHONPATH=/opt/fastvla:/opt/fastvla/fastvla/labeler
EXPOSE 9400
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9400/healthz',timeout=3).status==200 else 1)"

COPY --chown=fastvla:fastvla deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["fastvla-server"]
