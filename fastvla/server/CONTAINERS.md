# Containers on gpu5 without admin

Author: Varun Chandra
Date: 2026-10-14
Host: gpu5 (Ubuntu 20.04.6, kernel 5.4.0-139-generic, driver 530.30.02)

## Verdict

**Yes. Unprivileged Apptainer works, and NVIDIA Triton 23.07 runs under it.**
Docker is blocked and stays blocked, but it is not needed. Triton's
`tritonserver --help` has been executed successfully inside the container on
this machine, CPU only. The GPU path (`--nv`) is written down below but has
NOT been run yet, because GPU 1 was busy with another job.

No root was used. Nothing was installed outside `$HOME`.

## 1. Capability checks (all run as vchandr2)

    $ cat /proc/sys/user/max_user_namespaces
    1030624

    $ cat /proc/sys/kernel/unprivileged_userns_clone
    1

    $ grep vchandr2 /etc/subuid /etc/subgid
    /etc/subuid:vchandr2:1410720:65536
    /etc/subgid:vchandr2:1410720:65536

    $ which newuidmap newgidmap
    (nothing; exit 1)

    $ unshare -U -r true && echo userns-ok
    userns-ok

    $ stat -fc %T /sys/fs/cgroup
    tmpfs

    $ uname -r
    5.4.0-139-generic

    $ head -3 /etc/os-release
    NAME="Ubuntu"
    VERSION="20.04.6 LTS (Focal Fossa)"
    ID=ubuntu

    $ nvidia-smi --query-gpu=driver_version --format=csv
    driver_version
    530.30.02
    530.30.02

Two extra checks that actually decide the outcome, since unprivileged
Apptainer mounts SquashFS through FUSE rather than a setuid helper:

    $ ls -l /dev/fuse
    crw-rw-rw- 1 root root 10, 229 Aug  1 12:38 /dev/fuse     # world rw

    $ fusermount -V
    fusermount version: 2.9.9

### Reading of the checks

- Unprivileged user namespaces are enabled and usable. This is the one that
  matters, and it passes.
- `/dev/fuse` is world read/write and `fusermount` exists, so Apptainer can
  mount images without setuid.
- `newuidmap`/`newgidmap` are absent. Apptainer in unprivileged mode does not
  need them (it maps only the single calling UID). Rootless Docker *does*
  need them, so rootless Docker is not an option here despite the subuid and
  subgid entries existing.
- `/sys/fs/cgroup` is tmpfs, i.e. cgroup v1. Apptainer does not require
  cgroup v2 for plain `exec`/`run`; it only matters for resource limits.

## 2. Installing Apptainer (no root)

    curl -fsSL -o ~/install-unprivileged.sh \
      https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh
    chmod +x ~/install-unprivileged.sh
    ~/install-unprivileged.sh ~/apptainer

**Gotcha:** the installer unpacks Fedora/EPEL RPMs and needs `rpm2cpio`,
which Ubuntu 20.04 does not ship. There is no way to apt-install it without
root. A ~55-line Python stand-in lives at `~/bin/rpm2cpio` (it skips the
96-byte RPM lead and the two headers, then decompresses the cpio payload;
gzip/xz/bzip2/zstd). Put `~/bin` on `PATH` before running the installer:

    export PATH=$HOME/bin:$PATH

The installer must be given an empty target directory; a failed run leaves
`~/apptainer/x86_64` behind and the retry aborts with "is not empty". Delete
the directory and rerun.

Result: `apptainer version 1.5.4-1.el8` in `~/apptainer/bin`.

    export PATH=$HOME/apptainer/bin:$PATH
    export APPTAINER_CACHEDIR=$HOME/.apptainer

## 3. What has been verified to work

    $ apptainer exec docker://ubuntu:22.04 echo ok
    ok

    $ apptainer pull tritonserver_23.07-py3.sif \
        docker://nvcr.io/nvidia/tritonserver:23.07-py3
    # -> ~/images/tritonserver_23.07-py3.sif, 6.2 GB

    $ apptainer exec ~/images/tritonserver_23.07-py3.sif tritonserver --help
    Usage: tritonserver [options]
      --help
            Print usage
    ...

    $ apptainer exec ~/images/tritonserver_23.07-py3.sif nvcc --version | tail -2
    Cuda compilation tools, release 12.1, V12.1.105

The pull emits `warn rootless{...} ignoring (usually) harmless EPERM on
setxattr "user.rootlesscontainers"` many times. That is expected when
converting OCI layers as a non-root user and does not affect the result.

## 4. Why 23.07 and not something newer

From NVIDIA's framework support matrix:

- 23.03, 23.04       CUDA 12.1.0  -> requires driver >= 530
- 23.05, 23.06, 23.07 CUDA 12.1.1 -> requires driver >= 530
- 23.08, 23.09       CUDA 12.2.1  -> requires driver >= 535

This host has driver 530.30.02, so **23.07 is the newest usable tag** and the
CUDA 12.1 inside it matches the driver exactly. Anything 23.08 or later needs
a driver upgrade, which needs root.

## 5. The GPU command, to run once a GPU is free

NOT YET RUN. GPU 0 belongs to another user. GPU 1 was running another job.

    export PATH=$HOME/apptainer/bin:$PATH
    CUDA_VISIBLE_DEVICES=1 apptainer exec --nv \
      ~/images/tritonserver_23.07-py3.sif nvidia-smi

Expected: exactly one card listed (the A5000 that is GPU 1 on the host, which
appears as GPU 0 inside the container), driver 530.30.02, CUDA 12.1.

To serve a model repository:

    CUDA_VISIBLE_DEVICES=1 apptainer exec --nv \
      --bind /path/to/model_repository:/models \
      ~/images/tritonserver_23.07-py3.sif \
      tritonserver --model-repository=/models

`--nv` makes Apptainer bind the host driver libraries into the container. It
needs no root. Pick non-default ports if anyone else might run Triton:
`--http-port`, `--grpc-port`, `--metrics-port` (defaults 8000/8001/8002).

## 6. Disk

    ~/images/tritonserver_23.07-py3.sif   6.2 GB
    ~/apptainer                           157 MB
    ~/.apptainer (cache, after cleaning)   36 KB
    ~/bin                                 8 KB
    -------------------------------------------
    total                                 6.3 GB

The pull left 13 GB of OCI blobs in `~/.apptainer`. Clear it with:

    apptainer cache clean --force

The SIF is self-contained and still runs after the cache is cleared; this was
verified. Clear the cache after every pull or it will grow quickly. Also
remove `$APPTAINER_CACHEDIR/tmp` if a pull was interrupted.

## 7. Remaining caveats

- The GPU path is unverified. Everything else is confirmed working, and `--nv`
  is the standard mechanism, but it should be run once before anyone depends
  on it.
- Rootless Docker is not viable: `newuidmap`/`newgidmap` are missing and
  cannot be installed without root. Do not spend time on it.
- Kernel 5.4 is old but sufficient; unprivileged userns is what matters and it
  is enabled.
- `~/bin/rpm2cpio` is only needed to reinstall or upgrade Apptainer. It is not
  used at runtime.

---

## 8. The `--nv` GPU check: PASSES, with the right check

Added 2026-10-15, corrected the same day.

### The check that matters

An earlier version of this section reported the GPU path as FAILING because

    CUDA_VISIBLE_DEVICES=1 apptainer exec --nv <sif> nvidia-smi

lists **both** cards. That was the wrong test, and the conclusion drawn from it was
wrong. `nvidia-smi` ignores `CUDA_VISIBLE_DEVICES` **on the host as well**:

    $ CUDA_VISIBLE_DEVICES=1 nvidia-smi --query-gpu=index,uuid --format=csv,noheader
    0, GPU-2c6273fb-12cf-6171-8708-36c7225f4ba3
    1, GPU-040542a1-6ad7-1ddd-f7ca-595acfd703a7

and the device nodes are world read/write on the host too:

    $ ls -l /dev/nvidia0 /dev/nvidia1
    crw-rw-rw- 1 root root 195, 0 /dev/nvidia0
    crw-rw-rw- 1 root root 195, 1 /dev/nvidia1

So the container grants no access the host user did not already have. Confinement has
always rested on the **CUDA runtime** honouring the mask, inside a container or outside
one. The check therefore has to be the CUDA runtime's view plus proof of where an
allocation lands.

### Result: PASS

`fastvla/server/preflight_gpu.py`, run inside the container:

    [PREFLIGHT] device_count=1 (want 1) -> ok
    [PREFLIGHT] cuda:0 pci bus id=0000:C1:00.0 (want 00000000:C1:00.0) -> ok
    [PREFLIGHT] 256 MB allocated on cuda:0; our pid appears on 1 expected-card row(s),
                0 forbidden-card row(s) -> ok
    [PREFLIGHT]   ours: GPU-040542a1-6ad7-1ddd-f7ca-595acfd703a7, 482968, 982 MiB
    [PREFLIGHT] PASS

The same script passes on the host, which is the point: the container behaves exactly as
the host does.

Two details worth keeping:

- torch 2.2 exposes neither a UUID nor a PCI bus id on `get_device_properties`, so the
  bus id is read through `cudaDeviceGetPCIBusId` in libcudart, which is mask-aware.
- `cudaDeviceGetPCIBusId` pads the PCI domain to 4 digits (`0000:C1:00.0`) and
  `nvidia-smi` to 8 (`00000000:C1:00.0`). The same device, compared after normalising.

### The required invocation

    export PATH=$HOME/apptainer/bin:$PATH
    CUDA_VISIBLE_DEVICES=1 apptainer exec --nv \
      --bind /data_new/vchandr2/miniconda3:/data_new/vchandr2/miniconda3 \
      --bind /data_new/vchandr2/fastvla-eval:/data_new/vchandr2/fastvla-eval \
      ~/images/tritonserver_23.07-py3.sif \
      /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python <script>

Plain `--nv`, `CUDA_VISIBLE_DEVICES=1`, and the pre-flight above as the first thing every
in-container GPU job runs -- the same standing as `nvidia-smi` and `who` on the host.

### NEVER use `--contain` with `--nv` on this host

    $ CUDA_VISIBLE_DEVICES=0 apptainer exec --nv --contain \
        --bind /dev/nvidia1 --bind /dev/nvidiactl ... <sif> nvidia-smi --query-gpu=uuid
    WARNING: Skipping /dev/nvidia1 bind mount: already mounted
    GPU-2c6273fb-12cf-6171-8708-36c7225f4ba3        <-- the OTHER user's card, alone

`--contain` made the container present exactly one card and it was the wrong one, while
reporting that the explicit device binds had been skipped as already mounted. The device
set a `--contain` invocation ends up with is not the one asked for. `--nvccli` with
`NVIDIA_VISIBLE_DEVICES=1` (`nvidia-container-cli` is present at `/usr/bin/nvidia-container-cli`)
also failed to restrict the set. Neither is needed: plain `--nv` plus the mask plus the
pre-flight is correct and verified.

### Python compatibility for binding the host env

The container is CPython 3.10.6, the `libero-vla` env is CPython 3.10.20 -- the same cp310
ABI, so the env's extension modules load inside the container. Binding
`/data_new/vchandr2/miniconda3` and running the env's own interpreter gives the container
the identical stack to the host harness, which is what makes a Triton-vs-ZeroMQ comparison
a comparison of serving layers rather than of dependency stacks. The Triton image's own
`python3` has no torch (its PyTorch backend uses libtorch, not the Python package).
