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

## 8. The `--nv` GPU check: RUN, and it FAILS the safety condition

Added 2026-10-15. Section 5 above left one thing unverified and predicted
"Expected: exactly one card listed". That prediction is **wrong**, and the
check is a blocker for running anything GPU-bound in a container on this box.

GPU 1 was idle (3 MiB, 0%) and GPU 0 held another user's job, so the check ran
under the right conditions.

    $ nvidia-smi --query-gpu=index,uuid --format=csv,noheader     # host, for reference
    0, GPU-2c6273fb-12cf-6171-8708-36c7225f4ba3                   # chunyiy's card
    1, GPU-040542a1-6ad7-1ddd-f7ca-595acfd703a7                   # ours

    $ CUDA_VISIBLE_DEVICES=1 apptainer exec --nv \
        ~/images/tritonserver_23.07-py3.sif nvidia-smi --query-gpu=index,name,uuid --format=csv
    index, name, uuid
    0, NVIDIA RTX A5000, GPU-2c6273fb-12cf-6171-8708-36c7225f4ba3   <-- NOT OURS
    1, NVIDIA RTX A5000, GPU-040542a1-6ad7-1ddd-f7ca-595acfd703a7

**Two cards, including the one that belongs to another user.**

### Why

`CUDA_VISIBLE_DEVICES` is honoured inside the container (it reads back as `1`),
but it only filters what the **CUDA runtime** enumerates. `nvidia-smi` talks to
the driver through NVML and `/dev/nvidiactl`, and `--nv` binds **every** GPU
device node in:

    $ CUDA_VISIBLE_DEVICES=1 apptainer exec --nv <sif> ls -la /dev/nvidia*
    crw-rw-rw- 1 nobody nogroup 195,   0 /dev/nvidia0      <-- other user's card
    crw-rw-rw- 1 nobody nogroup 195,   1 /dev/nvidia1
    crw-rw-rw- 1 nobody nogroup 195, 255 /dev/nvidiactl
    crw-rw-rw- 1 nobody nogroup 510,   0 /dev/nvidia-uvm

Both nodes are world read/write, so a process in the container can open GPU 0
directly. Confinement rests entirely on every process inside honouring
`CUDA_VISIBLE_DEVICES`. The CUDA runtime does; NVML tools do not; and anything
that sets the variable itself, or uses the driver API directly, is unconstrained.
Against this project's hard rule -- GPU 0 belongs to someone else, never touch
it -- that is not good enough.

### Attempted fixes, none of which work

| attempt | result |
|---|---|
| `CUDA_VISIBLE_DEVICES=1 --nv` | both cards |
| `CUDA_VISIBLE_DEVICES=1 --nv --contain` | both cards |
| `--nv --nvccli` with `NVIDIA_VISIBLE_DEVICES=1` (`nvidia-container-cli` **is** present at `/usr/bin/nvidia-container-cli`) | both cards |
| `--nv --contain` + explicit `--bind /dev/nvidia1` | **one card, and it was GPU 0 -- the wrong one.** Apptainer reported `Skipping /dev/nvidia1 bind mount: already mounted`, so the explicit binds were ignored and the selection came from somewhere else |

The last row is the reason this is a hard stop rather than a cosmetic
complaint: the device set the container ends up with is not reliably the one
asked for, and in one configuration it was exactly the card that must not be
touched.

### What would fix it

- `nvidia-container-cli` exists but `--nvccli` did not restrict the device set
  here; it may need a host-side configuration change (`/etc/nvidia-container-runtime/config.toml`)
  that requires root.
- Apptainer 1.5 has no unprivileged way to drop a `/dev` node that `--nv`
  has already bound; `--contain` does not remove them.
- The dependable route is a host change: either a cgroup device rule, or
  running the container under a wrapper that sets up `/dev` before `--nv`.
  Both need the box's administrator.

### Consequence for M3b and M5

Container-hosted GPU work on gpu5 is **not cleared**. The CPU-only container
path is fine and is still verified (section 3), so the parts of M5 that do not
need a GPU inside a container -- image build, CI, CPU tests, manifests -- can
proceed. Anything that runs the policy inside a container on this box waits on
the administrator.
