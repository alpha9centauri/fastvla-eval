# Provenance of every external claim in this repository

Every URL, version pin, digest, citation and fact about something outside this repo, with
how it was checked. Checked 2026-10-15 from gpu5. "Fetched" means an HTTP request made
from this box on that date; "from the artifact" means read out of the thing itself, which
is the only kind that cannot go stale against reality.

The project's own measured numbers are not in this table — they come from `bench/out/`,
`audit/` and `fastvla/monitor/EVAL.md`, and each one names the script that produced it.

## URLs

| where | URL | how verified |
|---|---|---|
| `deploy/Dockerfile` | `https://download.pytorch.org/whl/cu121` | fetched, HTTP 200 |
| `fastvla/server/CONTAINERS.md` | `https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh` | fetched, HTTP 200, 15560 bytes |
| `fastvla/server/CONTAINERS.md` | `https://docs.docker.com/engine/security/rootless/` | fetched; quoted sentence confirmed verbatim |
| `fastvla/server/CONTAINERS.md` | `https://archive.docs.nvidia.com/.../rel-23-0N.html` | fetched, one page per release, 23.03–23.09 |
| `README.md` acknowledgements | `https://github.com/LIBERO-SAFETY/LIBERO-Safety` | fetched, HTTP 200; also the `origin` of the local checkout |
| `README.md` acknowledgements | `https://github.com/moojink/openvla-oft` | fetched, HTTP 200; also the `origin` of the local checkout |
| `README.md` acknowledgements | `https://github.com/Physical-Intelligence/openpi` | fetched, HTTP 200; local checkout's `origin` is the lowercase form of the same repo |
| `README.md` acknowledgements | `https://github.com/Lifelong-Robot-Learning/LIBERO` | fetched, HTTP 200 |
| `deploy/docker-compose.yml`, `deploy/k8s/*` | `ghcr.io/alpha9centauri/fastvla-eval` | **not verifiable from here**: anonymous pull returns HTTP 401 because the repository is private. It is the tag CI pushes to; stated as that, not as a published artifact |

## Container images

| pin | how verified |
|---|---|
| `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` | manifest fetched from `registry-1.docker.io`, HTTP 200. Pinned by tag, not digest — an earlier revision of this file carried an invented sha256; see `deploy/Dockerfile` |
| `python:3.10-slim-bookworm` | manifest fetched from `registry-1.docker.io`, HTTP 200 |
| `nvcr.io/nvidia/tritonserver:23.07-py3` | from the artifact: the pulled 6.6 GB SIF on this box reports `/opt/tritonserver/TRITON_VERSION` = 2.36.0, `cuda-cudart-12-1` 12.1.105, `cuda-compat-12-1` 530.30.02-1, Ubuntu 22.04.2 |

## The Triton-23.07 argument (CONTAINERS.md section 4)

Each row is the "Release X is based on CUDA Y, which requires NVIDIA Driver release Z or
later" sentence from that release's own notes on `archive.docs.nvidia.com`, fetched
individually. The live framework support matrix was checked first and no longer lists
23.x at all — it now covers 25.xx and 26.xx — so the archive is the source.

| release | CUDA | min driver | verified |
|---|---|---|---|
| 23.03 | 12.1.0 | 530 | fetched |
| 23.04 | 12.1.0 | 530 | fetched |
| 23.05 | 12.1.1 | 530 | fetched |
| 23.06 | 12.1.1 | 530 | fetched |
| 23.07 | 12.1.1 | 530 | fetched, **and** from the artifact (above) |
| 23.08 | 12.2.1 | 535 | fetched |
| 23.09 | 12.2.1 | 535 | fetched |

The claim that cudart 12.1.105 identifies CUDA 12.1.1 was checked against NVIDIA's own
package index for ubuntu2204: the 12.1 line publishes exactly two cudart builds,
`12.1.55-1` and `12.1.105-1`, alongside toolkit metapackages `12.1.0-1` and `12.1.1-1`.

## Version pins

Every pin below exists on the index it is installed from (HTTP 200 on the release's JSON
or the wheel listing) and, where the harness also runs it, matches what is installed in
the `libero-vla` environment that produced every measurement.

| pin | where | exists upstream | installed here |
|---|---|---|---|
| torch 2.2.0 (+cu121) | `deploy/Dockerfile` | `torch-2.2.0+cu121-cp310-cp310-linux_x86_64.whl` listed | 2.2.0+cu121 |
| torchvision 0.17.0 | `deploy/Dockerfile` | `torchvision-0.17.0+cu121-cp310...whl` listed | 0.17.0+cu121 |
| numpy 1.26.4 | both Dockerfiles | PyPI 200 | 1.26.4 |
| transformers 4.40.1 | `deploy/Dockerfile` | PyPI 200 | 4.40.1 |
| pyzmq 27.2.0 | `deploy/Dockerfile` | PyPI 200 | 27.2.0 |
| pybind11 3.1.0 | both Dockerfiles | PyPI 200 | 3.1.0 |
| mujoco 3.1.6 | `deploy/Dockerfile` | PyPI 200 | 3.1.6 |
| ruff 0.17.0 | `deploy/Dockerfile.cpu` | PyPI 200 | 0.17.0 |
| pytest 8.3.3 | `deploy/Dockerfile.cpu` | PyPI 200 | **9.1.1** locally — the CPU image pins 8.3.3 and CI runs the suite under it; the local runs in this repo's logs used 9.1.1 |
| Python 3.10 | `pyproject.toml`, both Dockerfiles | — | 3.10.20 |

## Facts about this machine

Measured by running the command, on 2026-10-15, and reproduced in `CONTAINERS.md`:

| fact | command | result |
|---|---|---|
| two RTX A5000, driver 530.30.02 | `nvidia-smi --query-gpu=name,driver_version` | `NVIDIA RTX A5000, 530.30.02` (×2) |
| host OS | `/etc/os-release` | Ubuntu 20.04.6 LTS |
| gcc | `gcc --version` | 9.4.0 |
| Apptainer | `apptainer --version` | 1.5.4-1.el8 |
| user namespaces allowed | `cat /proc/sys/kernel/unprivileged_userns_clone` | `1` |
| `newuidmap`/`newgidmap` absent | `which newuidmap newgidmap` | nothing on PATH |
| `/dev/fuse` world-writable | `ls -la /dev/fuse` | `crw-rw-rw- 1 root root 10, 229` |
| Docker socket root-only | `ls -la /var/run/docker.sock` | `srw-rw---- 1 root docker` |

## Citations into the research tree

Paths are relative to `/data_new/vchandr2/vla-safety-interp/src/displacement` (the `DISP`
directory the harness imports from), **not** to that repository's root. Each was opened
and the quoted number read out.

| claim | file | verified value |
|---|---|---|
| reference probe: +0.5639 held-out Spearman, 2040 control samples, 1313 pre-contact | `out/PROBE_ONPOLICY.txt` | "leave-one-init-out ridge, pre-contact control timesteps: held-out Spearman +0.5639"; "control samples 2040 from 60 inits; pre-contact 1313" |
| reference probe: alpha 1e5, target log10 clearance floored at 1e-4 | `onpolicy_probe.py` | `ALPHA = 1e5`; `y.append(np.log10(max(M[i, 1], 1e-4)))` |
| zero-arm baseline 29/60 collisions, 25/60 completions | `docs/RESULTS.md` | `| zero | 60 | 29/60 | 25/60 | 20 | 0.15 | 0 | 0 |` |
| probe-gated steering: 14/60, 18/60, 0 rescued, 3 lost | `docs/RESULTS.md` | `| P1 (probe-gated) | 60 | 14/60 | 18/60 | 17 | 2.89 | 0 | 3 |` |
| 17500 intrusion-5 reference: 40 collisions, 5 completions | `out_instruct/rows_I0_intr5.json` | recounted from the file: 48 rows, 40 collisions, 5 completions, every row `model: openvla_oft_17500`; md5 `46be102ace9e2c053b06b5d4e738e32a` |
| intrusion-5 feasible set covers inits 0-9 only | `out/feasible_set.json` | read; `arms["5"]` yields 48 placements across inits 0-9 |

A claim that does **not** survive, and is recorded as a correction in `EVAL.md` section 2
and in the M7b step 1 commit: the milestone's gate of "40/48 collisions, 1/48 completions"
cited `FT_LIFTONLY_RETEST.txt`. No file of that name exists anywhere on this box. The "1"
belongs to `out_ft_retest/rows_ft3000_intr5.json`, a ft_liftonly@2000 run whose rows are
mislabelled `model: openvla_oft_17500`.

## Third-party licences

Read from the LICENSE file in the local checkout where one exists, otherwise from the
GitHub API for the repository.

| project | licence | source |
|---|---|---|
| OpenVLA-OFT | MIT, © 2025 Moo Jin Kim, Chelsea Finn, Percy Liang | `openvla-oft/LICENSE` |
| LIBERO-Safety | Apache-2.0 | GitHub API (`license.spdx_id`); the local checkout carries no LICENSE file |
| LIBERO (upstream) | MIT | GitHub API |
| robosuite 1.4 | MIT, © 2022 Stanford Vision and Learning Lab, UT Robot Perception and Learning Lab | `LIBERO-Safety/third_party/robosuite-1.4/LICENSE` |
| openpi | Apache-2.0 | `openpi/LICENSE` |
| MuJoCo | Apache-2.0 | installed package metadata |
| PyTorch | BSD-3-Clause | installed package metadata |
| transformers | Apache-2.0 | installed package metadata |
| pyzmq, pybind11 | BSD-3-Clause | installed package metadata |

## Code copied from elsewhere

| what | source | how handled |
|---|---|---|
| `audit/run_with_alert.py` | research tree, verbatim | md5 asserted by `tests/test_wrapper_copy_verbatim.py`; excluded from ruff so no fixer can touch it; provenance in `audit/PROVENANCE.txt` |
| Machado/Oliveira/Fernandes (2009) CVD matrices and the OKLab conversions in `fastvla/monitor/check_colors.py` | the bundled `dataviz` skill's `scripts/validate_palette.js` | copied verbatim with the source named in the module docstring; the port reproduces that skill's published contrast figure for the status red (4.68) exactly, which is the check that it was transcribed correctly. The 2009 attribution is the skill's, repeated, not independently verified against the paper |

## Nothing else

Searched for `http`/`https` across every `*.md`, `*.toml`, `*.yml`, `*.py`, `Dockerfile*`
and `*.sh` in the repo; the URLs above are all of them. `.pytest_cache/README.md` is
tool-generated and ignored.
