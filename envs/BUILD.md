# Rebuilding the environments

Author: Varun Chandra. Written 2026-10-22, when the box was cleared down to the two repos, the
two kept checkpoints and `learn-gpu`.

`<name>.yml` and `<name>.requirements.txt` in this directory are `conda env export --no-builds`
and `pip freeze` taken from the live environments on 21 Oct 2026, before they were deleted.
They are necessary but **not sufficient**: four things in these environments cannot be expressed
in either file, and a `conda env create -f` alone produces an environment that imports but does
not work. Those four are below, in the order they must be done.

```
fastvla-eval/envs/     planner.yml  libero-vla.yml  learn-gpu.yml  (+ .requirements.txt each)
vla-safety-interp/envs/  libero-vla-finetune.yml  libero-safety.yml  (+ .requirements.txt each)
```

| env | python | numpy | torch | what it ran |
|---|---|---|---|---|
| `planner` | 3.10.22 | 1.26.4 | 2.2.0+cu121 | milestone 7a, cuRobo motion planning |
| `libero-vla` | 3.10.20 | 1.26.4 | 2.2.0+cu121 | rollouts, the displacement driver, **`make bench`** |
| `libero-vla-finetune` | 3.10.20 | 1.26.4 | 2.2.0 | fine-tuning, TFDS conversion, all analysis |
| `libero-safety` | 3.10.20 | 1.26.4 | 2.5.1+cu121 | the benchmark and the physics monitor |
| `learn-gpu` | 3.10.20 | 2.2.6 | 2.5.1+cu121 | lint and tests, CPU only |

---

## 1. cuRobo and warp — order matters, and the clone is required

**The `planner` environment did not install cuRobo from a wheel.** It is an *editable* install
whose `.pth` points straight into the repo:

```
site-packages/__editable__.nvidia_curobo-0.0.0.pth
  -> /data_new/vchandr2/fastvla-eval/fastvla/planner/third_party/curobo/src
```

`curobo.__file__` resolved to `fastvla/planner/third_party/curobo/src/curobo/__init__.py`. So
**a rebuild needs the vendored clone re-created**; `pip install nvidia-curobo` is not what this
project ran and is not what the measurements were taken against. `third_party/` was deleted in
the cleanup (614 M, and it is third-party code, not ours). Re-create it:

```bash
cd fastvla/planner
mkdir -p third_party && cd third_party
git clone https://github.com/NVlabs/curobo.git
cd curobo && git checkout v0.7.6          # commit 2fbffc35225398cf9d5f382804faa9de2608753b
```

Then, **in this order**:

```bash
pip install numpy==1.26.4                 # BEFORE warp and cuRobo
pip install warp-lang==1.4.2
cd fastvla/planner/third_party/curobo && pip install -e . --no-build-isolation
```

- **`numpy<2` is not optional.** torch 2.2.0+cu121 and warp 1.4.2 are both built against the
  numpy 1.x ABI. With numpy 2.x installed, warp kernels fail at load with an ABI error and
  cuRobo's first `MotionGen` call dies. Pin 1.26.4 first, and do not let a later `pip install`
  quietly upgrade it — check with `pip list | grep numpy` after every install step.
- `--no-build-isolation` keeps pip from fetching a fresh numpy into the build env and compiling
  cuRobo's extensions against the wrong ABI.
- First import compiles warp kernels and takes several minutes; the cache then lives in
  `~/.cache/warp`. A cold `MotionGen` construction after that is about 40 s on an RTX A5000 (sm_86).

## 2. flash-attn — not installed, deliberately

There is no flash-attn build step. `docs/SETUP_NOTES.md:172` in the research repo records the
decision: **flash-attn SKIPPED — inference uses `attn_implementation="sdpa"`.** It appears in no
`pip freeze` of any of the five environments. If a future run wants it, it is a from-source
build (`pip install flash-attn --no-build-isolation`) that needs a matching CUDA toolchain and
takes 20–40 minutes; nothing measured in either repo depends on it.

## 3. MuJoCo, EGL and the LIBERO config

These are environment variables, so `conda env export` cannot carry them. `libero-vla` and
`libero-safety` set them from `etc/conda/activate.d/libero_env.sh`; recreate that file:

```bash
cat > $CONDA_PREFIX/etc/conda/activate.d/libero_env.sh <<'EOF'
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LIBERO_CONFIG_PATH=/data_new/vchandr2/.libero
export MUJOCO_GL=egl
export PIP_CACHE_DIR=/data_new/vchandr2/.cache/pip
export HF_HOME=/data_new/vchandr2/.cache/huggingface
EOF
```

Per-run variables, which every GPU job in both repos sets explicitly:

```
MUJOCO_GL=egl                                 headless rendering
MUJOCO_EGL_DEVICE_ID=1                        GPU 1. Must match CUDA_VISIBLE_DEVICES, or the
                                              renderer and the policy land on different GPUs
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python protobuf C++ backend segfaults with this tfds build
PYTHONPATH=/data_new/vchandr2/rollout_stubs   shim every rollout script needs
```

For **CPU-only rendering** (no GPU, as the 21 Oct place-pose renders used) swap the first two for
`MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` and leave `CUDA_VISIBLE_DEVICES` empty. `libOSMesa.so.8`
is a system library, already present at `/usr/lib/x86_64-linux-gnu/`.

`LIBERO_CONFIG_PATH` points at `/data_new/vchandr2/.libero`, which holds the one-file config
naming the benchmark paths. It survived the cleanup. If it has not, LIBERO writes a fresh one on
first import and prompts for the dataset root.

## 4. LIBERO-Safety — clone, patch, editable install

**`libero` and `robosuite` were editable installs pointing into the checkout**, not PyPI packages:

```
__editable___libero_0_1_0_finder.py     -> /data_new/vchandr2/LIBERO-Safety/libero
__editable___robosuite_1_4_0_finder.py  -> /data_new/vchandr2/LIBERO-Safety/third_party/robosuite-1.4/robosuite
```

so the checkout must exist before either import works. The 18 G checkout was deleted; the local
changes are one patch file in the research repo.

```bash
git clone https://github.com/LIBERO-SAFETY/LIBERO-Safety.git
cd LIBERO-Safety
git checkout ef0f79b70fc50c5fb612a1bbc1cf8b6c033a702a      # upstream, 2026-06-23
git apply ../vla-safety-interp/src/displacement/patches/libero_safety_env_import_fix.patch
cp ../vla-safety-interp/src/displacement/patches/libero_safety__libero__init__.py libero/__init__.py
(cd libero/libero/bddl_files/libero_spatial && ln -s . L0)
(cd libero/libero/init_files/libero_spatial  && ln -s . L0)
pip install -e .
pip install -e third_party/robosuite-1.4
```

The patch's own header repeats this and explains what the diff does (it is an import fix for
`wand`/ImageMagick in `envs/env_wrapper.py`, not a scoring change, despite the filename).

## 5. Where the checkpoint goes

`make bench` resolves its checkpoint in `bench/triton_serve.sh:17`:

```
BASE=openvla-7b-oft-libero-combined+libero_safety+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--libero_safety--warmstart-combined--headinit
CKPT=${CKPT:-/data_new/vchandr2/runs/$BASE--17500_chkpt}
```

`runs` is a symlink to `checkpoints`, so that path resolves to
`/data_new/vchandr2/checkpoints/…--headinit--17500_chkpt`. **That directory is kept** and is the
reference policy every number in both repos is measured against. If the tree moves, either
recreate the `runs -> checkpoints` symlink or set `CKPT` explicitly; nothing else needs changing.

`bench/bench_all.sh:26` hardcodes its interpreter:

```
PY=${PY:-/data_new/vchandr2/miniconda3/envs/libero-vla/bin/python}
```

so `make bench` needs the **`libero-vla`** environment rebuilt from `libero-vla.yml`, not
`learn-gpu`. Override with `PY=<interpreter> make bench` if it lives elsewhere.

## 5b. Installing from these files — two things that bite, both measured

The `learn-gpu` environment was rebuilt from `libero-vla.yml` + `libero-vla.requirements.txt` on
22 Oct 2026 to check these specs actually work. They do, but not with the obvious commands.

**`conda env create -f libero-vla.yml` fails at its embedded pip stage.** The conda layer
installs fine (python 3.10.20 appears); the pip stage dies with `CondaEnvException: Pip failed`.
The real error is three screens up:

```
ERROR: Could not find a version that satisfies the requirement torch==2.2.0+cu121
ERROR: No matching distribution found for torch==2.2.0+cu121
```

`pip freeze` writes the local version suffix `+cu121` but not the index that wheel came from, and
`torch==2.2.0+cu121` is not on PyPI. Every environment here except `learn-gpu`'s original has a
`+cu121` torch, so every one of them hits this. Add the index:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r <name>.requirements.txt
```

**That still fails, with `ResolutionImpossible`.** pip's resolver rejects the set:

```
Cannot install ... requests==2.28.1, sentry-sdk==2.66.1, tritonclient==2.73.0
and urllib3==1.26.13 because these package versions have conflicting dependencies.
```

These environments were built incrementally over months, and the resulting closure is not one pip
can re-derive in a single resolve. A `pip freeze` *is* already the complete closure, so replay it
as one rather than asking pip to solve it again:

```bash
pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r <name>.requirements.txt
```

That installed all 163 non-VCS packages in one pass, with numpy 1.26.4 and torch 2.2.0+cu121
intact.

**The four VCS entries are pinned git URLs, not local paths** — `pip freeze` recorded them
properly, so they are installable from a clean box:

```
dlimp        git+https://github.com/moojink/dlimp_openvla@040105d256bd28866cc6620621a3d5f7b6b91b46
libero       git+https://github.com/LIBERO-SAFETY/LIBERO-Safety.git@ef0f79b70fc50c5fb612a1bbc1cf8b6c033a702a
robosuite    the same repo and commit, subdirectory=third_party/robosuite-1.4
openvla_oft  git+https://github.com/moojink/openvla-oft.git@e4287e94541f459edc4feabc4e181f537cd569a8
transformers git+https://github.com/moojink/transformers-openvla-oft.git@bc339d9ad707454c0c115970db43c260067c61ab
```

This is how `openvla-oft` is recovered after the cleanup deleted the local checkout — at a pinned
commit, which is better than what was on disk. **But note:** installing `libero` through that URL
gives you *upstream*, without the patch in section 4. For anything that creates environments, do
section 4 by hand and skip these two lines; `pip install --no-deps` leaves them out anyway.

They were skipped for `learn-gpu`: `tests/` imports only `fastvla`, `numpy`, `pytest` and stdlib,
and the `libero` line would clone an 18 G repository this cleanup had just removed.

**Dev tools added on top** (none were in the original `learn-gpu`; it had never run lint or tests):

```bash
pip install pytest ruff pybind11 line_profiler jupyter
#            9.1.1  0.17.0  3.1.0    5.1.0         1.1.1
```

Result in the rebuilt environment, CPU only:

```
make test   55 passed in 4.19s                                exit 0
make lint   141 errors, ALL in fastvla/planner/, 0 elsewhere  exit 2
```

The lint failure is **not** an environment problem — `libero-vla` carries the same ruff 0.17.0.
It is pre-existing: the planner module was merged to main in `6adaa8b` without `make lint` having
been run against it. Breakdown: E702 x27, I001 x25, SIM115 x23, E701 x14, E402 x13, SIM105 x9,
E401 x8, F401 x7, B023 x5, B007 x4, and 6 others. 43 are auto-fixable. Nothing outside
`fastvla/planner/` reports a single error, so the rest of the repo is clean.

## 6. The order that works

```
1  conda env create -f <name>.yml
2  pip install numpy==1.26.4                       (except learn-gpu, which is numpy 2.2.6)
3  clone + patch + editable-install LIBERO-Safety  (section 4)
4  clone + editable-install cuRobo, after warp     (section 1, planner only)
5  recreate activate.d/libero_env.sh               (section 3)
6  pip install -r <name>.requirements.txt          last, and re-check numpy afterwards
```

Step 6 last and re-checked, because the requirements file pins transitive versions that can pull
numpy forward; the pin in step 2 is what the measurements were taken under.
