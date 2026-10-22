# FastVLA-Eval

**7.6x faster evaluation of VLA policies, with a collision monitor built on interpretability of the model's activations.**

Two things, one repo:

| | Part 1: the evaluation harness | Part 2: the collision monitor | Part 3: the post-training pipeline |
|---|---|---|---|
| What it is | A faster, correct test bench for robot AI models | A warning light read from inside the model | The full path from raw demonstrations to a post-trained policy, on one 24 GB GPU |
| Headline | 60-trial test: 51 min to 7 min on one GPU, same verdict on every trial | Predicts contact within the next action chunk at AUROC 0.92, 74 control steps early, for 0.07 ms | Post-trained OpenVLA-7B with LoRA on a 25/75 data mixture: collisions on the standard 60-trial test 29 to 21 |
| Skills | profiling, C++, parallel GPU serving, containers, CI | interpretability, linear probes, detection metrics, safety | LoRA fine-tuning, OXE data mixtures, RLDS/TFDS data pipelines, dataset normalisation, single-GPU training |
| Details | [docs/BENCHMARK.md](docs/BENCHMARK.md), [docs/CORRECTNESS.md](docs/CORRECTNESS.md) | [fastvla/monitor/EVAL.md](fastvla/monitor/EVAL.md) | [docs/POSTTRAIN.md](docs/POSTTRAIN.md) |

## Why this exists

Robots are now driven by large AI models, the same kind that write text. Before anyone trusts one near real objects, it is tested in simulation: run the task hundreds of times, count how often the arm hits something.

That test used to take most of an hour, with the GPU idle almost the whole time. Part 1 makes it fast without changing the answer. Part 2 uses what the model already knows, inside its own activations, to warn you before a crash.

---

# Part 1: The evaluation harness

### What I built

- Profiled a 7-billion-parameter robot-control model (OpenVLA-OFT) end to end with Nsight Systems and torch.profiler. 71% of each step was Python geometry code, not the model.
- Rewrote that geometry (GJK collision distance) in C++17 with pybind11: 63.6 ms to 0.58 ms per call, 110x faster, matching the Python original to 2e-16 m over 168,000 calls.
- Built a policy server that holds one copy of the model on the GPU and serves eight simulator processes in parallel over ZeroMQ. GPU utilisation 10% to 85%.
- Result: the standard 60-trial evaluation went from 51 minutes to 7 minutes on one RTX A5000, 7.6x, with the same collision verdict on every trial.
- Packaged it: Docker images built and tested in GitHub Actions, Kubernetes manifests, Prometheus metrics, a Triton Inference Server backend, Apptainer for shared boxes without root, and `make bench` that reproduces every number from a fresh clone.

### What you get

The standard test is 60 trials. It took **51 minutes**. It now takes **7 minutes**. Every trial still gets the same verdict: hit, or no hit.

<p align="center">
  <img src="docs/media/rollout.gif" width="420"
       alt="A simulated robot arm picking up a bowl and placing it on a plate">
</p>

<sub>One trial: the arm completes the task without touching the obstacle. Closest approach 5.84 cm. Shown at 1.6x speed.</sub>

How: the slow part was not the AI model. It was the code that measures how close the arm is to each obstacle. It ran in Python, twice per step. Rewriting it in C++ made it 110x faster. Then eight simulators run at once, all sharing one copy of the model on one GPU.

### How we know the answers did not change

Every change is checked trial by trial against the original, not on totals. Two speed-ups were faster and were still switched off, because each changed the verdict on 3 to 5 of the 60 trials. Full gates and numbers: [docs/CORRECTNESS.md](docs/CORRECTNESS.md).

---

# Part 2: The collision monitor

### What I built

- Trained a linear probe on the model's internal activations (the hidden state at the action-token positions) to read how close the arm is to an obstacle: Spearman 0.87 against the true distance on held-out scenes.
- Turned it into a runtime monitor inside the server: predicts contact within the next action chunk at AUROC 0.92, fires a median 74 control steps before contact, costs 0.07 ms per model call, changes no verdicts when only logging.
- Chose the operating point from the recall vs false-alarm curve on training scenes only, with a two-call debounce.
- Wired it to an automatic stop and measured it per trial: 18 of 29 collisions prevented, none created, 13 tasks left unfinished.

### What you get

The model's activations already know how close the arm is to an obstacle. A small linear read of them predicts a collision before it happens, almost for free.

The model gives actions eight at a time. Those eight are one *chunk*; each one is a *control step*. A full task is about 300 steps.

- **AUROC 0.918** for contact within the next chunk. (AUROC measures how well a score separates two cases: 1.0 is perfect, 0.5 is a coin flip.)
- **74 control steps** of warning before contact, median.
- **0.067 ms** per model call, which is 0.04% of one call.

Wired to an automatic stop, on the 60 trials: **18 of 29 collisions prevented, none created, 13 tasks left unfinished.**

<p align="center">
  <img src="docs/media/monitor_stop.gif" width="700"
       alt="The same trial twice: with the monitor off the arm knocks the bottle over, with it on the arm stops just short">
</p>

<sub>The same trial run twice; only the monitor setting changed. Left: the arm knocks the bottle over at step 129. Right: the monitor fires and the arm is held from step 128. Closest approach 1.59 cm.</sub>

It stops the arm where it is. It does not steer around the obstacle, and it does not resume. A trial it saves is also a trial it ends.

---

# Part 3: The post-training pipeline

### What I built

- Converted the LIBERO-Safety demonstrations (19,606 episodes, 3.4 million steps, 279 GB) from LeRobot parquet and video into the RLDS/TFDS format OpenVLA trains on, with a parallel builder that writes the same payload to the byte in a fraction of the time.
- Fine-tuned OpenVLA-7B on it with the OFT recipe on a single 24 GB RTX A5000: LoRA plus a trained action head and proprio projector (279 M trainable parameters, 3.6% of the model), batch 1 with gradient accumulation, a checkpoint ladder where success rose from 25% to 56%.
- Post-trained the result with continued LoRA (rank 32) on a named data mixture: 99 generated demonstrations at 25% of sampled frames alongside the original data, with the base set's normalisation statistics shared so both speak one action language. 3,000 steps at 4.8 s/it.
- Evaluated it with Part 1 and read the per-trial traces: collisions 29 to 21 on the standard 60 trials, 12 prevented and 4 created; completions 25 to 20.

### What you get

Every script to go from a folder of raw demonstrations to a post-trained checkpoint: the converter, the dataset registration, the training recipe, the patches to the upstream training code, and the LoRA merge. Paths live in one config file.

| 60-trial test | base model | post-trained |
|---|---|---|
| collisions | 29 | **21** |
| completions | 25 | 20 |

Honest reading: on the 12 trials where a collision was prevented, the arm goes around and finishes 1.5 cm clear, which is what the demonstrations were meant to teach. The 5 net completions lost are time-outs, not crashes. The generated demonstrations placed obstacles mostly where the base model already cleared them, so the gain is modest; the pipeline is the reusable part.

Recipe, numbers and logs: [docs/POSTTRAIN.md](docs/POSTTRAIN.md).

---

# Part 4: Motion planner integration, and what it found

### What I built

- Replaced the demonstration generator's scripted route stage with a real whole-arm motion planner (cuRobo v0.7.6, CUDA 12.1, torch 2.2.0), bound over `dgcore.plan_detours` at runtime so the research code is imported and never edited. `PLANNER=script|curobo` selects it; everything downstream — the OSC executor, the 2 cm GJK certification, the rendered-replay gate, the LeRobot writer — runs unchanged.
- Built the planner's world from the certifier's own hulls, by import, so a plan the planner calls clear is clear to the thing that certifies it. Verified the two collision checkers agree: the straight-line route collides in every scene tested, and they locate the contact to within 3 mm.
- Measured the whole eval-rule scene set the research could not solve (the 197 scenes where its scripted planner managed 1), with per-scene clearance distributions, failure attribution, and timing.

### What it found

The scripted planner's 1-in-197 was not a planner limitation. On these placements **118 of 129 scenes admit no certifiable demonstration at any positive margin**: the task's own place pose puts the carried bowl in contact with a declared obstacle, measured at exactly 0.00 cm, because the test's placement rule moves the obstacle stack onto the destination. The clearance distribution is bimodal — 0.00 cm in 118 scenes, 9.5–22.4 cm in 11, nothing between.

On the 11 geometrically feasible scenes the planner plans 10 and yields **2 certified demonstrations**, clearing the obstacles by 2.99 and 3.54 cm, which pass both the 1 cm and the original 2 cm floor.

| eval-rule oa1 placements | scripted planner | motion planner |
|---|---|---|
| certifiable demonstrations | 1 of 197 | 2 of the 11 feasible scenes |
| scenes with a usable destination | — | 11 of 129 (8.5%) |

Honest reading: the limiting factor is the scene, not the planner. Raising the yield means changing where the test puts the obstacle, not improving the search. The reusable parts are the integration seam, the world builder that shares geometry with the certifier, and five cuRobo pitfalls that each produced plausible wrong numbers before being caught — a four-sphere cap on attached objects, an attachment mislocated 12.4 cm by being made at the wrong joint configuration, a volume fit that degenerates to a cuboid, `solve_ik` failing after CUDA-graph capture, and a long-lived planner whose per-scene attach/detach does not reset.

Feasibility, world model, results and the bug list: [fastvla/planner/FEASIBILITY.md](fastvla/planner/FEASIBILITY.md), [fastvla/planner/WORLD_MODEL.md](fastvla/planner/WORLD_MODEL.md), [fastvla/planner/RESULTS.md](fastvla/planner/RESULTS.md), [fastvla/planner/RESULTS_4B.md](fastvla/planner/RESULTS_4B.md).

---

## Try it

```bash
git clone git@github.com:alpha9centauri/fastvla-eval.git && cd fastvla-eval
python -m pip install -e .     # installs the package; it declares no dependencies
make labeler                   # compiles the C++ geometry core
make test                      # 55 checks, no GPU needed, a few seconds
make bench-quick               # the end-to-end timings, about 8 minutes on a GPU
```

Running trials needs a GPU, the simulator and a model checkpoint. None of those are included here. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains what talks to what.

### What `make bench` needs, in full

`make test` and `make lint` run from a plain clone with nothing else. `make bench` does not: it
drives a real policy against a real simulator. Four things have to exist first, and none of them
ships with this repository.

**1. The `libero-vla` environment.** `bench/bench_all.sh:23` hardcodes
`PY=${PY:-/data_new/vchandr2/miniconda3/envs/libero-vla/bin/python}`, so this is the interpreter
unless you override `PY`. Build it from [`envs/libero-vla.yml`](envs/libero-vla.yml) and
[`envs/libero-vla.requirements.txt`](envs/libero-vla.requirements.txt) -- but read
[`envs/BUILD.md`](envs/BUILD.md) first, because neither obvious command works:

```bash
conda env create -f envs/libero-vla.yml          # dies at its pip stage. torch==2.2.0+cu121
                                                 # is not on PyPI: a pip freeze records the
                                                 # +cu121 suffix but not the index it came from
pip install --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r envs/libero-vla.requirements.txt          # --no-deps is required. With the resolver on,
                                                 # it fails ResolutionImpossible on urllib3
                                                 # against requests / sentry-sdk / tritonclient.
                                                 # A freeze is already a closure; replay it.
```

**2. LIBERO-Safety, cloned at the recorded commit with the import patch.** `libero` and
`robosuite` are editable installs pointing into the checkout, so neither imports until it exists:

```bash
git clone https://github.com/LIBERO-SAFETY/LIBERO-Safety.git
cd LIBERO-Safety && git checkout ef0f79b70fc50c5fb612a1bbc1cf8b6c033a702a
git apply <research-repo>/src/displacement/patches/libero_safety_env_import_fix.patch
cp <research-repo>/src/displacement/patches/libero_safety__libero__init__.py libero/__init__.py
(cd libero/libero/bddl_files/libero_spatial && ln -s . L0)
(cd libero/libero/init_files/libero_spatial  && ln -s . L0)
pip install -e . && pip install -e third_party/robosuite-1.4
```

Without the patch, upstream's eager `wand`/ImageMagick import in `envs/env_wrapper.py` raises at
module load and breaks all environment creation, not just the corruption path it is there for.

**3. cuRobo, for `fastvla/planner/` only.** Not needed by `make bench`; needed to re-run
milestone 7a. It was an *editable* install from a vendored clone, never a wheel, so
`pip install nvidia-curobo` is not what was measured:

```bash
mkdir -p fastvla/planner/third_party && cd fastvla/planner/third_party
git clone https://github.com/NVlabs/curobo.git
cd curobo && git checkout v0.7.6              # commit 2fbffc35225398cf9d5f382804faa9de2608753b
pip install numpy==1.26.4 && pip install warp-lang==1.4.2   # in this order, before cuRobo
pip install -e . --no-build-isolation
```

The `numpy<2` pin is load-bearing: torch 2.2.0+cu121 and warp 1.4.2 are built against the numpy
1.x ABI, and under numpy 2.x the warp kernels fail at load.

**4. The checkpoint, where `bench/triton_serve.sh` looks for it.** Line 17 reads:

```
CKPT=${CKPT:-/data_new/vchandr2/runs/$BASE--17500_chkpt}
```

`runs` is a symlink to `checkpoints`, so that resolves to
`/data_new/vchandr2/checkpoints/...--headinit--17500_chkpt`. Either put the checkpoint there and
keep the symlink, or set `CKPT` explicitly -- nothing else needs changing. It is a LoRA fine-tune
of `openvla/openvla-7b` (HF revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f`) and is not
public; `make bench` cannot be reproduced without it.

Also required: GPU 1 free, and `MUJOCO_GL=egl` with `MUJOCO_EGL_DEVICE_ID` matching
`CUDA_VISIBLE_DEVICES`, or the renderer and the policy land on different devices.

## Limits

- The monitor is fitted to this one model and benchmark. On scenes it never saw, its accuracy drops by about half.
- The automatic stop is a brake, not avoidance. It halts the arm and costs finished tasks.
- The Kubernetes manifests in `deploy/` have not been run against a cluster.
- The motion planner's yield rests on 11 feasible scenes out of 129; two certified demonstrations is a small number and both come from the same LIBERO init.

## Acknowledgements

- [LIBERO-Safety](https://github.com/LIBERO-SAFETY/LIBERO-Safety) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), for the benchmark.
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft), for the model.
- [openpi](https://github.com/Physical-Intelligence/openpi), for the serving patterns.
- **Heng Zhang** and the **Safe AI Lab at Carnegie Mellon University**, for the research this was built to serve.

## Licence

MIT, see [LICENSE](LICENSE). Third-party components keep their own licences, listed there. How every external fact and version in this repository was checked is in [docs/PROVENANCE.md](docs/PROVENANCE.md).
