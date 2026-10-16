# FastVLA-Eval

**A fast, trustworthy test bench for robot AI models, with a built-in warning light that predicts a collision before it happens.**

## What I built

- Profiled a 7-billion-parameter robot-control model (OpenVLA-OFT) end to end with Nsight Systems and torch.profiler, and found that 71% of each step was Python geometry code, not the model.
- Rewrote that geometry (GJK collision distance) in C++17 with pybind11 bindings: 63.6 ms to 0.58 ms per call, 110x faster, matching the Python original to 2e-16 metres over 168,000 calls.
- Built a policy server that holds one copy of the model on the GPU and serves eight simulator processes in parallel over ZeroMQ. GPU utilisation went from 10% to 85%.
- Result: the standard 60-trial evaluation went from 51 minutes to 7 minutes on one RTX A5000, 7.6x faster, with the same collision verdict on every trial.
- Wrote a per-trial correctness gate that every change must pass. Two speed-ups (request batching, torch.compile) were faster but changed 3 to 5 verdicts out of 60, so they were rejected.
- Added a runtime collision monitor: a linear probe on the model's own activations that predicts contact within the next action chunk at AUROC 0.92, with 74 control steps of warning, for 0.07 ms per call.
- Packaged it: Docker images built and tested in GitHub Actions, Kubernetes manifests, Prometheus metrics, a Triton Inference Server backend for comparison, Apptainer for running on a shared box without root, and `make bench` that reproduces every number from a fresh clone.

## Why this exists

Robots are now driven by large AI models, the same kind that write text. Before anyone trusts one near real objects, it is tested in simulation. You run the task hundreds of times and count how often the arm hits something.

That test used to take most of an hour, and the graphics card sat idle almost the whole time. This tool runs the same test in seven minutes and gives the same answer on every trial. It also watches the model's internal signals and warns you before a crash.

## What you get

### 1. A faster test bench

The standard test is 60 trials. It took **51 minutes**. It now takes **7 minutes**. Every trial still gets the same verdict: hit, or no hit.

<p align="center">
  <img src="docs/media/rollout.gif" width="420"
       alt="A simulated robot arm picking up a bowl and placing it on a plate">
</p>

<sub>One trial: the arm completes the task without touching the obstacle. Closest approach 5.84 cm. Shown at 1.6x speed.</sub>

How: the slow part was not the AI model. It was the code that measures how close the arm is to each obstacle. It ran in Python, twice per step. Rewriting it in C++ made it 110x faster. Then eight simulators run at once, all sharing one copy of the model on one GPU.

Full numbers, each traceable to the file that produced it: **[docs/BENCHMARK.md](docs/BENCHMARK.md)**.

### 2. A warning light

The model's internal activations already know how close the arm is to an obstacle. A small linear read of them predicts a collision before it happens, almost for free.

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

Method and full results: **[fastvla/monitor/EVAL.md](fastvla/monitor/EVAL.md)**.

## How we know the answers did not change

Every change is checked trial by trial against the original, not on totals. Two speed-ups were faster and were still switched off: batching requests, and compiling the model. Each changed the verdict on 3 to 5 of the 60 trials. One of them matched the correct totals while getting four individual trials wrong, which is why the check is per trial. Details: **[docs/CORRECTNESS.md](docs/CORRECTNESS.md)**.

## Try it

```bash
git clone git@github.com:alpha9centauri/fastvla-eval.git && cd fastvla-eval
python -m pip install -e .     # installs the package; it declares no dependencies
make labeler                   # compiles the C++ geometry core
make test                      # 55 checks, no GPU needed, a few seconds
make bench-quick               # the end-to-end timings, about 8 minutes on a GPU
```

Running trials needs a GPU, the simulator and a model checkpoint. None of those are included here. `docs/ARCHITECTURE.md` explains what talks to what.

## Limits

- The monitor is fitted to this one model and benchmark. On scenes it never saw, its accuracy drops by about half.
- The automatic stop is a brake, not avoidance. It halts the arm and costs finished tasks.
- The Kubernetes manifests in `deploy/` have not been run against a cluster.

## Acknowledgements

- [LIBERO-Safety](https://github.com/LIBERO-SAFETY/LIBERO-Safety) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), for the benchmark.
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft), for the model.
- [openpi](https://github.com/Physical-Intelligence/openpi), for the serving patterns.
- **Heng Zhang** and the **Safe AI Lab at Carnegie Mellon University**, for the research this was built to serve.

## Licence

MIT, see [LICENSE](LICENSE). Third-party components keep their own licences, listed there. How every external fact and version in this repository was checked is in [docs/PROVENANCE.md](docs/PROVENANCE.md).
