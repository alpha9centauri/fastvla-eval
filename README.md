# FastVLA-Eval

**7.6x faster evaluation of VLA policies, with a collision monitor built on interpretability of the model's activations.**

Two things, one repo:

| | Part 1: the evaluation harness | Part 2: the collision monitor | Part 3: the post-training pipeline |
|---|---|---|---|
| What it is | A faster, correct test bench for robot AI models | A warning light read from inside the model | Teaching the model to lift over the obstacle |
| Headline | 60-trial test: 51 min to 7 min on one GPU, same verdict on every trial | Predicts contact within the next action chunk at AUROC 0.92, 74 control steps early, for 0.07 ms | Collisions 29 to 21 on the 60-trial test: 12 prevented, 4 created, 5 completions lost |
| Skills | profiling, C++, parallel GPU serving, containers, CI | interpretability, linear probes, detection metrics, safety | data pipelines, LoRA fine-tuning, mixture design, honest evaluation |
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

- Converted the benchmark from LeRobot parquet and MP4 to RLDS/TFDS: **19,606 trajectories, 3,431,183 transitions**, 278 GB. A parallel builder writes the same total payload to the byte as the single-process one, resharded, cutting a day of wall clock.
- Generated 99 demonstrations with a planner that lifts over the obstacle, and registered them as a 25/75 mixture with the original data. The generated set is normalised with the **base set's** statistics, not its own; its action range is half as wide, and normalising it separately would have taught two different action languages at once.
- Continued the LoRA fine-tune of the 7B policy on one 24 GB card: rank 32, alpha 16, lr 5e-5, batch 1 x 8 accumulation, action head and proprio projector warm-started from the base checkpoint, fresh zero-init adapter so step 0 is exactly the base policy. 278,764,295 trainable parameters, 4.83 s/it, 4 h 25 m for 3000 steps, L1 loss 0.0495 to 0.0209.
- Evaluated the result through Part 1's harness and reported what it actually did, per trial, including the parts that got worse.

### What you get

The post-trained checkpoint against the base, 60 trials, same harness:

| | base | post-trained |
|---|---|---|
| collisions | 29 | **21** |
| completions | 25 | **20** |
| per trial | — | **12 collisions prevented, 4 created, 8 completions lost, 3 gained** |

Eight fewer collisions, paid for with five completions net. The honest reading is in
[docs/POSTTRAIN.md](docs/POSTTRAIN.md): on the trials where contact is prevented the arm
does not stop — it runs longer and clears by 1.5 cm — while the lift the demonstrations
taught shows up mostly where it was least needed, 22 mm higher on trials that already
cleared the obstacle. The direction is right; the mechanism is not the one the data was
designed to teach.

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
