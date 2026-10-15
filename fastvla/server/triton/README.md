# M3b: the same policy as a Triton Python backend

`models/openvla_oft/` is a complete Triton model repository entry:

- `config.pbtxt` — Python backend, `max_batch_size 4`, `dynamic_batching` with
  `preferred_batch_size [2, 4]` and `max_queue_delay_microseconds 10000`, one GPU
  instance pinned to device 1.
- `1/model.py` — the backend. It wraps the **same** `BatchedPolicy` as M3a's ZeroMQ
  server, so a Triton-vs-M3a comparison measures the serving layer, not the model. Triton's
  dynamic batcher does the grouping; `execute` turns one group into one
  `BatchedPolicy.forward`.

## Why it was not run on gpu5

| requirement | state on this box |
|---|---|
| `tritonserver` binary | not installed |
| Docker | `/usr/bin/docker` exists, but `/var/run/docker.sock` is `root:docker` mode 660 and this user's only group is `vchandr2` |
| podman / singularity / apptainer / nerdctl / enroot | none present |
| `pip install tritonserver` | PyPI has only a 1.5 KB `0.0.1.dev0` placeholder; NVIDIA's in-process wheel is published on their own index, not PyPI |
| Triton for Linux x86_64 | distributed only as the `nvcr.io/nvidia/tritonserver` container image |

So M3b's measurement needs container access this account does not have. Getting it is a
one-line request to whoever administers the box (`usermod -aG docker vchandr2`), and it is
a prerequisite of M5 (containerised deployment) anyway, so M3b is parked against M5 rather
than worked around.

## What was measured instead

The comparison M3b exists to make is Triton's per-request serving overhead against M3a's.
M3a's own overhead is measured and decomposed in `bench/out/m3a_overhead.json` (transport
round-trip, queue wait, CPU preprocessing, GPU forward), so when Triton can run, the
comparison is a single number against a number that already exists.

What the M3 measurements already say about the likely outcome, recorded now so it is a
prediction rather than a rationalisation: batching this model is worth 1.27x at best and
flat past B=4, and any batching at all perturbs bf16 reduction order enough to flip
marginal starts. Triton's dynamic batcher is a better implementation of a mechanism that
is not the bottleneck here. Its real value for this project is the packaging, metrics and
model-repository story in M5, not throughput on one GPU.
