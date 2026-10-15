# Kubernetes manifests

| file | what |
|---|---|
| `pvc.yaml` | checkpoints (ReadOnlyMany, 64Gi) and results (ReadWriteMany, 16Gi) |
| `server-deployment.yaml` | the policy server, `nvidia.com/gpu: 1`, readiness and liveness on `/healthz`, Prometheus scrape annotations on `:9400/metrics` |
| `worker-job.yaml` | N workers, `parallelism: 8` in Indexed completion mode so each pod gets its own `WORKER_ID` |
| `configmap-manifest.yaml` | the run configuration as data: which manifest, N, B, D |

Apply in that order:

    kubectl apply -f pvc.yaml -f configmap-manifest.yaml -f server-deployment.yaml
    kubectl wait --for=condition=available deployment/fastvla-server --timeout=15m
    kubectl apply -f worker-job.yaml

## Not tested on a cluster

**These manifests have not been applied to a running cluster.** There is no Kubernetes on
gpu5, and no kind or minikube either -- both want a container runtime this account cannot
drive (Docker's socket is root-only; unprivileged Apptainer is not a CRI). They are
checked for schema validity only, by `kubectl --dry-run=client` in CI when a `kubectl` is
available, and they encode the configuration M3 and M4 measured. Treat them as a
reviewed starting point, not as a tested deployment, and say so in any write-up.

Two things a real cluster will need that a single box did not:

- **Workers requesting a whole GPU each.** On gpu5 eight workers share one card, because
  they need it only for EGL rendering (0.7 ms per camera). `nvidia.com/gpu: 1` per worker
  is what plain Kubernetes can express; sharing needs time-slicing or MPS through the
  NVIDIA device plugin, or the workers must be co-scheduled onto the server's node.
- **The ipc:// socket becomes tcp://.** On one box the server and workers share a Unix
  socket; across pods the endpoint is the Service. That costs the loopback-versus-TCP
  difference on a 301 KB payload, which has not been measured.
