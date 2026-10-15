"""Worker-side client for the Triton server, same interface as the ZeroMQ PolicyClient.

Swapping transports is then a one-line change in the worker, and the 60-start comparison
measures the serving layer rather than two different client designs.

Triton is served with max_batch_size 1 and no dynamic batching, matching the accepted
configuration, so the batch size reported back is always 1.
"""

import contextlib
import time

import numpy as np
import tritonclient.grpc as grpcclient


class TritonPolicyClient:
    def __init__(self, url, worker_id, run_id, model="openvla_oft", timeout_s=300.0):
        self.cli = grpcclient.InferenceServerClient(url=url, verbose=False)
        self.model = model
        self.worker_id = worker_id
        self.run_id = run_id
        self.timeout_s = timeout_s
        self.n_requests = 0
        self.batch_sizes = []
        self.queue_waits = []
        self.last_activation = None  # Triton path does not carry activations
        self.last_monitor_score = None  # nor the monitor score: M7b is the ZeroMQ path
        self.last_monitor_held = False
        self.last_monitor_alarm = False
        self.round_trips = []
        self.server_forwards = []

    def hello(self, n_rollouts):
        if not self.cli.is_server_live():
            raise RuntimeError("triton server is not live")
        if not self.cli.is_model_ready(self.model):
            raise RuntimeError(f"triton model {self.model} is not ready")

    def bye(self):
        pass

    def get_chunk(self, rollout_key, policy_step, instruction, full_image, wrist_image, state):
        """-> (8, 7) action chunk. max_batch_size 1, so every tensor carries a batch dim."""
        f = np.ascontiguousarray(full_image, dtype=np.uint8)[None, ...]
        w = np.ascontiguousarray(wrist_image, dtype=np.uint8)[None, ...]
        s = np.ascontiguousarray(state, dtype=np.float64).reshape(1, -1)
        instr = np.array([[instruction.encode("utf-8")]], dtype=object)
        ins = [
            grpcclient.InferInput("full_image", f.shape, "UINT8"),
            grpcclient.InferInput("wrist_image", w.shape, "UINT8"),
            grpcclient.InferInput("state", s.shape, "FP64"),
            grpcclient.InferInput("instruction", instr.shape, "BYTES"),
        ]
        ins[0].set_data_from_numpy(f)
        ins[1].set_data_from_numpy(w)
        ins[2].set_data_from_numpy(s)
        ins[3].set_data_from_numpy(instr)
        t0 = time.perf_counter()
        out = self.cli.infer(
            model_name=self.model,
            inputs=ins,
            outputs=[grpcclient.InferRequestedOutput("actions")],
            client_timeout=self.timeout_s,
        )
        rt = 1000.0 * (time.perf_counter() - t0)
        actions = out.as_numpy("actions")
        if actions.ndim == 3:
            actions = actions[0]
        self.n_requests += 1
        self.round_trips.append(rt)
        self.batch_sizes.append(1)
        # Triton reports queue and compute time only in aggregate statistics, not per
        # request, so the per-request split is read from the model statistics after the
        # run (see bench/triton_stats.py); these stay 0 here rather than being invented.
        self.queue_waits.append(0.0)
        self.server_forwards.append(0.0)
        return np.asarray(actions, dtype=np.float64)

    def close(self):
        with contextlib.suppress(Exception):
            self.cli.close()
