"""Triton Python backend for the OpenVLA-OFT policy (M3b).

Wraps the same `fastvla.server.batched_forward.BatchedPolicy` that M3a's ZeroMQ server
uses, so swapping servers changes the serving layer and nothing about the model: the same
two batch-1 assumptions are handled the same way (append the empty token for the whole
batch; the rebound reshape), and parity at batch 1 is therefore the same bit-exact result.

Triton hands `execute` a list of requests that its dynamic batcher has already grouped, so
this backend does not implement batching itself -- it converts the group into one
`BatchedPolicy.forward` call. `max_batch_size` and `max_queue_delay_microseconds` in
config.pbtxt are the B and D of M3a's sweep.

Runs on gpu5 under unprivileged Apptainer (fastvla/server/CONTAINERS.md): plain `--nv`
with `CUDA_VISIBLE_DEVICES=1`, the libero-vla env bound and reached through PYTHONPATH,
started by `bench/triton_serve.sh`. Docker is not available here and is not needed.
"""

import json
import os
import site
import sys

import numpy as np

try:
    import triton_python_backend_utils as pb_utils
except ImportError:  # pragma: no cover - only present inside Triton
    pb_utils = None

SITE_PACKAGES = "/data_new/vchandr2/miniconda3/envs/libero-vla/lib/python3.10/site-packages"

# libero, openvla_oft and robosuite are EDITABLE installs: their __editable__*.pth files
# install import finders, and .pth files are executed only for directories registered
# through `site`, never for a directory that merely sits on PYTHONPATH. Triton's Python
# backend runs the container's own interpreter, so without this the stub fails to
# initialise with "ModuleNotFoundError: No module named 'libero'" even though PYTHONPATH
# is correct -- which is exactly how this first failed.
site.addsitedir(SITE_PACKAGES)

DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for _p in (
    "/data_new/vchandr2",
    "/data_new/vchandr2/openvla-oft",
    DISP,
    "/data_new/vchandr2/fastvla-eval",
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class TritonPythonModel:
    def initialize(self, args):
        import rollout_bootstrap  # noqa: F401
        from experiments.robot.libero.run_libero_eval import GenerateConfig
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
        )
        from experiments.robot.robot_utils import get_model
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK

        from fastvla.server.batched_forward import BatchedPolicy

        base = (
            "openvla-7b-oft-libero-combined+libero_safety+b32+lr-0.0005+lora-r32"
            "+dropout-0.0--image_aug--libero_safety--warmstart-combined--headinit"
        )
        # `or` not `.get(default)`: triton_serve.sh may pass CKPT as an empty string, and
        # .get would then return "" rather than falling back.
        ckpt = os.environ.get("CKPT") or f"/data_new/vchandr2/runs/{base}--17500_chkpt"
        cfg = GenerateConfig(
            model_family="openvla",
            pretrained_checkpoint=ckpt,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_images_in_input=2,
            use_proprio=True,
            center_crop=True,
            num_open_loop_steps=NUM_ACTIONS_CHUNK,
            lora_rank=32,
            env_img_res=256,
        )
        model = get_model(cfg)
        pp = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
        ah = get_action_head(cfg, model.llm_dim)
        proc = get_processor(cfg)
        cfg.unnorm_key = "libero_safety"
        self.policy = BatchedPolicy(cfg, model, proc, ah, pp)
        self._log_device()
        self.model_config = json.loads(args["model_config"]) if "model_config" in args else {}
        self.n_requests = 0
        self.n_batches = 0

    def _log_device(self):
        """Prove the model instance landed on the masked device, as preflight_gpu.py does.

        config.pbtxt pins the instance to GPU 1 via `gpus: [1]`, but inside the container
        the mask has already renumbered the devices, so the instance must be confirmed
        rather than assumed. Logged at load so every server start carries the evidence.
        """
        import subprocess

        import torch

        from fastvla.server.preflight_gpu import norm_bus, pci_bus_id_of_cuda0

        bus = pci_bus_id_of_cuda0()
        uuid = None
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout
            for line in q.strip().splitlines():
                u, b = [x.strip() for x in line.split(",")]
                if norm_bus(b) == norm_bus(bus):
                    uuid = u
                    break
        except Exception as e:
            uuid = f"(nvidia-smi failed: {e})"
        print(
            f"[triton] model instance on cuda:0 device_count={torch.cuda.device_count()} "
            f"pci={bus} uuid={uuid}",
            flush=True,
        )
        self.device_uuid = uuid
        self.device_bus = bus

    def execute(self, requests):
        """One call per dynamic batch Triton has formed."""
        prepared, n_per = [], []
        for request in requests:
            full = pb_utils.get_input_tensor_by_name(request, "full_image").as_numpy()
            wrist = pb_utils.get_input_tensor_by_name(request, "wrist_image").as_numpy()
            state = pb_utils.get_input_tensor_by_name(request, "state").as_numpy()
            instr = pb_utils.get_input_tensor_by_name(request, "instruction").as_numpy()
            text = instr.reshape(-1)[0]
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            # Triton prepends the batch dimension that config.pbtxt's max_batch_size
            # implies, so each request arrives as (1, ...) and is squeezed back.
            full = full.reshape(-1, *full.shape[-3:])
            wrist = wrist.reshape(-1, *wrist.shape[-3:])
            state = state.reshape(-1, state.shape[-1])
            n = full.shape[0]
            n_per.append(n)
            for i in range(n):
                prepared.append(self.policy.prepare_request(full[i], wrist[i], state[i], text))

        chunks = self.policy.forward(prepared)
        self.n_requests += len(prepared)
        self.n_batches += 1

        responses, k = [], 0
        for n in n_per:
            actions = np.stack(chunks[k : k + n], axis=0).astype(np.float64)
            k += n
            out = pb_utils.Tensor("actions", actions)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses

    def finalize(self):
        print(
            f"[triton] {self.n_requests} requests in {self.n_batches} batches "
            f"(mean {self.n_requests / max(1, self.n_batches):.2f})",
            flush=True,
        )
