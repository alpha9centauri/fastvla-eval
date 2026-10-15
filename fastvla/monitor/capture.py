"""Capture the probe's activation at the site the on-policy probe used.

The site is not a free choice -- it is fixed by the reference probe this monitor packages
(vla-safety-interp/src/displacement, `out/PROBE_ONPOLICY.txt` and `onpolicy_probe.py`),
and matching it exactly is what lets the published Spearman be a comparison rather than a
coincidence:

  * tensor   the output of `model.language_model.model.norm`, i.e. the post-final-norm
             residual stream. The driver's PROBE_CAPTURE="norm" path hooks exactly this,
             and records that it is the same tensor as `hidden_states[32]`
             (probe_capture_check.py settled that).
  * positions `h[0, -(N_ACT + 1):-1, :]` with N_ACT = ACTION_DIM * NUM_ACTIONS_CHUNK = 56,
             i.e. the 56 action-token positions, then **mean-pooled** over them -> 4096-d.
  * dtype    float16 on the wire and on disk, as the driver's ACT_TRACE saved it.

`h[0, ...]` is batch element 0, which is assumption B2 in
audit/single_env_assumptions.md: at batch > 1 the hook silently returns the first
request's hidden state for every request. Capture is therefore refused unless the server
is running at batch size 1, the same rule DESIGN.md section 5 applies to PROBE=1.
"""

import numpy as np
import torch

ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8
N_ACT = ACTION_DIM * NUM_ACTIONS_CHUNK  # 56 action-token positions


class ActivationCapture:
    """Hook the post-final-norm tensor and keep the latest mean-pooled action-token slice."""

    def __init__(self, model, max_batch):
        if max_batch != 1:
            raise ValueError(
                f"activation capture needs batch size 1, got MAX_BATCH={max_batch}: the "
                "probe hook indexes h[0, ...] (assumption B2), so at batch > 1 it returns "
                "the first request's hidden state for every request"
            )
        self.latest = None
        self.n_captures = 0
        self._handle = model.language_model.model.norm.register_forward_hook(self._hook)

    def _hook(self, _module, _args, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] < N_ACT + 1:
            return
        # same slice and pooling as the reference probe; nothing is written back, so the
        # forward is unchanged -- which is what step 5's parity check proves.
        self.latest = (
            h[0, -(N_ACT + 1) : -1, :].detach().float().mean(0).to(torch.float16).cpu().numpy()
        )
        self.n_captures += 1
        return

    def take(self):
        """The activation from the most recent forward, or None."""
        a = self.latest
        self.latest = None
        return a

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def pack(activation):
    return np.ascontiguousarray(activation, dtype=np.float16).tobytes()


def unpack(buf, dim=4096):
    return np.frombuffer(buf, dtype=np.float16).reshape(dim)
