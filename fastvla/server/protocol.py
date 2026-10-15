"""Wire format for the policy server.

ZeroMQ multipart, so the big arrays travel as their own frames and are never copied into
a JSON string. A request is one re-plan; the response is one 8-step action chunk.

  request   [ b"REQ", header_json, full_image, wrist_image, state ]
  response  [ header_json, actions ]
  hello     [ b"HELLO", header_json ]          worker announcing itself
  bye       [ b"BYE",   header_json ]          worker finished, lets the server exit

Header fields are in DESIGN.md section 2. `request_id` is
"<worker_id>:<rollout_key>:<policy_step>" and is asserted on arrival of the reply: a
mismatched reply would otherwise look exactly like policy nondeterminism, which on this
benchmark would be misread as the documented oa6_L1 coin flip.
"""

import json

import numpy as np

REQ = b"REQ"
HELLO = b"HELLO"
BYE = b"BYE"
IMG_DTYPE = np.uint8
STATE_DTYPE = np.float64
ACT_DTYPE = np.float64


def make_request_id(worker_id, rollout_key, policy_step):
    return f"{worker_id}:{rollout_key}:{policy_step}"


def pack_request(
    worker_id, run_id, rollout_key, policy_step, instruction, full_image, wrist_image, state
):
    full_image = np.ascontiguousarray(full_image, dtype=IMG_DTYPE)
    wrist_image = np.ascontiguousarray(wrist_image, dtype=IMG_DTYPE)
    state = np.ascontiguousarray(state, dtype=STATE_DTYPE).reshape(-1)
    header = dict(
        request_id=make_request_id(worker_id, rollout_key, policy_step),
        run_id=run_id,
        worker_id=worker_id,
        rollout_key=rollout_key,
        policy_step=int(policy_step),
        instruction=instruction,
        full_shape=list(full_image.shape),
        wrist_shape=list(wrist_image.shape),
        state_len=int(state.size),
    )
    return [
        REQ,
        json.dumps(header).encode(),
        full_image.tobytes(),
        wrist_image.tobytes(),
        state.tobytes(),
    ]


def unpack_request(frames):
    header = json.loads(frames[1].decode())
    full = np.frombuffer(frames[2], dtype=IMG_DTYPE).reshape(header["full_shape"])
    wrist = np.frombuffer(frames[3], dtype=IMG_DTYPE).reshape(header["wrist_shape"])
    state = np.frombuffer(frames[4], dtype=STATE_DTYPE).reshape(header["state_len"])
    return header, full, wrist, state


def pack_response(header, actions, activation=None):
    """Response frames; a third frame carries the probe activation when captured.

    float16, 4096-d = 8 KB, against a 301 KB request -- small enough that collection runs
    at the same rate as a normal run, which step 1 measures rather than assumes.
    """
    actions = np.ascontiguousarray(actions, dtype=ACT_DTYPE)
    h = dict(header)
    h["action_shape"] = list(actions.shape)
    frames = [None, actions.tobytes()]
    if activation is not None:
        act = np.ascontiguousarray(activation, dtype=np.float16)
        h["activation_dim"] = int(act.size)
        frames.append(act.tobytes())
    frames[0] = json.dumps(h).encode()
    return frames


def unpack_response(frames):
    header = json.loads(frames[0].decode())
    actions = np.frombuffer(frames[1], dtype=ACT_DTYPE).reshape(header["action_shape"])
    activation = None
    if header.get("activation_dim") and len(frames) > 2:
        activation = np.frombuffer(frames[2], dtype=np.float16).reshape(header["activation_dim"])
    return header, actions, activation


def pack_hello(worker_id, run_id, n_rollouts):
    return [
        HELLO,
        json.dumps(dict(worker_id=worker_id, run_id=run_id, n_rollouts=n_rollouts)).encode(),
    ]


def pack_bye(worker_id, run_id):
    return [BYE, json.dumps(dict(worker_id=worker_id, run_id=run_id)).encode()]
