"""Worker-side stub: one blocking request/reply per re-plan."""

import time

import zmq

from fastvla.server import protocol as P


class PolicyClient:
    def __init__(self, endpoint, worker_id, run_id, timeout_s=300.0):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.DEALER)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(endpoint)
        self.worker_id = worker_id
        self.run_id = run_id
        self.timeout_ms = int(timeout_s * 1000)
        self.n_requests = 0
        self.batch_sizes = []
        self.queue_waits = []
        self.last_activation = None  # probe activation from the most recent reply
        self.last_monitor_score = None  # monitor score from the most recent reply, if any
        self.last_monitor_held = False
        self.last_monitor_alarm = False
        self.round_trips = []  # wall time of get_chunk, worker side
        self.server_forwards = []  # the forward the server reported for this request

    def hello(self, n_rollouts):
        self.sock.send_multipart(P.pack_hello(self.worker_id, self.run_id, n_rollouts))

    def bye(self):
        self.sock.send_multipart(P.pack_bye(self.worker_id, self.run_id))

    def get_chunk(self, rollout_key, policy_step, instruction, full_image, wrist_image, state):
        """-> (8, 7) action chunk. Asserts the reply is the one we asked for."""
        frames = P.pack_request(
            self.worker_id,
            self.run_id,
            rollout_key,
            policy_step,
            instruction,
            full_image,
            wrist_image,
            state,
        )
        t0 = time.perf_counter()
        self.sock.send_multipart(frames)
        if not self.sock.poll(self.timeout_ms, zmq.POLLIN):
            raise TimeoutError(
                f"no reply within {self.timeout_ms} ms for {rollout_key} step {policy_step}"
            )
        header, actions, activation = P.unpack_response(self.sock.recv_multipart())
        self.last_activation = activation
        want = P.make_request_id(self.worker_id, rollout_key, policy_step)
        if header.get("request_id") != want:
            # A crossed reply would present as policy nondeterminism, which on this
            # benchmark would be misread as the documented oa6_L1 coin flip.
            raise RuntimeError(f"reply mismatch: asked {want}, got {header.get('request_id')}")
        self.n_requests += 1
        self.round_trips.append(1000.0 * (time.perf_counter() - t0))
        self.batch_sizes.append(header.get("server_batch_size", 0))
        self.queue_waits.append(header.get("queue_wait_ms", 0.0))
        self.server_forwards.append(header.get("forward_ms", 0.0))
        self.last_monitor_score = header.get("monitor_score")
        self.last_monitor_held = bool(header.get("monitor_held", False))
        self.last_monitor_alarm = bool(header.get("monitor_alarm", False))
        return actions

    def close(self):
        self.sock.close(0)
