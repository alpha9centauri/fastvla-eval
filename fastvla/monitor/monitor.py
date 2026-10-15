"""The runtime collision monitor, as the server runs it.

Three modes, chosen by the MONITOR env var and off by default:

  off    nothing happens; no hook, no score, no cost
  log    every forward is scored and the score goes back in the response header (so the
         worker writes it into the rollout record) and into the Prometheus metrics.
         Actions are untouched -- this mode cannot change a rollout's verdict.
  stop   as `log`, and the first time a rollout's score crosses the threshold the arm is
         held for the rest of that rollout: the pose delta is zeroed and the gripper
         command is passed through unchanged.

The hold is applied to the RAW chunk the policy returned, before the worker's
process_action, so the gripper goes through normalize/invert exactly as it otherwise
would -- "unchanged" means the policy's own gripper command, not a fabricated one.

A held arm is a brake, not an avoidance: zero pose delta tells the OSC controller to stay
where it is. The monitor has no notion of which way is safe.

The probe reads the activation captured at the on-policy probe's site, which indexes batch
element 0 (assumption B2), so any mode but `off` requires MAX_BATCH == 1.
"""

import os
import threading

import numpy as np

MODES = ("off", "log", "stop")
POSE_DIMS = 6  # dims 0..5 are the pose delta; dim 6 is the gripper
DEFAULT_PROBE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "bench",
    "out",
    "m7b_probe.npz",
)


class RuntimeMonitor:
    def __init__(self, mode, probe_path=None, threshold=None, debounce=None):
        if mode not in MODES:
            raise ValueError(f"MONITOR must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.lock = threading.Lock()
        self.n_scored = 0
        self.n_crossings = 0  # forwards over the threshold
        self.n_alarms = 0  # forwards where the debounce was satisfied
        self.score_ms = []
        self.latched = {}  # rollout_key -> the score that latched it
        self.consecutive = {}  # rollout_key -> crossings in a row, the online debounce
        self.probe_path = probe_path or DEFAULT_PROBE
        if mode == "off":
            self.w = None
            self.threshold = float("inf")
            self.debounce = 1
            return
        z = np.load(self.probe_path)
        self.w = np.asarray(z["w"], np.float64)
        self.mu = np.asarray(z["mu"], np.float64)
        self.sd = np.asarray(z["sd"], np.float64)
        self.ybar = float(z["ybar"][0])
        self.alpha = float(z["alpha"][0])
        self.threshold = float(z["threshold"][0]) if threshold is None else float(threshold)
        # The debounce travels with the probe, so the server's default operating point is
        # the one EVAL.md reports rather than something a caller has to remember to pass.
        stored = int(z["debounce"][0]) if "debounce" in z.files else 1
        self.debounce = stored if debounce is None else int(debounce)
        if self.debounce < 1:
            raise ValueError(f"debounce must be >= 1, got {self.debounce}")

    @property
    def enabled(self):
        return self.mode != "off"

    def score(self, activation):
        """Alarm score for one activation: minus the predicted log10 clearance.

        Higher is more alarming, so a single `>=` comparison is the alarm, with no sign
        flip anywhere else in the server.
        """
        x = np.asarray(activation, np.float64).reshape(-1)
        return float(-(((x - self.mu) / self.sd) @ self.w + self.ybar))

    def observe(self, rollout_key, activation):
        """Score one forward.

        Returns (score, hold, alarm). `alarm` is the debounced alarm -- the threshold
        crossed on this forward AND on the `debounce - 1` before it, for this rollout --
        which is what `stop` acts on and what the alarm metric counts. `hold` is True only
        in `stop` mode, and stays True for the rest of a rollout once latched.
        """
        s = self.score(activation)
        cross = s >= self.threshold
        with self.lock:
            run = (self.consecutive.get(rollout_key, 0) + 1) if cross else 0
            self.consecutive[rollout_key] = run
            alarm = run >= self.debounce
            self.n_scored += 1
            self.n_crossings += int(cross)
            self.n_alarms += int(alarm)
            if self.mode == "stop" and alarm and rollout_key not in self.latched:
                self.latched[rollout_key] = s
            hold = self.mode == "stop" and rollout_key in self.latched
        return s, hold, alarm

    @staticmethod
    def hold(chunk):
        """Zero the pose delta over the whole chunk; leave the gripper command alone."""
        out = np.array(chunk, dtype=np.float64, copy=True)
        out[..., :POSE_DIMS] = 0.0
        return out

    def summary(self):
        with self.lock:
            sm = np.array(self.score_ms) if self.score_ms else np.zeros(1)
            return dict(
                mode=self.mode,
                probe=self.probe_path,
                threshold=self.threshold,
                debounce=self.debounce,
                scored=self.n_scored,
                crossings=self.n_crossings,
                crossing_rate=self.n_crossings / max(1, self.n_scored),
                alarms=self.n_alarms,
                alarm_rate=self.n_alarms / max(1, self.n_scored),
                latched_rollouts=len(self.latched),
                latched=dict(sorted(self.latched.items())),
                score_ms_mean=float(sm.mean()),
                score_ms_max=float(sm.max()),
            )
