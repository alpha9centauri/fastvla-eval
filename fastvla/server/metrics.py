"""Prometheus metrics for the policy server, with no third-party dependency.

The exposition format is simple enough to emit directly, and the harness has to run in an
Apptainer container where adding packages means rebuilding a 6 GB image, so a ~90-line
handler is the cheaper choice than a dependency. If `prometheus_client` is ever already
present, this still matches what it would produce for these metric types.

Exposed on METRICS_PORT (default 9400), path /metrics, plus /healthz for a container
readiness probe:

  fastvla_requests_total              counter, requests served
  fastvla_forwards_total              counter, GPU forwards (< requests when batching)
  fastvla_queue_depth                 gauge, requests waiting for a forward right now
  fastvla_workers_active              gauge, workers that have said hello and not bye
  fastvla_forward_latency_seconds      histogram, GPU forward
  fastvla_queue_wait_seconds           histogram, time between arrival and forward
  fastvla_preprocess_seconds           histogram, crop/normalise/tokenise on the server
  fastvla_batch_size                   histogram, requests per forward
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# seconds; chosen around the measured values -- forward ~0.18 s, queue up to ~1.2 s
LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0)
BATCH_BUCKETS = (1, 2, 4, 8, 16)


class _Hist:
    def __init__(self, buckets):
        self.buckets = tuple(buckets)
        self.counts = [0] * len(self.buckets)
        self.inf = 0
        self.total = 0.0
        self.n = 0

    def observe(self, v):
        self.n += 1
        self.total += v
        for i, b in enumerate(self.buckets):
            if v <= b:
                self.counts[i] += 1
                return
        self.inf += 1

    def render(self, name, unit_suffix=""):
        out = [f"# TYPE {name} histogram"]
        cum = 0
        for b, c in zip(self.buckets, self.counts, strict=True):
            cum += c
            out.append(f'{name}_bucket{{le="{b}"}} {cum}')
        cum += self.inf
        out.append(f'{name}_bucket{{le="+Inf"}} {cum}')
        out.append(f"{name}_sum {self.total:.6f}")
        out.append(f"{name}_count {self.n}")
        return out


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests_total = 0
        self.forwards_total = 0
        self.queue_depth = 0
        self.workers_active = 0
        self.forward = _Hist(LATENCY_BUCKETS)
        self.queue = _Hist(LATENCY_BUCKETS)
        self.preprocess = _Hist(LATENCY_BUCKETS)
        self.batch = _Hist(BATCH_BUCKETS)
        self.monitor_scored_total = 0
        self.monitor_alarms_total = 0  # debounce satisfied -- what the monitor acts on
        self.monitor_crossings_total = 0  # threshold crossed, debounce or not
        self.monitor_held_rollouts = 0
        self.monitor_last_score = float("nan")
        self.started = time.time()

    def observe_forward(self, batch_size, forward_s, waits_s):
        with self.lock:
            self.forwards_total += 1
            self.requests_total += batch_size
            self.forward.observe(forward_s)
            self.batch.observe(batch_size)
            for w in waits_s:
                self.queue.observe(w)

    def observe_preprocess(self, seconds):
        with self.lock:
            self.preprocess.observe(seconds)

    def observe_monitor(self, score, alarm, held_rollouts, crossing=None):
        """One monitor evaluation. `alarm` counts crossings, not rollouts: the alarm rate
        a dashboard wants is alarms_total / scored_total."""
        with self.lock:
            self.monitor_scored_total += 1
            self.monitor_alarms_total += int(bool(alarm))
            self.monitor_crossings_total += int(bool(alarm if crossing is None else crossing))
            self.monitor_held_rollouts = held_rollouts
            self.monitor_last_score = float(score)

    def set_queue_depth(self, n):
        with self.lock:
            self.queue_depth = n

    def set_workers(self, n):
        with self.lock:
            self.workers_active = n

    def render(self):
        with self.lock:
            lines = [
                "# TYPE fastvla_requests_total counter",
                f"fastvla_requests_total {self.requests_total}",
                "# TYPE fastvla_forwards_total counter",
                f"fastvla_forwards_total {self.forwards_total}",
                "# TYPE fastvla_queue_depth gauge",
                f"fastvla_queue_depth {self.queue_depth}",
                "# TYPE fastvla_workers_active gauge",
                f"fastvla_workers_active {self.workers_active}",
                "# TYPE fastvla_uptime_seconds gauge",
                f"fastvla_uptime_seconds {time.time() - self.started:.3f}",
            ]
            if self.monitor_scored_total:
                lines += [
                    "# TYPE fastvla_monitor_scored_total counter",
                    f"fastvla_monitor_scored_total {self.monitor_scored_total}",
                    "# TYPE fastvla_monitor_alarms_total counter",
                    f"fastvla_monitor_alarms_total {self.monitor_alarms_total}",
                    "# TYPE fastvla_monitor_crossings_total counter",
                    f"fastvla_monitor_crossings_total {self.monitor_crossings_total}",
                    "# TYPE fastvla_monitor_alarm_rate gauge",
                    f"fastvla_monitor_alarm_rate "
                    f"{self.monitor_alarms_total / self.monitor_scored_total:.6f}",
                    "# TYPE fastvla_monitor_held_rollouts gauge",
                    f"fastvla_monitor_held_rollouts {self.monitor_held_rollouts}",
                    "# TYPE fastvla_monitor_last_score gauge",
                    f"fastvla_monitor_last_score {self.monitor_last_score:.6f}",
                ]
            lines += self.forward.render("fastvla_forward_latency_seconds")
            lines += self.queue.render("fastvla_queue_wait_seconds")
            lines += self.preprocess.render("fastvla_preprocess_seconds")
            lines += self.batch.render("fastvla_batch_size")
        return "\n".join(lines) + "\n"


def serve(metrics, port):
    """Start the exposition server on a daemon thread; returns the HTTPServer."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/metrics"):
                body = metrics.render().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/healthz"):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok\n")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a):
            pass  # scrape traffic would otherwise flood the server log

    srv = HTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
