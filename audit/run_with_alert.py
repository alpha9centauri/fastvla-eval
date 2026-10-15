# COPIED VERBATIM from
#   /data_new/vchandr2/vla-safety-interp/src/displacement/run_with_alert.py
# (source md5 recorded in audit/PROVENANCE.txt). Author: Varun Chandra.
#
# WHY A COPY AND NOT AN IMPORT. The wrapper derives every path it writes from
# HERE = Path(__file__).parent: RUN_LOG, SUMMARY.txt, DONE and FAILED. Imported or
# invoked from the research tree it would unlink and rewrite that tree's DONE and
# SUMMARY.txt (which exist, dated Oct 12 and Sep 13) and dirty a clean git repo.
# This project may not modify anything outside fastvla-eval, so the file sits here
# instead and all four markers land in audit/. The body below is unmodified; only
# this comment was added. The child is run with cwd=HERE, so audit scripts cd to
# the driver's directory themselves.
"""Run the obstacle-displacement test under a wandb watchdog that emails on exit.

Author: Varun Chandra.

The point of this wrapper is that the displacement test is long (600 closed-loop rollouts across
two models) and must not fail silently overnight. It therefore reports on exit no matter how it
exits: clean, non-zero, crash, or Ctrl-C. The alert is in a `finally`, so the only way to lose it
is SIGKILL.

  python run_with_alert.py --alert-test     send one "alert test" alert and exit, nothing else
  python run_with_alert.py                  run the real thing
  python run_with_alert.py --no-wandb       run with NO credential: markers + run.log only

--no-wandb exists because this box has no mail transport and no wandb key, and a W&B API key is
a bearer credential for the whole account that should not be pasted into a chat transcript to
get a notification. In that mode the wrapper still does everything except talk to wandb: it
streams the child to run.log, keeps the heartbeat on stdout, and writes DONE/FAILED with the exit
code, wall time and progress. Something outside this process then watches for those markers.

CHILD CONTRACT. The test is run as a subprocess and its combined stdout/stderr is streamed to
run.log (and echoed, so an interactive caller sees it live). For the heartbeat to report progress
the child should print lines of the form

    [PROGRESS] done=<k> total=<N>

any time either number changes. If it never does, the wrapper falls back to counting lines that
match ROLLOUT_RE and to DISPLACEMENT_TOTAL for the denominator, and says which source it used --
it never invents a number.

WANDB. Nothing heavy is logged: a start alert, a heartbeat every 30 min, and one final alert.
`console="off"` keeps wandb from hijacking the child's stdout, which this wrapper is already
capturing itself.

Env: [DISPLACEMENT_CMD] the child command (default: the driver in this directory)
     [DISPLACEMENT_TOTAL=600] fallback denominator
     [HEARTBEAT_SEC=1800]
"""
import os, re, sys, time, signal, subprocess, threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_LOG = HERE / os.environ.get("RUN_LOG_NAME", "run.log")  # per-stage, so a later
                                                           # wrapper cannot overwrite
                                                           # an earlier stage's log
SUMMARY = HERE / "SUMMARY.txt"
DONE, FAILED = HERE / "DONE", HERE / "FAILED"
HEARTBEAT_SEC = int(os.environ.get("HEARTBEAT_SEC", "1800"))
TOTAL_FALLBACK = int(os.environ.get("DISPLACEMENT_TOTAL", "600"))
DEFAULT_CMD = f"{sys.executable} {HERE / 'run_displacement.sh'}"
CHILD_CMD = os.environ.get("DISPLACEMENT_CMD") or str(HERE / "run_displacement.sh")
PROGRESS_RE = re.compile(r"\[PROGRESS\]\s+done=(\d+)\s+total=(\d+)")
# The child can request an interim email at a stage boundary by printing
#   [ALERT] <title> | <text, newlines written as \n>
# so a long pipeline reports its milestones instead of going quiet for hours.
STAGE_RE = re.compile(r"^\[ALERT\]\s+([^|]{1,120})\|(.*)$")
MAX_STAGE_ALERTS = 12
ROLLOUT_RE = re.compile(r"^\[ROLLOUT\]")
TAIL_N = 60

_state = {"done": 0, "total": None, "counted": 0, "src": "none"}
_lock = threading.Lock()


def tail(path, n=TAIL_N):
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except Exception as e:
        return f"(could not read {path}: {e})"
    return "\n".join(lines[-n:]) if lines else f"({path} is empty)"


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m{sec % 60:02d}s"


def progress_str():
    with _lock:
        d, t, src = _state["done"], _state["total"], _state["src"]
        if src == "none":
            d, t, src = _state["counted"], TOTAL_FALLBACK, "counted [ROLLOUT] lines"
    return f"{d}/{t if t else '?'} rollouts ({src})", d, t


class _NoWandb:
    """Stand-in so the run path is identical with and without a credential."""

    AlertLevel = type("L", (), {"INFO": "INFO", "ERROR": "ERROR"})

    @staticmethod
    def alert(title, text, level="INFO"):
        print(f"[ALERT/{level}] {title}\n{text}", flush=True)

    def log(self, *a, **k):
        pass

    def finish(self):
        pass


def main():
    no_wandb = "--no-wandb" in sys.argv
    alert_test = "--alert-test" in sys.argv
    if no_wandb:
        wandb = run = _NoWandb()
        print("[ALERT] --no-wandb: markers + run.log only, no credential used.", flush=True)
    else:
        import wandb
        if not wandb.api.api_key:
            print("[ALERT] wandb is not logged in (no api key resolved). Refusing to start.\n"
                  "        Either authenticate, or re-run with --no-wandb.", file=sys.stderr)
            return 2
        run = wandb.init(project="vla-safety-interp",
                         name="displacement_17500_pi05" + ("_alerttest" if alert_test else ""),
                         job_type="eval",
                         settings=wandb.Settings(console="off"),
                         config=dict(child_cmd=CHILD_CMD, total_expected=TOTAL_FALLBACK,
                                     heartbeat_sec=HEARTBEAT_SEC, alert_test=alert_test))
    try:
        if alert_test:
            wandb.alert(title="alert test",
                        text=("Wrapper is wired up and wandb alerts reach this address.\n"
                              f"host {os.uname().nodename}\ndir  {HERE}\n"
                              f"child would be: {CHILD_CMD}\n"
                              "No rollouts were run and GPU 0 was not touched."),
                        level=wandb.AlertLevel.INFO)
            print("[ALERT] sent 'alert test'. No rollouts run, GPU 0 untouched.")
            return 0

        wandb.alert(title="displacement test started",
                    text=f"host {os.uname().nodename}\ncmd  {CHILD_CMD}\n"
                         f"expecting ~{TOTAL_FALLBACK} closed-loop rollouts",
                    level=wandb.AlertLevel.INFO)

        for m in (DONE, FAILED):
            m.unlink(missing_ok=True)
        t0 = time.time()
        rc, err = None, None
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(HEARTBEAT_SEC):
                msg, d, t = progress_str()
                run.log({"rollouts_done": d, "rollouts_total": t or 0,
                         "elapsed_sec": time.time() - t0})
                print(f"[HEARTBEAT] {msg}, elapsed {hms(time.time() - t0)}", flush=True)

        sent = {"n": 0}

        def stage_alert(title, text):
            if sent["n"] >= MAX_STAGE_ALERTS:
                return
            sent["n"] += 1
            try:
                wandb.alert(title=title.strip()[:120],
                            text=text.replace("\\n", "\n").strip()[:4000],
                            level=wandb.AlertLevel.INFO)
                print(f"[ALERT] forwarded stage alert: {title.strip()}", flush=True)
            except Exception as e:
                print(f"[ALERT] stage alert failed: {e}", file=sys.stderr)

        hb = threading.Thread(target=heartbeat, daemon=True)
        hb.start()
        proc = None
        try:
            with open(RUN_LOG, "w", buffering=1) as log:
                log.write(f"# displacement test, started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log.write(f"# cmd: {CHILD_CMD}\n")
                proc = subprocess.Popen(CHILD_CMD, shell=True, cwd=str(HERE),
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, start_new_session=True)
                for line in proc.stdout:
                    log.write(line)
                    sys.stdout.write(line)
                    sm = STAGE_RE.match(line.rstrip())
                    if sm:
                        stage_alert(sm.group(1), sm.group(2))
                    m = PROGRESS_RE.search(line)
                    with _lock:
                        if m:
                            _state["done"] = int(m.group(1))
                            _state["total"] = int(m.group(2))
                            _state["src"] = "child [PROGRESS]"
                        elif ROLLOUT_RE.search(line):
                            _state["counted"] += 1
                rc = proc.wait()
        except BaseException as e:                     # KeyboardInterrupt included
            err = f"{type(e).__name__}: {e}"
            if proc is not None and proc.poll() is None:
                try:                                   # kill the child's own group, by pid
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=60)
                except Exception:
                    pass
                rc = proc.poll()
            raise
        finally:
            stop.set()
            wall = time.time() - t0
            msg, d, t = progress_str()
            ok = (rc == 0) and err is None
            body = SUMMARY if SUMMARY.exists() else RUN_LOG
            text = (f"exit code: {rc}\nwall time: {hms(wall)}\nprogress: {msg}\n"
                    + (f"exception: {err}\n" if err else "")
                    + f"\n--- last {TAIL_N} lines of {body.name} ---\n{tail(body)}")
            (DONE if ok else FAILED).write_text(
                f"exit={rc} wall={hms(wall)} progress={msg}\n" + (f"error={err}\n" if err else ""))
            try:
                wandb.alert(title="displacement test finished" if ok
                            else "displacement test FAILED",
                            text=text,
                            level=wandb.AlertLevel.INFO if ok else wandb.AlertLevel.ERROR)
            except Exception as ae:
                print(f"[ALERT] could not send final alert: {ae}", file=sys.stderr)
            print(f"[ALERT] {'DONE' if ok else 'FAILED'} exit={rc} wall={hms(wall)} {msg}",
                  flush=True)
        return rc
    finally:
        try:
            run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
