"""Pre-flight for any GPU job, in a container or on the host.

What this checks, and why it is this and not `nvidia-smi`:

`nvidia-smi` ignores `CUDA_VISIBLE_DEVICES` -- on the host as much as inside a container --
and the `/dev/nvidia*` nodes are world read/write on the host too, so a container adds no
access the host user did not already have. Confinement has always rested on the CUDA
runtime honouring the mask. So the check that matters is the CUDA runtime's view, plus
proof that a real allocation lands on the intended card.

Three assertions:

  1. `torch.cuda.device_count() == 1` -- the mask is in effect.
  2. CUDA device 0 is the expected physical card. torch 2.2 exposes neither a UUID nor a
     PCI bus id on device properties, so the bus id is read through `cudaDeviceGetPCIBusId`
     in libcudart, which is mask-aware, and compared against the expected one.
  3. With a live allocation on CUDA device 0, `nvidia-smi --query-compute-apps` shows this
     process on the expected card and **nothing of ours on the forbidden card**. This is
     the assertion that actually proves where the memory went.

Exit 0 if all hold, 3 otherwise. Prints a one-line summary either way, so it can be the
first thing in every job's log.

Env: EXPECT_UUID, EXPECT_BUS_ID, FORBID_UUID  (defaults are gpu5's GPU 1 and GPU 0)
"""

import ctypes
import os
import subprocess
import sys

EXPECT_UUID = os.environ.get("EXPECT_UUID", "GPU-040542a1-6ad7-1ddd-f7ca-595acfd703a7")
EXPECT_BUS_ID = os.environ.get("EXPECT_BUS_ID", "00000000:C1:00.0")
FORBID_UUID = os.environ.get("FORBID_UUID", "GPU-2c6273fb-12cf-6171-8708-36c7225f4ba3")
ALLOC_MB = int(os.environ.get("PREFLIGHT_ALLOC_MB", "256"))


def pci_bus_id_of_cuda0():
    """cudaDeviceGetPCIBusId, which respects CUDA_VISIBLE_DEVICES."""
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            lib = None
    if lib is None:
        try:  # torch ships one
            import glob

            import torch

            cands = glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart*"))
            lib = ctypes.CDLL(cands[0]) if cands else None
        except Exception:
            lib = None
    if lib is None:
        return None
    buf = ctypes.create_string_buffer(64)
    rc = lib.cudaDeviceGetPCIBusId(buf, 64, 0)
    return buf.value.decode() if rc == 0 else None


def norm_bus(b):
    """cudaDeviceGetPCIBusId pads the domain to 4 digits, nvidia-smi to 8.

    "0000:C1:00.0" and "00000000:C1:00.0" are the same device; compare on the
    domain:bus:device.function with the domain's leading zeros stripped.
    """
    if not b:
        return None
    parts = b.strip().upper().split(":")
    if len(parts) == 3:
        return f"{int(parts[0], 16)}:{parts[1]}:{parts[2]}"
    return b.strip().upper()


def main():
    import torch

    res = {}
    n = torch.cuda.device_count()
    res["device_count"] = n
    ok_count = n == 1

    bus = pci_bus_id_of_cuda0()
    res["cuda0_pci_bus_id"] = bus
    ok_bus = bus is not None and norm_bus(bus) == norm_bus(EXPECT_BUS_ID)

    # hold a real allocation on cuda:0 while asking the driver who is using what
    buf = torch.empty(int(ALLOC_MB * 1024 * 1024 // 2), dtype=torch.float16, device="cuda:0")
    buf.fill_(1.0)
    torch.cuda.synchronize()
    me = os.getpid()
    q = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    ours_on_expected, ours_on_forbidden, others = [], [], []
    for line in q.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        uuid, pid = parts[0], parts[1]
        if pid == str(me):
            (
                ours_on_expected
                if uuid == EXPECT_UUID
                else ours_on_forbidden
                if uuid == FORBID_UUID
                else others
            ).append(line)
        elif uuid == FORBID_UUID:
            others.append(line)
    res["compute_apps"] = q.strip().splitlines()
    res["ours_on_expected_card"] = ours_on_expected
    res["ours_on_forbidden_card"] = ours_on_forbidden
    ok_alloc = bool(ours_on_expected) and not ours_on_forbidden
    del buf
    torch.cuda.empty_cache()

    ok = ok_count and ok_bus and ok_alloc
    print(f"[PREFLIGHT] device_count={n} (want 1) -> {'ok' if ok_count else 'FAIL'}")
    print(
        f"[PREFLIGHT] cuda:0 pci bus id={bus} (want {EXPECT_BUS_ID}) "
        f"-> {'ok' if ok_bus else 'FAIL'}"
    )
    print(
        f"[PREFLIGHT] {ALLOC_MB} MB allocated on cuda:0; our pid {me} appears on "
        f"{len(ours_on_expected)} expected-card row(s), {len(ours_on_forbidden)} "
        f"forbidden-card row(s) -> {'ok' if ok_alloc else 'FAIL'}"
    )
    for line in ours_on_expected:
        print(f"[PREFLIGHT]   ours: {line}")
    if ours_on_forbidden:
        for line in ours_on_forbidden:
            print(f"[PREFLIGHT]   ON FORBIDDEN CARD: {line}")
    print(f"[PREFLIGHT] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
