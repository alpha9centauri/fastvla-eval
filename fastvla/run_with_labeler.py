"""Run an existing research script with the labeler selected by the LABELER flag.

The driver and the gate are regression references and are never edited or copied. This
installs the labeler (fastvla.labeler.patch) and then executes the target with runpy, so
the target binds the patched `physics_safety` module at its own import time.

REDIRECT_WRITE exists for one specific reason: `datagen/curve_gate.py` ends with
`json.dump(rows, open(os.path.join(HERE, "curve_gate.json"), "w"))`, which would write
into the research repo -- the file is already there, dated 13 Oct. This project may not
modify anything outside fastvla-eval, and there is no env var for that path, so writes to
a named basename are redirected into this repo instead. Reads are untouched, so the gate
still loads its own references (out/feasible_set.json, out_instruct/rows_I0_intr5.json)
from the research tree.

Env: TARGET (script to run), LABELER, [LABELER_LEVEL, LABELER_THREADS,
     REDIRECT_WRITE="<basename>:<dest path>"], plus whatever the target reads.
"""
import builtins
import os
import runpy
import sys

DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for p in ("/data_new/vchandr2", "/data_new/vchandr2/openvla-oft", DISP,
          os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    sys.path.insert(0, p)

import rollout_bootstrap  # noqa: F401, E402

from fastvla.labeler import patch  # noqa: E402

TARGET = os.environ["TARGET"]
level = patch.install()
print(f"[LABELER] LABELER={os.environ.get('LABELER', 'python')} level={level} "
      f"threads={os.environ.get('LABELER_THREADS', '1')} target={TARGET}", flush=True)

_redirect = os.environ.get("REDIRECT_WRITE", "")
if _redirect:
    _base, _dest = _redirect.split(":", 1)
    _real_open = builtins.open

    def _open(file, mode="r", *a, **kw):
        if isinstance(file, (str, os.PathLike)) and "w" in str(mode) \
                and os.path.basename(str(file)) == _base:
            print(f"[LABELER] redirecting write {file} -> {_dest}", flush=True)
            os.makedirs(os.path.dirname(_dest) or ".", exist_ok=True)
            return _real_open(_dest, mode, *a, **kw)
        return _real_open(file, mode, *a, **kw)

    builtins.open = _open

runpy.run_path(TARGET, run_name="__main__")
