"""Install the C++ labeler into SafetyMonitor, behind the LABELER flag.

    LABELER=python   (default) nothing is patched; the reference path runs unchanged
    LABELER=cpp      the C++ labeler is used
    LABELER_LEVEL=gjk    swap only the GJK core at both call sites (step 4a)
                  =batch  also run the whole branch-and-bound in C++ (step 4b)
    LABELER_THREADS=N    threads for the piece scan (step 4c)
    CONTACTS=1       also install the restructured contact scan (M3 step 0, bit-exact)

The Python path is never modified, only shadowed: `install()` rebinds module globals and
`uninstall()` puts the originals back, so the fallback is the untouched reference and a
single process can measure both.

At `batch` level the replacement `_update_clearance` keeps the reference's contract
exactly: it reads `best` from the incoming record (so `distmax` is respected), writes
`min_clearance` only when something beat it, and updates the actor/geom labels only when
the record is not already a contact.
"""

import os

import numpy as np

_STATE = {"installed": None, "orig": {}}


def _labeler_for(mon):
    """Build (once per SafetyMonitor) a C++ Labeler holding every hull this scene needs."""
    L = getattr(mon, "_fastgjk", None)
    if L is not None:
        return L
    from fastvla.labeler import load_fastgjk

    fastgjk = load_fastgjk()
    L = fastgjk.Labeler()
    hulls = mon.dist_engine.hulls
    # obstacle pieces are known up front; actor geoms are registered lazily because the
    # carried-object set changes as the gripper picks things up
    for gs in mon.obstacle_geoms.values():
        for g in gs:
            verts, pad = hulls.local(g)
            L.add_geom(int(g), np.ascontiguousarray(verts, dtype=np.float64), float(pad))
    union_keys = {}
    for k, (name, hb) in enumerate(mon._obs_hull.items()):
        body, P = hb
        L.set_union_hull(k, np.ascontiguousarray(P, dtype=np.float64))
        union_keys[name] = (k, int(body))
    mon._fastgjk = L
    mon._fastgjk_union = union_keys
    mon._fastgjk_hulls = hulls
    return L


def _ensure_geoms(mon, L, gids):
    hulls = mon._fastgjk_hulls
    for g in gids:
        if not L.has_geom(int(g)):
            verts, pad = hulls.local(g)
            L.add_geom(int(g), np.ascontiguousarray(verts, dtype=np.float64), float(pad))


_ZERO9 = np.zeros(9, dtype=np.float64)
_ZERO3 = np.zeros(3, dtype=np.float64)


def _make_update_clearance(n_threads):
    def _update_clearance(self, dd, actor_of):
        agl = list(actor_of.keys())
        if not agl:
            return
        L = _labeler_for(self)
        _ensure_geoms(self, L, agl)
        aid = np.asarray(agl, dtype=np.int32)
        gx = np.asarray(dd.geom_xmat).reshape(-1, 9)
        gp = np.asarray(dd.geom_xpos).reshape(-1, 3)
        a_xmat = np.ascontiguousarray(gx[aid])
        a_xpos = np.ascontiguousarray(gp[aid])

        for o, ogeoms in self.obstacle_geoms.items():
            rec = self._acc["per_obstacle"][o]
            pid = np.asarray(ogeoms, dtype=np.int32)
            ku = self._fastgjk_union.get(o)
            if ku is None:
                key, u_xmat, u_xpos = -1, _ZERO9, _ZERO3
            else:
                key, body = ku
                u_xmat = np.ascontiguousarray(np.asarray(dd.xmat).reshape(-1, 9)[body])
                u_xpos = np.ascontiguousarray(np.asarray(dd.xpos).reshape(-1, 3)[body])
            r = L.min_clearance(
                aid,
                a_xmat,
                a_xpos,
                pid,
                np.ascontiguousarray(gx[pid]),
                np.ascontiguousarray(gp[pid]),
                key,
                u_xmat,
                u_xpos,
                float(rec["min_clearance"]),
                n_threads,
            )
            if r["actor_idx"] >= 0:
                rec["min_clearance"] = r["best"]
                if not rec["contact"]:
                    ag = agl[r["actor_idx"]]
                    rec["actor"] = actor_of[ag]
                    rec["actor_geom"] = self.idx.geom_label(ag)
                    rec["obstacle_geom"] = self.idx.geom_label(ogeoms[r["piece_idx"]])

    return _update_clearance


def install(which=None, level=None, n_threads=None, contacts=None):
    """Patch according to the flags. Returns the level actually installed, or None."""
    which = (which or os.environ.get("LABELER", "python")).lower()
    level = (level or os.environ.get("LABELER_LEVEL", "batch")).lower()
    if n_threads is None:
        n_threads = int(os.environ.get("LABELER_THREADS", "1"))
    if contacts is None:
        contacts = os.environ.get("CONTACTS", "0") == "1"
    if contacts:
        from fastvla.labeler import contacts as contacts_mod

        contacts_mod.install()
    if which == "python":
        return None
    if which != "cpp":
        raise ValueError(f"LABELER must be 'python' or 'cpp', got {which!r}")

    import convex_distance as cd
    import physics_safety as ps

    from fastvla.labeler import load_fastgjk

    fastgjk = load_fastgjk()
    _STATE["orig"] = dict(
        ps_gjk=ps.gjk_distance,
        cd_gjk=cd.gjk_distance,
        uc=ps.SafetyMonitor._update_clearance,
    )

    def cpp_gjk(A, B, max_iter=64, tol=1e-12):
        return fastgjk.gjk_distance(A, B, max_iter, tol)

    ps.gjk_distance = cpp_gjk
    cd.gjk_distance = cpp_gjk
    if level == "batch":
        ps.SafetyMonitor._update_clearance = _make_update_clearance(n_threads)
    elif level != "gjk":
        raise ValueError(f"LABELER_LEVEL must be 'gjk' or 'batch', got {level!r}")
    _STATE["installed"] = level
    return level


def uninstall():
    from fastvla.labeler import contacts as contacts_mod

    contacts_mod.uninstall()
    if _STATE["installed"] is None:
        return
    import convex_distance as cd
    import physics_safety as ps

    o = _STATE["orig"]
    ps.gjk_distance = o["ps_gjk"]
    cd.gjk_distance = o["cd_gjk"]
    ps.SafetyMonitor._update_clearance = o["uc"]
    _STATE["installed"] = None
