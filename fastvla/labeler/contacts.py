"""M3 step 0: restructured `SafetyMonitor._on_substep`, same reads, less Python.

The reference (physics_safety.py md5 78a026bb7f8ce81d2e1e6e3589bb04e2, lines 208-335) walks
`data.ncon` twice per MuJoCo substep -- once in `_carried_now` and once for the contact
accounting -- and each iteration builds a `data.contact[i]` wrapper object and reads
`.geom1`, `.geom2`, `.dist` off it. At 25 substeps per control step that is 50 passes and
up to a few hundred wrapper constructions per control step.

What changes here, and only this:

  * the contact fields are read in bulk, once per substep, as the numpy arrays the
    bindings already expose (`dd.contact.geom`, `dd.contact.dist`), and the
    `dist <= CONTACT_DIST_EPS` filter is a vectorised comparison. Only surviving
    contacts are visited in Python, in ascending original index order, and
    `mj_contactForce` is still called with the original contact index.
  * `fingers`, the robot half of `actor_of`, and the obstacle geom set are built once per
    monitor instead of once per substep. `_carried_now` rebuilt `set(self.idx.finger_geoms)`
    on every call; the robot loop re-inserted ~33 dict entries on every call.

What does NOT change: which contacts are examined, the order they are examined in, the
`(a, b)` / `(b, a)` orientation check and its `break`, the entity lookups, the force
reads, the fields written, or the order of the `pairs` list. `actor_of` is built in the
same order (robot geoms in `self.idx.robot_geoms` order, then carried geoms in the same
set-iteration order), which matters because the C++ labeler indexes actors positionally.

Acceptance: every monitor record identical, field for field, including `pairs` order --
see `bench_contacts.py MODE=parity`.
"""

import numpy as np

_STATE = {"orig": None}


def _prepare(mon):
    """Per-monitor constants the reference recomputed on every substep."""
    c = getattr(mon, "_fastcontacts", None)
    if c is not None:
        return c
    obstacle_geoms_union = set()
    for o in mon.obstacles:
        obstacle_geoms_union |= set(mon.obstacle_geoms[o])
    base_actor = {}
    for g in mon.idx.robot_geoms:
        base_actor[int(g)] = "robot"
    c = dict(
        fingers=frozenset(int(g) for g in mon.idx.finger_geoms),
        obstacle_union=frozenset(obstacle_geoms_union),
        base_actor=base_actor,
        geom2entity=mon.idx.geom2entity,
        entity_geoms=mon.idx.entity_geoms,
    )
    mon._fastcontacts = c
    return c


def _make_on_substep():
    import mujoco
    from physics_safety import CONTACT_DIST_EPS

    def _on_substep(self):
        if self._acc is None:
            return
        self._substeps_seen += 1
        self.dist_engine.begin_frame()
        data = self.sim.data
        dd = data._data if hasattr(data, "_data") else data
        mm = self.idx.mm
        c = _prepare(self)
        g2e = c["geom2entity"]
        per_obstacle = self._acc["per_obstacle"]

        ncon = int(dd.ncon)
        if ncon:
            # one bulk read per substep instead of a wrapper object per contact
            geom = np.asarray(dd.contact.geom)[:ncon]
            dist = np.asarray(dd.contact.dist)[:ncon]
            kept = np.nonzero(dist <= CONTACT_DIST_EPS)[0]
        else:
            geom = None
            kept = ()

        # ---- carried: movable entities touching a gripper finger pad -------------
        carried = set()
        fingers = c["fingers"]
        if fingers and len(kept):
            for i in kept:
                g1 = int(geom[i, 0])
                g2 = int(geom[i, 1])
                if g1 in fingers and g2 in g2e:
                    carried.add(g2e[g2])
                elif g2 in fingers and g1 in g2e:
                    carried.add(g2e[g1])
        self._acc["carried"] |= carried

        carried_geoms = set()
        for name in carried:
            carried_geoms.update(c["entity_geoms"].get(name, []))
        carried_geoms -= c["obstacle_union"]

        actor_of = c["base_actor"].copy()
        for g in carried_geoms:
            actor_of[int(g)] = g2e.get(g, "carried")

        # ---- contact accounting --------------------------------------------------
        if len(kept):
            geom_label = self.idx.geom_label
            acc = self._acc
            pairs = acc["pairs"]
            for i in kept:
                g1 = int(geom[i, 0])
                g2 = int(geom[i, 1])
                for a, b in ((g1, g2), (g2, g1)):
                    if a in actor_of:
                        ent = g2e.get(b)
                        if ent in per_obstacle:
                            f = np.zeros(6, dtype=np.float64)
                            # mirrors the reference's bare guard. contextlib.suppress
                            # would be equivalent but builds a context manager per
                            # contact, which is the cost this restructure is removing.
                            try:  # noqa: SIM105
                                mujoco.mj_contactForce(mm, dd, int(i), f)
                            except Exception:
                                pass
                            fn = float(abs(f[0]))
                            rec = per_obstacle[ent]
                            rec["contact"] = True
                            rec["max_force"] = max(rec["max_force"], fn)
                            rec["actor"] = actor_of[a]
                            rec["actor_geom"] = geom_label(a)
                            rec["obstacle_geom"] = geom_label(b)
                            if not acc["contact"]:
                                acc["contact_substep"] = self._substeps_seen
                            acc["contact"] = True
                            pairs.append(
                                dict(
                                    substep=self._substeps_seen,
                                    actor=actor_of[a],
                                    actor_geom=geom_label(a),
                                    obstacle=ent,
                                    obstacle_geom=geom_label(b),
                                    dist=round(float(dist[i]), 6),
                                    force=round(fn, 4),
                                )
                            )
                        break

        if self.track_clearance and (
            self._substeps_seen % self.clearance_every == 0 or self._substeps_seen == 1
        ):
            self._update_clearance(dd, actor_of)

    return _on_substep


def install():
    import physics_safety as ps

    if _STATE["orig"] is None:
        _STATE["orig"] = ps.SafetyMonitor._on_substep
    ps.SafetyMonitor._on_substep = _make_on_substep()
    return True


def uninstall():
    import physics_safety as ps

    if _STATE["orig"] is not None:
        ps.SafetyMonitor._on_substep = _STATE["orig"]
