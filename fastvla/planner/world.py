"""Step 2: the planner's world, built from the labeler's own hulls.

Milestone 7a step 2. The planner has to see the same geometry the certification sees, or a
"collision-free" plan means nothing downstream. So nothing here re-derives geometry: the hulls come
from `vla-safety-interp/src/physics`, by import, through the `SafetyMonitor` the generator already
builds.

What goes into the world:

  obstacles        the declared obstacles' UNION convex hulls (SafetyMonitor._obs_hull), in world
                   coordinates via SafetyMonitor._obstacle_hull_world
  table            the `table_collision` geom's hull
  receptacle       the destination's geoms (on oa1, plate_1_g0..g5)
  carried object   after the grasp, the payload's hull attached to the gripper link

Two deliberate approximations, both conservative, both stated here rather than buried:

1. **Union hull, not the 244 pieces.** oa1's perfume bottle is a 244-piece convex decomposition.
   Planning against the union of those pieces over-approximates the obstacle, so a plan that
   clears the union clears every piece -- the direction of the error is safe. The labeler's own
   S3 stage already uses this hull as an exact lower bound for the same reason. Certification
   downstream still runs against the exact pieces, unchanged.

2. **Spheres on the robot side.** cuRobo represents the arm as spheres fitted to the Panda meshes;
   the labeler represents it as MuJoCo geom hulls. The two therefore do not agree to the
   millimetre and are not expected to. What `verify_world.py` checks is that they agree on the
   decision (does the straight-line route collide) and on the culprit (which link), evaluated at
   the same joint configuration.

Frames. The LIBERO Panda's base `robot0_link0` sits at world [-0.66, 0, 0.912] with identity
orientation, so world -> robot base is a pure translation. `robot_base_offset` reads it from the
model rather than hard-coding it, because a scene that mounts the robot elsewhere would otherwise
be silently wrong.

Author: Varun Chandra
"""
from __future__ import annotations

import os
import sys

import numpy as np

_VSI = "/data_new/vchandr2/vla-safety-interp/src"
for _p in (os.path.join(_VSI, "physics"), "/data_new/vchandr2"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TABLE_GEOM = "table_collision"


def robot_base_offset(dom):
    """World position and rotation of the arm's base link, read from the model."""
    m, dd = dom.sim.model, dom.sim.data
    bid = m.body_name2id("robot0_link0")
    p = np.asarray(dd.body_xpos[bid], dtype=np.float64).copy()
    R = np.asarray(dd.body_xmat[bid], dtype=np.float64).reshape(3, 3).copy()
    return p, R


def to_base(points, base_p, base_R):
    """World points -> robot base frame."""
    q = np.asarray(points, dtype=np.float64) - base_p[None, :]
    return q @ base_R


def geom_hull_world(mon, geom_id):
    """World-space hull vertices and swept pad of one MuJoCo geom, from the labeler's cache."""
    v, pad = mon.dist_engine._w(mon.sim.data, geom_id)
    return np.asarray(v, dtype=np.float64), float(pad)


def entity_geom_ids(dom, prefix):
    """Collision geom ids whose name starts with `prefix` (LIBERO names pieces `<name>_g<N>`)."""
    m = dom.sim.model
    out = []
    for g in range(m.ngeom):
        nm = m.geom_id2name(g)
        if not nm or not nm.startswith(prefix):
            continue
        if nm.endswith("_vis") or "visual" in nm:
            continue
        out.append(g)
    return out


def _hull_mesh(points):
    """Convex hull of `points` as (vertices, triangle faces) for cuRobo's Mesh.

    scipy's ConvexHull gives simplices already triangulated and indexed into the input array; the
    vertex set is reduced to the hull vertices so the mesh carries no interior points.
    """
    from scipy.spatial import ConvexHull
    P = np.asarray(points, dtype=np.float64)
    if len(P) < 4:
        return P, np.zeros((0, 3), dtype=np.int32)
    h = ConvexHull(P)
    keep = np.asarray(h.vertices, dtype=np.int64)
    remap = -np.ones(len(P), dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    faces = remap[np.asarray(h.simplices, dtype=np.int64)]
    return np.ascontiguousarray(P[keep]), np.ascontiguousarray(faces.astype(np.int32))


def scene_meshes(dom, mon, sg, tag, carried=None, include_receptacle=True):
    """Every piece of static world geometry the planner must avoid, in the robot base frame.

    Returns a list of dicts: name, vertices (Nx3, base frame), faces (Mx3), pad (m), source.
    `carried` names an entity to EXCLUDE from the world because it is attached to the gripper
    instead -- passing the carried object as both an obstacle and an attachment makes every
    post-grasp configuration self-colliding.
    """
    base_p, base_R = robot_base_offset(dom)
    m = dom.sim.model
    out = []

    # 1. the declared obstacles, as union hulls
    for name in sorted(mon._obs_hull):
        if carried is not None and name == carried:
            continue
        W = mon._obstacle_hull_world(mon.sim.data, name)
        if W is None or len(W) < 4:
            continue
        v, f = _hull_mesh(to_base(W, base_p, base_R))
        out.append(dict(name=f"obstacle__{name}", vertices=v, faces=f, pad=0.0,
                        source="SafetyMonitor._obstacle_hull_world (union hull)"))

    # 2. the table
    for g in entity_geom_ids(dom, TABLE_GEOM):
        W, pad = geom_hull_world(mon, g)
        v, f = _hull_mesh(to_base(W, base_p, base_R))
        out.append(dict(name=f"table__{m.geom_id2name(g)}", vertices=v, faces=f, pad=pad,
                        source="DistanceEngine._w"))

    # 3. the receptacle the object is placed into, piece by piece (it is small)
    if include_receptacle:
        for _obj, _kind, dest in MOVES_FOR(tag):
            for g in entity_geom_ids(dom, dest):
                W, pad = geom_hull_world(mon, g)
                if len(W) < 4:
                    continue
                v, f = _hull_mesh(to_base(W, base_p, base_R))
                out.append(dict(name=f"receptacle__{m.geom_id2name(g)}", vertices=v, faces=f,
                                pad=pad, source="DistanceEngine._w"))
    return out


def MOVES_FOR(tag):
    """dgcore's MOVES table, by import -- the destination is part of the task definition."""
    import dgcore as D
    return D.MOVES[tag]


def carried_mesh_abs(dom, mon, obj):
    """The carried object's hull in the robot BASE frame, absolute -- not re-centred.

    cuRobo's `attach_external_objects_to_robot` takes the object where it actually is, together
    with the joint state at which it is being grasped, and computes the gripper-relative transform
    itself. That is strictly better than re-centring the hull on the grasp point by hand, which is
    what `carried_mesh` below does: the grasp point is an eef-SITE position, so hand-rolling the
    offset would inherit the same 9.65 cm tool-offset error that made every planner goal 10 cm too
    far out. Letting cuRobo do it removes the question.
    """
    base_p, base_R = robot_base_offset(dom)
    pts = []
    for g in entity_geom_ids(dom, obj):
        Wv, _pad = geom_hull_world(mon, g)
        if len(Wv):
            pts.append(Wv)
    if not pts:
        return None
    v, f = _hull_mesh(to_base(np.vstack(pts), base_p, base_R))
    return dict(name=f"carried__{obj}", vertices=v, faces=f, pad=0.0,
                source="DistanceEngine._w, base frame, absolute")


def carried_mesh(dom, mon, obj, grasp_point):
    """The carried object's hull, expressed relative to the grasp point.

    cuRobo attaches an object to the gripper as spheres in the gripper frame; this returns the
    hull in a frame centred on the grasp point so the caller can hand it to
    `MotionGen.attach_external_objects_to_robot` without re-deriving the offset.
    """
    base_p, base_R = robot_base_offset(dom)
    pts = []
    for g in entity_geom_ids(dom, obj):
        W, _pad = geom_hull_world(mon, g)
        if len(W):
            pts.append(W)
    if not pts:
        return None
    P = np.vstack(pts)
    rel = P - np.asarray(grasp_point, dtype=np.float64)[None, :]
    v, f = _hull_mesh(rel @ base_R)
    return dict(name=f"carried__{obj}", vertices=v, faces=f, pad=0.0,
                source="DistanceEngine._w, re-centred on the grasp point")


def summarise(meshes):
    """One line per mesh, for the record in a report."""
    rows = []
    for d in meshes:
        v = d["vertices"]
        rows.append(dict(name=d["name"], n_vertices=int(len(v)), n_faces=int(len(d["faces"])),
                         pad_m=round(float(d["pad"]), 5),
                         bbox_min=[round(float(x), 4) for x in v.min(axis=0)],
                         bbox_max=[round(float(x), 4) for x in v.max(axis=0)],
                         source=d["source"]))
    return rows
