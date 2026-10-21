# Step 2 — the planner's world, and whether it matches the certifier's

Author: Varun Chandra, 2026-10-19. Milestone 7a step 2.

## What the planner sees

`fastvla/planner/world.py`. Nothing re-derives geometry: every hull comes from
`vla-safety-interp/src/physics` by import, through the `SafetyMonitor` the generator already
builds. On an eval-rule oa1 scene the world is **14 meshes**:

| group | count | source |
|---|---|---|
| obstacles | 2 | `SafetyMonitor._obstacle_hull_world` — the union convex hull of `bottle_of_perfume_1` (824 hull vertices over a 244-piece decomposition) and `white_place_box_1` (16) |
| table | 1 | `DistanceEngine._w(table_collision)` |
| receptacle | 11 | `plate_1_g0..g10`, piece by piece |

Everything is expressed in the robot base frame. `robot_base_offset` reads `robot0_link0` from the
model — world `[-0.66, 0, 0.912]`, identity rotation on oa1 — rather than hard-coding it, so a
scene that mounts the arm elsewhere fails loudly instead of silently planning in the wrong place.
The carried object is handled separately (`carried_mesh`), hull re-centred on the grasp point, for
attachment to the gripper after the grasp.

**Two approximations, both conservative, both deliberate.** The obstacle goes in as its union hull
rather than its 244 pieces, so a plan that clears the union clears every piece — the labeler's own
S3 stage already uses this hull as an exact lower bound. And cuRobo's arm is 65 spheres fitted to
the Panda meshes where the labeler's arm is MuJoCo collision hulls. Certification downstream is
unchanged and still runs on exact pieces.

## Does it match the certifier?

`verify_world.py`, three eval-rule oa1 scenes (inits 0, 1, 2 — shifts 18.97 / 18.70 / 17.00 cm),
24 samples along the carried traverse each, 70 samples where IK converged. Both checkers are asked
about **the same joint configuration**: cuRobo IK solves each sampled end-effector pose once, then
cuRobo answers over its spheres and the labeler answers over MuJoCo hulls after `sim.forward()` at
those joint angles. Artefacts: `out/verify_world.json`, `out/verify_world_i0*.npz`.

### 1. The straight-line route collides — both agree, in all three scenes

```
  straight-line route, per scene:
    init  0: planner says collides on  9/23 samples, labeler on 6/23
    init  1: planner says collides on  8/24 samples, labeler on 7/24
    init  2: planner says collides on 10/23 samples, labeler on 7/23

  agree on collide/clear      63/70
```

The 7 disagreements are all the same shape: the planner calls contact where the labeler still
measures 0.7–3.8 cm of clearance. The planner is **conservative**, which is the safe direction —
it will refuse routes the certifier would have passed, never the reverse. Nothing in the 70 samples
has the planner calling a configuration clear that the labeler calls a collision.

### 2. Which link hits — same place, different name

Exact link-name agreement is only **6 of 27** colliding samples. The pattern is systematic, not
random:

```
  culprit link, planner vs labeler:
         panda_link7  vs  panda_link6        14 samples  differ
         panda_link7  vs  panda_hand          7 samples  differ
         panda_link6  vs  panda_link6         4 samples  MATCH
          panda_hand  vs  panda_hand          2 samples  MATCH
  same or neighbouring link   24/27
```

A systematic offset could mean the two models put the same metal in different places, which would
make the world model wrong. `check_link_offset.py` measures it directly — the world position of the
planner's deepest sphere against the closest point on the labeler's worst geom:

```
  centre-to-surface gap:  median 4.79 cm   range 4.68-5.23
  sphere radius:          median 4.90 cm   range 4.90-4.90
  gap minus radius:       median -0.11 cm  range -0.22-0.33
```

**The two models locate the contact to within about 3 mm.** The apparent 4.79 cm offset was the
wrist sphere's own radius — I was comparing a sphere centre against a surface point, and the first
version of this check reported that radius as if it were a discrepancy. With the radius subtracted
the gap is ±3 mm, so cuRobo's `panda_link7` sphere set simply covers the metal MuJoCo calls
`robot0_link6_collision`. The disagreement is a naming partition across the wrist, not a geometry
error.

That is the answer step 2 needed: **the planner and the certifier see the same obstacle in the same
place, and agree that the straight-line route drives the wrist into it.** Which is the collision
the scripted planner could not see, because it checked only the end effector and the carried bowl.

## Two bugs found and fixed here, both of which produced plausible numbers

1. **Stale hull cache.** `DistanceEngine` caches each geom's world-space hull for the duration of a
   frame; `begin_frame()` clears it. Without that call after moving the arm, every sample was
   measured against the first configuration's geometry and the labeler reported *the same distance,
   0.25857 m, at every point on the route*. A constant column is the tell.
2. **`get_collision_distance` reduces over spheres.** It answers "does this configuration collide",
   not "where", so `argmax` over its output is always index 0 → `panda_link0`, the immovable base,
   named as the culprit on every sample. Per-link attribution needs the spheres queried one at a
   time. Both of those passed casual inspection; neither survived asking whether the number could
   possibly be right.

## Carried into step 3

- The planner's conservatism (contact called 0.7–3.8 cm early) is in the useful direction but eats
  into the 2 cm certification margin from the other side. Combined with step 1's finding that
  returned paths sit 7–18 mm from obstacles at `collision_activation_distance=0.02`, the activation
  distance has to go up — roughly 0.04 m is the first thing to try.
- The table must be in the planning world but **must not** be in a comparison world: the Panda is
  mounted at table height, so its base link sits inside the table box and every query otherwise
  blames `panda_link0`.
- The carried bowl must be attached to the gripper *and* removed from the obstacle set. Leaving it
  in both makes every post-grasp configuration self-colliding.
