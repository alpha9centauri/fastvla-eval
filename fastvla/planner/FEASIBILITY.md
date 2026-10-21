# Step 1 — planner feasibility on this box

Author: Varun Chandra, 2026-10-19. Milestone 7a step 1. Half-day budget; finished inside it.

**cuRobo works. The OMPL fallback was not needed and was not installed.**

## What was installed

| | |
|---|---|
| conda env | `/data_new/vchandr2/miniconda3/envs/planner` (python 3.10) |
| torch | **2.2.0+cu121** — the same version the rollout env uses, so no second torch |
| CUDA | system toolkit **12.1** (`nvcc` 12.1.66), driver 12.1, GPU RTX A5000 `sm_86` |
| cuRobo | **v0.7.6** (tag, commit `2fbffc3`, 2024-11-22), `pip install -e .` from source at `fastvla/planner/third_party/curobo` |
| build | `TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=16 CUDA_HOME=/usr/local/cuda-12.1`, CUDA extensions compiled clean |
| warp-lang | **pinned 1.4.2** |
| numpy | **pinned <2** (1.26.4) |

### Three things that had to be pinned, and why

1. **cuRobo v0.7.6, not HEAD.** Current cuRobo (`v0.8.0`, Sep 2026) declares `torch>=2.5` for its
   cu12 extra. v0.7.6 declares `torch>=1.10`, which is what allows the planner to share torch
   2.2.0 with the rollout environment instead of needing a second, incompatible torch in-process.
2. **warp-lang 1.4.2, not 1.18.0.** With warp 1.18 (what pip resolves by default),
   `wp.get_cuda_devices()` returns `[]` and every cuRobo kinematics call dies in
   `device_from_torch` with `IndexError: list index out of range`. 1.4.2 is contemporary with
   cuRobo v0.7.6 and detects the card (`"cuda:0": NVIDIA RTX A5000 (24 GiB, sm_86)`).
3. **numpy < 2.** torch 2.2.0 is compiled against numpy 1.x; with numpy 2.2.6 installed, importing
   torch emits `_ARRAY_API not found` and the module is unusable.

One unrelated repair: `conda create` first failed with
`CondaVerificationError: the package for wheel ... appears to be corrupted`. That cached package
was left half-removed by the `conda clean --packages --tarballs` run during the previous
milestone's cleanup. Deleting `miniconda3/pkgs/wheel-0.47.0-py310h06a4308_0` and re-creating the
env fixed it.

## The proof

`fastvla/planner/feasibility_curobo.py`, 20 plans, GPU 1. Panda from cuRobo's own `franka.yml`.
Artefacts: `out/feasibility_curobo.json` and `out/curobo_plan_*.npz` (every returned joint path).

**The problem is well-posed, and the script asserts it rather than assuming it:**

```
box placed at [0.434 0.008 0.542], dims 0.08 x 0.08 x 0.30
well-posedness:  start +0.0510 m   goal +0.0510 m   straight line -0.0890 m
  -> both endpoints clear, the straight line collides: there is something to avoid
```

The two configurations differ only in the base joint, mirrored (`j1 = -1.1` → `+1.1`), which
swings the end effector through a 77 cm arc. The box is **derived** from that arc — centred on the
end-effector position at the midpoint of the straight joint-space line — not placed by hand.

**Result:**

```
SUCCESS 20/20   mean 222.4 ms   median 187.3 ms   range 154.8-700.4 ms
returned paths: worst min distance to world +0.0073 m (ALL CLEAR)
```

The first plan costs 700 ms and the second 422 ms (CUDA graph capture and kernel cache); from the
third onwards it is 155–212 ms. Build and warmup is a one-off **11.3 s** per process, which matters
for step 3: the planner must be constructed once and reused across scenes, not per demo.

Every returned path was re-checked **outside cuRobo's collision engine** — `_aabb_sphere_distance`
in the script is my own sphere-to-box arithmetic over cuRobo's forward kinematics — so "the planner
says it succeeded" is not the evidence. The worst case over 20 paths and ~950 waypoints is +7.3 mm.

### Two false starts, recorded because they are the interesting part

The first two versions of this test were **vacuous**, each in a different way, and both reported
numbers that looked fine:

- **Box not in the way.** I placed the box at a guessed pose. 3/3 plans succeeded at ~110 ms and
  every path measured exactly +0.0510 m clearance — the same value as the straight line, because
  the clearance was to the *table*, and the box was 5 cm off the path. "Collision-free path
  returned" was true and meant nothing.
- **Box over the goal.** Deriving the pose from the path fixed that, but the first derivation
  centred the box on a wrist sphere at the path midpoint while the end effector only travelled
  23 cm in total, so the box overlapped the **goal configuration** by 6.7 cm. All 20 plans failed
  in 37 ms. That looked like a planner limitation and was an ill-posed query: no plan exists.

The well-posedness assertion now in the script is the fix for both. It is worth keeping for the
same reason the rollout-loop gate is worth keeping: a test that cannot fail tells you nothing, and
both of these produced plausible output while testing nothing.

## What this implies for the integration

- **Clearance margin.** With `collision_activation_distance=0.02` the returned paths sit
  7–18 mm from the obstacle in true distance. Downstream certification needs **2 cm of GJK
  clearance**, so step 3 must ask for a larger activation distance (or post-check and re-plan);
  at this setting a certified demo is not guaranteed. This is the first thing to tune.
- **One planner per process.** 11.3 s of warmup against ~0.2 s per plan means the `MotionGen`
  object is built once and the world is swapped per scene
  (`update_world` / `WorldConfig`), never rebuilt per demo.
- **13 collision links, and the right one is there.** cuRobo checks
  `panda_link0..7`, `panda_link8`, `panda_hand`, `panda_leftfinger`, `panda_rightfinger` and
  `attached_object`. **`panda_link5` is among them** — the elbow link whose collisions the scripted
  planner never checked and which defeated every lateral attempt — and `attached_object` is the
  slot for the carried bowl in step 2.
- **Shared torch.** Because v0.7.6 runs on torch 2.2.0, the planner can be imported in the same
  process as the LIBERO environment rather than run as a service. Whether it *should* be is a step
  3 question (the `libero-vla` env would need cuRobo and warp added to it, or the generator moved
  into this env with libero installed); nothing in step 1 forces the answer.

## Reproduce

```bash
cd /data_new/vchandr2/fastvla-eval/fastvla/planner
env CUDA_VISIBLE_DEVICES=1 N_PLANS=20 \
  /data_new/vchandr2/miniconda3/envs/planner/bin/python -u feasibility_curobo.py
```

Exits after printing `FEASIBILITY_CUROBO_DONE`; the assertions fail loudly if the test stops being
well-posed.
