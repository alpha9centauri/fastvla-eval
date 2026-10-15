"""Fit and evaluate the collision monitor's probe.

The probe is a ridge regression from the policy's own activation to log clearance, which
is the detection half of the probe-gated pullback (vla-safety-interp RESULTS.md section 2,
arm P1). It is deliberately the same construction as the reference on-policy probe
(`onpolicy_probe.py`, `out/PROBE_ONPOLICY.txt`, held-out Spearman +0.5639) so that the
number here is comparable rather than merely similar:

  * features  the post-final-norm residual stream, mean-pooled over the 56 action-token
              positions -- see fastvla/monitor/capture.py
  * target    log10(min clearance in metres) at the control step of that forward,
              floored at 1e-4 as the reference floors it
  * fit       standardise, then closed-form ridge on the centred target
  * CV        grouped by SCENE (task + init), never by frame. Frames within a rollout are
              heavily autocorrelated, so a frame-split would report a number that says
              nothing about a new scene. The reference used leave-one-init-out for the
              same reason.

What differs on purpose: the reference fixed alpha at 1e5; here it is chosen by
scene-split CV on the training set only, which is what the milestone asks for.
"""

import glob
import json
import os

import numpy as np

CLEARANCE_FLOOR_M = 1e-4
# Extended below 1e3 after the first fit chose the grid's edge: an optimum at an
# endpoint is not an optimum, it is an unexplored direction. The reference used a
# fixed 1e5, which this grid contains.
ALPHA_GRID = (1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 5:
        return float("nan")

    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        r[o] = np.arange(len(v))
        _u, st, ct = np.unique(v[o], return_index=True, return_counts=True)
        for s_, c_ in zip(st, ct, strict=True):
            if c_ > 1:
                r[o[s_ : s_ + c_]] = r[o[s_ : s_ + c_]].mean()
        return r

    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def auroc(scores, labels):
    """Rank-based AUROC; returns nan if one class is absent."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.all() or not y.any():
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    _u, st, ct = np.unique(s[o], return_index=True, return_counts=True)
    for s_, c_ in zip(st, ct, strict=True):
        if c_ > 1:
            r[o[s_ : s_ + c_]] = r[o[s_ : s_ + c_]].mean()
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def load_set(run_dirs, k_list=(8, 16, 32)):
    """Gather every captured forward from one or more run directories.

    Returns a dict of arrays, one row per forward:
      X          (n, 4096)  activation
      y          (n,)       log10 clearance at that control step
      scene      (n,)       "<tag>__i<NN>" -- the CV group
      key        (n,)       rollout key
      step       (n,)       control step of the forward
      pre        (n,)       True if the forward precedes first contact
      hit_k      (n,)       per k: contact within the next k control steps
      collided   (n,)       whether that rollout ever contacts
    """
    X, y, scene, key, step, pre, collided = [], [], [], [], [], [], []
    hit = {k: [] for k in k_list}
    n_rollouts = 0
    for rd in run_dirs:
        merged = os.path.join(rd, "merged.json")
        if os.path.exists(merged):
            with open(merged) as fh:
                rows = json.load(fh)["rows"]
        else:
            # merged.json is written when the run ends; the per-rollout records are
            # written as each rollout finishes (assumption A1: written once, never
            # rewritten), so a run still in flight can be read from those.
            rows = []
            for f in sorted(glob.glob(os.path.join(rd, "rollouts", "*.json"))):
                with open(f) as fh:
                    rows.append(json.load(fh))
            if not rows:
                continue
        for r in rows:
            p = os.path.join(rd, "arrays", f"{r['key']}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p)
            A = np.asarray(z["activations"], np.float32)
            steps = np.asarray(z["activation_steps"], np.int64)
            clr = np.asarray(z["clearance"], np.float64)
            con = np.asarray(z["contact"], bool)
            if len(A) == 0 or len(clr) == 0:
                continue
            n_rollouts += 1
            first = int(np.argmax(con)) if con.any() else None
            for i, s in enumerate(steps):
                if s >= len(clr):
                    continue
                X.append(A[i])
                y.append(np.log10(max(float(clr[s]), CLEARANCE_FLOOR_M)))
                scene.append(f"{r['tag']}__i{int(r['init_idx']):02d}")
                key.append(r["key"])
                step.append(int(s))
                pre.append(first is None or s < first)
                collided.append(bool(con.any()))
                for k in k_list:
                    hit[k].append(bool(con[s + 1 : s + 1 + k].any()))
    out = dict(
        X=np.asarray(X, np.float64),
        y=np.asarray(y, np.float64),
        scene=np.asarray(scene),
        key=np.asarray(key),
        step=np.asarray(step),
        pre=np.asarray(pre, bool),
        collided=np.asarray(collided, bool),
        n_rollouts=n_rollouts,
    )
    for k in k_list:
        out[f"hit_{k}"] = np.asarray(hit[k], bool)
    return out


def _standardise(X):
    mu, sd = X.mean(0), X.std(0)
    sd = np.maximum(sd, max(1e-6, 1e-3 * np.median(sd)))
    return mu, sd, (X - mu) / sd


def _solve(gram, rhs, alpha):
    return np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)


def fit_ridge(X, y, alpha):
    """Standardise, then closed-form ridge on the centred target -- the reference's form."""
    mu, sd, Z = _standardise(X)
    ybar = y.mean()
    w = _solve(Z.T @ Z, Z.T @ (y - ybar), alpha)
    return dict(w=w, mu=mu, sd=sd, ybar=ybar, alpha=float(alpha))


def predict(probe, X):
    Z = (np.asarray(X, np.float64) - probe["mu"]) / probe["sd"]
    return Z @ probe["w"] + probe["ybar"]


def scene_split_cv(X, y, scene, alphas=ALPHA_GRID, n_folds=5, seed=0):
    """Choose alpha by grouped CV. Folds are whole scenes, never split frames."""
    uniq = np.array(sorted(set(scene.tolist())))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    folds = np.array_split(order, min(n_folds, len(uniq)))
    # Folds outside, alphas inside: the standardisation and the 4096x4096 Gram matrix
    # depend only on the fold, so they are formed once per fold and every alpha reuses
    # them. Identical arithmetic to calling fit_ridge per (fold, alpha) -- same Gram, same
    # solve -- but it does the expensive part 5 times instead of 35.
    rhos = {float(a): [] for a in alphas}
    for f in folds:
        held = set(uniq[f].tolist())
        te = np.array([s in held for s in scene])
        tr = ~te
        if tr.sum() < 10 or te.sum() < 5:
            continue
        mu, sd, Z = _standardise(X[tr])
        ybar = y[tr].mean()
        gram, rhs = Z.T @ Z, Z.T @ (y[tr] - ybar)
        for a in alphas:
            p = dict(w=_solve(gram, rhs, a), mu=mu, sd=sd, ybar=ybar, alpha=float(a))
            rhos[float(a)].append(spearman(predict(p, X[te]), y[te]))
    results = [
        dict(
            alpha=float(a),
            mean_spearman=float(np.nanmean(rhos[float(a)])) if rhos[float(a)] else float("nan"),
            per_fold=[float(r) for r in rhos[float(a)]],
        )
        for a in alphas
    ]
    best = max(results, key=lambda r: r["mean_spearman"])
    return best, results
