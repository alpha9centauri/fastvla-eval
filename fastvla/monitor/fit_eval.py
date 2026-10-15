"""Fit the monitor's probe on the training scenes and evaluate it on the held-out ones.

Train: unshifted inits 10-49 (240 rollouts). Test: unshifted inits 0-9 (60) and the
48-start intrusion-5 set. Disjoint by scene, so no scene the probe was fitted on appears
in any test number.

Note on the specification: the milestone asked for "intrusion-5 where feasible" in the
TRAINING set too, but out/feasible_set.json only covers inits 0-9, so intrusion-5 placements
exist only for the test inits. Training is therefore unshifted-only and every intrusion-5
scene is held out -- which makes the intrusion-5 numbers a genuine distribution shift the
probe never saw, rather than a weaker in-domain result.

Env: TRAIN_RUNS, TEST_UNSHIFTED_RUN, TEST_INTR5_RUN, OUT
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastvla.monitor import probe as P  # noqa: E402

K_LIST = (8, 16, 32)
RECALL_TARGET = 0.90
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "bench", "runs")
OUT = os.environ.get("OUT", os.path.join(ROOT, "bench", "out", "m7b_probe"))


def alarm_score(pred_log_clearance):
    """Higher = more alarming. The probe predicts log clearance, so negate it."""
    return -np.asarray(pred_log_clearance, float)


def threshold_for_recall(scores, labels, target):
    """Lowest threshold achieving at least `target` recall (most permissive wins)."""
    s, y = np.asarray(scores), np.asarray(labels, bool)
    if not y.any():
        return float("nan")
    pos = np.sort(s[y])
    idx = max(0, int(np.floor((1.0 - target) * len(pos))))
    return float(pos[idx])


def per_rollout_alarms(d, scores, thr):
    """Group forwards by rollout; return first-alarm step and first-contact step."""
    out = {}
    for k in np.unique(d["key"]):
        m = d["key"] == k
        st, sc, pre = d["step"][m], scores[m], d["pre"][m]
        o = np.argsort(st)
        st, sc, pre = st[o], sc[o], pre[o]
        fired = np.where((sc >= thr) & pre)[0]
        first_alarm = int(st[fired[0]]) if len(fired) else None
        out[str(k)] = dict(
            first_alarm=first_alarm,
            n_forwards=int(m.sum()),
            any_alarm=bool(len(fired)),
            collided=bool(d["collided"][m][0]),
        )
    return out


def first_contact_steps(run_dirs):
    out = {}
    for rd in run_dirs:
        mp = os.path.join(rd, "merged.json")
        if not os.path.exists(mp):
            continue
        with open(mp) as fh:
            for r in json.load(fh)["rows"]:
                p = os.path.join(rd, "arrays", f"{r['key']}.npz")
                if not os.path.exists(p):
                    continue
                con = np.asarray(np.load(p)["contact"], bool)
                out[r["key"]] = int(np.argmax(con)) if con.any() else None
    return out


def main():
    train_runs = [
        os.path.join(RUNS, r) for r in os.environ.get("TRAIN_RUNS", "m7b_train").split(",")
    ]
    te_un = os.path.join(RUNS, os.environ.get("TEST_UNSHIFTED_RUN", "m7b_test_unshifted"))
    te_in = os.path.join(RUNS, os.environ.get("TEST_INTR5_RUN", "m7b_test_intr5"))

    tr = P.load_set(train_runs, K_LIST)
    te_u = P.load_set([te_un], K_LIST)
    te_i = P.load_set([te_in], K_LIST)
    print(
        f"[PROBE] train {tr['n_rollouts']} rollouts, {len(tr['y'])} forwards, "
        f"{len(set(tr['scene'].tolist()))} scenes"
    )
    print(f"[PROBE] test unshifted {te_u['n_rollouts']} rollouts, {len(te_u['y'])} forwards")
    print(f"[PROBE] test intrusion-5 {te_i['n_rollouts']} rollouts, {len(te_i['y'])} forwards")
    assert not (set(tr["scene"].tolist()) & set(te_u["scene"].tolist())), (
        "train and test share a scene"
    )

    # alpha by scene-split CV on the TRAINING set only, pre-contact frames (the reference's
    # fitting population)
    m = tr["pre"]
    best, grid = P.scene_split_cv(tr["X"][m], tr["y"][m], tr["scene"][m])
    print(
        "[PROBE] alpha grid: "
        + ", ".join(f"{g['alpha']:.0e}:{g['mean_spearman']:+.3f}" for g in grid)
    )
    print(f"[PROBE] chosen alpha {best['alpha']:.0e} (CV Spearman {best['mean_spearman']:+.4f})")
    fitted = P.fit_ridge(tr["X"][m], tr["y"][m], best["alpha"])

    res = dict(
        alpha=best["alpha"],
        cv=best,
        cv_grid=grid,
        n_train_rollouts=tr["n_rollouts"],
        n_train_forwards=int(len(tr["y"])),
        n_train_scenes=len(set(tr["scene"].tolist())),
        sets={},
    )
    fc = first_contact_steps([te_un, te_in])

    # threshold from TRAINING, so the test false-alarm rate is not fitted on test
    tr_scores = alarm_score(P.predict(fitted, tr["X"]))
    thr = threshold_for_recall(tr_scores[tr["pre"]], tr["hit_16"][tr["pre"]], RECALL_TARGET)
    res["threshold"] = dict(
        value=float(thr), chosen_on="train", target_recall=RECALL_TARGET, at_k=16
    )
    print(f"[PROBE] alarm threshold {thr:+.4f} ({RECALL_TARGET:.0%} recall at k=16, on train)")

    for name, d, _rd in (("test_unshifted", te_u, te_un), ("test_intr5", te_i, te_in)):
        sc = alarm_score(P.predict(fitted, d["X"]))
        pre = d["pre"]
        s = dict(
            n_rollouts=d["n_rollouts"],
            n_forwards=int(len(d["y"])),
            n_forwards_pre_contact=int(pre.sum()),
            spearman_pre_contact=P.spearman(P.predict(fitted, d["X"][pre]), d["y"][pre]),
            spearman_all=P.spearman(P.predict(fitted, d["X"]), d["y"]),
        )
        for k in K_LIST:
            s[f"auroc_k{k}"] = P.auroc(sc[pre], d[f"hit_{k}"][pre])
            s[f"positives_k{k}"] = int(d[f"hit_{k}"][pre].sum())
        # lead time on rollouts that collide
        alarms = per_rollout_alarms(d, sc, thr)
        leads, missed = [], 0
        for key, a in alarms.items():
            if not a["collided"]:
                continue
            c = fc.get(key)
            if c is None:
                continue
            if a["first_alarm"] is None or a["first_alarm"] >= c:
                missed += 1
            else:
                leads.append(c - a["first_alarm"])
        leads = np.asarray(leads, float)
        s["lead_time"] = dict(
            n_collision_rollouts=int(sum(a["collided"] for a in alarms.values())),
            n_with_alarm_before_contact=int(len(leads)),
            n_missed=int(missed),
            steps_mean=float(leads.mean()) if len(leads) else None,
            steps_median=float(np.median(leads)) if len(leads) else None,
            steps_p10=float(np.percentile(leads, 10)) if len(leads) else None,
            steps_p90=float(np.percentile(leads, 90)) if len(leads) else None,
            steps_min=float(leads.min()) if len(leads) else None,
            steps_max=float(leads.max()) if len(leads) else None,
            all_steps=[int(x) for x in leads],
        )
        # false alarms on rollouts that never collide
        clean = [k for k, a in alarms.items() if not a["collided"]]
        n_alarm_clean = sum(1 for k in clean if alarms[k]["any_alarm"])
        steps_clean = 0
        fa_events = 0
        for k in clean:
            mm = d["key"] == k
            fa_events += int(((sc[mm] >= thr) & d["pre"][mm]).sum())
            steps_clean += int(d["step"][mm].max() + 1) if mm.any() else 0
        s["false_alarms"] = dict(
            n_clean_rollouts=len(clean),
            rollouts_with_any_alarm=n_alarm_clean,
            fraction_of_clean_rollouts_alarming=(n_alarm_clean / len(clean)) if clean else None,
            alarming_forwards=fa_events,
            control_steps_in_clean_rollouts=steps_clean,
            alarms_per_100_control_steps=(100.0 * fa_events / steps_clean) if steps_clean else None,
        )
        res["sets"][name] = s
        print(
            f"\n[PROBE] {name}: Spearman(pre-contact) {s['spearman_pre_contact']:+.4f}  "
            f"all {s['spearman_all']:+.4f}"
        )
        print(
            f"         AUROC  k=8 {s['auroc_k8']:.3f}  k=16 {s['auroc_k16']:.3f}  "
            f"k=32 {s['auroc_k32']:.3f}"
        )
        lt = s["lead_time"]
        print(
            f"         lead time: {lt['n_with_alarm_before_contact']}/"
            f"{lt['n_collision_rollouts']} collisions alarmed first; median "
            f"{lt['steps_median']} steps, p10 {lt['steps_p10']}, p90 {lt['steps_p90']}"
        )
        fa = s["false_alarms"]
        print(
            f"         false alarms: {fa['rollouts_with_any_alarm']}/{fa['n_clean_rollouts']} "
            f"clean rollouts, {fa['alarms_per_100_control_steps']:.2f} per 100 control steps"
            if fa["alarms_per_100_control_steps"] is not None
            else "         false alarms: n/a"
        )

    np.savez_compressed(
        OUT + ".npz",
        w=fitted["w"],
        mu=fitted["mu"],
        sd=fitted["sd"],
        ybar=np.array([fitted["ybar"]]),
        alpha=np.array([fitted["alpha"]]),
        threshold=np.array([thr]),
    )
    with open(OUT + ".json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n[PROBE] -> {OUT}.json and {OUT}.npz")
    print("PROBE FIT DONE", flush=True)


if __name__ == "__main__":
    main()
