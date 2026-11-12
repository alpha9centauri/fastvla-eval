"""Step 2 result table: collision rate per intrusion bin, against the pre-registered predictions,
plus the kept-position list Step 3 plans at. CPU only.

Also audits the pattern that produced the scoring defect in the last chapter: a trial whose
minimum clearance is under 5 mm but which no contact was scored for. The monitor is the fixed one
(gripper fingers in the watched set), so these should now be rare; any that remain are listed
rather than assumed benign.

Author: Varun Chandra.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR", "/data_new/vchandr2/data/planner_v2/step2")
PRED = {"9-1000": 0.90, "7-9": 0.80, "5-7": 0.60, "3-5": 0.35}
LAB = {"3-5": "3-5 cm", "5-7": "5-7 cm", "7-9": "7-9 cm", "9-1000": "above 9 cm"}
ORDER = ["9-1000", "7-9", "5-7", "3-5"]


def main():
    rows = [r for r in json.load(open(os.path.join(OUT, "rows_step2.json")))
            if not r.get("skipped")]
    print(f"\n{len(rows)} rollouts scored\n")
    print(f"{'bin':<12}{'n':>4}{'collide':>9}{'rate':>8}{'predicted':>11}{'complete':>10}"
          f"{'median minclr cm':>19}")
    for b in ORDER:
        rs = [r for r in rows if r["bin"] == b]
        if not rs:
            continue
        col = sum(r["collision"] for r in rs)
        comp = sum(r["completion"] for r in rs)
        med = float(np.median([r["min_clearance_cm"] for r in rs]))
        print(f"{LAB[b]:<12}{len(rs):>4}{col:>9}{col/len(rs):>8.0%}{PRED[b]:>11.0%}"
              f"{comp:>10}{med:>19.3f}")
    col = sum(r["collision"] for r in rows)
    comp = sum(r["completion"] for r in rows)
    print(f"{'overall':<12}{len(rows):>4}{col:>9}{col/len(rows):>8.0%}{0.65:>11.0%}{comp:>10}"
          f"{float(np.median([r['min_clearance_cm'] for r in rows])):>19.3f}")

    print("\nper init:")
    print(f"{'init':<6}{'n':>3}{'collide':>9}{'complete':>10}   bins that collided")
    for ii in range(10):
        rs = [r for r in rows if r["init"] == ii]
        if not rs:
            continue
        hit = sorted({LAB[r["bin"]] for r in rs if r["collision"]})
        print(f"i{ii:02d}{'':<3}{len(rs):>3}{sum(r['collision'] for r in rs):>9}"
              f"{sum(r['completion'] for r in rs):>10}   {', '.join(hit) or '-'}")

    pairs = {}
    for r in rows:
        for p in r.get("pairs") or []:
            k = f"{p.get('actor_geom')} x {p.get('obstacle_geom')}"
            pairs[k] = pairs.get(k, 0) + 1
    if pairs:
        print("\ncontact pairs, over every scored collision:")
        for k, v in sorted(pairs.items(), key=lambda kv: -kv[1]):
            print(f"  {v:>3}  {k}")

    near = [r for r in rows if not r["collision"] and r["min_clearance_cm"] < 0.5]
    print(f"\nnear misses: {len(near)} trials under 5 mm with no contact scored")
    for r in sorted(near, key=lambda r: r["min_clearance_cm"]):
        print(f"  {r['cid']}  {LAB[r['bin']]:<10} intrusion {r['intrude_cm']:5.2f} cm  "
              f"min clearance {r['min_clearance_cm']:.3f} cm  complete {r['completion']}")

    kept = [r for r in rows if r["collision"]]
    json.dump(kept, open(os.path.join(HERE, "kept_positions.json"), "w"), indent=1)
    print(f"\nkept for Step 3: {len(kept)} positions where the base policy collides, over "
          f"{len({r['init'] for r in kept})} distinct inits")
    print("STEP2_TABLE_DONE")


if __name__ == "__main__":
    main()
