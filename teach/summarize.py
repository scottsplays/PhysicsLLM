"""Table of all runs in teach/results: mean and spread across seeds.

  python teach/summarize.py
"""

import json
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"


def main():
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RES.glob("*.json"))]
    by = defaultdict(list)
    for r in runs:
        by[r["name"]].append(r)
    keys = ["choice_acc", "top1_acc", "acc_fact", "acc_relation", "acc_novel", "train_loss"]
    rows = []
    # Untrained floor = step 0 of every run.
    floor = [r["curve"][0] for r in runs if r["curve"][0]["step"] == 0]
    if floor:
        rows.append(("untrained (step 0)", len(floor), floor))
    for name, rs in sorted(by.items()):
        rows.append((name, len(rs), [r["curve"][-1] for r in rs]))
    chance = runs[0]["curve"][-1]["chance"] if runs else 0
    print(f"{'condition':<20} {'seeds':>5}  " + "  ".join(f"{k:>13}" for k in keys))
    for name, n, cs in rows:
        cells = []
        for k in keys:
            vals = [c[k] for c in cs if c.get(k) is not None]
            if not vals:
                cells.append(f"{'—':>13}")
                continue
            m = sum(vals) / len(vals)
            spread = (max(vals) - min(vals)) / 2
            fmt = (f"{m:.3f}±{spread:.3f}" if k == "train_loss" else f"{m:.0%}±{spread:.0%}")
            cells.append(f"{fmt:>13}")
        print(f"{name:<20} {n:>5}  " + "  ".join(cells))
    print(f"\nchance (random option) = {chance:.0%};  ± is half the range across seeds")


if __name__ == "__main__":
    main()
