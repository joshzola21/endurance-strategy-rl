"""How much does the agent row move when only the training seed changes?

Every agent number this project has reported is one training run. The human
strategies are deterministic functions of the race, so their two-hundred-seed
interval is the whole of their uncertainty. An agent row carries that interval
*plus* the spread across training seeds, and nothing has ever measured the
second.

This reads the per-seed evaluation tables written by the recipe in the
handover and reports the spread beside the evaluation width, so the two can be
compared. If the training spread is the larger of the two, then every
comparison drawn between runs in this project so far was drawn across noise.

    python scripts/collect_seed_spread.py

Reads `outputs/seed_sweep/{series}_{bank}_summary_s{N}.csv`. Writes nothing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "outputs" / "seed_sweep"
SERIES = ("imsa", "wec")
BANKS = ("headline", "held_out")

# The columns worth watching. `gained` is what every claim in this project is
# phrased in; `lost` because a strategy can gain and lose at once; the two
# diagnostics because they say *what the policy did* rather than how it scored,
# and two runs that score alike for different reasons are not the same result.
WATCH = ("gained", "lost", "median_d_pos", "median_stops")


def load(series: str, bank: str) -> pd.DataFrame | None:
    rows = []
    for path in sorted(SWEEP.glob(f"{series}_{bank}_summary_s*.csv")):
        m = re.search(r"_s(\d+)\.csv$", path.name)
        if not m:
            continue
        d = pd.read_csv(path)
        d["train_seed"] = int(m.group(1))
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else None


def report(series: str, bank: str) -> dict | None:
    d = load(series, bank)
    if d is None:
        return None

    seeds = [int(x) for x in sorted(d["train_seed"].unique())]
    print(f"\n=== {series.upper()} / {bank} — {len(seeds)} training seeds "
          f"{seeds} ===")

    agent = d[d["strategy"] == "agent"].sort_values("train_seed")
    if agent.empty:
        print("  no agent row in these tables")
        return None

    for col in WATCH:
        if col not in agent.columns:
            continue
        v = agent[col].to_numpy(dtype=float)
        print(f"  {col:14} " + "  ".join(f"s{s}={x:6.3f}"
                                         for s, x in zip(seeds, v)))
        print(f"  {'':14} mean {v.mean():6.3f}  sd {v.std(ddof=1):6.3f}  "
              f"range {v.min():.3f}..{v.max():.3f}  spread {np.ptp(v):.3f}")

    # The comparison that decides whether any of this was ever readable.
    gained = agent["gained"].to_numpy(dtype=float)
    width = None
    if {"gained_lo", "gained_hi"} <= set(agent.columns):
        width = float((agent["gained_hi"] - agent["gained_lo"]).mean())
        print(f"\n  evaluation interval, one run   {width:.3f} wide "
              f"(200 paired races)")
        print(f"  training-seed spread           {np.ptp(gained):.3f} wide "
              f"({len(seeds)} runs)")
        if np.ptp(gained) > width:
            print("  -> the training seed moves the answer further than the "
                  "evaluation can resolve.\n     A single run is not a "
                  "measurement of this configuration.")
        else:
            print("  -> the training seed moves the answer less than the "
                  "evaluation interval.\n     A single run is readable, and "
                  "the differences between configurations were real.")

    # Where the agent sits among the humans, using the worst and best runs
    # rather than the mean: a range is the honest summary of five runs.
    humans = d[(d["strategy"] != "agent") & (d["train_seed"] == seeds[0])]
    if not humans.empty:
        print("\n  against the roster (which does not move with training seed):")
        for _, row in humans.sort_values("gained", ascending=False).iterrows():
            print(f"    {row['strategy']:16} {row['gained']:.3f}")
        print(f"    {'agent':16} {gained.min():.3f}..{gained.max():.3f} "
              f"across training seeds")

    return {"series": series, "bank": bank, "spread": float(np.ptp(gained)),
            "width": width}


def main() -> int:
    if not SWEEP.is_dir():
        print(f"no {SWEEP.relative_to(ROOT)}/ — run the training loop first")
        return 2

    found = [r for s in SERIES for b in BANKS
             if (r := report(s, b)) is not None]
    if not found:
        print(f"no per-seed tables in {SWEEP.relative_to(ROOT)}/. Expected "
              f"names like imsa_headline_summary_s0.csv")
        return 2

    print("\n=== verdict ===")
    for r in found:
        if r["width"] is None:
            print(f"  {r['series']}/{r['bank']}: spread {r['spread']:.3f} "
                  f"(no interval in the table to compare against)")
            continue
        verdict = ("training noise dominates" if r["spread"] > r["width"]
                   else "single runs are readable")
        print(f"  {r['series']}/{r['bank']}: spread {r['spread']:.3f} against "
              f"interval {r['width']:.3f} — {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
