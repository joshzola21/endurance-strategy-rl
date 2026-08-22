"""Sweep `pit_transit_frac`, the assumed dial 03b's agent found a hole in.

    python scripts/sweep_pit_transit.py              # both series
    python scripts/sweep_pit_transit.py imsa         # one
    python scripts/sweep_pit_transit.py --no-agent   # the humans alone

What this is asking
-------------------
03b's IMSA policy converged on **164 stops in a 189-lap race**, at 12.8 s a
stop against the null's 41.9 s. That is not a bug in the policy; it is the
policy being right about the model. Work it through the stand-in dials:
lane transit costs `pit_time_mean_s * pit_transit_frac` = 47 x 0.25 =
11.75 s, and `stop_cost` charges for the fuel actually taken. A car stopping
every lap is nearly full each time, so it adds one lap's worth - a thirtieth
of a tank - for 35.25 / 30 = 1.18 s. Total 12.93 s against 12.85 observed.

`pit_transit_frac` is in `ASSUMED_FIELDS`. Lap timing records how long a
stop took, never what happened during it, so this number was never measured
and decision 2 says an assumed parameter gets swept rather than chosen. The
agent reached a corner of the action space the five parameter-free
strategies had no reason to visit, and found the assumption there. That is
what an adversarial policy search is for, and it is a result rather than an
embarrassment.

**The question is where the roster's conclusions stop depending on it.** If
the human rows move as this dial moves, 02c's findings are partly about an
assumption. If only the agent's row moves, the exploit is the agent's and
the humans were never near it.

What the sweep can and cannot reach
-----------------------------------
The pit model anchors a full service at the measured mean and splits it into
shares, so raising transit lowers the refuel share and **a full stop always
costs the same**. Only the ratio moves, which is exactly the quantity the
exploit turns on.

That also fixes the ceiling. `test_a_full_service_costs_the_measured_mean_
in_both_series` holds because a full stop is transit plus the larger of the
tyre and refuel jobs; with `pit_tyre_frac` at 0.35 that requires
`pit_transit_frac <= 0.65`. Multipliers past 2.6 break the anchor and are
refused here rather than producing a table that silently violates it.

**Worth stating plainly: 0.65 may not be enough.** A real Daytona pit delta
- lane entry, the speed limit down the lane, exit - is most of a stop, and
the current shape cannot express a large fixed overhead on top of a
proportional service at all. If the sweep runs out of room before the
exploit dies, the finding is about the *shape* of `stop_cost` rather than
about the value of this dial, and that is a bigger thing than a number.

The arm this does not run
-------------------------
The policy is held fixed across the sweep. That answers "how much of the
agent's result is an artefact of the dial it trained on". It does not answer
"what would an agent do if the dial were right", which needs a retrain at
each point - five points, two series, half a million steps each - and is a
separate decision rather than a flag, because the retrained agent is a
different agent and its row would not belong in this table.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def find_project_root(marker: str = "src/endurance") -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(
        f"could not find {marker!r} at or above {here.parent}.\n\n"
        "This script belongs in `scripts/`, beside `src/` and `notebooks/`.")


ROOT = find_project_root()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd                                             # noqa: E402

from endurance import harness, scale_dials                      # noqa: E402
from endurance.assets import (dials_fingerprint,               # noqa: E402
                               dials_source)
from endurance.policy import (                                  # noqa: E402
    PolicyCard,
    agent_roster,
    bank_fingerprint,
    load_policy,
)
from endurance.strategies import ROSTER                         # noqa: E402
from freeze_assets import SERIES, load_assets                   # noqa: E402


DIAL = "pit_transit_frac"
POLICIES = ROOT / "outputs" / "policies"
OUT = ROOT / "outputs" / "sweeps"

# 0.25 -> 0.65 in five steps. The top of the range is not a taste: past it a
# full service stops costing the measured mean, which is the one thing the
# pit layer is not allowed to change.
MULTIPLIERS = (1.0, 1.4, 1.8, 2.2, 2.6)
DELTA_COLUMNS = ("d_class_pos", "d_laps", "d_race_time_s", "d_stops",
                 "d_pit_time_s")


def check_range(config, multipliers) -> None:
    """Refuse a multiplier that breaks the anchor rather than reporting it."""
    for cls in config.classes:
        for m in multipliers:
            transit = cls.pit_transit_frac * m
            if transit + cls.pit_tyre_frac > 1.0 + 1e-9:
                raise ValueError(
                    f"multiplier {m} puts {DIAL} at {transit:.3f} on "
                    f"{cls.class_name}, and {transit:.3f} + "
                    f"{cls.pit_tyre_frac} exceeds 1.0 - a full service would "
                    f"stop costing pit_time_mean_s. Cap the sweep, or change "
                    f"pit_tyre_frac too and accept that the point is no "
                    f"longer one-at-a-time.")


def behaviour(rows: pd.DataFrame) -> pd.DataFrame:
    """Stops, laps and seconds a stop - where the exploit is visible.

    `harness.sweep_dial` returns `summarise()` alone, and the gained/lost
    shares cannot show a policy stopping a hundred and sixty times. So this
    mirrors `sweep_dial`'s contract - one fresh `NullRuns` per point, so a
    scaled race is never scored against another point's baseline - and keeps
    the rows as well. The deviation is the output, not the method.
    """
    out = rows.groupby("strategy").agg(
        laps=("laps", "mean"), stops=("stops", "mean"),
        pit_time_s=("pit_time_s", "mean"), class_pos=("class_pos", "mean"),
    )
    out["s_per_stop"] = (out["pit_time_s"] / out["stops"]).where(out["stops"] > 0)
    return out.round(2).reset_index()


def sweep(series_code: str, agent: bool = True,
          multipliers=MULTIPLIERS, out: Path = OUT) -> dict:
    config, bank, field = load_assets(series_code)
    check_range(config, multipliers)
    seeds = bank.sweep                       # decision 10: sweeps run on the fifty

    roster, card = dict(ROSTER), None
    if agent:
        checkpoint = POLICIES / f"{series_code}_maskable_ppo.zip"
        # Checked once, against the dials the policy was trained on. Every
        # point after the first is deliberately off-fingerprint - that is
        # what a sweep is - so the check cannot be repeated per point and
        # must not be silently skipped either.
        strategy = load_policy(checkpoint, config=config, bank=bank)
        card = PolicyCard.load(checkpoint)
        roster = agent_roster(roster, strategy)

    rows, summaries, behaviours = [], [], []
    for m in multipliers:
        scaled = scale_dials(config, **{DIAL: m})
        value = scaled.classes[0].pit_transit_frac
        comparison = harness.compare_roster(scaled, seeds, field, roster=roster,
                                            nulls=harness.NullRuns())

        point = comparison.rows
        null = point[point["strategy"] == "fuel_window"]
        for col in DELTA_COLUMNS:
            if not (null[col] == 0).all():
                raise AssertionError(
                    f"{series_code} at {DIAL}={value:.3f}: the null moved on "
                    f"{col}. Nothing at this point is readable.")

        for frame, sink in ((point, rows),
                            (comparison.summarise(), summaries),
                            (behaviour(point), behaviours)):
            frame = frame.copy()
            frame.insert(0, DIAL, round(value, 4))
            frame.insert(0, "multiplier", m)
            sink.append(frame)
        print(f"  {series_code} {DIAL}={value:.3f} (x{m}) done")

    rows = pd.concat(rows, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    behave = pd.concat(behaviours, ignore_index=True)

    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{series_code}_{DIAL}"
    rows.to_csv(f"{stem}_rows.csv", index=False)
    summary.to_csv(f"{stem}_summary.csv", index=False)
    behave.to_csv(f"{stem}_behaviour.csv", index=False)
    Path(f"{stem}_provenance.json").write_text(json.dumps({
        "dial": DIAL,
        "series_code": series_code,
        "race": config.name,
        "dials_fingerprint_at_multiplier_1": dials_fingerprint(config),
        "bank": "sweep",
        "bank_fingerprint": bank_fingerprint(bank),
        "n_seeds": len(seeds),
        "multipliers": list(multipliers),
        "values": [round(config.classes[0].pit_transit_frac * m, 4)
                   for m in multipliers],
        "dials_source": dials_source(config),
        "policy_held_fixed": bool(agent),
        "policy_card": card.__dict__ if card else None,
        "null_per_point": "fresh NullRuns at every point",
        "swept_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))

    return {"series": series_code, "rows": rows, "summary": summary,
            "behaviour": behave}


def report(result: dict) -> None:
    series_code = result["series"]
    behave, summary = result["behaviour"], result["summary"]

    print(f"\n=== {series_code}: what each strategy does as {DIAL} rises ===")
    for name in ("agent", "fuel_window", "caution_gambler"):
        sub = behave[behave["strategy"] == name]
        if sub.empty:
            continue
        print(f"  {name}")
        print(sub[[DIAL, "stops", "s_per_stop", "laps", "class_pos"]]
              .to_string(index=False))

    print(f"\n=== {series_code}: gained share as {DIAL} rises ===")
    wide = summary.pivot_table(index=DIAL, columns="strategy", values="gained")
    print(wide.round(3).to_string())

    # The question the sweep exists to answer, stated rather than left to the
    # reader: does the roster's ranking depend on an assumption?
    humans = [c for c in wide.columns if c not in ("agent", "fuel_window")]
    swing = (wide[humans].max() - wide[humans].min()).round(3)
    print(f"\n  human rows swing across the sweep: "
          f"{', '.join(f'{k} {v}' for k, v in swing.items())}")
    if "agent" in wide.columns:
        a = wide["agent"]
        print(f"  agent swings {a.max() - a.min():.3f} "
              f"({a.iloc[0]:.3f} at {wide.index[0]:.2f} -> "
              f"{a.iloc[-1]:.3f} at {wide.index[-1]:.2f})")

    agent_rows = behave[behave["strategy"] == "agent"]
    if not agent_rows.empty:
        # Rate, not count. An absolute stop count falls across the sweep for
        # a reason that has nothing to do with the policy: dearer stops mean
        # fewer laps fit in six hours, so there are fewer laps to stop on.
        # The first version of this block compared counts and announced that
        # the exploit had survived, which a frozen policy guarantees whatever
        # the dial does.
        rate = (agent_rows["stops"] / agent_rows["laps"]).round(3)
        print(f"  agent stops per lap: "
              f"{' '.join(f'{r:.3f}' for r in rate)}")
        if rate.max() - rate.min() < 0.02:
            print("  Flat, as a frozen policy must be. **This arm cannot say "
                  "whether the exploit is dial-dependent** - only a retrain "
                  "at each point can, and that is a different agent per "
                  "point. What it does show is what the exploit costs: laps "
                  f"{agent_rows['laps'].iloc[0]:.0f} -> "
                  f"{agent_rows['laps'].iloc[-1]:.0f}.")


def monotone_movers(summary: pd.DataFrame, tol: float = 1e-9) -> list[str]:
    """Strategies whose gained share moves one way at every step.

    The discriminator this sweep needs. At fifty seeds a share carries an
    interval around +-0.14, so any single pair of points proves nothing and
    the raw swing flatters a strategy that merely wandered. A share that
    falls at all four steps in both series is a different kind of evidence -
    not a formal test, but not noise either.
    """
    wide = summary.pivot_table(index=DIAL, columns="strategy", values="gained")
    out = []
    for name in wide.columns:
        v = wide[name].to_numpy()
        if v.max() - v.min() < 0.05 or name == "fuel_window":
            continue
        if all(v[i] >= v[i + 1] - tol for i in range(len(v) - 1)) or \
           all(v[i] <= v[i + 1] + tol for i in range(len(v) - 1)):
            out.append(name)
    return out


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("series", nargs="*", default=list(SERIES),
                        help="imsa, wec, or nothing for both")
    parser.add_argument("--no-agent", action="store_true",
                        help="the five humans alone")
    args = parser.parse_args(argv[1:])

    unknown = [s for s in args.series if s not in SERIES]
    if unknown:
        raise SystemExit(f"unknown series {unknown}; choose from {list(SERIES)}")

    movers = {}
    for series_code in args.series:
        result = sweep(series_code, agent=not args.no_agent)
        report(result)
        movers[series_code] = monotone_movers(result["summary"])
        print(f"  monotone across the sweep: "
              f"{', '.join(movers[series_code]) or 'nobody'}")

    both = set.intersection(*map(set, movers.values())) if len(movers) > 1 else set()
    if both:
        print(f"\nMoves one way in every series: {', '.join(sorted(both))}. "
              f"That result is about this assumption, not about racing, and "
              f"belongs beside 02c's rows in the decision record.")

    print(f"\nwrote to {OUT.relative_to(ROOT)}")
    print("Read the human swing first. If the roster's ranking moves as this "
          "dial moves, 02c's findings are partly about an assumption.")


if __name__ == "__main__":
    main(sys.argv)
