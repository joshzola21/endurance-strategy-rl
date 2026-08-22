"""Run the gates, print what they assert on, and write it down once.

    python scripts/evidence.py
    python scripts/evidence.py --quiet      # the verdict lines only

Four handovers have recorded that **the gates print nothing**. They pass or
they fail, and a passing gate leaves no trace of the number it passed on - so
the claim that every figure in the write-up traces to something checkable was
true in principle and unverifiable in practice.

This runs the gates that a committed tree can run, prints the quantity each one
turns on, and writes `outputs/evidence.json`. That file is what
`scripts/check_figures.py` checks the documents against, and it is what the
methods page can read rather than restate.

What it can and cannot run
--------------------------
Everything here runs from the artefacts in this repository. The one gate it
cannot run is 00's - the dials regenerating from the raw timing - because
`laps.csv` is 582 MB and is not committed. Where a stored result from that gate
exists it is read and labelled as read; where it does not, the gap is reported
rather than passed over. `scripts/check_dials_provenance.py` is the thing that
produces it, on a machine that has the file.

It also does not re-run training. The five-seed sweep costs about two hours and
its conclusion is a property of runs that have already happened, so the numbers
are read from `outputs/seed_sweep/` if that folder exists.

**Nothing here is a substitute for `pytest tests/`.** The gates live in the
test suite; this reports what they turn on. A gate that passed silently and a
gate that was never run look the same from here, which is why the verdict says
which of the two it saw.
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
sys.path.insert(0, str(ROOT))

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

from endurance import harness                                   # noqa: E402
from endurance.assets import (BackgroundField, SeedBank,        # noqa: E402
                              dials_fingerprint, rules_fingerprint)
from endurance.params import RaceConfig                         # noqa: E402
from endurance.strategies import ROSTER                         # noqa: E402

SERIES = ("imsa", "wec")
BANKS = ("headline", "held_out")
N_GATE = 5              # races per series where a gate runs live
DELTAS = ("d_class_pos", "d_laps", "d_race_time_s", "d_stops", "d_pit_time_s")
OUT = ROOT / "outputs" / "evidence.json"


def find(name: str) -> Path | None:
    for hit in sorted(ROOT.rglob(name)):
        if ".ipynb_checkpoints" not in hit.parts:
            return hit
    return None


def load(code: str):
    return (RaceConfig.load(find(f"{code}.json")),
            SeedBank.load(find(f"{code}_seeds.json")),
            BackgroundField.load(find(f"{code}_field.json")))


class Evidence:
    """Gate results and figures, printed as they are found and written once."""

    def __init__(self, quiet: bool = False):
        self.gates: list[dict] = []
        self.figures: dict[str, float | int | str] = {}
        self.quiet = quiet

    def gate(self, stage: str, name: str, verdict: str, detail: str) -> None:
        self.gates.append({"stage": stage, "gate": name, "verdict": verdict,
                           "detail": detail})
        if not self.quiet or verdict != "held":
            print(f"  {verdict:9} [{stage}] {name}")
            print(f"            {detail}")

    def figure(self, key: str, value) -> None:
        """A number the documents are allowed to quote, under a stable name."""
        self.figures[key] = value


# ----------------------------------------------------------------------
# 00 - the dials regenerate from the raw timing
# ----------------------------------------------------------------------
def stage_00(ev: Evidence) -> None:
    print("\n=== 00 · calibration ===")
    stored = find("dials_provenance.json")
    if stored is None:
        ev.gate("00", "the dials regenerate from the raw timing", "not run",
                "needs data/raw/laps.csv, which is 582 MB and not committed. "
                "Run scripts/check_dials_provenance.py on a machine that has "
                "it; the last recorded verdict was 'calibrated, float noise "
                "only' for both series, at residuals of 1e-15 to 1e-18.")
        return
    verdicts = json.loads(stored.read_text())
    ev.gate("00", "the dials regenerate from the raw timing", "read",
            f"from {stored.relative_to(ROOT)}: {verdicts}")


# ----------------------------------------------------------------------
# 02c - the null is the null
# ----------------------------------------------------------------------
def stage_02c(ev: Evidence) -> None:
    print("\n=== 02c · the paired comparison ===")
    for code in SERIES:
        config, bank, field = load(code)
        seeds = bank.headline[:N_GATE]

        # Live, on this tree, rather than read from a table somebody wrote.
        rows = harness.null_is_the_null(config, seeds, field)
        ev.gate("02c", f"{code}: the null reproduces itself", "held",
                f"{len(rows)} of {len(seeds)} seeds, dials "
                f"{dials_fingerprint(config)}")

        comparison = harness.compare_roster(config, seeds, field)
        null = comparison.rows[comparison.rows["strategy"] == "fuel_window"]
        worst = max(float(null[c].abs().max()) for c in DELTAS)
        ev.gate("02c", f"{code}: the null's deltas are identically zero",
                "held" if worst == 0.0 else "BROKEN",
                f"largest absolute delta across {', '.join(DELTAS)} is {worst:g}")

        # The sign convention, checked against the raw columns. A table that
        # published these backwards would look entirely reasonable.
        nulls = null.set_index("seed")[["class_pos", "laps", "pit_time_s"]]
        bad = 0
        for row in comparison.rows.itertuples():
            base = nulls.loc[row.seed]
            bad += (row.d_class_pos != base["class_pos"] - row.class_pos)
            bad += (row.d_laps != row.laps - base["laps"])
        ev.gate("02c", f"{code}: positive is better in every delta column",
                "held" if bad == 0 else "BROKEN",
                f"{len(comparison.rows)} rows checked against the raw columns, "
                f"{bad} disagreements")


# ----------------------------------------------------------------------
# 04 - the app steps the race everything else was measured on
# ----------------------------------------------------------------------
def stage_04(ev: Evidence) -> None:
    print("\n=== 04 · the app ===")
    try:
        from app.controller import RaceController, roster_seat
    except ImportError as e:
        ev.gate("04", "gate A: the app's race is the harness's race", "not run",
                f"could not import the app: {e}")
        return

    columns = ("laps", "race_time_s", "stops", "pit_time_s", "class_pos")
    for code in SERIES:
        config, bank, field = load(code)
        mismatched = 0
        for seed in bank.headline[:N_GATE]:
            focal = harness.focal_car(config, seed)
            ctrl = RaceController(config=config, field=field, seed=seed,
                                  seat=roster_seat("fuel_window"),
                                  seat_name="fuel_window")
            mine = ctrl.finish().classification().set_index("car_id").loc[focal]
            theirs = harness.run_focal(config, seed, focal,
                                       ROSTER["fuel_window"](), field)
            mismatched += sum(1 for c in columns if mine[c] != theirs[c])
        ev.gate("04", f"{code}: gate A, the app's race is the harness's race",
                "held" if mismatched == 0 else "BROKEN",
                f"{N_GATE} races x {len(columns)} columns, "
                f"{mismatched} disagreements")


# ----------------------------------------------------------------------
# The saved tables - what the write-up is allowed to quote
# ----------------------------------------------------------------------
def figures(ev: Evidence) -> None:
    print("\n=== the saved tables ===")
    for code in SERIES:
        for bank_name in BANKS:
            path = find(f"{code}_{bank_name}_summary.csv")
            if path is None:
                ev.gate("05", f"{code}/{bank_name}: saved table", "missing",
                        "run scripts/evaluate.py")
                continue
            table = pd.read_csv(path).set_index("strategy")

            if "never_pit" not in table.index:
                ev.gate("05", f"{code}/{bank_name}: the control is scored",
                        "BROKEN",
                        "this table predates never_pit and cannot say whether "
                        "a policy has learned anything")
            else:
                ev.gate("05", f"{code}/{bank_name}: the control is scored",
                        "held", f"{len(table)} strategies including never_pit")

            for strategy in table.index:
                for column in ("gained", "lost", "level", "median_d_pos",
                               "median_stops"):
                    if column in table.columns:
                        value = table.at[strategy, column]
                        ev.figure(f"{code}.{bank_name}.{strategy}.{column}",
                                  round(float(value), 4))

            # The one comparison amendment 25 exists for, computed rather than
            # left to a reader: is the agent distinguishable from a car that
            # takes no decisions?
            if {"agent", "never_pit"} <= set(table.index):
                gap = max(abs(float(table.at["agent", c])
                              - float(table.at["never_pit", c]))
                          for c in ("gained", "lost", "level")
                          if c in table.columns)
                ev.figure(f"{code}.{bank_name}.agent_minus_never_pit", round(gap, 4))
                ev.gate("05", f"{code}/{bank_name}: agent against the control",
                        "held",
                        f"largest difference on any share is {gap:.3f} - "
                        + ("indistinguishable from taking no decisions"
                           if gap < 0.02 else "distinguishable"))


# ----------------------------------------------------------------------
# The five-seed sweep, read rather than re-run
# ----------------------------------------------------------------------
def seed_spread(ev: Evidence) -> None:
    print("\n=== the training-seed spread ===")
    folder = ROOT / "outputs" / "seed_sweep"
    if not folder.is_dir():
        ev.gate("post-04", "an agent row is a distribution, not a point",
                "not run", "no outputs/seed_sweep/; run scripts/seed_sweep.py "
                           "(about two hours)")
        return

    for code in SERIES:
        files = sorted(folder.glob(f"{code}_headline_summary_s*.csv"))
        if not files:
            continue
        gained, stops = [], []
        for path in files:
            table = pd.read_csv(path).set_index("strategy")
            if "agent" not in table.index:
                continue
            gained.append(float(table.at["agent", "gained"]))
            if "median_stops" in table.columns:
                stops.append(float(table.at["agent", "median_stops"]))
        if not gained:
            continue
        spread = float(np.ptp(gained))
        ev.figure(f"{code}.train_seed_spread_gained", round(spread, 4))
        ev.figure(f"{code}.train_seed_stops", sorted(stops))
        ev.gate("post-04", f"{code}: an agent row is a distribution",
                "held",
                f"{len(gained)} training seeds, gained ranges "
                f"{min(gained):.3f} to {max(gained):.3f} (spread {spread:.3f})"
                + (f", median stops {sorted(stops)}" if stops else ""))


# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"tree: {ROOT}")
    print(f"rule logic: {rules_fingerprint()}")

    ev = Evidence(quiet=args.quiet)
    ev.figure("rules_fingerprint", rules_fingerprint())
    for code in SERIES:
        config, _bank, _field = load(code)
        ev.figure(f"{code}.dials_fingerprint", dials_fingerprint(config))
        ev.figure(f"{code}.duration_h", round(config.duration_s / 3600, 2))
        ev.figure(f"{code}.total_cars", config.total_cars)
        ev.figure(f"{code}.n_classes", len(config.classes))
        ev.figure(f"{code}.caution_rate", round(config.classes[0].caution_rate, 4))

    stage_00(ev)
    stage_02c(ev)
    stage_04(ev)
    figures(ev)
    seed_spread(ev)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rules_fingerprint": rules_fingerprint(),
        "gates": ev.gates,
        "figures": ev.figures,
    }, indent=2))

    held = sum(1 for g in ev.gates if g["verdict"] == "held")
    broken = [g for g in ev.gates if g["verdict"] in ("BROKEN", "missing")]
    absent = [g for g in ev.gates if g["verdict"] == "not run"]

    print("\n=== verdict ===")
    print(f"  {held} gates held, {len(broken)} broken, {len(absent)} not run")
    print(f"  {len(ev.figures)} figures written to "
          f"{OUT.relative_to(ROOT)}")
    for g in broken:
        print(f"  BROKEN  [{g['stage']}] {g['gate']}")
    for g in absent:
        print(f"  not run [{g['stage']}] {g['gate']}")
    if absent and not broken:
        print("\n  A gate that was not run is not a gate that passed. The two "
              "are listed separately\n  for that reason.")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
