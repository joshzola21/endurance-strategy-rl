"""Score the agent the only way it may be scored: as a sixth roster member.

    python scripts/evaluate.py                      # both series, headline + held out
    python scripts/evaluate.py imsa --bank sweep    # a selection run
    python scripts/evaluate.py --no-agent           # the five humans alone

There is no agent-specific evaluation code in this file and there must not
be. The policy is wrapped in `gym_env.PolicyStrategy`, inserted into
`ROSTER` under the name `agent`, and handed to `harness.compare_roster` with
the same banks, the same frozen background field, the same headline class
and the same pace rank as the five humans. Every column in the output is
computed by the same lines that computed the humans' columns. A second
evaluation path that can differ from the roster's is the failure decision 6
exists to prevent, and it is the kind that produces plausible numbers rather
than errors.

Which bank, and what each one can be used for
---------------------------------------------
`--bank sweep` is the fifty the design is selected on: hyperparameters,
architecture, timestep budget, anything chosen by looking at a number. The
sweep fifty are the first fifty of the headline two hundred, so this is a
slice of the training races and says nothing about generalisation. That is
what it is for.

`--bank headline` is the two hundred the roster was scored on, and the
comparable table. The agent trained on these, which the write-up has to say
in the same sentence as the number: the five humans did not train at all, so
the headline row is the agent at its most flattering.

`--bank held_out` is the fifty nothing was chosen on and nothing trained on.
It is the honest generalisation row, and it is reported beside the headline
rather than instead of it. Note that the refusal in `EnduranceEnv` does not
apply here - evaluation never touches the env, it runs `PolicyStrategy`
through `harness.run_focal` - so the discipline on this bank is procedural
rather than enforced. Do not select on it.

Two tables, never one
---------------------
Decision 10's budget is per strategy per series, and a rulebook difference
makes the caution lever live in IMSA and nearly dead in WEC. Every output
here is per series. Nothing is pooled, and `summarise` is called per series
frame rather than on a concatenation.

The spread check
----------------
02c published a roster finding on a two-place gap in a statistic whose
per-seed spread ran to three places, and two hundred seeds reversed its
direction. So `--bootstrap` resamples the seed set - paired, so the
strategies stay on the same races within a resample - and reports an
interval beside every share and median. **A ranking whose intervals overlap
is not a ranking**, and the seed count belongs in the sentence next to the
number.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def find_project_root(marker: str = "src/endurance") -> Path:
    """Walk up until the project appears, as the notebooks do."""
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

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

from endurance import harness                                   # noqa: E402
from endurance.assets import dials_fingerprint                  # noqa: E402
from endurance.policy import (                                  # noqa: E402
    PolicyCard,
    agent_roster,
    bank_fingerprint,
    load_policy,
)
from endurance.strategies import ROSTER                         # noqa: E402
from freeze_assets import SERIES, load_assets                   # noqa: E402


POLICIES = ROOT / "outputs" / "policies"
OUT = ROOT / "outputs" / "evaluation"

BANKS = ("headline", "sweep", "held_out")
DELTA_COLUMNS = ("d_class_pos", "d_laps", "d_race_time_s", "d_stops",
                 "d_pit_time_s")


# ----------------------------------------------------------------------
# The spread check
# ----------------------------------------------------------------------
def paired_bootstrap(rows: pd.DataFrame, n_resamples: int = 2000,
                     seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """An interval on each strategy's share and median, over the seed set.

    The seeds are resampled **once per draw and shared across strategies**,
    not independently per strategy. Resampling independently would break the
    pairing the whole comparison rests on and would widen every interval by
    the between-race variance that common random numbers exist to remove.

    This says how much of a difference is the seed set rather than the
    strategy. It says nothing about how much of it is the model, which is
    02b's ratio check and needs the per-seed benchmark artefacts that do not
    yet exist.
    """
    wide = rows.pivot_table(index="seed", columns="strategy",
                            values="d_class_pos")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(wide), size=(n_resamples, len(wide)))
    values = wide.to_numpy()

    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    out = []
    for j, strategy in enumerate(wide.columns):
        column = values[:, j]
        sampled = column[draws]                      # (n_resamples, n_seeds)
        gained = (sampled > 0).mean(axis=1)
        lost = (sampled < 0).mean(axis=1)
        median = np.median(sampled, axis=1)
        out.append({
            "strategy": strategy,
            "n_seeds": len(wide),
            "gained_lo": round(float(np.quantile(gained, lo_q)), 3),
            "gained_hi": round(float(np.quantile(gained, hi_q)), 3),
            "lost_lo": round(float(np.quantile(lost, lo_q)), 3),
            "lost_hi": round(float(np.quantile(lost, hi_q)), 3),
            "median_d_pos_lo": float(np.quantile(median, lo_q)),
            "median_d_pos_hi": float(np.quantile(median, hi_q)),
        })
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# One series, one bank
# ----------------------------------------------------------------------
def load_benchmark(series_code: str, bank_name: str) -> dict | None:
    """02b's per-seed causal reference, if it has ever been produced.

    It has not been, at the time of writing - the blueprint records the
    per-seed artefacts as outstanding - so the join is optional and the
    absence is reported rather than passed over. When they arrive, the
    expected shape is `{seed: {"class_pos": int, "laps": int}}` from the
    **causal** reference: scoring against the clairvoyant one alone would
    penalise a policy for failing to predict the future.
    """
    path = ROOT / "data" / "processed" / f"{series_code}_benchmark_causal.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


def evaluate(series_code: str, bank_name: str, agent: bool = True,
             bootstrap: bool = True, out: Path = OUT) -> dict:
    config, bank, field = load_assets(series_code)
    seeds = list(getattr(bank, bank_name))

    roster = dict(ROSTER)
    card = None
    if agent:
        checkpoint = POLICIES / f"{series_code}_maskable_ppo.zip"
        # `config` and `bank` are passed so the card is a check rather than
        # a note: a policy trained against other dials, or against another
        # draw of the bank, is refused here instead of quietly producing a
        # complete and meaningless table.
        strategy = load_policy(checkpoint, config=config, bank=bank)
        card = PolicyCard.load(checkpoint)
        roster = agent_roster(roster, strategy)

    comparison = harness.compare_roster(config, seeds, field, roster=roster)
    rows = harness.attach_benchmark(comparison.rows,
                                    load_benchmark(series_code, bank_name))
    summary = harness.summarise(rows)

    if len(set(rows["series"])) != 1:
        raise AssertionError("a frame covering two series has been built; "
                             "the two are never pooled")

    spread = paired_bootstrap(rows) if bootstrap else None
    if spread is not None:
        summary = summary.merge(spread, on="strategy", how="left")

    # The null arm has to be identically zero in every delta column. It is
    # 02c's gate seen from inside the table, it costs nothing to assert, and
    # a non-zero null puts a systematic offset under every number here.
    null = rows[rows["strategy"] == "fuel_window"]
    for col in DELTA_COLUMNS:
        if not (null[col] == 0).all():
            raise AssertionError(
                f"{series_code}/{bank_name}: the null moved on {col}. The "
                f"apparatus is wrong and nothing in this table is readable.")

    stem = out / f"{series_code}_{bank_name}"
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(f"{stem}_rows.csv", index=False)
    summary.to_csv(f"{stem}_summary.csv", index=False)

    provenance = {
        **comparison.provenance,
        "bank": bank_name,
        "bank_fingerprint": bank_fingerprint(bank),
        "dials_source": "STAND-IN, not calibrated",
        "agent": bool(agent),
        "policy_card": card.__dict__ if card else None,
        "benchmark_joined": "gap_to_benchmark_pos" in rows.columns,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bootstrap_resamples": 2000 if bootstrap else 0,
    }
    Path(f"{stem}_provenance.json").write_text(json.dumps(provenance, indent=2))

    return {"series": series_code, "bank": bank_name, "rows": rows,
            "summary": summary, "provenance": provenance}


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def report(result: dict) -> None:
    series_code, bank_name = result["series"], result["bank"]
    summary, provenance = result["summary"], result["provenance"]
    n = provenance["n_seeds"]

    print(f"\n=== {series_code} / {bank_name} ({n} seeds, "
          f"dials {provenance['dials_fingerprint']}) ===")
    columns = [c for c in ("strategy", "gained", "gained_lo", "gained_hi",
                           "level", "lost", "median_d_pos",
                           "median_d_pos_lo", "median_d_pos_hi")
               if c in summary.columns]
    print(summary[columns].to_string(index=False))

    if not provenance["benchmark_joined"]:
        print("  benchmark: NOT JOINED - 02b's per-seed causal artefacts do "
              "not exist, so the gap to the reference is unmeasured")

    if "agent" in set(summary["strategy"]) and "gained_lo" in summary.columns:
        agent = summary[summary["strategy"] == "agent"].iloc[0]
        others = summary[~summary["strategy"].isin(["agent", "fuel_window"])]
        overlapping = [r.strategy for r in others.itertuples()
                       if r.gained_lo <= agent.gained_hi
                       and agent.gained_lo <= r.gained_hi]
        if overlapping:
            print(f"  the agent's interval overlaps {', '.join(overlapping)} "
                  f"at {n} seeds: not an ordering, and must not be reported "
                  f"as one")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("series", nargs="*", default=list(SERIES),
                        help="imsa, wec, or nothing for both")
    parser.add_argument("--bank", nargs="*", default=["headline", "held_out"],
                        help=f"any of {list(BANKS)}")
    parser.add_argument("--no-agent", action="store_true",
                        help="the five humans alone, before a policy exists")
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args(argv[1:])

    for name, given, allowed in (("series", args.series, SERIES),
                                 ("bank", args.bank, BANKS)):
        unknown = [g for g in given if g not in allowed]
        if unknown:
            raise SystemExit(f"unknown {name} {unknown}; "
                             f"choose from {list(allowed)}")

    for series_code in args.series:
        for bank_name in args.bank:
            report(evaluate(series_code, bank_name, agent=not args.no_agent,
                            bootstrap=not args.no_bootstrap))

    print(f"\nwrote tables to {OUT.relative_to(ROOT)}")
    if "held_out" in args.bank and "headline" in args.bank:
        print("Report both. The headline row is the agent on races it "
              "trained on; the held-out row is the one that generalises.")


if __name__ == "__main__":
    main(sys.argv)
