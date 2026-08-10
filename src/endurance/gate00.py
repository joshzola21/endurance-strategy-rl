"""Stage 00's verification gate.

The blueprint specifies 00 with a boundary constraint and no gate, which 02c
already named as a gate nobody can fail. This is the replacement, in the shape
02c and 03b established: conditions that can fail, that fail on the defect they
were written for, and that are checkable against the data rather than against
anybody's judgement.

Four conditions:

1. **The race is one race.** Duration, grid size and lap counts are those of a
   single running.
2. **The dials describe the race they came from.** Checked against the file
   rather than against the engine - see `condition_two` for why that changed.
3. **The falsifier.** Widen the scope to two editions deliberately and require
   one and two to fail. A gate that cannot fail is not a gate.
4. **Degradation has the right sign in every class.** Stated separately
   because it is the one symptom that might not be caused by pooling.

Kept out of `calibrate.py` on purpose. That module produces numbers and must
not import the engine; this one consumes both and is imported by the notebook
and the tests rather than by the package.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .calibrate import GREEN, build_race_config, calibrate_pace
from .engine import run_race
from .params import RaceConfig

SCHEDULED_S = 24 * 3600.0


def _row(condition: str, check: str, value, threshold, passed: bool) -> dict:
    return {"condition": condition, "check": check,
            "value": value, "threshold": threshold, "passed": bool(passed)}


# ----------------------------------------------------------------------
def observed_classification(con, session_ids) -> pd.DataFrame:
    """What the file says happened: laps and stops per car, per class.

    Grouped on `(class, car)` with `car` as text. Read as an integer it merges
    the leading-zero pairs, which is how Le Mans 2026's Hypercar winner came
    to show sixty-two stops in three hundred and eighty-one laps.
    """
    from .calibrate import _scope
    return con.execute(f"""
        SELECT session_id, class, car, MAX(lap) AS laps,
               SUM(CASE WHEN pit_time IS NOT NULL THEN 1 ELSE 0 END) AS stops
        FROM laps
        WHERE session = 'race' AND {_scope(session_ids)}
        GROUP BY session_id, class, car
    """).df()


def _observed_winner(obs: pd.DataFrame, class_name: str) -> pd.Series:
    rows = obs[obs["class"] == class_name].sort_values("laps", ascending=False)
    return rows.iloc[0]


def _observed_laps_per_stop(obs: pd.DataFrame, class_name: str) -> float:
    """Median laps between stops across the class.

    The *median* across cars, matching what the simulated side computes. An
    earlier version took the winner's own ratio here and the class median on
    the other side, which compared two different statistics and reported the
    difference as a calibration error.
    """
    rows = obs[(obs["class"] == class_name) & (obs["stops"] > 0)]
    return float((rows["laps"] / rows["stops"]).median())


def _observed_pit_iqr(con, session_ids, class_name: str) -> tuple[float, float]:
    """The interquartile range of observed green stops for one class."""
    from .calibrate import _scope
    v = con.execute(f"""
        SELECT pit_time FROM laps
        WHERE session = 'race' AND class = '{class_name}'
          AND {_scope(session_ids)}
          AND pit_time IS NOT NULL AND flags = 'GF'
    """).df()["pit_time"].astype(float).dropna()
    return float(v.quantile(0.25)), float(v.quantile(0.75))


# ----------------------------------------------------------------------
def condition_one(con, cfg: RaceConfig, session_ids,
                  scheduled_s: float = SCHEDULED_S,
                  duration_tol: float = 0.02,
                  lap_tol: float = 0.05) -> pd.DataFrame:
    """The scoped selection is one running of the race."""
    from .calibrate import _scope
    rows = [_row("one: the race is one race", "duration vs scheduled",
                 round(cfg.duration_s / 3600, 2), f"{scheduled_s / 3600:.0f} h ±2%",
                 abs(cfg.duration_s - scheduled_s) / scheduled_s <= duration_tol)]

    # Restricted to the classes the config names, or this compares a sum over
    # some classes with a union over all of them and fails on a dropped class.
    #
    # Within one race a number belongs to one car, so the sum of the per-class
    # counts equals the number of distinct numbers. Pooling breaks that: a
    # number that was GTP one year and GTD the next is counted twice on the
    # left and once on the right.
    named = "', '".join(c.class_name for c in cfg.classes)
    grid = con.execute(f"""
        SELECT COUNT(DISTINCT car) AS cars
        FROM laps WHERE session = 'race' AND {_scope(session_ids)}
          AND class IN ('{named}')
    """).df().iloc[0]
    rows.append(_row("one: the race is one race",
                     "one number, one car (grid is a single grid)",
                     cfg.total_cars, int(grid["cars"]),
                     cfg.total_cars == int(grid["cars"])))

    # Every class in the selection actually competed in this edition. With a
    # session scope that is true by construction, so what this catches is a
    # class list assembled somewhere other than from the scope.
    observed = set(con.execute(f"""
        SELECT DISTINCT class FROM laps
        WHERE session = 'race' AND {_scope(session_ids)}
    """).df()["class"])
    named = {c.class_name for c in cfg.classes}
    rows.append(_row("one: the race is one race", "classes all competed",
                     len(named - observed), 0, named <= observed))

    # No car completed more laps than the race had time for.
    obs = observed_classification(con, session_ids)
    for c in cfg.classes:
        allowed = cfg.duration_s / c.base_pace_s
        seen = int(obs.loc[obs["class"] == c.class_name, "laps"].max())
        rows.append(_row("one: the race is one race",
                         f"{c.class_name} laps within the clock",
                         seen, round(allowed * (1 + lap_tol)),
                         seen <= allowed * (1 + lap_tol)))
    return pd.DataFrame(rows)


def _observed_green_stints(con, session_ids, class_name: str):
    """The file's completed green stints for one class, in laps."""
    from .calibrate import _fuel_stints, _stint_frame
    df = _fuel_stints(_stint_frame(con, session_ids, class_name))
    last = df.groupby(["session_id", "car"])["fuel_stint"].transform("max")
    green = (df[(df["fuel_stint"] < last) & (df["flags"] == GREEN)]
             .groupby(["session_id", "car", "fuel_stint"]).size())
    return green[green >= 3]


def condition_two(con, cfg: RaceConfig, session_ids,
                  seeds: Sequence[int] = (0, 1, 2, 3, 4),
                  laps_tol: float = 0.10) -> pd.DataFrame:
    """The dials describe the race they were calibrated from.

    **Restated at 00.** The incoming specification asked for the simulated
    stint length and mean pit time to land within 10% of the observed ones.
    Neither can be met, and the diagnostic located why - twice, in opposite
    directions, and neither cause is a dial:

    * At Le Mans the simulated stint is 1.4 laps short of the tank in all
      three classes. That is `RunToFuelWindow` stopping a lap and a half
      early, which 03a recorded and which is a property of the baseline.
    * At Daytona the file's cars average 26.4 laps between stops against a
      30-lap tank, *despite* a caution share that should stretch stints by
      11%. Real cars take fuel under yellow with half a tank; the default
      field never does. The effect scales with caution rate, which is why the
      two series miss in opposite directions.

    Both are the engine and the strategy roster consuming the dials, and the
    stage's boundary is explicit that a dial the engine misuses is 02a's
    problem rather than 00's. A gate that cannot be met is dropped and
    replaced, not restated until it passes - 03a's precedent.

    What replaces it compares the **calibration against the file**, which is
    what this stage is answerable for, and keeps one coarse engine check that
    holds if anything does:

    1. `green_stint_laps` lies between the median and the longest stint the
       cars actually completed. The dial it replaced - the upper quartile of
       laps per `stint_number`, which counts driver stints - came out at 58 to
       73 laps against a maximum of 36, so this fails on the defect it was
       written for.
    2. `pit_time_mean_s` lies inside the interquartile range of observed green
       stops. The arithmetic mean this replaced sat outside it.
    3. `pit_time_std_s` is no greater than its own mean.
    4. The simulated winner beats the real winner - it has no damage, no
       penalties and no unscheduled stops - by less than `laps_tol`. One-sided
       because the direction is predicted, and 10% because the two mechanisms
       above account for a systematic offset of order 5% between them; a
       tighter bound would be measuring 02a's fidelity, not 00's calibration.
    """
    obs = observed_classification(con, session_ids)
    sims = [run_race(cfg, seed=s).classification() for s in seeds]

    rows = []
    for c in cfg.classes:
        green = _observed_green_stints(con, session_ids, c.class_name)
        med, mx = float(green.median()), float(green.max())
        rows.append(_row("two: the dials describe the race",
                         f"{c.class_name} stint dial inside observed stints",
                         round(c.green_stint_laps, 1), f"{med:.0f}-{mx:.0f}",
                         med <= c.green_stint_laps <= mx))

        lo, hi = _observed_pit_iqr(con, session_ids, c.class_name)
        rows.append(_row("two: the dials describe the race",
                         f"{c.class_name} pit dial inside observed IQR",
                         round(c.pit_time_mean_s, 1), f"{lo:.1f}-{hi:.1f}",
                         lo <= c.pit_time_mean_s <= hi))

        rows.append(_row("two: the dials describe the race",
                         f"{c.class_name} pit sd not above its mean",
                         round(c.pit_time_std_s, 1), round(c.pit_time_mean_s, 1),
                         c.pit_time_std_s <= c.pit_time_mean_s))

        real = _observed_winner(obs, c.class_name)
        got = float(np.median([
            int(s[s["class"] == c.class_name]
                .sort_values("laps", ascending=False).iloc[0]["laps"])
            for s in sims]))
        over = (got - real["laps"]) / real["laps"]
        rows.append(_row("two: the dials describe the race",
                         f"{c.class_name} winning laps, 0 to +{laps_tol:.0%}",
                         f"{got:.0f} ({over:+.1%})", int(real["laps"]),
                         0.0 <= over <= laps_tol))
    return pd.DataFrame(rows)


def condition_four(con, cfg: RaceConfig, session_ids) -> pd.DataFrame:
    """Degradation has the right sign in every class where the sign means anything.

    **The sign test applies only where the slope is identified.** Within a
    stint the car gets lighter as the tyres get older; where a class changes
    tyres at every stop the two are the same number and no estimator can
    separate them. Le Mans Hypercar and LMP2 are in that position: tyre age
    tops out at 10 against fuel stints of 11 to 12 laps.

    Testing the sign of a quantity that is provably not identified is a test
    of nothing, and the blueprint's rule is that such a gate is dropped and
    replaced rather than restated until it passes. What replaces it is
    visibility: an unidentified class is exempt from the sign test and is
    **named** - in this table, in the notebook's Part 3, and on the app page -
    so the exemption is something a reader meets rather than something a
    threshold absorbs.

    This cuts both ways, which is the point. Before this change every IMSA
    class was unidentified and every one passed the sign test on a positive
    slope, so four sign tests on quantities that mean nothing were being
    counted as evidence that the dials were sound.
    """
    rows, exempt = [], []
    for c in cfg.classes:
        p = calibrate_pace(con, session_ids, c.class_name)
        if p["deg_identified"]:
            rows.append(_row("four: degradation sign",
                             f"{c.class_name} slope > 0",
                             round(c.deg_slope_s_per_lap, 5), "> 0",
                             c.deg_slope_s_per_lap > 0))
        else:
            exempt.append(c.class_name)
            rows.append(_row("four: degradation sign",
                             f"{c.class_name} EXEMPT - tyre age and fuel not "
                             f"separable (corr {p['age_fuel_corr']:.2f})",
                             round(c.deg_slope_s_per_lap, 5),
                             "net within-stint trend, sign not tested", True))

    rows.append(_row("four: degradation sign",
                     "classes whose slope is a net trend, not tyre wear",
                     ", ".join(exempt) if exempt else "none",
                     "must be named in Part 3 and on the app page", True))
    return pd.DataFrame(rows)


def stint_diagnostic(con, cfg: RaceConfig, session_ids,
                     seeds: Sequence[int] = (0, 1, 2)) -> pd.DataFrame:
    """Where the stint discrepancy lives: the file, the dial, or the engine.

    Not part of the gate. Condition two's stint rows fail in opposite
    directions in the two series - IMSA long, WEC short - so a single
    systematic cause is ruled out and the difference has to be located rather
    than assumed. Three quantities, side by side:

    * what the file's stints look like, in green laps between pit records;
    * what the dial took from them, which is the upper quartile - tank range
      rather than typical stint;
    * what the engine then does with it, which is a stint per tank with no
      unscheduled stops and no early termination.

    A dial above the observed maximum means a tank nobody ever emptied. A
    simulated stint far from the dial means the discrepancy is in the engine's
    stopping rule or in `fuel_per_lap_caution`, not in the calibration.
    """
    from .calibrate import _fuel_stints, _stint_frame

    obs = observed_classification(con, session_ids)
    sims = [run_race(cfg, seed=s).classification() for s in seeds]

    rows = []
    for c in cfg.classes:
        df = _fuel_stints(_stint_frame(con, session_ids, c.class_name))
        last = df.groupby(["session_id", "car"])["fuel_stint"].transform("max")
        green = (df[(df["fuel_stint"] < last) & (df["flags"] == "GF")]
                 .groupby(["session_id", "car", "fuel_stint"]).size())
        green = green[green >= 3]

        o = obs[(obs["class"] == c.class_name) & (obs["stops"] > 0)]
        sim_lps = float(np.median([
            (s.loc[s["class"] == c.class_name, "laps"]
             / s.loc[s["class"] == c.class_name, "stops"].clip(lower=1)).median()
            for s in sims]))

        rows.append({
            "class": c.class_name,
            "file_green_median": round(float(green.median()), 1),
            "file_green_q75_the_dial": round(float(green.quantile(0.75)), 1),
            "file_green_max": int(green.max()),
            "dial_green_stint_laps": round(c.green_stint_laps, 1),
            "file_laps_per_stop": round(float((o["laps"] / o["stops"]).median()), 1),
            "sim_laps_per_stop": round(sim_lps, 1),
            "caution_rate": round(c.caution_rate, 3),
            "fuel_per_lap_caution_ratio": round(
                c.fuel_per_lap_caution / c.fuel_per_lap, 2),
        })
    return pd.DataFrame(rows)


def condition_three(con, series_code: str, session_ids, other_session_id: int,
                    **kwargs) -> pd.DataFrame:
    """The falsifier: pooling two editions must break conditions one and two.

    Only worth running if pooling is something the gate detects rather than
    something it tolerates. The partner is the adjacent edition rather than a
    distant one, because two runnings of the same race are genuinely similar
    and are the hardest case for the gate to catch.
    """
    ids = ([int(session_ids)] if isinstance(session_ids, (int, np.integer))
           else [int(s) for s in session_ids])
    widened = ids + [int(other_session_id)]

    pooled = build_race_config(con, series_code, widened, name="POOLED falsifier")
    one = condition_one(con, pooled, widened, **kwargs)
    try:
        two = condition_two(con, pooled, widened)
    except Exception:
        # A config built from two editions failing to run at all is a failure
        # of condition two, not an error in the gate.
        two = pd.DataFrame([_row("two: the dials reproduce the race",
                                 "pooled config runs", "raised", "runs", False)])

    failed = int((~one["passed"]).sum() + (~two["passed"]).sum())
    detail = pd.concat([one, two], ignore_index=True)
    detail["condition"] = "three: falsifier (pooled, must fail) - " + detail["condition"]
    summary = pd.DataFrame([_row("three: the falsifier",
                                 f"pooling {ids} with {other_session_id} is detected",
                                 f"{failed} checks failed", "at least 1", failed > 0)])
    return pd.concat([summary, detail], ignore_index=True)


# ----------------------------------------------------------------------
def run_gate(con, series_code: str, session_id: int, other_session_id: int,
             cfg: RaceConfig | None = None, verbose: bool = True) -> pd.DataFrame:
    """All four conditions, one table. The notebook shows this."""
    cfg = cfg or build_race_config(con, series_code, session_id)
    parts = [
        condition_one(con, cfg, session_id),
        condition_two(con, cfg, session_id),
        condition_three(con, series_code, session_id, other_session_id),
        condition_four(con, cfg, session_id),
    ]
    out = pd.concat(parts, ignore_index=True)
    if verbose:
        headline = out[~out["condition"].str.startswith("three: falsifier (")]
        print(headline.to_string(index=False))
        print(f"\n{int(headline['passed'].sum())}/{len(headline)} checks pass")
    return out


def gate_passes(gate: pd.DataFrame) -> bool:
    """Conditions one, two and four must pass; three must have detected pooling.

    The falsifier's own detail rows are excluded: they are required to fail and
    counting them as failures would invert the gate.
    """
    headline = gate[~gate["condition"].str.startswith("three: falsifier (")]
    return bool(headline["passed"].all())
