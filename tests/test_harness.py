"""02c's verification gate, and the properties the comparison rests on.

The gate itself - `null_is_the_null` - is a function in `harness.py` rather
than only a test, because the notebook has to run it and show it passing.
What is here is the gate exercised on both series, plus the three things the
gate cannot see: that the focal car is chosen from the seed and not from the
arm, that the delta signs mean what the column names say, and that the null
cache is keyed on the dials so a sweep point cannot be scored against
another point's baseline.

The last of those is the one worth having. A cache that ignored the dials
would serve every sweep point the baseline from the first, and the result
would be a beautifully smooth response curve that measured nothing.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from endurance import ClassDials, RaceConfig, run_race, scale_dials  # noqa: E402
from endurance.assets import freeze_background  # noqa: E402
from endurance.strategies import ROSTER, FixedLapStint  # noqa: E402
from endurance import harness  # noqa: E402


SEEDS = [11, 12, 13, 14]


def config(series="imsa", duration_s=3 * 3600.0) -> RaceConfig:
    fast = ClassDials(
        series_code=series, class_name="GTP", base_pace_s=97.5,
        deg_slope_s_per_lap=0.012, pace_spread_s=0.5, lap_noise_s=0.5,
        caution_rate=0.20, caution_mean_dur_s=600.0, green_stint_laps=30.0,
        fuel_per_lap=1 / 30, fuel_per_lap_caution=0.6 / 30, tyre_life_laps=60.0,
        pit_time_mean_s=47.0, pit_time_std_s=2.0, n_cars=6)
    slow = ClassDials(
        series_code=series, class_name="GTD", base_pace_s=112.0,
        deg_slope_s_per_lap=0.02, pace_spread_s=0.9, lap_noise_s=0.6,
        caution_rate=0.20, caution_mean_dur_s=600.0, green_stint_laps=28.0,
        fuel_per_lap=1 / 28, fuel_per_lap_caution=0.6 / 28, tyre_life_laps=56.0,
        pit_time_mean_s=45.0, pit_time_std_s=2.0, n_cars=5)
    return RaceConfig(name=f"{series} 3h", series_code=series,
                      duration_s=duration_s, classes=[fast, slow])


# ----------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------
def test_the_null_is_the_null_in_both_series():
    """02c's verification gate, run where it has to hold.

    Both series, because the background field is resolved through
    `PitRules` and a fault in that resolution could show in one and not the
    other.
    """
    for series in ("imsa", "wec"):
        cfg = config(series)
        rows = harness.null_is_the_null(cfg, SEEDS, freeze_background(cfg))
        assert len(rows) == len(SEEDS), series


def test_the_gate_would_notice_a_null_that_is_not_the_null():
    """The gate is only worth running if it can fail.

    A different strategy in the focal seat has to be caught, or the gate is
    asserting that `run_race` agrees with itself.
    """
    cfg = config()
    with pytest.raises(AssertionError):
        harness.null_is_the_null(cfg, SEEDS, freeze_background(cfg),
                                 null_strategy=lambda: FixedLapStint(stint_laps=9))


def test_the_null_arm_of_the_comparison_is_identically_zero():
    """The gate again, from inside the table rather than beside it.

    The null scored against itself must move nothing at all, on every seed.
    This is the row a reader can check in the published figure without
    running anything.
    """
    cfg = config()
    rows = harness.compare_roster(cfg, SEEDS, freeze_background(cfg)).rows
    null = rows[rows["strategy"] == "fuel_window"]
    for col in ("d_class_pos", "d_laps", "d_race_time_s", "d_stops",
                "d_pit_time_s"):
        assert (null[col] == 0).all(), col


# ----------------------------------------------------------------------
# The focal car
# ----------------------------------------------------------------------
def test_the_focal_car_is_chosen_from_the_seed_and_not_from_the_arm():
    """Pace rank picks the same car whatever is driving it.

    This is what makes the comparison paired. A focal car chosen anywhere
    downstream of the race - by finishing position, by who happened to lead
    - would break the pairing while leaving every number looking plausible.
    """
    cfg = config()
    for seed in SEEDS:
        chosen = harness.focal_car(cfg, seed)
        assert chosen == harness.focal_car(cfg, seed)
        # and it does not depend on anything the strategies do
        field = freeze_background(cfg)
        a = harness.run_focal(cfg, seed, chosen, ROSTER["fuel_window"](), field)
        b = harness.run_focal(cfg, seed, chosen, FixedLapStint(stint_laps=9), field)
        assert a["focal"] == b["focal"] == chosen


def test_pace_rank_is_a_rank_and_is_bounds_checked():
    cfg = config()
    ranks = {harness.focal_car(cfg, 11, pace_rank=r)
             for r in range(1, cfg.class_by_name("GTP").n_cars + 1)}
    assert len(ranks) == cfg.class_by_name("GTP").n_cars
    with pytest.raises(ValueError):
        harness.focal_car(cfg, 11, pace_rank=99)


def test_the_headline_class_is_the_quick_one():
    """Taken from pace, so renaming a class cannot change what is measured."""
    assert harness.headline_class(config()) == "GTP"


# ----------------------------------------------------------------------
# The deltas
# ----------------------------------------------------------------------
def test_positive_is_better_in_every_delta_column():
    """The easiest way to publish a table that says the opposite.

    A gained place is a *lower* `class_pos`, so `d_class_pos` has to be the
    null minus the treatment while `d_laps` is the other way round. Checked
    against the raw columns rather than trusted.
    """
    cfg = config()
    rows = harness.compare_roster(cfg, SEEDS, freeze_background(cfg)).rows
    nulls = (rows[rows["strategy"] == "fuel_window"]
             .set_index("seed")[["class_pos", "laps", "pit_time_s"]])

    for row in rows.itertuples():
        null = nulls.loc[row.seed]
        assert row.d_class_pos == null["class_pos"] - row.class_pos
        assert row.d_laps == row.laps - null["laps"]
        assert row.d_pit_time_s == null["pit_time_s"] - row.pit_time_s


# ----------------------------------------------------------------------
# The null cache
# ----------------------------------------------------------------------
def test_the_null_cache_is_keyed_on_the_dials_and_not_only_the_seed():
    """A sweep point must not be scored against another point's baseline.

    Without this the response curve is smooth, plausible and meaningless.
    """
    cfg = config()
    twisted = scale_dials(cfg, fuel_per_lap=2.0)
    field = freeze_background(cfg)
    cache = harness.NullRuns()

    focal = harness.focal_car(cfg, 11)
    base = cache.get(cfg, 11, focal, field, ROSTER["fuel_window"])
    other = cache.get(twisted, 11, focal, field, ROSTER["fuel_window"])
    assert base["stops"] != other["stops"], "the twisted race reused the baseline"


def test_the_cache_returns_the_same_run_for_the_same_race():
    cfg = config()
    field = freeze_background(cfg)
    cache = harness.NullRuns()
    focal = harness.focal_car(cfg, 11)
    assert (cache.get(cfg, 11, focal, field, ROSTER["fuel_window"])
            == cache.get(cfg, 11, focal, field, ROSTER["fuel_window"]))


# ----------------------------------------------------------------------
# The two tables
# ----------------------------------------------------------------------
def test_the_summary_is_grouped_by_series_and_never_pooled():
    """Decision 10's budget is per strategy per series.

    A rulebook difference makes the lap-down defender's wave clause live in
    IMSA and dead in WEC, so a pooled row would report the mean of a
    measurement and a non-measurement.
    """
    frames = []
    for series in ("imsa", "wec"):
        cfg = config(series)
        frames.append(harness.compare_roster(cfg, SEEDS,
                                             freeze_background(cfg)).rows)
    summary = harness.summarise(pd.concat(frames, ignore_index=True))
    assert set(summary["series"]) == {"imsa", "wec"}
    assert len(summary) == 2 * len(ROSTER)


def test_the_summary_reports_shares_and_not_a_mean_delta():
    """Decision 10 asks for a distribution. The mean is absent on purpose."""
    cfg = config()
    summary = harness.compare_roster(cfg, SEEDS,
                                     freeze_background(cfg)).summarise()
    assert {"gained", "level", "lost", "median_d_pos"} <= set(summary.columns)
    assert not any("mean" in c for c in summary.columns)
    for row in summary.itertuples():
        assert abs(row.gained + row.level + row.lost - 1.0) < 1e-9


def test_provenance_records_what_the_comparison_was_run_against():
    """A table without its dials is a table about an unknown race."""
    cfg = config()
    comparison = harness.compare_roster(cfg, SEEDS, freeze_background(cfg))
    for key in ("dials_fingerprint", "series_code", "class", "pace_rank",
                "null", "n_seeds"):
        assert key in comparison.provenance, key


# ----------------------------------------------------------------------
# The benchmark join
# ----------------------------------------------------------------------
def test_the_benchmark_join_is_optional_and_does_nothing_when_absent():
    """The notebook has to run before 02b's per-seed script has."""
    cfg = config()
    rows = harness.compare_roster(cfg, SEEDS, freeze_background(cfg)).rows
    assert harness.attach_benchmark(rows, None).equals(rows)

    joined = harness.attach_benchmark(
        rows, {s: {"class_pos": 1, "laps": 200} for s in SEEDS})
    assert "gap_to_benchmark_pos" in joined.columns
    assert (joined["gap_to_benchmark_pos"] == joined["class_pos"] - 1).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
