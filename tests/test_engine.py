"""Invariants the engine must not break.

These are not "does it produce nice numbers" tests - they are the properties
everything built later leans on. The paired-comparison property in
particular (same seed, same race) is what makes the strategy comparison in
02 and the agent evaluation in 03 mean anything at all, so it is tested
explicitly rather than assumed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from endurance import (  # noqa: E402
    ClassDials,
    FixedLapStint,
    OpportunistUnderCaution,
    RaceConfig,
    RunToFuelWindow,
    assign_strategy,
    run_race,
    scale_dials,
)
from endurance.engine import CautionTimeline  # noqa: E402


def make_config(duration_s=3600.0, caution_rate=0.1, n_cars=6, n_cars_b=5):
    """A small two-class race - big enough for traffic, quick enough to test."""
    fast = ClassDials(
        series_code="test", class_name="FAST",
        base_pace_s=100.0, deg_slope_s_per_lap=0.02,
        pace_spread_s=0.8, lap_noise_s=0.3,
        caution_rate=caution_rate, caution_mean_dur_s=300.0,
        green_stint_laps=20.0, fuel_per_lap=1 / 20, fuel_per_lap_caution=0.6 / 20,
        tyre_life_laps=40.0, pit_time_mean_s=40.0, pit_time_std_s=2.0,
        n_cars=n_cars,
    )
    slow = ClassDials(
        series_code="test", class_name="SLOW",
        base_pace_s=115.0, deg_slope_s_per_lap=0.03,
        pace_spread_s=1.0, lap_noise_s=0.4,
        caution_rate=caution_rate, caution_mean_dur_s=300.0,
        green_stint_laps=22.0, fuel_per_lap=1 / 22, fuel_per_lap_caution=0.6 / 22,
        tyre_life_laps=44.0, pit_time_mean_s=42.0, pit_time_std_s=2.0,
        n_cars=n_cars_b,
    )
    return RaceConfig(name="Test 1h", series_code="test",
                      duration_s=duration_s, classes=[fast, slow])


# ----------------------------------------------------------------------
# Basic integrity
# ----------------------------------------------------------------------
def test_race_runs_and_every_car_classifies():
    cfg = make_config()
    result = run_race(cfg, seed=0)
    c = result.classification()

    assert len(c) == cfg.total_cars
    assert (c["laps"] > 0).all()
    assert c["overall_pos"].tolist() == list(range(1, len(c) + 1))


def test_fuel_never_goes_negative():
    result = run_race(make_config(), seed=1)
    assert (result.laps["fuel"] >= -1e-9).all()


def test_tyre_age_never_exceeds_its_life():
    cfg = make_config()
    result = run_race(cfg, seed=2)
    for cls in cfg.classes:
        sub = result.laps[result.laps["class"] == cls.class_name]
        assert sub["tyre_age"].max() <= cls.tyre_life_laps


def test_no_car_races_past_the_chequered_flag():
    cfg = make_config(duration_s=1800.0)
    result = run_race(cfg, seed=3)
    # A car may cross the line once after time expires - that is how a timed
    # race ends - but it must not start another lap after that.
    for car_id, sub in result.laps.groupby("car_id"):
        past = sub[sub["t"] >= cfg.duration_s]
        assert len(past) <= 1, f"{car_id} ran on past the flag"


def test_faster_class_completes_more_laps():
    result = run_race(make_config(), seed=4)
    c = result.classification()
    assert c[c["class"] == "FAST"]["laps"].mean() > c[c["class"] == "SLOW"]["laps"].mean()


# ----------------------------------------------------------------------
# The property that makes comparisons fair
# ----------------------------------------------------------------------
def test_same_seed_gives_an_identical_race():
    a = run_race(make_config(), seed=7).classification()
    b = run_race(make_config(), seed=7).classification()
    assert a.equals(b)


def test_different_seeds_give_different_races():
    a = run_race(make_config(), seed=7).classification()
    b = run_race(make_config(), seed=8).classification()
    assert not a.equals(b)


def test_caution_timeline_is_independent_of_strategy():
    """The whole point of drawing cautions in advance.

    Change what every car does; the race that happens to them must not
    change. Without this, a strategy could look good simply by having been
    handed an easier race.
    """
    cfg = make_config()
    a = run_race(cfg, default_strategy=RunToFuelWindow(), seed=11)
    b = run_race(cfg, default_strategy=FixedLapStint(stint_laps=12), seed=11)
    assert a.cautions.periods == b.cautions.periods


def test_strategy_actually_changes_the_outcome():
    cfg = make_config()
    a = run_race(cfg, default_strategy=RunToFuelWindow(), seed=11).classification()
    b = run_race(cfg, default_strategy=FixedLapStint(stint_laps=10), seed=11).classification()
    # Stopping every 10 laps instead of every ~20 must cost laps.
    assert b["stops"].mean() > a["stops"].mean()
    assert b["laps"].mean() < a["laps"].mean()


# ----------------------------------------------------------------------
# Cautions
# ----------------------------------------------------------------------
def test_caution_timeline_hits_its_target_share():
    rng = np.random.default_rng(0)
    shares = []
    for _ in range(60):
        tl = CautionTimeline.draw(24 * 3600, caution_rate=0.20,
                                  mean_dur_s=600.0, rng=rng)
        shares.append(tl.total_caution_s() / (24 * 3600))
    # The alternating draw realises the share it was given, so this band is
    # tight on both sides. It used to be generous on the low side because
    # the old draw merged overlaps away; that allowance would now hide a
    # regression rather than describe one.
    assert 0.18 < np.mean(shares) < 0.22


def test_zero_caution_rate_means_no_cautions():
    rng = np.random.default_rng(0)
    tl = CautionTimeline.draw(3600, caution_rate=0.0, mean_dur_s=300.0, rng=rng)
    assert tl.periods == []
    assert not tl.is_caution(100.0)


def test_caution_periods_do_not_overlap_and_are_ordered():
    rng = np.random.default_rng(3)
    tl = CautionTimeline.draw(24 * 3600, 0.3, 400.0, rng)
    for (s1, e1), (s2, e2) in zip(tl.periods, tl.periods[1:]):
        assert s1 < e1 <= s2 < e2


def test_more_cautions_means_more_caution_laps():
    cfg = make_config(caution_rate=0.05)
    quiet = run_race(cfg, seed=5)
    busy = run_race(scale_dials(cfg, caution_rate=6.0), seed=5)
    assert busy.laps["under_caution"].sum() > quiet.laps["under_caution"].sum()


# ----------------------------------------------------------------------
# Levers
# ----------------------------------------------------------------------
def test_more_fuel_per_lap_means_more_stops():
    cfg = make_config()
    base = run_race(cfg, seed=6).classification()
    thirsty = run_race(scale_dials(cfg, fuel_per_lap=2.0), seed=6).classification()
    assert thirsty["stops"].mean() > base["stops"].mean()


def test_harsher_degradation_slows_the_field():
    cfg = make_config()
    base = run_race(cfg, seed=9).classification()
    worn = run_race(scale_dials(cfg, deg_slope_s_per_lap=20.0), seed=9).classification()
    assert worn["laps"].mean() < base["laps"].mean()


def test_traffic_penalty_is_felt_and_can_be_switched_off():
    cfg = make_config()
    with_traffic = run_race(cfg, seed=10)
    assert with_traffic.laps["traffic_s"].sum() > 0

    without = run_race(scale_dials(cfg, traffic_penalty_s=0.0), seed=10)
    assert without.laps["traffic_s"].sum() == 0


def test_traffic_only_ever_costs_time():
    result = run_race(make_config(), seed=12)
    assert (result.laps["traffic_s"] >= 0).all()


def test_scale_dials_leaves_the_original_untouched():
    cfg = make_config()
    before = cfg.classes[0].fuel_per_lap
    scale_dials(cfg, fuel_per_lap=3.0)
    assert cfg.classes[0].fuel_per_lap == before


def test_unknown_dial_is_rejected():
    with pytest.raises(AttributeError):
        scale_dials(make_config(), not_a_real_dial=2.0)


# ----------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------
def test_caution_opportunist_stops_under_caution_more_often():
    cfg = make_config(caution_rate=0.25)
    plain = run_race(cfg, default_strategy=RunToFuelWindow(), seed=13)
    oppo = run_race(cfg, default_strategy=OpportunistUnderCaution(), seed=13)

    plain_share = plain.laps.loc[plain.laps["pitted"], "under_caution"].mean()
    oppo_share = oppo.laps.loc[oppo.laps["pitted"], "under_caution"].mean()
    assert oppo_share > plain_share


def test_strategies_can_be_assigned_per_class():
    cfg = make_config()
    strats = assign_strategy(cfg, FixedLapStint(stint_laps=8), class_name="FAST")
    assert all(k.startswith("FAST-") for k in strats)
    assert len(strats) == cfg.class_by_name("FAST").n_cars

    result = run_race(cfg, strategies=strats, seed=14).classification()
    fast_stops = result[result["class"] == "FAST"]["stops"].mean()
    slow_stops = result[result["class"] == "SLOW"]["stops"].mean()
    assert fast_stops > slow_stops


# ----------------------------------------------------------------------
# Results plumbing
# ----------------------------------------------------------------------
def test_positions_are_a_valid_ranking_at_every_moment():
    result = run_race(make_config(duration_s=1200.0), seed=15)
    pos = result.positions()
    assert pos["position"].min() == 1
    assert pos["position"].max() <= result.config.total_cars
    assert (pos["class_position"] >= 1).all()


def test_config_survives_a_save_and_load(tmp_path):
    cfg = make_config()
    path = tmp_path / "cfg.json"
    cfg.save(path)
    back = RaceConfig.load(path)
    assert back.to_dict() == cfg.to_dict()
    assert run_race(back, seed=1).classification().equals(
        run_race(cfg, seed=1).classification()
    )


def test_summary_reports_a_plausible_race():
    result = run_race(make_config(), seed=16)
    s = result.summary()
    assert s["cars"] == make_config().total_cars
    assert 0.0 <= s["caution_share"] <= 1.0
    assert s["winning_laps"] > 0
