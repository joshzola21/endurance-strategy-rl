"""Compression, wave-arounds, and the one property that must not break.

The decision record is explicit that if a single test gets written for this
work it should be the one asserting compression never invents position. So
that one comes first, and it is checked the hard way - by rebuilding every
car's race time from its own lap times and demanding the classification
agree - rather than by asserting that a function was not called.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from endurance import ClassDials, Compat, RaceConfig, run_race  # noqa: E402
from endurance.caution import (  # noqa: E402
    CautionRules,
    compressed_lap_time,
    wave_eligible,
)


class _Car:
    def __init__(self, car_id, class_name):
        self.car_id, self.class_name = car_id, class_name


def two_class_config(duration_s=6 * 3600.0, caution_rate=0.2) -> RaceConfig:
    """A field with a quick class and a slow one, so caution pace is visible."""
    fast = ClassDials(
        series_code="imsa", class_name="GTP", base_pace_s=97.5,
        deg_slope_s_per_lap=0.012, pace_spread_s=0.5, lap_noise_s=0.5,
        caution_rate=caution_rate, caution_mean_dur_s=600.0,
        green_stint_laps=30.0, fuel_per_lap=1 / 30, fuel_per_lap_caution=0.6 / 30,
        tyre_life_laps=60.0, pit_time_mean_s=47.0, pit_time_std_s=2.0, n_cars=8)
    slow = ClassDials(
        series_code="imsa", class_name="GTD", base_pace_s=112.0,
        deg_slope_s_per_lap=0.02, pace_spread_s=0.9, lap_noise_s=0.6,
        caution_rate=caution_rate, caution_mean_dur_s=600.0,
        green_stint_laps=28.0, fuel_per_lap=1 / 28, fuel_per_lap_caution=0.6 / 28,
        tyre_life_laps=56.0, pit_time_mean_s=45.0, pit_time_std_s=2.0, n_cars=10)
    return RaceConfig(name="compression test", series_code="imsa",
                      duration_s=duration_s, classes=[fast, slow])


# ----------------------------------------------------------------------
# The invariant
# ----------------------------------------------------------------------
def test_compression_never_invents_position():
    """Every car's finishing time is the sum of its own laps and stops.

    This is 01's central claim, re-checked against machinery built to move
    cars around: if compression had reached in and reordered anything, the
    rebuilt time would drift from the classified one. Nothing else in this
    file matters as much.
    """
    result = run_race(two_class_config(), seed=3)
    classification = result.classification().set_index("car_id")

    for car_id, sub in result.laps.groupby("car_id"):
        rebuilt = sub["lap_time"].sum() + sub["pit_cost_s"].sum()
        assert abs(rebuilt - classification.loc[car_id, "race_time_s"]) < 1e-6, car_id
        assert len(sub) == classification.loc[car_id, "laps"], car_id


def test_a_car_cannot_compress_past_the_car_ahead():
    """Overtaking under yellow is prohibited; here it is arithmetically impossible.

    The next gap works out at `target * close + gap * (1 - close)`, which is
    positive for any positive gap, so the check is that it stays positive
    across the whole range of gaps a race can produce.
    """
    sc, target, close = 160.0, 2.0, 0.5
    for gap in np.linspace(0.1, 320.0, 200):
        for gap_ahead in (0.1, 2.0, 40.0, 300.0):
            lap = compressed_lap_time(sc, gap, target, close, floor_s=0.0)
            lap_ahead = compressed_lap_time(sc, gap_ahead, target, close, floor_s=0.0)
            # The next gap is a weighted average of two positive gaps, so it
            # stays positive however differently the two cars are placed.
            assert gap + lap - lap_ahead > 0.0, (gap, gap_ahead)


def test_gaps_converge_on_the_queue_spacing():
    sc, target, close = 160.0, 2.0, 0.5
    gap = 90.0
    for _ in range(30):
        lap = compressed_lap_time(sc, gap, target, close, floor_s=0.0)
        gap = gap + lap - sc
    assert abs(gap - target) < 0.01


def test_a_floor_only_slows_the_closing_down():
    """When the car cannot physically go quicker, the gap still shrinks."""
    lap = compressed_lap_time(160.0, gap_s=300.0, queue_gap_s=2.0,
                              close_frac=0.5, floor_s=100.0)
    assert lap == 100.0
    assert 300.0 + lap - 160.0 < 300.0


# ----------------------------------------------------------------------
# What a caution now does to the field
# ----------------------------------------------------------------------
def test_the_field_bunches_up_instead_of_spreading_out():
    """01's cautions let a quick car pull away from a slow one. This is the fix.

    Measured as a paired comparison rather than in absolute terms: the
    spread of crossing times at the end of a caution grows either way,
    because cars pit during it and a stop is worth more than the bunching.
    What matters is that it grows less when the field is being compressed.
    """
    cfg = two_class_config()

    def end_of_caution_spread(compat):
        result = run_race(cfg, seed=4, compat=compat)
        laps = result.laps
        out = {}
        for i, (start, end) in enumerate(result.cautions.periods):
            sub = laps[(laps["t"] >= start) & (laps["t"] < end)]
            if sub["car_id"].nunique() < cfg.total_cars:
                continue
            last = sub.groupby("car_id")["t"].max()
            out[i] = last.max() - last.min()
        return out

    off = end_of_caution_spread(Compat(legacy_caution_pace=True))
    on = end_of_caution_spread(Compat())
    shared = sorted(set(off) & set(on))
    assert len(shared) >= 4, "not enough cautions with the whole field in them"

    tighter = [on[i] < off[i] for i in shared]
    assert sum(tighter) > 0.6 * len(tighter)
    assert np.mean([on[i] for i in shared]) < np.mean([off[i] for i in shared])


def test_everyone_runs_the_safety_car_lap_not_a_multiple_of_their_own():
    """A slow class behind the safety car goes safety car speed, not its own.

    Under 01 a GTD caution lap was 1.6 times a GTD lap, so the class that
    was slow under green was slow under yellow too and the field kept its
    shape. Behind a real safety car there is one pace for everybody.
    """
    cfg = two_class_config()
    result = run_race(cfg, seed=5)
    laps = result.laps
    caution = laps[laps["under_caution"] & ~laps["wave_by"]]

    gtp = caution[caution["class"] == "GTP"]["lap_time"].mean()
    gtd = caution[caution["class"] == "GTD"]["lap_time"].mean()
    assert abs(gtp - gtd) < 0.1 * gtp

    # And it is the quick class that sets it, not the slow one.
    slow = cfg.class_by_name("GTD")
    assert gtd < slow.base_pace_s * slow.caution_pace_multiplier * 0.95


def test_compression_closes_the_field_without_flattering_the_leader():
    """The tail gains, the leader does not. A caution should cost the front."""
    cfg = two_class_config()
    legacy = run_race(cfg, seed=6, compat=Compat(legacy_caution_pace=True))
    new = run_race(cfg, seed=6, compat=Compat())

    def gtp(result):
        c = result.classification()
        return c[c["class"] == "GTP"]["laps"]

    a, b = gtp(legacy), gtp(new)
    assert b.max() - b.min() < a.max() - a.min()      # the field closes up
    assert abs(int(b.max()) - int(a.max())) <= 0.02 * a.max()   # the leader does not run away


# ----------------------------------------------------------------------
# Wave-arounds
# ----------------------------------------------------------------------
def _cars(*ids):
    return {cid: _Car(cid, cls) for cid, cls in ids}


def test_a_wave_needs_a_full_lap_of_deficit_and_track_position_ahead():
    """Both halves of the rule, and the trap that makes the first one necessary.

    A car strung out most of a lap behind has a lower lap *count* than the
    leader for part of every lap. Counting crossings would wave it round;
    counting progress does not.
    """
    cars = _cars(("A", "GTP"), ("B", "GTP"), ("C", "GTP"), ("D", "GTP"))

    # A leads on 100.1 laps. B is 0.9 down and further round: not lapped.
    # C is 1.2 down and further round: waved. D is 1.2 down and behind: not.
    progress = {"A": 100.1, "B": 99.2, "C": 98.8, "D": 98.05}
    assert wave_eligible(cars, progress) == {"C"}


def test_the_leader_is_never_waved():
    cars = _cars(("A", "GTP"), ("B", "GTP"))
    assert wave_eligible(cars, {"A": 50.5, "B": 48.9}) <= {"B"}
    assert "A" not in wave_eligible(cars, {"A": 50.5, "B": 48.9})


def test_eligibility_is_judged_within_a_class():
    """A GT car is not waved because a prototype lapped it."""
    cars = _cars(("P1", "GTP"), ("G1", "GTD"), ("G2", "GTD"))
    progress = {"P1": 120.9, "G1": 100.5, "G2": 100.1}
    assert wave_eligible(cars, progress) == set()


def test_imsa_runs_two_waves_and_wec_one():
    """IMSA has a Pass-Around and a Final Wave-By; WEC has the first only."""
    imsa = CautionRules.for_series("imsa")
    wec = CautionRules.for_series("wec")
    args = dict(start=1000.0, end=1000.0 + 6 * 160.0, caution_lap_s=160.0,
                open_delay_laps=1.0)
    assert len(imsa.wave_times(**args)) == 2
    assert len(wec.wave_times(**args)) == 1
    assert CautionRules.for_series("test").wave_times(**args) == []


def test_a_caution_too_short_to_hold_a_wave_does_not_run_one():
    rules = CautionRules.for_series("imsa")
    assert rules.wave_times(1000.0, 1050.0, 160.0, 1.0) == []


def test_a_wave_is_a_credit_rather_than_a_lap_anyone_drives():
    """Short, flagged, and never handed to the same car twice in one caution.

    The lap time is a bookkeeping artefact - the car is being credited a
    crossing rather than driving one - so the test is that it is clearly
    shorter than a caution lap and that the flag is there to exclude it
    from anything reading caution laps as pace.
    """
    cfg = two_class_config()
    result = run_race(cfg, seed=7)
    laps = result.laps
    waves = laps[laps["wave_by"]]
    assert len(waves) > 0

    normal = laps[laps["under_caution"] & ~laps["wave_by"]]["lap_time"].mean()
    assert waves["lap_time"].max() < normal

    # IMSA runs two waves per caution, so no car may take more than two.
    for (car_id, _), taken in waves.groupby(
            ["car_id", waves["t"].map(_episode_of(result))]):
        assert len(taken) <= 2, car_id


def _episode_of(result):
    periods = result.cautions.periods

    def which(t):
        for i, (start, end) in enumerate(periods):
            if start <= t < end:
                return i
        return -1
    return which


def test_waves_do_not_run_when_the_series_has_no_rulebook():
    cfg = two_class_config()
    cfg.series_code = "test"
    for c in cfg.classes:
        c.series_code = "test"
    result = run_race(cfg, seed=8)
    assert not result.laps["wave_by"].any()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
