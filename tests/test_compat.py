"""The 02a switchboard, and the property it exists to protect.

Two things are being guarded here. First, that every 02a change can be
switched off, and that with all of them off the engine is bit-for-bit the
engine notebook 01 validated - otherwise "here is what the change did" is
not a claim anyone can check. Second, that noise is a function of the seed
and never of the strategy, which is what makes a paired comparison paired.

The second one is easy to believe and hard to keep: the defect it replaces
looked fine for the whole of 01.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from endurance import (  # noqa: E402
    ClassDials,
    Compat,
    PitRules,
    FixedLapStint,
    RaceConfig,
    RunToFuelWindow,
    run_race,
    stop_cost,
)
from endurance.engine import (  # noqa: E402
    CarState,
    NoiseStream,
    RaceEngine,
    RaceState,
    _STREAM_LAP_NOISE,
)


# ----------------------------------------------------------------------
# The regression gate
# ----------------------------------------------------------------------
# Captured from the engine as it stood before 02a, by tests/capture_golden.py.
# These are not values anybody chose - they are what the old engine did, and
# the only way a comparison against 01 can mean anything is by holding them.
# If numpy ever changes its Generator stream these will move together and
# harmlessly; re-capture with the pre-02a engine rather than editing by hand.
GOLDEN_01 = {
    0: (('FAST-06', 36, 1, 3614.694), ('FAST-05', 36, 1, 3632.337),
        ('FAST-01', 36, 1, 3636.015), ('FAST-02', 36, 1, 3652.206),
        ('FAST-04', 36, 1, 3674.09), ('FAST-03', 36, 1, 3690.321),
        ('SLOW-03', 32, 1, 3654.318), ('SLOW-05', 32, 1, 3687.862),
        ('SLOW-01', 31, 1, 3609.238), ('SLOW-02', 31, 1, 3619.463),
        ('SLOW-04', 31, 1, 3621.854)),
    3: (('FAST-01', 37, 1, 3674.554), ('FAST-06', 37, 1, 3688.111),
        ('FAST-03', 36, 1, 3632.901), ('FAST-05', 36, 1, 3646.493),
        ('FAST-02', 36, 1, 3655.316), ('FAST-04', 36, 1, 3656.468),
        ('SLOW-02', 32, 1, 3702.792), ('SLOW-01', 31, 1, 3607.669),
        ('SLOW-05', 31, 1, 3614.391), ('SLOW-04', 31, 1, 3623.449),
        ('SLOW-03', 31, 1, 3716.417)),
    7: (('FAST-05', 32, 1, 3660.939), ('FAST-03', 32, 1, 3670.963),
        ('FAST-01', 32, 1, 3672.843), ('FAST-02', 32, 1, 3676.395),
        ('FAST-06', 32, 1, 3685.325), ('FAST-04', 31, 1, 3604.898),
        ('SLOW-05', 28, 1, 3691.532), ('SLOW-03', 28, 1, 3692.717),
        ('SLOW-04', 28, 1, 3706.523), ('SLOW-01', 28, 1, 3709.312),
        ('SLOW-02', 27, 1, 3620.097)),
}


def make_config(duration_s=3600.0, caution_rate=0.1, deg=True, traffic=True,
                lap_noise_s=0.3):
    """The same two-class race test_engine uses, so the golden is comparable."""
    fast = ClassDials(
        series_code="test", class_name="FAST",
        base_pace_s=100.0, deg_slope_s_per_lap=0.02 if deg else 0.0,
        pace_spread_s=0.8, lap_noise_s=lap_noise_s,
        caution_rate=caution_rate, caution_mean_dur_s=300.0,
        green_stint_laps=20.0, fuel_per_lap=1 / 20, fuel_per_lap_caution=0.6 / 20,
        tyre_life_laps=40.0, pit_time_mean_s=40.0, pit_time_std_s=2.0,
        n_cars=6, traffic_penalty_s=0.8 if traffic else 0.0,
    )
    slow = ClassDials(
        series_code="test", class_name="SLOW",
        base_pace_s=115.0, deg_slope_s_per_lap=0.03 if deg else 0.0,
        pace_spread_s=1.0, lap_noise_s=lap_noise_s + 0.1,
        caution_rate=caution_rate, caution_mean_dur_s=300.0,
        green_stint_laps=22.0, fuel_per_lap=1 / 22, fuel_per_lap_caution=0.6 / 22,
        tyre_life_laps=44.0, pit_time_mean_s=42.0, pit_time_std_s=2.0,
        n_cars=5, traffic_penalty_s=0.8 if traffic else 0.0,
    )
    return RaceConfig(name="Test 1h", series_code="test",
                      duration_s=duration_s, classes=[fast, slow])


def fingerprint(result) -> tuple:
    c = result.classification()
    return tuple((row.car_id, int(row.laps), int(row.stops),
                  round(float(row.race_time_s), 3)) for row in c.itertuples())


def test_v01_reproduces_the_pre_02a_engine():
    """The gate the whole of 02a runs behind.

    Not "the new engine agrees with itself" - these numbers were taken off
    the old engine before a line of 02a was written.
    """
    for seed, expected in GOLDEN_01.items():
        assert fingerprint(run_race(make_config(), seed=seed,
                                    compat=Compat.v01())) == expected, seed


def test_every_02a_change_can_be_switched_off_independently():
    """All four have landed, so all four flags are live in both directions.

    The guard that made a flag refuse to be switched off before its change
    existed is kept but empty: a switch that quietly does nothing is worse
    than no switch, and the next stage to add one will want it back.
    """
    assert Compat._NOT_YET == {}
    for flag in ("legacy_cautions", "legacy_noise", "legacy_pit",
                 "legacy_caution_pace", "legacy_traffic"):
        assert getattr(Compat(), flag) is False
        assert getattr(Compat(**{flag: True}), flag) is True
        assert getattr(Compat.v01(), flag) is True


def test_compat_and_the_old_keywords_are_not_mixed():
    with pytest.raises(TypeError):
        RaceEngine(make_config(), seed=0, split_streams=False,
                   compat=Compat.v01())


def test_old_keywords_still_mean_what_they_meant():
    eng = RaceEngine(make_config(), seed=0, legacy_cautions=True,
                     split_streams=False)
    assert eng.legacy_cautions and not eng.split_streams
    assert not eng.compat.legacy_noise      # only the two named flags move


# ----------------------------------------------------------------------
# Noise is a function of the seed, not the strategy
# ----------------------------------------------------------------------
def _green_lap_times(result, car_id):
    """Lap time by lap number, on a config where lap time is pace + noise."""
    sub = result.laps[result.laps["car_id"] == car_id]
    return dict(zip(sub["lap"], sub["lap_time"]))


def test_lap_noise_is_identical_under_two_strategies():
    """02a's headline property, end to end rather than by introspection.

    Degradation, traffic and cautions are switched off, so a green lap time
    is exactly base pace plus that lap's noise. Two strategies that pit at
    completely different moments must therefore produce the same lap time
    at the same lap number, for every car, or the noise moved when the
    strategy did.
    """
    cfg = make_config(caution_rate=0.0, deg=False, traffic=False)
    a = run_race(cfg, default_strategy=RunToFuelWindow(), seed=11)
    b = run_race(cfg, default_strategy=FixedLapStint(stint_laps=7), seed=11)

    checked = 0
    for car_id in a.laps["car_id"].unique():
        la, lb = _green_lap_times(a, car_id), _green_lap_times(b, car_id)
        for lap in set(la) & set(lb):
            assert abs(la[lap] - lb[lap]) < 1e-9, (car_id, lap)
            checked += 1
    assert checked > 100, "the comparison has to actually overlap somewhere"


def test_the_old_shared_generator_did_not_have_that_property():
    """The defect, demonstrated rather than asserted.

    Kept because 01 passed every test it had while this was true, and the
    only thing that makes the fix legible is being able to show the thing
    it fixed.
    """
    cfg = make_config(caution_rate=0.0, deg=False, traffic=False)
    legacy = Compat.v01()
    a = run_race(cfg, default_strategy=RunToFuelWindow(), seed=11, compat=legacy)
    b = run_race(cfg, default_strategy=FixedLapStint(stint_laps=7), seed=11,
                 compat=legacy)

    car_id = a.laps["car_id"].iloc[0]
    la, lb = _green_lap_times(a, car_id), _green_lap_times(b, car_id)
    shared = sorted(set(la) & set(lb))
    assert any(abs(la[lap] - lb[lap]) > 1e-9 for lap in shared)


def test_pit_cost_is_keyed_to_the_stop_index():
    """A car's third stop gets its third noise draw, whenever it takes it.

    The cost itself is no longer expected to match: with the pit layer in,
    a stop is priced on how much fuel it actually took, and two strategies
    stop with different amounts left. What must match is the noise on top,
    so the residual is what gets compared.
    """
    cfg = make_config(caution_rate=0.0, deg=False, traffic=False)
    a = run_race(cfg, default_strategy=RunToFuelWindow(), seed=13)
    b = run_race(cfg, default_strategy=FixedLapStint(stint_laps=7), seed=13)

    def residuals(result, car_id):
        cls = cfg.class_by_name(result.laps[result.laps["car_id"] == car_id]["class"].iloc[0])
        rules = PitRules.for_series(cfg.series_code)
        stops = result.laps[(result.laps["car_id"] == car_id) & result.laps["pitted"]]
        return [row.pit_cost_s - stop_cost(cls, rules, 1.0 - row.fuel, True)
                for row in stops.itertuples()]

    checked = 0
    for car_id in a.laps["car_id"].unique():
        ra, rb = residuals(a, car_id), residuals(b, car_id)
        for i in range(min(len(ra), len(rb))):
            assert abs(ra[i] - rb[i]) < 1e-9, (car_id, i)
        checked += min(len(ra), len(rb))
    assert checked > 10


def test_cars_do_not_share_a_stream():
    a = NoiseStream(5, _STREAM_LAP_NOISE, 0, 32)
    b = NoiseStream(5, _STREAM_LAP_NOISE, 1, 32)
    assert [a[i] for i in range(32)] != [b[i] for i in range(32)]


def test_a_grown_stream_holds_what_a_long_one_would_have():
    """The sizing hint must not be able to change the numbers.

    If growing a stream drew different values, every result would depend on
    a guess about how many laps a car might complete.
    """
    short = NoiseStream(9, _STREAM_LAP_NOISE, 2, size=4, block=4)
    long = NoiseStream(9, _STREAM_LAP_NOISE, 2, size=200, block=4)
    assert [short[i] for i in range(200)] == [long[i] for i in range(200)]


def test_noise_size_is_independent_of_the_dial_that_scales_it():
    """Standard normals scaled at use, so a noise sweep is paired too.

    With no noise at all a green lap is exactly the car's base pace, which
    gives the baseline the other two runs are measured from. Quadrupling
    `lap_noise_s` must then quadruple every lap's deviation rather than
    redraw it.
    """
    def run(noise):
        return run_race(make_config(caution_rate=0.0, deg=False, traffic=False,
                                    lap_noise_s=noise), seed=4)

    zero, quiet, loud = run(0.0), run(0.1), run(0.4)
    car = quiet.laps["car_id"].iloc[0]
    base = _green_lap_times(zero, car)[1]
    q, l = _green_lap_times(quiet, car), _green_lap_times(loud, car)

    laps = sorted(set(q) & set(l))[:20]
    assert len(laps) >= 10
    for lap in laps:
        assert abs((l[lap] - base) - 4.0 * (q[lap] - base)) < 1e-9, lap


# ----------------------------------------------------------------------
# Lap-down status, in one place
# ----------------------------------------------------------------------
def _state(cars, cfg):
    return RaceState(t=100.0, duration_s=3600.0, under_caution=False,
                     cars={c.car_id: c for c in cars}, config=cfg)


def test_class_leader_is_derived_the_same_way_position_is():
    cfg = make_config()
    a = CarState(car_id="FAST-01", class_name="FAST", base_pace_s=100.0,
                 laps_done=10, race_time_s=1000.0)
    b = CarState(car_id="FAST-02", class_name="FAST", base_pace_s=100.0,
                 laps_done=10, race_time_s=990.0)     # same lap, got there sooner
    c = CarState(car_id="SLOW-01", class_name="SLOW", base_pace_s=115.0,
                 laps_done=12, race_time_s=1010.0)    # more laps, other class
    st = _state([a, b, c], cfg)

    assert st.class_leader("FAST").car_id == "FAST-02"
    assert st.class_leader("SLOW").car_id == "SLOW-01"
    assert st.class_leader("NOBODY") is None


def test_laps_down_counts_against_the_class_leader_and_never_goes_negative():
    cfg = make_config()
    leader = CarState(car_id="FAST-01", class_name="FAST", base_pace_s=100.0,
                      laps_done=12, race_time_s=1000.0)
    down = CarState(car_id="FAST-02", class_name="FAST", base_pace_s=100.0,
                    laps_done=10, race_time_s=1005.0)
    st = _state([leader, down], cfg)

    assert st.laps_down(down) == 2
    assert st.laps_down(leader) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
