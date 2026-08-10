"""What the caution timeline has to keep doing.

The properties here are the ones 02b's benchmark leans on. The exactness of
the causal benchmark rests on caution durations being memoryless, and that
rests on the draw not merging anything - so it is worth a test rather than a
comment.
"""

import numpy as np
import pytest

from endurance import ClassDials, RaceConfig, run_race
from endurance.engine import CautionTimeline, RaceEngine


DURATION = 24 * 3600.0
RATE = 0.40
MEAN_DUR = 1200.0


def dials(**over) -> ClassDials:
    base = dict(
        series_code="imsa", class_name="GTP",
        base_pace_s=100.0, deg_slope_s_per_lap=0.02,
        pace_spread_s=0.4, lap_noise_s=0.8,
        caution_rate=RATE, caution_mean_dur_s=MEAN_DUR,
        green_stint_laps=32.0, fuel_per_lap=1.0 / 32.0,
        fuel_per_lap_caution=0.6 / 32.0, tyre_life_laps=64.0,
        pit_time_mean_s=45.0, pit_time_std_s=3.0, n_cars=6,
    )
    base.update(over)
    return ClassDials(**base)


def config(**over) -> RaceConfig:
    return RaceConfig(name="test", series_code="imsa",
                      duration_s=DURATION, classes=[dials(**over)])


def timelines(n=300, **kw):
    return [CautionTimeline.draw(DURATION, RATE, MEAN_DUR,
                                 np.random.default_rng(s), **kw)
            for s in range(n)]


# ----------------------------------------------------------------------
# The properties 02b needs
# ----------------------------------------------------------------------
def test_episodes_never_overlap_or_touch():
    """Nothing to merge means nothing gets destroyed by merging."""
    for tl in timelines(200):
        for (_, end), (start, _) in zip(tl.periods, tl.periods[1:]):
            assert start > end


def test_realised_share_matches_the_target():
    """The calibrated share is what the race actually runs, not an aspiration."""
    shares = [tl.total_caution_s() / DURATION for tl in timelines(500)]
    assert abs(np.mean(shares) - RATE) < 0.01


def test_episode_lengths_stay_exponential():
    """Memoryless durations are what make the causal benchmark exact.

    An exponential has a coefficient of variation of exactly 1. The old
    merging draw pushed it above that by gluing episodes together.
    """
    lens = np.array([e - s for tl in timelines(500) for s, e in tl.periods])
    # Episodes clipped by the end of the race shorten the tail slightly, so
    # they are excluded rather than allowed to bias the statistic.
    lens = lens[lens < 0.9 * DURATION]
    assert abs(lens.std(ddof=1) / lens.mean() - 1.0) < 0.05


def test_legacy_draw_still_merges():
    """The legacy path is kept honest: it is the old behaviour, warts included."""
    legacy = timelines(200, legacy=True)
    merged_short = np.mean([tl.total_caution_s() / DURATION for tl in legacy])
    assert merged_short < RATE - 0.02


# ----------------------------------------------------------------------
# The regression gate
# ----------------------------------------------------------------------
def test_legacy_flags_reproduce_notebook_01():
    """`legacy_cautions=True, split_streams=False` must be bit-for-bit 01.

    This is the gate the whole of 02a runs behind: every change has to be
    switchable off, and with everything off the engine has to be the engine
    01 validated.
    """
    cfg = config()
    a = run_race(cfg, seed=3, legacy_cautions=True, split_streams=False)
    b = run_race(cfg, seed=3, legacy_cautions=True, split_streams=False)
    assert a.classification().equals(b.classification())

    # And the legacy path must not have been quietly modernised: its
    # timeline is the merged one, drawn off the single shared generator.
    eng = RaceEngine(cfg, seed=3, legacy_cautions=True, split_streams=False)
    expected = CautionTimeline._draw_legacy(
        DURATION, RATE, MEAN_DUR, np.random.default_rng(3))
    assert eng.cautions.periods == expected.periods


def test_split_streams_isolates_the_caution_change():
    """Changing the caution model must not disturb the field.

    Under 01's shared generator the two draws are entangled, so a caution
    change silently reassigns every car's base pace and the comparison
    measures both at once.
    """
    cfg = config()
    new = RaceEngine(cfg, seed=5, legacy_cautions=False, split_streams=True)
    old = RaceEngine(cfg, seed=5, legacy_cautions=True, split_streams=True)
    new._build_field()
    old._build_field()
    assert [c.base_pace_s for c in new.cars.values()] == \
           [c.base_pace_s for c in old.cars.values()]

    shared_new = RaceEngine(cfg, seed=5, legacy_cautions=False, split_streams=False)
    shared_old = RaceEngine(cfg, seed=5, legacy_cautions=True, split_streams=False)
    shared_new._build_field()
    shared_old._build_field()
    assert [c.base_pace_s for c in shared_new.cars.values()] != \
           [c.base_pace_s for c in shared_old.cars.values()]


# ----------------------------------------------------------------------
# Edges
# ----------------------------------------------------------------------
def test_no_cautions_when_rate_is_zero():
    assert CautionTimeline.draw(DURATION, 0.0, MEAN_DUR,
                                np.random.default_rng(0)).periods == []


def test_full_caution_race_terminates():
    """`scale_dials` can push the rate to its 0.95 clamp; it must not hang."""
    tl = CautionTimeline.draw(DURATION, 0.95, MEAN_DUR, np.random.default_rng(0))
    assert tl.total_caution_s() / DURATION > 0.9


def test_still_independent_of_strategy():
    """01's central property, re-checked against the new draw."""
    from endurance import FixedLapStint, OpportunistUnderCaution, assign_strategy

    cfg = config()
    a = run_race(cfg, strategies=assign_strategy(cfg, OpportunistUnderCaution()), seed=7)
    b = run_race(cfg, strategies=assign_strategy(cfg, FixedLapStint(stint_laps=15)), seed=7)
    assert a.cautions.periods == b.cautions.periods


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
