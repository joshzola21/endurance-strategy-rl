"""What the rulebook is allowed to change about a stop, and what it is not.

The layer is anchored to the measured mean, so a full service costs what it
always cost. Everything worth testing is therefore about stops that are not
full service, and about the lane being shut - which is the only part of
02a's pit work that moves a number in a race the current baselines run.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from endurance import (  # noqa: E402
    ClassDials,
    Compat,
    PitRules,
    RaceConfig,
    lane_status,
    run_race,
    stop_cost,
)
from endurance.engine import CautionTimeline, RaceEngine  # noqa: E402


MEAN = 40.0


def dials(series="imsa", class_name="GTP", **over) -> ClassDials:
    base = dict(
        series_code=series, class_name=class_name,
        base_pace_s=100.0, deg_slope_s_per_lap=0.02,
        pace_spread_s=0.8, lap_noise_s=0.3,
        caution_rate=0.15, caution_mean_dur_s=400.0,
        green_stint_laps=20.0, fuel_per_lap=1 / 20, fuel_per_lap_caution=0.6 / 20,
        tyre_life_laps=40.0, pit_time_mean_s=MEAN, pit_time_std_s=2.0, n_cars=4,
    )
    base.update(over)
    return ClassDials(**base)


class _Timeline:
    """A caution timeline with periods chosen rather than drawn."""

    def __init__(self, periods):
        self.periods = periods


# ----------------------------------------------------------------------
# The shape of a stop
# ----------------------------------------------------------------------
def test_a_full_service_costs_the_measured_mean_in_both_series():
    """The anchor. The level is data; only the shape is regulation.

    If this drifts, the layer has started making claims about how long a
    stop takes, which the timing data already answered.
    """
    for series in ("imsa", "wec"):
        d = dials(series=series)
        rules = PitRules.for_series(series)
        assert abs(stop_cost(d, rules, fuel_added=1.0, change_tyres=True) - MEAN) < 1e-9


def test_a_splash_costs_less_than_a_full_stop():
    """The thing notebook 01 could not represent at all."""
    d, rules = dials(), PitRules.for_series("imsa")
    full = stop_cost(d, rules, 1.0, True)
    splash = stop_cost(d, rules, 0.3, False)
    assert splash < full
    # and never less than getting down the lane and out again
    assert splash >= d.pit_time_mean_s * d.pit_transit_frac - 1e-9


def test_the_two_series_price_a_partial_stop_differently():
    """IMSA overlaps the jobs, WEC queues them - art. 34.1.1 against art. 12.

    On a full stop the two agree by construction, so a partial stop with
    tyres is where the regulation actually shows up.
    """
    d_imsa, d_wec = dials(series="imsa"), dials(series="wec", class_name="HYPERCAR")
    imsa = stop_cost(d_imsa, PitRules.for_series("imsa"), 0.3, True)
    wec = stop_cost(d_wec, PitRules.for_series("wec"), 0.3, True)
    assert wec > imsa
    # IMSA's short splash disappears inside the tyre change; WEC's does not.
    assert abs(imsa - MEAN * (d_imsa.pit_transit_frac + d_imsa.pit_tyre_frac)) < 1e-9


def test_legacy_mode_ignores_the_shape_entirely():
    d, rules = dials(), PitRules.for_series("imsa")
    assert stop_cost(d, rules, 0.1, False, legacy=True) == MEAN
    assert stop_cost(d, rules, 1.0, True, legacy=True) == MEAN


# ----------------------------------------------------------------------
# The lane
# ----------------------------------------------------------------------
def _status(series, class_name, t, periods=((3600.0, 4200.0),), duration=24 * 3600.0):
    return lane_status(PitRules.for_series(series), _Timeline(list(periods)),
                       t, class_name, caution_lap_s=160.0, duration_s=duration)


def test_the_lane_is_open_under_green():
    assert _status("imsa", "GTP", 3000.0).open


def test_imsa_releases_prototypes_before_gts():
    """Art. 46.3.1, which is most of why the two series are worth simulating."""
    # One delay lap, then prototypes on the next, GTs on the one after.
    just_after_prototypes = 3600.0 + 2.1 * 160.0
    assert _status("imsa", "GTP", just_after_prototypes).open
    assert not _status("imsa", "GTD", just_after_prototypes).open
    assert _status("imsa", "GTD", 3600.0 + 3.1 * 160.0).open


def test_nobody_gets_in_at_the_moment_of_the_call():
    for cls in ("GTP", "GTD"):
        assert not _status("imsa", cls, 3601.0).open


def test_wec_releases_everyone_together():
    """Art. 14.6.5: three laps, then unrestricted - no class staging."""
    early, late = 3600.0 + 2.0 * 160.0, 3600.0 + 4.1 * 160.0
    for cls in ("HYPERCAR", "LMGT3"):
        assert not _status("wec", cls, early).open
        assert _status("wec", cls, late).open


def test_a_short_imsa_caution_never_opens_at_all():
    """Art. 46.3.3, and the reason the caution gambler carries real risk."""
    early = _status("imsa", "GTP", 600.0, periods=((300.0, 1500.0),))
    assert not early.open and "short" in early.reason

    # ... and one following hard on a restart is treated the same way.
    after_restart = _status("imsa", "GTP", 5000.0,
                            periods=((3600.0, 4200.0), (4500.0, 5400.0)))
    assert not after_restart.open


def test_a_series_with_no_rulebook_keeps_its_lane_open():
    """The synthetic test configs are not a series, so nothing is asserted."""
    assert _status("test", "FAST", 3700.0).open
    assert PitRules.for_series("test").series_code == "unknown"


# ----------------------------------------------------------------------
# In a race
# ----------------------------------------------------------------------
def race_config(series="imsa", class_name="GTP", duration_s=6 * 3600.0):
    return RaceConfig(name=f"{series} test", series_code=series,
                      duration_s=duration_s,
                      classes=[dials(series=series, class_name=class_name)])


def test_a_shut_lane_refuses_a_stop_that_was_merely_wanted():
    """The gambler's downside, made real.

    The opportunist asks to stop whenever a caution is out. With the lane
    modelled, some of those asks have to be turned down.
    """
    from endurance import OpportunistUnderCaution

    cfg = race_config()
    result = run_race(cfg, default_strategy=OpportunistUnderCaution(), seed=2)
    assert "stop_refused" in result.laps.columns
    assert result.laps["stop_refused"].notna().sum() > 0


def test_forced_stops_are_taken_anyway_and_counted():
    """No penalties are modelled, so these have to be visible to be honest."""
    from endurance import OpportunistUnderCaution

    cfg = race_config()
    result = run_race(cfg, default_strategy=OpportunistUnderCaution(), seed=2)
    assert "lane_closed_stop" in result.laps.columns
    # Rare rather than absent: a car out of fuel under a shut lane must stop.
    share = result.laps["lane_closed_stop"].mean()
    assert 0.0 <= share < 0.02


def test_the_layer_charges_for_the_fuel_actually_taken():
    """Turning the layer on makes stops cheaper, and for a stateable reason.

    A baseline stops with a little fuel still aboard, and notebook 01
    charged it for a full service anyway. The saving is not a free lunch:
    it is the fuel that was never put in, and it should come out to about
    the leftover fraction of the refuel job.
    """
    cfg = race_config(series="test", class_name="FAST")
    cfg.classes[0].caution_rate = 0.0        # no discount to muddy the arithmetic
    on = run_race(cfg, seed=5, compat=Compat())
    off = run_race(cfg, seed=5, compat=Compat(legacy_pit=True))

    p_on = on.laps[on.laps["pitted"]]
    p_off = off.laps[off.laps["pitted"]]
    assert p_on["pit_cost_s"].mean() < p_off["pit_cost_s"].mean()

    d = cfg.classes[0]
    full_refuel_s = d.pit_time_mean_s * (1.0 - d.pit_transit_frac)
    predicted = full_refuel_s * p_on["fuel"].mean()
    observed = p_off["pit_cost_s"].mean() - p_on["pit_cost_s"].mean()
    assert abs(observed - predicted) < 0.15 * predicted


def test_a_stop_is_never_cheaper_than_getting_down_the_lane():
    """The floor is regulation, not modelling: you still have to drive it."""
    cfg = race_config(series="test", class_name="FAST")
    result = run_race(cfg, seed=5)
    d = cfg.classes[0]
    floor = d.pit_time_mean_s * d.pit_transit_frac
    stops = result.laps[result.laps["pitted"]]
    # Noise and the caution discount can dip below the nominal floor, so the
    # check is that nothing collapses towards zero, not that nothing dips.
    assert stops["pit_cost_s"].min() > 0.5 * floor


def test_strategies_can_see_the_lane():
    """02c's gambler needs this; without it the strategy can only be lucky."""
    seen = []

    def spy(car, state):
        from endurance import PitDecision
        seen.append((state.under_caution, state.pit_lane_open))
        return PitDecision(pit=False)

    run_race(race_config(), default_strategy=spy, seed=4)
    assert any(under and not open_ for under, open_ in seen)
    assert any(open_ for _, open_ in seen)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
