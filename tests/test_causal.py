"""The one property that makes the causal reference causal.

Everything else about it is a modelling choice that can be argued over. This
cannot: if a decision taken at minute ten changes when a caution at minute
fifty is deleted, the policy is reading the future and the gap between it and
the clairvoyant reference measures nothing.

Tested by deletion rather than by inspection. Reading the code and concluding
that no future value is consulted is exactly the reasoning that let the
wave-eligibility defect stand.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from endurance import ClassDials, RaceConfig  # noqa: E402
from endurance.benchmark import FocalContext, anchor, forced_only_plan  # noqa: E402
from endurance.causal import causal_plan, hazards, solve_policy  # noqa: E402


FOCAL = "GTP-03"


def cfg1h() -> RaceConfig:
    cls = ClassDials(
        series_code="imsa", class_name="GTP", base_pace_s=100.0,
        deg_slope_s_per_lap=0.012, pace_spread_s=0.5, lap_noise_s=0.35,
        caution_rate=0.30, caution_mean_dur_s=400.0, green_stint_laps=15.0,
        fuel_per_lap=1 / 15, fuel_per_lap_caution=0.6 / 15, tyre_life_laps=30.0,
        pit_time_mean_s=45.0, pit_time_std_s=3.0, n_cars=6, traffic_penalty_s=0.8)
    return RaceConfig(name="imsa 1h", series_code="imsa", duration_s=3600.0,
                      classes=[cls])


def _context(seed=3):
    ctx = FocalContext(cfg1h(), seed, FOCAL)
    anchor(ctx, plan=forced_only_plan(ctx))
    return ctx


def test_deleting_a_later_caution_changes_no_earlier_decision():
    """Cut the timeline in half; the first half's stops must be identical."""
    ctx = _context()
    policy = solve_policy(ctx, bucket_s=60.0)
    full_trace: list = []
    full = causal_plan(ctx, policy, trace=full_trace)

    periods = ctx.cautions.periods
    assert len(periods) >= 2, "need a timeline with something to delete"
    cut = periods[len(periods) // 2][0]

    trimmed = _context()
    trimmed.cautions.periods = [(s, e) for s, e in periods if e <= cut]
    partial_trace: list = []
    partial = causal_plan(trimmed, policy, trace=partial_trace)

    assert _before(full, full_trace, cut) == _before(partial, partial_trace, cut)


def _before(plan, trace, cut: float):
    """The stops taken before `cut`, by the clock rather than by lap number."""
    laps_before = {lap for lap, t in trace if t < cut}
    return [s for s in plan.stops if s.after_lap in laps_before]


def test_the_hazards_come_out_of_the_calibrated_dials():
    """Rate and mean duration are measured; the two hazards are arithmetic.

    Nothing new is assumed here, which matters: if the causal reference
    needed a dial of its own it would belong in `ASSUMED_FIELDS`, and it
    does not.
    """
    cls = cfg1h().classes[0]
    lam_g, lam_c = hazards(cls)
    mean_dur = 1.0 / lam_c
    mean_gap = 1.0 / lam_g
    assert abs(mean_dur - cls.caution_mean_dur_s) < 1e-9
    # The share of time under caution the two rates imply is the dial itself.
    assert abs(mean_dur / (mean_dur + mean_gap) - cls.caution_rate) < 1e-9


def test_the_policy_depends_on_the_seed_only_through_three_numbers():
    """Base pace and the two anchored levels, and nothing else from the draw.

    Two different seeds give different policies, because the focal car's
    drawn pace and its measured caution lap differ. Force those three to
    agree and the policies must be identical - if they are not, the solver
    is reading the realised timeline and the causal reference is not causal.
    """
    import numpy as np

    a, b = _context(3), _context(9)
    assert (a.base_pace_s, a.caution_lap_s) != (b.base_pace_s, b.caution_lap_s)

    b.base_pace_s = a.base_pace_s
    b.floor_s = a.floor_s
    b.caution_lap_s = a.caution_lap_s
    b.green_offset_s = a.green_offset_s
    assert np.array_equal(solve_policy(a, bucket_s=60.0).action,
                          solve_policy(b, bucket_s=60.0).action)


def test_the_value_function_knows_that_fuel_is_worth_something():
    """A full tank must be worth more than two laps of fuel.

    It was not. With a fuel grid coarser than a lap's burn, rounding to the
    nearest bucket returned the car to the bucket it started in, so fuel
    never fell and the value function came out flat - identical to three
    decimals from two laps of fuel to a full tank. The policy then had no
    reason to prefer a big fill and took the cheapest stop available, nine
    times, every seven laps.

    Nothing about that looked like a bug from the outside: the solver ran,
    the value function was smooth in time, and the plans were legal.
    """
    import numpy as np

    ctx = _context()
    policy = solve_policy(ctx, bucket_s=60.0)
    mid = policy.value.shape[0] // 3
    by_fuel = policy.value[mid, 0, :, 0]

    assert np.all(np.diff(by_fuel) >= -1e-9), "value is not monotone in fuel"
    assert by_fuel[-1] > by_fuel[1] + 1e-3, "a full tank is worth no more than two laps"


def test_a_fuel_grid_coarser_than_the_burn_is_refused():
    """The failure above cannot be reintroduced quietly."""
    ctx = _context()
    with pytest.raises(ValueError):
        solve_policy(ctx, bucket_s=60.0,
                     fuel_step=ctx.cls.fuel_per_lap_caution * 2)


def test_the_causal_reference_is_never_worse_than_stopping_when_forced():
    """A reference beaten by the trivial plan is not a reference.

    Checked on laps rather than position, because position depends on the
    rest of the field and this is a statement about the plan.
    """
    for seed in (3, 4, 5, 6):
        ctx = _context(seed)
        policy = solve_policy(ctx, bucket_s=60.0)
        causal = causal_plan(ctx, policy)
        reference = forced_only_plan(ctx)
        assert causal.laps >= reference.laps, seed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
