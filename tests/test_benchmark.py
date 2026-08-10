"""The two gates 02b is not allowed to skip.

The first is the one the 02 decision record insists on: the DP has to agree
with brute force in the no-caution limit, and agree about the *plan* rather
than only about the total. In the F1 work the same gate caught a benchmark
that passed a total-time comparison while reconstructing a suboptimal plan,
which is why the assertion below is on the stops and not just the seconds.

The second cannot be provided by the first, because the no-caution race has
no closed windows to stop through. It is checked twice over: once on the
plan the search returns, and once by replaying that plan through the real
engine and requiring that the engine never had to refuse a stop. A benchmark
whose stops get turned down at the lane is not a reference, it is a wish.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from endurance import ClassDials, RaceConfig, run_race  # noqa: E402
from endurance.benchmark import (  # noqa: E402
    FocalContext,
    brute_force,
    forced_only_plan,
    search_plans,
)


def dials(series="imsa", class_name="GTP", **over) -> ClassDials:
    base = dict(
        series_code=series, class_name=class_name,
        base_pace_s=100.0, deg_slope_s_per_lap=0.02,
        pace_spread_s=0.8, lap_noise_s=0.3,
        caution_rate=0.0, caution_mean_dur_s=400.0,
        green_stint_laps=6.0, fuel_per_lap=1 / 6, fuel_per_lap_caution=0.6 / 6,
        tyre_life_laps=12.0, pit_time_mean_s=40.0, pit_time_std_s=2.0,
        n_cars=4, traffic_penalty_s=0.0,
    )
    base.update(over)
    return ClassDials(**base)


def tiny_race(series="imsa", class_name="GTP", duration_s=1500.0, **over):
    """Short enough to enumerate exhaustively, long enough to need stops."""
    return RaceConfig(name="gate", series_code=series, duration_s=duration_s,
                      classes=[dials(series=series, class_name=class_name, **over)])


FILLS = (0.5, 1.0)
FOCAL = "GTP-02"


# ----------------------------------------------------------------------
# Gate one: the DP is the brute force
# ----------------------------------------------------------------------
def test_dp_matches_brute_force_with_no_cautions():
    for seed in range(5):
        _one_gate_one_seed(seed)


def _one_gate_one_seed(seed):
    cfg = tiny_race()
    # `per_family` is raised to k so the selection degenerates to the plain
    # time ordering. Family selection is right for feeding stage two and
    # wrong for this gate, which is asking whether the DP's arithmetic finds
    # the optimum - a question about the ordering, not about coverage.
    kw = dict(clairvoyant=True, fill_levels=FILLS, k=12, per_family=12)
    dp = search_plans(cfg, seed, FOCAL, labels_per_state=12, fuel_quantum=0.01, **kw)
    bf_kw = {k: v for k, v in kw.items() if k != "per_family"}
    bf = brute_force(cfg, seed, FOCAL, max_stops=4, **bf_kw)

    assert bf, "the enumeration found nothing to compare against"
    best_dp, best_bf = dp.best(), bf[0]

    # The value, and then the thing the value is not allowed to stand in for.
    assert best_dp.laps == best_bf.laps
    assert abs(best_dp.race_time_s - best_bf.race_time_s) < 1e-9
    optima = {p.stops for p in bf if p.sort_key == best_bf.sort_key}
    assert best_dp.stops in optima, (best_dp.stops, sorted(optima)[:3])

    # And the ordering below the winner, since stage two re-scores the top k.
    a = [(p.laps, round(p.race_time_s, 9)) for p in dp.plans]
    b = [(p.laps, round(p.race_time_s, 9)) for p in bf[:len(a)]]
    assert a == b


def test_the_gate_would_notice_a_worse_plan():
    """The gate is only worth running if it can fail.

    A plan one lap later than the optimum has to be strictly worse on the
    ranking, or the comparison above is measuring nothing.
    """
    cfg = tiny_race()
    bf = brute_force(cfg, 0, FOCAL, clairvoyant=True, fill_levels=FILLS,
                     k=200, max_stops=4)
    assert len(bf) > 5
    assert bf[0].sort_key < bf[-1].sort_key
    assert bf[0].stops != bf[-1].stops


# ----------------------------------------------------------------------
# The DP's arithmetic is the engine's arithmetic
# ----------------------------------------------------------------------
def test_dp_reproduces_the_engine_exactly_when_the_field_cannot_interfere():
    """With cautions and traffic off, the reduced race is the real one.

    This is what licenses the reduced model at all: everything stage one
    leaves out is switched off here, so any difference is arithmetic rather
    than modelling. Once cautions are on, the two part company by design and
    stage two is what closes the gap.
    """
    cfg = tiny_race(duration_s=2400.0)
    for seed in range(3):
        plan = search_plans(cfg, seed, FOCAL, clairvoyant=True,
                            fill_levels=FILLS, k=1).best()

        result = run_race(cfg, strategies={FOCAL: plan.runner()}, seed=seed)
        row = result.classification().set_index("car_id").loc[FOCAL]

        assert int(row["laps"]) == plan.laps, seed
        assert abs(float(row["race_time_s"]) - plan.race_time_s) < 1e-6, seed


# ----------------------------------------------------------------------
# Gate two: no plan stops through a shut lane
# ----------------------------------------------------------------------
def test_no_plan_stops_through_a_closed_lane():
    for series, class_name in (("imsa", "GTP"), ("imsa", "GTD"),
                               ("wec", "HYPERCAR"), ("wec", "LMGT3")):
        _closed_lane_case(series, class_name)


def _closed_lane_case(series, class_name):
    """Both series, and a staged class as well as a leading one.

    IMSA releases prototypes before GTs and WEC releases everyone together,
    so a search that had quietly reimplemented the rule instead of asking
    `pitstop.lane_status` would pass one of these and fail the other.
    """
    focal = f"{class_name}-02"
    cfg = tiny_race(series=series, class_name=class_name, duration_s=7200.0,
                    caution_rate=0.35, caution_mean_dur_s=420.0)
    ctx = FocalContext(cfg, 5, focal)
    assert ctx.cautions.periods, "no cautions drawn - the gate would be vacuous"
    assert any(not ctx.lane_open(t)
               for start, end in ctx.cautions.periods
               for t in (start + 1.0, (start + end) / 2)), \
        "the lane was never shut - the gate would be vacuous"

    report = search_plans(cfg, 5, focal, clairvoyant=True, fill_levels=FILLS, k=10)
    for plan in report.plans:
        assert plan.voluntary_closed_lane_stops == 0, plan.stops


def _refusals(cfg, seed, focal, plan, defer=True) -> int:
    result = run_race(cfg, strategies={focal: plan.runner(defer=defer)}, seed=seed)
    mine = result.laps[result.laps["car_id"] == focal]
    if "stop_refused" not in mine.columns:
        return 0
    return int(mine["stop_refused"].notna().sum())


def test_the_engine_never_has_to_refuse_a_planned_stop():
    """Gate two where it actually bites: the plan as run, not as computed.

    `engine.run` turns a stop down when the lane is shut and records
    `stop_refused`, so a legal plan leaves that column empty for the focal
    car. This holds for the constrained search and *not* for the
    unconstrained one - see the test below, which is the reason the
    constraint exists rather than a tidying-up.
    """
    cfg = tiny_race(duration_s=7200.0, caution_rate=0.35, caution_mean_dur_s=420.0)
    ctx = FocalContext(cfg, 5, FOCAL)
    cap = len(forced_only_plan(ctx).stops)

    report = search_plans(cfg, 5, FOCAL, clairvoyant=True, fill_levels=FILLS,
                          k=5, max_stops=cap, ctx=ctx)
    for plan in report.plans:
        assert _refusals(cfg, 5, FOCAL, plan) == 0, plan.stops


def test_a_plan_played_back_rigidly_walks_into_shut_lanes():
    """The defect deferral exists for, demonstrated rather than asserted.

    Stage one places its stops against the arrival times of the *reduced*
    race, and the engine's differ - compression and the real caution lap
    move them by tens of seconds, which is enough to land a stop in a window
    that opens a caution lap later or never opens at all.

    Played back rigidly the plan loses stops. Played back the way a crew
    would run it - come round again and take it at the next open lane - it
    does not. Both halves are asserted here, because the first is the reason
    the second is not merely a convenience.
    """
    cfg = tiny_race(duration_s=7200.0, caution_rate=0.35, caution_mean_dur_s=420.0)
    ctx = FocalContext(cfg, 5, FOCAL)
    plans = search_plans(cfg, 5, FOCAL, clairvoyant=True, fill_levels=FILLS,
                         k=5, ctx=ctx).plans

    rigid = [_refusals(cfg, 5, FOCAL, p, defer=False) for p in plans]
    assert sum(rigid) > 0, "nothing was refused - the gate would be vacuous"
    assert all(_refusals(cfg, 5, FOCAL, p, defer=True) == 0 for p in plans)


def test_deferring_is_execution_and_not_a_decision():
    """A deferred stop is still the plan's stop, taken at the next open lane.

    The runner must not quietly become a strategy: it may only move a stop
    later, never cancel one, invent one, or change what it asks for.
    """
    cfg = tiny_race(duration_s=7200.0, caution_rate=0.35, caution_mean_dur_s=420.0)
    plan = search_plans(cfg, 5, FOCAL, clairvoyant=True, fill_levels=FILLS,
                        k=1).best()
    runner = plan.runner(defer=True)
    result = run_race(cfg, strategies={FOCAL: runner}, seed=5)
    mine = result.laps[result.laps["car_id"] == FOCAL]

    taken = mine[mine["pitted"]]
    assert len(taken) >= len(plan.stops), "a stop went missing"
    for lap in (s.after_lap for s in plan.stops):
        assert (taken["lap"] >= lap).any(), f"the stop planned at {lap} never happened"


# ----------------------------------------------------------------------
# Housekeeping the boundary constraint depends on
# ----------------------------------------------------------------------
def test_the_plan_runner_is_not_in_the_roster():
    """The benchmark is a reference, not a strategy.

    Stated as a test because the failure mode is a later thread importing
    it for convenience, and by then it looks like it belongs there.
    """
    from endurance import strategies

    assert "benchmark" not in strategies.BASELINES
    assert not any(cls.__name__ == "PlanRunner"
                   for cls in strategies.BASELINES.values())


def test_a_stop_never_asks_for_less_fuel_than_the_car_already_has():
    """`_apply_pit` sets fuel to `refuel_to`, so asking low throws fuel away.

    The search must not be able to find that as an optimisation, because it
    is an artefact of how the engine applies a decision rather than
    something a pit crew can do.
    """
    cfg = tiny_race(duration_s=2400.0)
    report = search_plans(cfg, 1, FOCAL, clairvoyant=True, fill_levels=FILLS, k=10)
    cls = cfg.classes[0]
    for plan in report.plans:
        fuel = 1.0
        lap = 0
        for stop in plan.stops:
            fuel = max(fuel - (stop.after_lap - lap) * cls.fuel_per_lap, 0.0)
            assert stop.refuel_to >= fuel - 1e-9, (stop, fuel)
            fuel = stop.refuel_to
            lap = stop.after_lap


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
