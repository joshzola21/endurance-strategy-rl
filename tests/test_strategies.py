"""What the roster has to keep being.

Two things are guarded here, and only one of them is behaviour.

The first is the **boundary constraint**: parameter-free, meaning each
strategy derives its numbers from the dials rather than being handed them.
Tested by moving a dial and requiring the decision to move with it, which is
the machine-checkable form of the claim - reading the code and concluding
there are no magic numbers is exactly the reasoning that let the
wave-eligibility defect stand.

The second is the **at-the-line arithmetic**. Two defects have now come out
of the same place: the engine reading its own progress from a stale lap
window, and the track-position defender differencing two `race_time_s`
values across different laps. Both looked correct in review and neither
raised anything. They are tested on constructed state, where the answer is
known, rather than on a race average that would hide them.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from endurance import ClassDials, RaceConfig, run_race, scale_dials  # noqa: E402
from endurance.engine import CarState, RaceState  # noqa: E402
from endurance.strategies import (  # noqa: E402
    BASELINES,
    ROSTER,
    CautionGambler,
    LapDownDefender,
    SplashAndDashPlanner,
    TrackPositionDefender,
    _would_be_passed,
)


FOCAL = "GTP-03"


def dials(series="imsa", class_name="GTP", **over) -> ClassDials:
    base = dict(
        series_code=series, class_name=class_name, base_pace_s=100.0,
        deg_slope_s_per_lap=0.012, pace_spread_s=0.5, lap_noise_s=0.35,
        caution_rate=0.20, caution_mean_dur_s=600.0, green_stint_laps=30.0,
        fuel_per_lap=1 / 30, fuel_per_lap_caution=0.6 / 30, tyre_life_laps=60.0,
        pit_time_mean_s=45.0, pit_time_std_s=3.0, n_cars=6)
    base.update(over)
    return ClassDials(**base)


def config(series="imsa", duration_s=6 * 3600.0, **over) -> RaceConfig:
    return RaceConfig(name=f"{series} test", series_code=series,
                      duration_s=duration_s, classes=[dials(series, **over)])


def car(car_id=FOCAL, laps_done=50, race_time_s=5000.0, fuel=0.03,
        lap_start_t=4900.0, lap_expected_s=100.0, **over) -> CarState:
    c = CarState(car_id=car_id, class_name="GTP", base_pace_s=100.0,
                 laps_done=laps_done, race_time_s=race_time_s)
    c.fuel = fuel
    c.lap_start_t = lap_start_t
    c.lap_expected_s = lap_expected_s
    for k, v in over.items():
        setattr(c, k, v)
    return c


def state(cars, cfg=None, t=5000.0, under_caution=False, lane_open=True,
          duration_s=6 * 3600.0) -> RaceState:
    return RaceState(t=t, duration_s=duration_s, under_caution=under_caution,
                     cars={c.car_id: c for c in cars},
                     config=cfg or config(), pit_lane_open=lane_open)


# ----------------------------------------------------------------------
# The two mappings
# ----------------------------------------------------------------------
def test_the_roster_is_parameter_free_in_the_shape_it_has_to_be():
    """Every member constructs with no arguments.

    A strategy that can be tuned has somewhere to put the tuning, so the
    absence of a constructor argument is the structural half of the boundary
    constraint. The behavioural half is the dial-dependence tests below.
    """
    for name, cls in ROSTER.items():
        cls()                                    # must not raise
        fields = getattr(cls, "__dataclass_fields__", {})
        assert not fields, f"{name} carries a tunable field: {list(fields)}"


def test_the_background_baselines_are_not_the_roster():
    """`freeze_background` reads `BASELINES`, so the two must not merge.

    Stated as a test because the failure mode is somebody adding a roster
    strategy to `BASELINES` for convenience, at which point the field a
    strategy is measured against contains the strategy. `never_pit` is the
    newest way that could happen and the worst: a whole field asking for
    nothing is a different experiment, not a background.
    """
    assert set(BASELINES) - set(ROSTER) == {"caution_opportunist", "fixed_stint"}
    assert set(ROSTER) - set(BASELINES) == {
        "caution_gambler", "track_position", "splash_and_dash", "lap_down",
        "never_pit"}
    assert set(BASELINES) & set(ROSTER) == {"fuel_window"}


def test_the_parameterised_baselines_stay_out_of_the_roster():
    """The two that carry chosen constants are exactly the two excluded."""
    for name in ("caution_opportunist", "fixed_stint"):
        assert getattr(BASELINES[name], "__dataclass_fields__", {}), name
        assert name not in ROSTER


# ----------------------------------------------------------------------
# Parameter-freeness, as behaviour
# ----------------------------------------------------------------------
def test_the_gambler_moves_when_its_dials_move():
    """Its threshold is a count over `fuel_per_lap` and the clock.

    Doubling the burn halves the range on a tank, which changes how many
    more stops are needed and therefore whether this one is free. If the
    decision does not move, the threshold is not coming from the dials.
    """
    gambler = CautionGambler()
    seen = set()
    for mult in (1.0, 2.0, 4.0):
        cfg = scale_dials(config(), fuel_per_lap=mult)
        cfg.classes[0].fuel_per_lap_caution *= mult
        me = car(fuel=0.55)
        decision = gambler(me, state([me], cfg, under_caution=True))
        seen.add(decision.pit)
    assert len(seen) == 2, "the gambler ignored a dial it claims to read"


def test_the_gambler_reads_the_clock_and_not_only_the_tank():
    """One tank state, the whole race: the answer has to depend on the clock.

    Asserted by scanning rather than on two chosen moments. A stop is free
    when the fuel still aboard does not push the remaining requirement over
    a tank boundary, and where that lands is a property of the *remainder*,
    not of how much time is left - so "early" and "late" are not reliably
    opposite and a two-point test can pass or fail on which two points it
    picks. What must hold is that the clock changes the answer at all.
    """
    gambler = CautionGambler()
    answers = set()
    for t in range(0, int(6 * 3600), 600):
        me = car(fuel=0.55)
        answers.add(gambler(me, state([me], under_caution=True,
                                      t=float(t))).pit)
    assert answers == {True, False}, "the gambler ignored the clock"


def test_the_gambler_does_not_ask_through_a_shut_lane():
    """Art. 46.3.3 is the reason the gamble carries risk.

    Without this the strategy is not gambling, it is relying on the engine
    to refuse it - and a refusal is recorded as a defect elsewhere.
    """
    me = car(fuel=0.55)
    shut = CautionGambler()(me, state([me], under_caution=True, lane_open=False))
    assert not shut.pit


# ----------------------------------------------------------------------
# The at-the-line arithmetic
# ----------------------------------------------------------------------
def test_a_rival_is_found_by_arrival_and_not_by_differencing_crossings():
    """The defect this replaced, on state where the two disagree.

    The focal car is at the line on lap 50 at t = 5000. The rival is on lap
    49 and will arrive at 5030, so a 45 s stop lets it past and a 20 s one
    does not. Differencing `race_time_s` instead compares a lap-50 crossing
    with a lap-49 one and returns a negative number, which fires the defence
    whatever the stop costs.
    """
    me = car()
    rival = car("GTP-04", laps_done=49, race_time_s=4930.0,
                lap_start_t=4930.0, lap_expected_s=100.0)   # arrives 5030
    st = state([me, rival])

    assert _would_be_passed(me, st, cost_s=45.0) is rival
    assert _would_be_passed(me, st, cost_s=20.0) is None

    # And the naive reading, stated so the test says what it guards against
    # rather than only that the right answer comes out. Differencing the two
    # crossings gives +70 s, which reads as "the rival is 70 s behind, a
    # 45 s stop is safe" - the opposite of the truth, because the 70 s is
    # measured between a lap-50 crossing and a lap-49 one.
    naive_gap = me.race_time_s - rival.race_time_s
    assert naive_gap == 70.0
    assert naive_gap > 45.0, "the naive comparison would have allowed the stop"


def test_a_rival_more_than_a_lap_down_cannot_be_let_past_by_one_stop():
    me = car()
    far = car("GTP-05", laps_done=47, race_time_s=4700.0,
              lap_start_t=4700.0, lap_expected_s=100.0)
    assert _would_be_passed(me, state([me, far]), cost_s=45.0) is None


def test_a_rival_already_ahead_is_not_defended_against():
    """A stop cannot drop the car behind something it is already behind.

    This is decision 9 item 3's original wording, restated as the thing the
    amendment exists to exclude.
    """
    me = car()
    ahead = car("GTP-01", laps_done=50, race_time_s=4950.0,
                lap_start_t=4950.0, lap_expected_s=100.0)
    assert _would_be_passed(me, state([me, ahead]), cost_s=45.0) is None


def test_another_class_is_not_a_rival():
    me = car()
    other = CarState(car_id="GTD-01", class_name="GTD", base_pace_s=112.0,
                     laps_done=49, race_time_s=4990.0)
    other.lap_start_t, other.lap_expected_s = 4990.0, 20.0    # arrives 5010
    assert _would_be_passed(me, state([me, other]), cost_s=45.0) is None


# ----------------------------------------------------------------------
# The defenders defend only what they may
# ----------------------------------------------------------------------
def test_a_defender_never_declines_past_the_forced_point():
    """Below one lap's burn, declining does not avoid the stop.

    It hands the decision to `_must_pit`, which takes a full service through
    whatever the lane is doing and records `lane_closed_stop`. A defender
    that defends its way into that has defended nothing, so the discretion
    stops where the fuel does.
    """
    cls = config().classes[0]
    dry = car(fuel=cls.fuel_per_lap * 0.5)
    rival = car("GTP-04", laps_done=49, race_time_s=4930.0,
                lap_start_t=4930.0, lap_expected_s=100.0)

    for strategy in (TrackPositionDefender(), LapDownDefender()):
        assert strategy(dry, state([dry, rival])).pit, type(strategy).__name__


def test_the_lap_down_defender_holds_station_only_when_lapped_and_yellow():
    """Clause two is `under_caution and laps_down >= 1`, and nothing else.

    Both halves are needed: lapped under green is a different situation, and
    on the lead lap under caution there is no credit to protect.
    """
    leader = car("GTP-01", laps_done=52, race_time_s=4990.0,
                 lap_start_t=4990.0, lap_expected_s=100.0)
    me = car(laps_done=50)
    strategy = LapDownDefender()

    held = strategy(me, state([me, leader], under_caution=True))
    assert not held.pit and "eligibility" in held.reason

    assert strategy(me, state([me, leader], under_caution=False)).reason != held.reason

    level = car(laps_done=52)
    on_lead_lap = strategy(level, state([level, leader], under_caution=True))
    assert on_lead_lap.reason != held.reason


# ----------------------------------------------------------------------
# The splash
# ----------------------------------------------------------------------
def test_the_splash_never_asks_for_less_fuel_than_is_aboard():
    """`_apply_pit` sets fuel *to* `refuel_to`, so a low ask throws it away.

    The same defect `test_benchmark.py` guards the search against. Checked
    across the whole last hour, because the ask only goes short near the end
    and a single sample would miss where it bites.
    """
    cls = config().classes[0]
    for t in range(int(6 * 3600 - 3600), int(6 * 3600), 120):
        me = car(fuel=cls.fuel_per_lap * 1.2)
        d = SplashAndDashPlanner()(me, state([me], t=float(t)))
        if d.pit:
            assert d.refuel_to >= me.fuel - 1e-9, (t, d.refuel_to, me.fuel)


def test_the_splash_is_short_near_the_flag_and_full_early():
    """The whole point: the last stop takes what the race still needs."""
    cls = config().classes[0]
    me = car(fuel=cls.fuel_per_lap * 1.2)
    early = SplashAndDashPlanner()(me, state([me], t=600.0))
    late = SplashAndDashPlanner()(me, state([me], t=6 * 3600.0 - 400.0))
    assert early.pit and early.refuel_to == 1.0
    assert late.pit and late.refuel_to < 1.0


def test_the_splash_keeps_tyres_when_their_life_covers_what_is_left():
    cls = config().classes[0]
    me = car(fuel=cls.fuel_per_lap * 1.2, tyre_age=1)
    late = SplashAndDashPlanner()(me, state([me], t=6 * 3600.0 - 400.0))
    assert late.pit and not late.change_tyres


# ----------------------------------------------------------------------
# In a race
# ----------------------------------------------------------------------
def test_every_roster_strategy_runs_a_whole_race_without_being_refused():
    """A strategy the engine has to turn down is not a strategy.

    Forced stops through a shut lane are the engine overriding the decision,
    and a roster member should not be generating them: they mean the
    strategy ran itself dry somewhere it could not stop.

    **`never_pit` is exempt, and the exemption is the point of it.** It runs
    itself dry every stint by construction and takes every stop as a forced
    one, so some of those will land on a shut lane. That is what it is for -
    it is a control, not a plan - and asserting otherwise would be asserting
    that the null for a learner is a competent strategy. Exempted by name
    rather than by loosening the condition, so the other five still have to
    pass it.
    """
    cfg = config(duration_s=3 * 3600.0)
    for name, cls in ROSTER.items():
        result = run_race(cfg, strategies={FOCAL: cls()}, seed=4)
        mine = result.laps[result.laps["car_id"] == FOCAL]
        assert len(mine) > 0, name
        if name == "never_pit":
            continue
        if "lane_closed_stop" in mine.columns:
            assert mine["lane_closed_stop"].sum() == 0, name


def test_never_pit_takes_every_stop_as_a_forced_one():
    """The other half of the exemption above, asserted rather than assumed.

    When a strategy declines a stop the rules require, the engine replaces the
    decision with its own and writes its reason into `pit_reason`. So every
    stop this control takes has to carry one of `_must_pit`'s reasons and
    never a reason of its own. If that ever stops being true, `never_pit` has
    started asking for something and is no longer the control it was added as.
    """
    cfg = config(duration_s=3 * 3600.0)
    result = run_race(cfg, strategies={FOCAL: ROSTER["never_pit"]()}, seed=4)
    mine = result.laps[result.laps["car_id"] == FOCAL]
    stops = mine[mine["pitted"]]
    assert len(stops) > 0, "the control never stopped at all"
    assert set(stops["pit_reason"]) <= {"out of fuel", "tyres done",
                                        "driver change"}


def test_the_roster_is_not_all_the_same_strategy():
    """Five rows that agree everywhere are one row printed five times."""
    cfg = config(duration_s=3 * 3600.0)
    fingerprints = set()
    for cls in ROSTER.values():
        c = run_race(cfg, strategies={FOCAL: cls()}, seed=4).classification()
        row = c.set_index("car_id").loc[FOCAL]
        fingerprints.add((int(row["laps"]), int(row["stops"]),
                          round(float(row["pit_time_s"]), 3)))
    assert len(fingerprints) >= 3, fingerprints


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
