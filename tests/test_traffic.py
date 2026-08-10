"""Traffic that a stop can do something about.

Under 01 a car counted as an obstruction if its *base* pace was slower than
yours, so the same cars blocked you on lap two and lap two hundred and no
stop you made could change it. "I will come out into traffic if I stop now"
- one of the two arguments on any pit wall - was not merely unimplemented,
it was unrepresentable.

The effect is small in aggregate, because degradation over a stint is worth
a few tenths while the gap between classes is worth seconds. So it is
tested on controlled state, where the mechanism either works or does not,
rather than on a race average that would mostly be measuring the class
structure.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from endurance import ClassDials, Compat, RaceConfig, run_race  # noqa: E402
from endurance.engine import RaceEngine  # noqa: E402


LAP = 100.0


def one_class_config(deg=0.05, window=0.05, penalty=1.0) -> RaceConfig:
    cls = ClassDials(
        series_code="test", class_name="ONE", base_pace_s=LAP,
        deg_slope_s_per_lap=deg, pace_spread_s=0.0, lap_noise_s=0.0,
        caution_rate=0.0, caution_mean_dur_s=300.0,
        green_stint_laps=20.0, fuel_per_lap=1 / 20, fuel_per_lap_caution=0.6 / 20,
        tyre_life_laps=40.0, pit_time_mean_s=40.0, pit_time_std_s=0.0,
        n_cars=3, traffic_window_frac=window, traffic_penalty_s=penalty)
    return RaceConfig(name="traffic", series_code="test",
                      duration_s=3600.0, classes=[cls])


def staged(compat, base_paces, tyre_ages, fracs, t=1000.0):
    """An engine with the field placed and worn exactly as asked.

    Positions are set through the arrays the penalty actually reads, so the
    test exercises the real comparison rather than a paraphrase of it.
    """
    eng = RaceEngine(one_class_config(), seed=0, compat=compat)
    eng._build_field()
    ids = list(eng.cars)
    for car_id, pace, age, frac in zip(ids, base_paces, tyre_ages, fracs):
        i = eng._idx[car_id]
        eng.cars[car_id].base_pace_s = pace
        eng.cars[car_id].tyre_age = age
        eng._arr_pace[i] = pace
        eng._arr_tyre[i] = age
        eng._arr_lap_expected[i] = LAP
        eng._arr_lap_start[i] = t - frac * LAP
    return eng, ids


# ----------------------------------------------------------------------
# The comparison
# ----------------------------------------------------------------------
def test_a_car_ahead_that_has_worn_out_becomes_an_obstruction():
    """The car ahead started quicker and is now slower. That is the point.

    Base paces 100.0 against 99.9, so on 01's test the car ahead is never a
    blocker. Twenty laps of wear at 0.05s a lap puts it a second down on
    where it started, and it is one now.
    """
    args = dict(base_paces=[100.0, 99.9, 100.0], tyre_ages=[0, 20, 0],
                fracs=[0.50, 0.52, 0.90])
    focal = 0

    eng, ids = staged(Compat(), **args)
    _, blockers = eng._traffic_penalty(eng.cars[ids[focal]], 1000.0)
    assert blockers == 1

    old, ids = staged(Compat(legacy_traffic=True), **args)
    _, was = old._traffic_penalty(old.cars[ids[focal]], 1000.0)
    assert was == 0


def test_a_worn_car_stops_being_held_up_by_cars_it_used_to_catch():
    """The other half: wear slows you into the traffic rather than through it."""
    args = dict(base_paces=[100.0, 100.4, 100.0], tyre_ages=[20, 0, 0],
                fracs=[0.50, 0.52, 0.90])

    eng, ids = staged(Compat(), **args)
    _, blockers = eng._traffic_penalty(eng.cars[ids[0]], 1000.0)
    assert blockers == 0

    old, ids = staged(Compat(legacy_traffic=True), **args)
    _, was = old._traffic_penalty(old.cars[ids[0]], 1000.0)
    assert was == 1


def test_legacy_traffic_cannot_see_tyre_age_at_all():
    """Same field, wildly different wear, identical answer - that was the defect."""
    fresh, _ = staged(Compat(legacy_traffic=True),
                      base_paces=[100.0, 100.4, 100.0], tyre_ages=[0, 0, 0],
                      fracs=[0.50, 0.52, 0.90])
    worn, ids = staged(Compat(legacy_traffic=True),
                       base_paces=[100.0, 100.4, 100.0], tyre_ages=[39, 39, 39],
                       fracs=[0.50, 0.52, 0.90])
    a = fresh._traffic_penalty(fresh.cars[list(fresh.cars)[0]], 1000.0)
    b = worn._traffic_penalty(worn.cars[ids[0]], 1000.0)
    assert a == b


def test_a_car_ahead_but_outside_the_window_is_not_traffic():
    eng, ids = staged(Compat(), base_paces=[100.0, 100.4, 100.0],
                      tyre_ages=[0, 0, 0], fracs=[0.50, 0.80, 0.90])
    _, blockers = eng._traffic_penalty(eng.cars[ids[0]], 1000.0)
    assert blockers == 0


def test_a_slower_car_behind_is_somebody_elses_problem():
    eng, ids = staged(Compat(), base_paces=[100.0, 100.4, 100.0],
                      tyre_ages=[0, 0, 0], fracs=[0.50, 0.48, 0.90])
    _, blockers = eng._traffic_penalty(eng.cars[ids[0]], 1000.0)
    assert blockers == 0


# ----------------------------------------------------------------------
# The bookkeeping the comparison depends on
# ----------------------------------------------------------------------
def test_the_field_view_of_tyre_age_never_goes_stale():
    """A stale array would make the whole change silently do nothing.

    Checked at every single traffic comparison of a whole race, for every
    car except the one asking - its own age is read straight off the car,
    and its array entry is one lap behind by construction because the
    mirror is written when the lap is set, which happens next.
    """
    seen = {"checks": 0}

    class Spy(RaceEngine):
        def _traffic_penalty(self, car, t):
            for car_id, j in self._idx.items():
                if car_id == car.car_id or self.cars[car_id].finished:
                    continue
                assert self._arr_tyre[j] == self.cars[car_id].tyre_age, car_id
                seen["checks"] += 1
            return super()._traffic_penalty(car, t)

    Spy(one_class_config(), seed=1, compat=Compat()).run()
    assert seen["checks"] > 100


def test_traffic_still_only_ever_costs_time():
    result = run_race(one_class_config(), seed=2)
    assert (result.laps["traffic_s"] >= 0).all()
    assert (result.laps["traffic_s"] == result.laps["blockers"]).all()


def test_a_field_on_different_stint_plans_meets_different_traffic():
    """The strategic case, and the reason the change was worth making.

    A field all running the same plan pits in lockstep, so every car carries
    the same tyre age and degradation cancels out of the comparison entirely
    - which is why this needs mixed strategies to show up at all. Once the
    stints are out of phase, a car that has just stopped is quicker than the
    ones around it and meets traffic the old comparison could not see.
    """
    from endurance import FixedLapStint, RunToFuelWindow

    cfg = one_class_config(deg=0.3, window=0.2)
    cfg.classes[0].pace_spread_s = 1.0
    cfg.classes[0].n_cars = 12
    cfg.duration_s = 7200.0

    plans = {f"ONE-{i + 1:02d}": (FixedLapStint(stint_laps=8 + 3 * (i % 4))
                                  if i % 2 else RunToFuelWindow())
             for i in range(12)}

    old = run_race(cfg, plans, seed=3, compat=Compat(legacy_traffic=True))
    new = run_race(cfg, plans, seed=3, compat=Compat())
    assert old.laps["blockers"].sum() != new.laps["blockers"].sum()


def test_a_field_in_lockstep_is_unaffected_by_the_change():
    """Stated as a test so it is a known limit rather than a surprise later.

    Everyone on one plan wears at the same rate, so current pace and base
    pace rank the field identically and the change does nothing. Any 02
    comparison run on a single-strategy field will see none of this.
    """
    cfg = one_class_config(deg=0.3, window=0.2)
    cfg.classes[0].pace_spread_s = 1.0
    cfg.classes[0].n_cars = 12
    cfg.duration_s = 7200.0

    old = run_race(cfg, seed=3, compat=Compat(legacy_traffic=True))
    new = run_race(cfg, seed=3, compat=Compat())
    assert old.laps["blockers"].sum() == new.laps["blockers"].sum()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
