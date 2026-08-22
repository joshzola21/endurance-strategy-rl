"""Amendment 14's floor: a stop cannot cost less than driving down the lane.

Four conditions. The first two are the fix; the third is its falsifier and is
the one that says this is a **dial** rather than a rewrite; the fourth keeps
the compat flag honest.

Run with pytest, or `python tests/test_stop_cost_floor.py` for a bare report.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from endurance.engine import Compat, RaceEngine
from endurance.params import ASSUMED_FIELDS, RaceConfig
from endurance.pitstop import PitRules, stop_cost, transit_s

SERIES = ("imsa", "wec")
FUELS = (0.0, 0.05, 0.1, 0.35, 0.7, 1.0)
TYRES = (False, True)


class _FixedNoise:
    """A `NoiseStream` that always answers the same standard normal."""

    def __init__(self, z: float):
        self._z = float(z)

    def __getitem__(self, i: int) -> float:
        return self._z


def find(name: str) -> Path:
    for hit in ROOT.rglob(name):
        if ".ipynb_checkpoints" not in hit.parts:
            return hit
    raise FileNotFoundError(name)


@pytest.fixture(scope="module", params=SERIES)
def race(request):
    code = request.param
    return code, RaceConfig.load(find(f"{code}.json"))


# ----------------------------------------------------------------------
def test_no_stop_costs_less_than_the_lane(race):
    """The floor, across every class and every shape of stop.

    Under caution and green, refuelling nothing or everything, tyres or not.
    The failing case amendment 14 found was a caution top-up, which is the
    cheapest stop the engine can price and therefore the one nearest the floor.
    """
    code, config = race
    rules = PitRules.for_series(code)
    for cls in config.classes:
        floor = transit_s(cls)
        assert floor > 0.0, f"{cls.class_name}: no transit to floor against"
        for fuel in FUELS:
            for tyres in TYRES:
                for caution in (False, True):
                    cost = stop_cost(cls, rules, fuel, tyres,
                                     under_caution=caution)
                    assert cost >= floor - 1e-9, (
                        f"{code} {cls.class_name}: fuel={fuel} tyres={tyres} "
                        f"caution={caution} costs {cost:.2f} s against a "
                        f"{floor:.2f} s lane transit")


def test_the_floor_holds_through_the_noise(race):
    """A left-tail draw cannot violate it either.

    `pit_time_std_s` runs two to five times its mean in places, so a stop
    priced at the floor and then given a normal draw would go under it
    routinely. The clamp is in `_apply_pit` after the noise for this reason,
    and this asserts on the engine rather than on `stop_cost`.
    """
    code, config = race
    engine = RaceEngine(config, seed=7)
    engine._build_field()
    for car_id, car in list(engine.cars.items())[:6]:
        cls = config.class_by_name(car.class_name)
        floor = transit_s(cls)
        # Drive the noise itself rather than hunting for a race that draws a
        # bad one. `NoiseStream` is read-only by design - the index is the
        # whole point of it - so the stream is swapped for one that answers
        # every index with an eight-sigma low. Same code path, known input.
        engine._pit_noise[car_id] = _FixedNoise(-8.0)
        from endurance.engine import PitDecision

        cost = engine._apply_pit(car, PitDecision(pit=True, refuel_to=0.05,
                                                  change_tyres=False),
                                 t=0.0, forced_reason="")
        assert cost >= floor - 1e-9, (
            f"{code} {car_id}: an eight-sigma low draw priced a stop at "
            f"{cost:.2f} s against a {floor:.2f} s floor")


def test_it_is_a_dial_not_a_rewrite(race):
    """The falsifier. Set the new dial to the old value, get the old engine.

    Before amendment 14 the discount multiplied the whole stop, transit
    included. `pit_transit_caution_discount = pit_caution_discount` says
    exactly that, so `stop_cost` must reproduce the superseded arithmetic to
    the last bit. If this fails, the change is not the dial it claims to be and
    something else moved with it.
    """
    code, config = race
    rules = PitRules.for_series(code)
    for cls in config.classes:
        old_style = RaceConfig.from_dict(config.to_dict()).class_by_name(
            cls.class_name)
        old_style.pit_transit_caution_discount = old_style.pit_caution_discount
        for fuel in FUELS:
            for tyres in TYRES:
                undiscounted = stop_cost(cls, rules, fuel, tyres)
                assert (stop_cost(old_style, rules, fuel, tyres,
                                  under_caution=True)
                        == pytest.approx(
                            undiscounted * (1.0 - cls.pit_caution_discount),
                            rel=0, abs=1e-12))


def test_the_new_dial_is_assumed_and_moves_the_fingerprint(race):
    """Amendment 21: a rules change that leaves the fingerprint alone is worse
    than no change, because every frozen artefact goes on matching."""
    from endurance.assets import dials_fingerprint

    _code, config = race
    assert "pit_transit_caution_discount" in ASSUMED_FIELDS

    moved = RaceConfig.from_dict(config.to_dict())
    for cls in moved.classes:
        cls.pit_transit_caution_discount = 0.4
    assert dials_fingerprint(moved) != dials_fingerprint(config)


def test_legacy_pit_is_untouched(race):
    """A compat flag that quietly acquired amendment 14 would stop being a way
    of showing what 02a changed."""
    _code, config = race
    engine = RaceEngine(config, seed=7, compat=Compat(legacy_pit=True,
                                                      legacy_noise=False))
    engine._build_field()
    from endurance.engine import PitDecision

    car = next(iter(engine.cars.values()))
    cls = config.class_by_name(car.class_name)
    engine._pit_noise[car.car_id] = _FixedNoise(0.0)
    engine.cautions.periods = [(0.0, 1000.0)]

    cost = engine._apply_pit(car, PitDecision(pit=True, refuel_to=1.0,
                                              change_tyres=True),
                             t=1.0, forced_reason="")
    assert cost == pytest.approx(
        cls.pit_time_mean_s * (1.0 - cls.pit_caution_discount), abs=1e-9)


if __name__ == "__main__":
    for code in SERIES:
        config = RaceConfig.load(find(f"{code}.json"))
        rules = PitRules.for_series(code)
        print(f"=== {code.upper()} ===")
        for cls in config.classes:
            floor = transit_s(cls)
            worst = min(stop_cost(cls, rules, f, t, under_caution=True)
                        for f in FUELS for t in TYRES)
            was = min(stop_cost(cls, rules, f, t) for f in FUELS for t in TYRES
                      ) * (1.0 - cls.pit_caution_discount)
            print(f"  {cls.class_name:8} lane {floor:6.2f} s | cheapest "
                  f"caution stop was {was:6.2f} s, now {worst:6.2f} s "
                  f"{'OK' if worst >= floor - 1e-9 else 'BELOW FLOOR'}")
