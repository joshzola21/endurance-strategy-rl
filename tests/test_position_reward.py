"""The position reward: does it sum to the thing every table reports?

Four conditions. The first is the whole design; the second and third are its
falsifiers; the fourth is the finding that motivated the change, asserted so
that a future engine which quietly stops refunding pit time is noticed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from endurance.assets import BackgroundField, SeedBank
from endurance.engine import PitDecision, RaceEngine
from endurance.gym_env import REWARD, EnduranceEnv
from endurance.harness import ROSTER, focal_car, run_focal
from endurance.params import RaceConfig, scale_dials

SERIES = ("imsa", "wec")


def find(name: str) -> Path:
    for hit in ROOT.rglob(name):
        if ".ipynb_checkpoints" not in hit.parts:
            return hit
    raise FileNotFoundError(name)


def assets(code: str):
    return (RaceConfig.load(find(f"{code}.json")),
            SeedBank.load(find(f"{code}_seeds.json")),
            BackgroundField.load(find(f"{code}_field.json")))


@pytest.fixture(scope="module", params=SERIES)
def race(request):
    return request.param, *assets(request.param)


def _episode(env, actions):
    """Run one episode under a fixed action, returning the return and the info."""
    obs, info = env.reset(seed=None)
    total, steps = 0.0, 0
    while True:
        obs, reward, done, _trunc, info = env.step(actions(info))
        total += reward
        steps += 1
        if done:
            return total, steps, info


# ----------------------------------------------------------------------
def test_the_return_is_places_gained(race):
    """The design. The rewards telescope to `start - finish`, and the finish
    is the number `classification()` publishes - so an episode's return *is*
    the headline statistic, not a proxy for it."""
    code, config, bank, field = race
    env = EnduranceEnv(config, field, bank)

    for seed in bank.headline[:4]:
        _obs, info = env.reset(seed=seed)
        start_pos = info["class_pos"]

        total = 0.0
        while True:
            _obs, reward, done, _t, info = env.step(0)   # stay out, always
            total += reward
            if done:
                break
        final = info["class_pos"]
        assert total == pytest.approx(start_pos - final, abs=1e-9), (
            f"{code} seed {seed}: rewards summed to {total}, but the car went "
            f"from P{start_pos} to P{final}")


def test_a_worse_policy_scores_worse(race):
    """Falsifier one. A reward that cannot tell two policies apart is the
    failure this replaced: laps could not, and scored 65 stops as costless."""
    code, config, bank, field = race
    env = EnduranceEnv(config, field, bank)

    returns = {}
    for action, label in ((0, "stay out"), (1, "stop for everything")):
        total = 0.0
        for seed in bank.headline[:4]:
            env.reset(seed=seed)
            while True:
                _o, reward, done, _t, _i = env.step(action)
                total += reward
                if done:
                    break
        returns[label] = total
    assert returns["stay out"] > returns["stop for everything"], (
        f"{code}: stopping on every lap scored {returns['stop for everything']} "
        f"against {returns['stay out']} for never stopping - the reward cannot "
        f"see a stop")


def test_the_live_rule_is_the_scored_rule(race):
    """Falsifier two. One sort key, or the return is about a different race
    from the table.

    Asserted on the *rule* rather than on a mid-race snapshot. Mid-race the two
    legitimately disagree, because `RaceState.cars` holds each car at its own
    last crossing and a car that has just crossed outranks one nine tenths of
    the way round - see `class_position`. That wobble cancels in the sum, which
    `test_the_return_is_places_gained` asserts. What must not differ is how the
    two rank cars that are directly comparable.
    """
    code, config, bank, field = race
    seed = bank.headline[0]
    engine = RaceEngine(config, seed=seed)
    engine._build_field()

    focal = focal_car(config, seed)
    car = engine.cars[focal]
    in_class = [c for c in engine.cars.values()
                if c.class_name == car.class_name]

    # Give the class a spread of laps and times, then rank it both ways.
    rng = np.random.default_rng(0)
    for i, other in enumerate(in_class):
        other.laps_done = int(rng.integers(40, 45))
        other.race_time_s = float(4000.0 + rng.random() * 400.0)

    state = engine._state(engine.cars[focal].race_time_s) if hasattr(
        engine, "_state") else None
    from endurance.engine import RaceState

    state = RaceState(t=4400.0, duration_s=config.duration_s,
                      under_caution=False, cars=engine.cars, config=config)

    mine = {c.car_id: state.class_position(c) for c in in_class}
    theirs = {c.car_id: i for i, c in enumerate(
        sorted(in_class, key=lambda c: (-c.laps_done, c.race_time_s)), start=1)}
    assert mine == theirs, f"{code}: two rankings of the same class"


def test_caution_compression_still_refunds_pit_time(race):
    """The finding this change rests on, kept as a regression.

    A car that stops constantly loses far fewer laps than the time it spends
    stopped, and the refund tracks the caution rate. If an engine change ever
    removes that, the argument for a position reward weakens and somebody
    should know rather than discover it later.
    """
    code, config, bank, field = race

    class StopOften:
        def __call__(self, car, state):
            if car.stint_laps >= 8:
                return PitDecision(pit=True, refuel_to=1.0,
                                   change_tyres=False, reason="often")
            return PitDecision(pit=False, reason="")

    refunds = {}
    for mult in (1.0, 0.05):
        cfg = config if mult == 1.0 else scale_dials(config, caution_rate=mult)
        lap_s = min(c.base_pace_s for c in cfg.classes)
        d_pit, d_laps = [], []
        for seed in bank.headline[:3]:
            focal = focal_car(cfg, seed)
            null = run_focal(cfg, seed, focal, ROSTER["fuel_window"](), field)
            mine = run_focal(cfg, seed, focal, StopOften(), field)
            d_pit.append(mine["pit_time_s"] - null["pit_time_s"])
            d_laps.append(null["laps"] - mine["laps"])
        spent = float(np.median(d_pit)) / lap_s
        lost = float(np.median(d_laps))
        refunds[mult] = 1.0 - lost / spent

    assert refunds[1.0] > refunds[0.05] + 0.15, (
        f"{code}: the refund no longer tracks the caution rate "
        f"({refunds[1.0]:.0%} at nominal against {refunds[0.05]:.0%} at "
        f"near-green); the reason for this reward has changed")


def test_the_card_will_not_lie():
    """`train.py` writes `REWARD` onto every policy card. It has now been
    wrong twice in this project's history, both times because it was a string
    literal somewhere else."""
    assert "position" in REWARD or "place" in REWARD
    assert "lap" not in REWARD.replace("laps of", "")
