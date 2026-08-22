"""What the wrapper has to keep being.

Three things are guarded here, and only one of them is behaviour.

The first is the **boundary constraint**: zero physics. The wrapper reads
quantities the engine computes and never computes one itself. Tested by
scanning what the module imports rather than by reading it and concluding it
looks fine, which is the reasoning that let two at-the-line defects stand.

The second is the **stage gate**: a policy returning `RunToFuelWindow`'s
decision reproduces that baseline exactly on the same seed. The blueprint's
original gate - a never-pit policy reproducing the baseline - is dropped
rather than restated: measured over twelve seeds it differs on laps in four
and on stop count in two, so there is no exact statement of it to keep.

The third is the **at-the-line arithmetic**, again. The observation's two gap
rows walk straight into the invariant that caught the track-position
defender, so they are tested on constructed state where the naive answer and
the right one are known to disagree, rather than on a race average where a
sign error averages away.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from endurance import ClassDials, PitDecision, RaceConfig, scale_dials  # noqa: E402
from endurance.assets import draw_seed_bank, freeze_background  # noqa: E402
from endurance.engine import CarState, RaceEngine, RaceState, run_race  # noqa: E402
from endurance.strategies import ROSTER, FixedLapStint, RunToFuelWindow  # noqa: E402
from endurance import gym_env, harness  # noqa: E402
from endurance.gym_env import (  # noqa: E402
    FLAG_KEEP,
    FLAG_TYRES,
    FULL_KEEP,
    FULL_TYRES,
    N_ACTIONS,
    N_OBS,
    OBS_ROWS,
    STAY,
    EnduranceEnv,
    PolicyStrategy,
    action_mask,
    gap_scale_s,
    observe,
    to_decision,
)


FOCAL = "GTP-03"
SEEDS = [11, 12, 13]


def config(series="imsa", duration_s=3 * 3600.0) -> RaceConfig:
    """The 02c stand-in's shape, shortened so the suite stays quick."""
    fast = ClassDials(
        series_code=series, class_name="GTP", base_pace_s=97.5,
        deg_slope_s_per_lap=0.012, pace_spread_s=0.5, lap_noise_s=0.5,
        caution_rate=0.20, caution_mean_dur_s=600.0, green_stint_laps=30.0,
        fuel_per_lap=1 / 30, fuel_per_lap_caution=0.6 / 30, tyre_life_laps=60.0,
        pit_time_mean_s=47.0, pit_time_std_s=2.0, n_cars=6)
    slow = ClassDials(
        series_code=series, class_name="GTD", base_pace_s=112.0,
        deg_slope_s_per_lap=0.02, pace_spread_s=0.9, lap_noise_s=0.6,
        caution_rate=0.20, caution_mean_dur_s=600.0, green_stint_laps=28.0,
        fuel_per_lap=1 / 28, fuel_per_lap_caution=0.6 / 28, tyre_life_laps=56.0,
        pit_time_mean_s=45.0, pit_time_std_s=2.0, n_cars=5)
    return RaceConfig(name=f"{series} 3h", series_code=series,
                      duration_s=duration_s, classes=[fast, slow])


def car(car_id=FOCAL, laps_done=50, race_time_s=5000.0, fuel=0.5,
        lap_start_t=4900.0, lap_expected_s=100.0, **over) -> CarState:
    c = CarState(car_id=car_id, class_name="GTP", base_pace_s=97.5,
                 laps_done=laps_done, race_time_s=race_time_s)
    c.fuel, c.lap_start_t, c.lap_expected_s = fuel, lap_start_t, lap_expected_s
    for k, v in over.items():
        setattr(c, k, v)
    return c


def state(cars, cfg=None, t=5000.0, under_caution=False, lane_open=True,
          duration_s=3 * 3600.0) -> RaceState:
    return RaceState(t=t, duration_s=duration_s, under_caution=under_caution,
                     cars={c.car_id: c for c in cars},
                     config=cfg or config(), pit_lane_open=lane_open)


def env(cfg=None, **over) -> EnduranceEnv:
    cfg = cfg or config()
    return EnduranceEnv(cfg, freeze_background(cfg), draw_seed_bank(cfg), **over)


def drive(e: EnduranceEnv, policy, seed=None, limit=10_000):
    """Run one episode, choosing with `policy(obs, mask)`."""
    obs, info = e.reset(seed=seed)
    steps, rewards = 0, []
    for _ in range(limit):
        obs, r, term, trunc, info = e.step(policy(obs, info["action_mask"]))
        steps += 1
        rewards.append(r)
        if term or trunc:
            return steps, rewards, info
    raise AssertionError("episode did not finish")


# ----------------------------------------------------------------------
# The boundary constraint
# ----------------------------------------------------------------------
def test_the_wrapper_imports_no_physics_and_no_ui():
    """One simulator. The wrapper adapts it and computes nothing itself.

    Checked on the import graph rather than by reading the code, because
    "it looks like it only reads state" is exactly the review that passed
    the wave-eligibility defect. Anything the wrapper needs it must get by
    asking a module that owns it.
    """
    source = Path(gym_env.__file__).read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0] if node.level == 0
                         else node.module)

    allowed = {"__future__", "gymnasium", "numpy",
               "assets", "engine", "harness", "strategies"}
    assert imported <= allowed, sorted(imported - allowed)
    for banned in ("streamlit", "matplotlib", "viz", "benchmark"):
        assert banned not in imported


def test_the_wrapper_never_prices_a_lap_or_a_stop():
    """The dials it may read are scales; the ones that price things are not.

    `pit_time_mean_s` is here as a *unit* for the gap row and nothing else,
    so it is allowed. A wrapper reaching for `deg_slope_s_per_lap`,
    `lap_noise_s` or `fuel_per_lap` is computing something the engine
    already computes, which is the failure this constraint exists to catch.
    """
    source = Path(gym_env.__file__).read_text()
    for dial in ("deg_slope_s_per_lap", "lap_noise_s", "pace_spread_s",
                 "caution_pace_multiplier", "pit_transit_frac"):
        assert dial not in source, dial


# ----------------------------------------------------------------------
# The engine inversion
# ----------------------------------------------------------------------
def test_run_is_a_drain_of_the_stream_rather_than_a_second_loop():
    """With no focal car the generator returns without ever yielding.

    That is what makes `run` a drain: if it yielded, `run` would need a loop
    of its own and the project would have two race loops that could drift
    apart.
    """
    stream = RaceEngine(config(), seed=5).run_stream(focal=None)
    with pytest.raises(StopIteration):
        next(stream)


def test_the_stream_reproduces_the_race_run_produces():
    """Driving the focal car by hand must not change the race.

    The strongest available statement that the inversion is behaviour
    preserving: same seed, same field, same decisions, same laps - for the
    whole field, not only the focal car.
    """
    cfg = config()
    field = freeze_background(cfg)
    for seed in SEEDS:
        plain = run_race(cfg, strategies=field.resolve(), seed=seed)

        stream = RaceEngine(cfg, seed=seed).run_stream(
            field.resolve(focal=FOCAL), focal=FOCAL)
        decision, driven = None, None
        while driven is None:
            try:
                car_, state_, _forced, _lane = (stream.send(decision) if decision
                                                else next(stream))
            except StopIteration as done:
                driven = done.value
                break
            decision = RunToFuelWindow()(car_, state_)

        assert driven.laps.equals(plain.laps), seed


def test_a_half_run_race_can_be_thrown_away():
    """`reset` mid-episode must not leak a suspended race.

    The property the generator was chosen for over a worker thread: closing
    it is a method call rather than a rendezvous with a blocked thread.
    """
    e = env()
    e.reset(seed=SEEDS[0])
    e.step(STAY)
    e.reset(seed=SEEDS[1])          # must not raise
    e.close()
    e.close()                       # and must be idempotent


# ----------------------------------------------------------------------
# The at-the-line arithmetic
# ----------------------------------------------------------------------
def test_gaps_are_projected_to_a_common_lap_and_not_differenced():
    """The invariant, on state where the two readings disagree by their sign.

    The focal car is at the line on lap 50 at t = 5000. The rival is on lap
    49 and arrives at 5030, so it is 30 s *behind*. Differencing the two
    `race_time_s` values compares a lap-50 crossing with a lap-49 one and
    gives +70 s, which reads as a car 70 s ahead.
    """
    me = car()
    rival = car("GTP-04", laps_done=49, race_time_s=4930.0,
                lap_start_t=4930.0, lap_expected_s=100.0)
    st = state([me, rival])

    assert abs(st.gap_behind_s(me) - 30.0) < 1e-9
    assert st.gap_ahead_s(me) is None

    naive = me.race_time_s - rival.race_time_s
    assert naive == 70.0, "the naive reading would have called this car ahead"


def test_a_car_a_lap_up_is_placed_by_when_its_current_lap_began():
    """One lap up crossed my lap when its own lap started - exactly, not by
    estimate, because `lap_start_t` is that crossing."""
    me = car()
    ahead = car("GTP-01", laps_done=51, race_time_s=5040.0,
                lap_start_t=4960.0, lap_expected_s=98.0)
    assert abs(state([me, ahead]).gap_ahead_s(me) - 40.0) < 1e-9


def test_nobody_within_a_lap_reads_as_no_gap_rather_than_a_large_one():
    me = car()
    far = car("GTP-06", laps_done=47, race_time_s=4700.0,
              lap_start_t=4700.0, lap_expected_s=100.0)
    st = state([me, far])
    assert st.gap_ahead_s(me) is None and st.gap_behind_s(me) is None
    obs = observe(me, st)
    assert obs[OBS_ROWS.index("gap_ahead")] == 1.0
    assert obs[OBS_ROWS.index("gap_behind")] == 1.0


def test_another_class_is_not_a_gap():
    """The score is class position, so a GT car alongside is traffic."""
    me = car()
    other = CarState(car_id="GTD-01", class_name="GTD", base_pace_s=112.0,
                     laps_done=49, race_time_s=4990.0)
    other.lap_start_t, other.lap_expected_s = 4990.0, 20.0
    st = state([me, other])
    assert st.gap_ahead_s(me) is None and st.gap_behind_s(me) is None


def test_no_gap_comes_out_negative_over_a_whole_race():
    """The defect this replaced fired about a quarter of the time."""
    cfg = config()
    seen = 0
    for car_, state_, _f, _l in _decisions(cfg, SEEDS[0]):
        for gap in (state_.gap_ahead_s(car_), state_.gap_behind_s(car_)):
            if gap is not None:
                assert gap >= 0.0
                seen += 1
    assert seen > 50, "the check has to actually meet some rivals"


def _decisions(cfg, seed, focal=FOCAL, policy=None):
    """Every decision point the focal car reaches, as (car, state, forced, lane)."""
    policy = policy or RunToFuelWindow()
    field = freeze_background(cfg)
    stream = RaceEngine(cfg, seed=seed).run_stream(
        field.resolve(focal=focal), focal=focal)
    decision = None
    while True:
        try:
            item = stream.send(decision) if decision else next(stream)
        except StopIteration:
            return
        yield item
        decision = policy(item[0], item[1])


# ----------------------------------------------------------------------
# The observation
# ----------------------------------------------------------------------
def test_the_observation_is_ten_rows_in_the_unit_box():
    """Ten since amendment 24, and the tenth is the reason for the change.

    The reward was defined on class position while the observation could not
    see it: `laps_down` was the row meant to carry it and correlates 0.091
    with it, being clipped at three laps with a front-runner at zero all race.
    The value function was estimating "how many places will I gain from here"
    without being told where here is. `class_position` correlates 1.000, which
    is the least surprising sentence in this file and took four retrains to
    reach.
    """
    e = env()
    obs, _ = e.reset(seed=SEEDS[0])
    assert obs.shape == (N_OBS,) == (10,)
    assert obs.dtype == np.float32
    assert e.observation_space.contains(obs)
    assert len(OBS_ROWS) == N_OBS


def test_every_row_stays_in_the_box_for_a_whole_race():
    """A row that leaves the box is a normalisation that does not hold."""
    cfg = config()
    checked = 0
    for car_, state_, _f, _l in _decisions(cfg, SEEDS[0]):
        obs = observe(car_, state_)
        assert ((obs >= 0.0) & (obs <= 1.0)).all(), obs
        checked += 1
    assert checked > 50


def test_the_rows_are_in_the_order_their_names_claim():
    """Nine numbers in a vector are indistinguishable once they are wrong."""
    cfg = config()
    me = car(fuel=0.25, tyre_age=15, stint_laps=10, laps_down=0)
    leader = car("GTP-01", laps_done=52, race_time_s=4990.0,
                 lap_start_t=4990.0, lap_expected_s=100.0)
    st = state([me, leader], cfg=cfg, t=5400.0, under_caution=True,
               lane_open=False)
    obs = dict(zip(OBS_ROWS, observe(me, st)))
    cls = cfg.classes[0]

    assert abs(obs["race_progress"] - 5400.0 / (3 * 3600.0)) < 1e-6
    assert abs(obs["fuel"] - 0.25) < 1e-6
    assert abs(obs["tyre_age"] - 15 / cls.tyre_life_laps) < 1e-6
    assert obs["under_caution"] == 1.0
    assert abs(obs["stint_laps"] - 10 / 40.0) < 1e-6
    assert abs(obs["laps_down"] - 2 / 3.0) < 1e-6
    assert obs["pit_lane_open"] == 0.0


def test_the_gap_scale_is_a_dial_and_not_a_constant_somebody_chose():
    """Move the dial the scale is taken from and the row has to move.

    The same shape as the roster's parameter-freeness tests: a normalisation
    that ignores the dials is a magic number with a docstring.
    """
    cfg = config()
    twisted = scale_dials(cfg, pit_time_mean_s=2.0)
    me = car()
    rival = car("GTP-04", laps_done=49, race_time_s=4930.0,
                lap_start_t=4930.0, lap_expected_s=100.0)

    a = observe(me, state([me, rival], cfg=cfg))
    b = observe(me, state([me, rival], cfg=twisted))
    i = OBS_ROWS.index("gap_behind")
    assert abs(b[i] - a[i] / 2.0) < 1e-5
    assert gap_scale_s(twisted.classes[0]) == 2 * gap_scale_s(cfg.classes[0])


# ----------------------------------------------------------------------
# The mask
# ----------------------------------------------------------------------
def test_a_forced_stop_removes_staying_out():
    mask = action_mask(state([car()]), forced="out of fuel")
    assert not mask[STAY]
    assert mask[FULL_TYRES] and mask[FLAG_KEEP]


def test_tyre_life_also_removes_the_keep_tyre_actions():
    """`_apply_pit` fits tyres on that reason whatever the decision said.

    Leaving them in the mask would let the policy pick an action whose
    stated effect the engine then contradicts, which is the discarded
    answer the mask exists to avoid.
    """
    mask = action_mask(state([car()]), forced="tyres done")
    assert not mask[STAY] and not mask[FULL_KEEP] and not mask[FLAG_KEEP]
    assert mask[FULL_TYRES] and mask[FLAG_TYRES]


def test_a_shut_lane_removes_every_stop_unless_one_is_forced():
    shut = state([car()], lane_open=False)
    assert list(action_mask(shut, forced="")) == [True, False, False, False, False]
    # ... and a forced stop goes ahead anyway, recorded as `lane_closed_stop`.
    assert action_mask(shut, forced="out of fuel")[FULL_TYRES]


def test_the_mask_is_never_empty_over_a_whole_race():
    """An empty mask is a hang rather than an error, so it gets a race."""
    cfg = config()
    checked = 0
    for _car, state_, forced, _lane in _decisions(cfg, SEEDS[0]):
        assert action_mask(state_, forced).any()
        checked += 1
    assert checked > 50


# ----------------------------------------------------------------------
# The actions
# ----------------------------------------------------------------------
def test_there_is_no_refuel_level_for_the_agent_to_choose():
    """The structural half of the same constraint the roster carries.

    A level is a number, and a number the agent picks is a tuning surface
    the five human strategies were forbidden. Five discrete actions have
    nowhere to put one.
    """
    assert env().action_space.n == N_ACTIONS == 5


def test_every_action_maps_to_a_decision_a_strategy_could_have_returned():
    me, st = car(), None
    st = state([car()])
    assert not to_decision(STAY, me, st).pit
    for action in (FULL_TYRES, FULL_KEEP, FLAG_TYRES, FLAG_KEEP):
        d = to_decision(action, me, st)
        assert isinstance(d, PitDecision) and d.pit
        assert d.change_tyres == (action in (FULL_TYRES, FLAG_TYRES))
    assert to_decision(FULL_TYRES, me, st).refuel_to == 1.0


def test_the_flag_fill_is_short_near_the_end_and_full_early():
    """The lever 02c measured at 18.0 s a stop in WEC, reachable by the agent."""
    me = car(fuel=0.05)
    early = to_decision(FLAG_KEEP, me, state([me], t=600.0))
    late = to_decision(FLAG_KEEP, me, state([me], t=3 * 3600.0 - 400.0))
    assert early.refuel_to == 1.0
    assert late.refuel_to < 1.0


def test_the_flag_fill_never_asks_for_less_than_is_aboard():
    """`_apply_pit` sets fuel *to* `refuel_to`, so a low ask throws it away.

    Guarded in `strategies.fuel_to_the_flag`, which is why the agent and the
    splash planner call one function - but asserted here as well, because
    the action is the thing that would be wrong.
    """
    cls = config().classes[0]
    for t in range(int(2 * 3600), int(3 * 3600), 120):
        me = car(fuel=cls.fuel_per_lap * 1.2)
        d = to_decision(FLAG_TYRES, me, state([me], t=float(t)))
        assert d.refuel_to >= me.fuel - 1e-9, (t, d.refuel_to)


# ----------------------------------------------------------------------
# The stage gate
# ----------------------------------------------------------------------
def test_the_gate_a_policy_echoing_the_baseline_reproduces_it():
    """03a's verification gate, restated and passing.

    The agent's plumbing sits in the middle - the observation is built, the
    action is chosen, the decision is rebuilt - and the classification row
    has to come out bit for bit against the baseline run directly. It gates
    the wrapper rather than the engine, which is the right target: every
    number 03b reports is measured through this path.
    """
    cfg = config()
    field = freeze_background(cfg)

    def echo(car_, state_):
        return RunToFuelWindow()(car_, state_)

    for seed in SEEDS:
        focal = harness.focal_car(cfg, seed)
        a = harness.run_focal(cfg, seed, focal, RunToFuelWindow(), field)
        b = harness.run_focal(cfg, seed, focal, echo, field)
        for col in ("laps", "race_time_s", "stops", "pit_time_s",
                    "traffic_time_s", "class_pos"):
            assert a[col] == b[col], (seed, col)


def test_the_gate_would_notice_a_policy_that_is_not_the_baseline():
    """The gate is only worth running if it can fail."""
    cfg = config()
    field = freeze_background(cfg)
    seed = SEEDS[0]
    focal = harness.focal_car(cfg, seed)
    a = harness.run_focal(cfg, seed, focal, RunToFuelWindow(), field)
    b = harness.run_focal(cfg, seed, focal, FixedLapStint(stint_laps=9), field)
    assert a["stops"] != b["stops"]


# ----------------------------------------------------------------------
# The agent gets no privileged path
# ----------------------------------------------------------------------
def test_the_engine_still_forces_a_stop_the_agent_declined():
    """`PolicyStrategy` has no mask, so the override is what holds.

    Fuel exists in this model only through `_must_pit` - `_next_lap` never
    reads it - so an agent allowed to decline forever would lap on an empty
    tank at full pace. The engine must take the stop instead.
    """
    cfg = config()
    never = PolicyStrategy(lambda obs, deterministic=True: (STAY, None))
    result = run_race(cfg, strategies={FOCAL: never}, seed=SEEDS[0])
    mine = result.laps[result.laps["car_id"] == FOCAL]
    assert mine["pitted"].sum() > 0, "the agent lapped on an empty tank"
    assert (mine["fuel"] >= -1e-9).all()


def test_the_agent_is_scored_through_the_harness_like_everyone_else():
    """Inserted into the roster as a sixth member, not a second code path.

    Decision 6's failure mode is an evaluation path that can differ from the
    roster's, so the agent's row has to be produced by the same call the
    five humans' rows are.
    """
    cfg = config()
    roster = dict(ROSTER)
    roster["agent"] = lambda: PolicyStrategy(
        lambda obs, deterministic=True: (FULL_TYRES, None))

    rows = harness.compare_roster(cfg, SEEDS, freeze_background(cfg),
                                  roster=roster).rows
    assert set(rows["strategy"]) == set(roster)
    agent = rows[rows["strategy"] == "agent"]
    assert len(agent) == len(SEEDS)
    assert agent["stops"].min() > 0


# ----------------------------------------------------------------------
# The environment
# ----------------------------------------------------------------------
def test_a_thousand_random_masked_steps_run_clean():
    """The blueprint's headless check, with the mask respected."""
    e = env()
    rng = np.random.default_rng(0)
    obs, info = e.reset(seed=e.bank.headline[0])
    episodes = 0
    for _ in range(1000):
        legal = np.flatnonzero(info["action_mask"])
        obs, reward, term, trunc, info = e.step(int(rng.choice(legal)))
        assert e.observation_space.contains(obs)
        if term:
            assert "classification" in info
            episodes += 1
            obs, info = e.reset()
    assert episodes >= 1, "a thousand steps did not finish a single race"


def test_the_same_seed_gives_the_same_episode():
    """One seed is one race, which is what pairs the agent with the roster."""
    e = env()
    a = drive(e, lambda obs, mask: STAY if mask[STAY] else FULL_TYRES,
              seed=SEEDS[0])
    b = drive(e, lambda obs, mask: STAY if mask[STAY] else FULL_TYRES,
              seed=SEEDS[0])
    assert a[0] == b[0]
    assert a[1] == b[1]
    assert a[2]["classification"] == b[2]["classification"]


def test_the_held_out_bank_is_refused_unless_it_is_asked_for():
    """The only set that can answer whether a design generalises."""
    e = env()
    with pytest.raises(ValueError):
        e.reset(seed=e.bank.held_out[0])
    allowed = env(allow_held_out=True)
    allowed.reset(seed=allowed.bank.held_out[0])      # must not raise


def test_the_reward_is_the_change_in_class_position():
    """Places, credited as they change.

    Supersedes `test_the_reward_credits_one_lap_a_lap`, which asserted a lap
    a lap. **That is the second reward this file has outlived**, and this time
    the superseded test was left in place for a whole pass and went on failing
    quietly: `tests/test_position_reward.py` was written beside it rather than
    over it. Two tests asserting two different rewards is one test lying.

    The lap reward was withdrawn at amendment 24 for a measured reason, not a
    stylistic one. A car that stops forty extra times at Daytona spends 1,072 s
    more in the pits and loses *zero* laps: caution compression refunds 70% of
    the time a stop costs at that caution rate, so laps and the score came
    apart exactly where the strategy lives. The IMSA policy took 65 stops
    against the null's 24.5, scored as having lost nothing in laps, and fell a
    place.

    Three things follow from crediting position instead, and all three are
    asserted here because each has a different way of going wrong:
    """
    e = env()
    _obs, info = e.reset(seed=SEEDS[0])
    start = info["class_pos"]

    rewards = []
    while True:
        _obs, r, term, trunc, info = e.step(
            STAY if info["action_mask"][STAY] else FULL_TYRES)
        rewards.append(r)
        if term or trunc:
            break

    # A place is a whole thing. A fractional reward would mean position had
    # stopped being derived the way `RaceResult.positions` derives it.
    assert all(float(r).is_integer() for r in rewards)

    # Most laps change nothing, which is what makes this reward sparse and is
    # the cost of the change. A reward that moved on every step would be
    # measuring something other than position.
    assert sum(1 for r in rewards if r == 0.0) > 0.5 * len(rewards)

    # And the whole design: the steps telescope to the number the table
    # reports, so an episode's return *is* the headline statistic rather than
    # a proxy for it.
    assert sum(rewards) == pytest.approx(start - info["class_pos"], abs=1e-9)


def test_the_return_tells_two_policies_apart():
    """A reward that does not vary with the policy is not a reward.

    This is the question the suite was missing, and the reason a degenerate
    return survived 03a and half a million training steps. Every reward
    assertion here was about the *shape of a step* - is it negative, is it
    about a lap long - and all of them are true of a return that is
    constant. None of them asked whether the episode score changes when the
    behaviour does.

    Two policies a whole pit strategy apart: one stopping whenever it may,
    one staying out until the engine forces it. The first throws away laps,
    and the return has to notice.
    """
    stop_always = drive(env(),
                        lambda obs, mask: FULL_TYRES if mask[FULL_TYRES] else STAY,
                        seed=SEEDS[0])
    stay_out = drive(env(),
                     lambda obs, mask: STAY if mask[STAY] else FULL_TYRES,
                     seed=SEEDS[0])

    assert stop_always[0] < stay_out[0], "stopping every lap cost no laps"
    assert sum(stay_out[1]) > sum(stop_always[1]) * 1.02, (
        f"two policies a pit strategy apart scored {sum(stay_out[1])} and "
        f"{sum(stop_always[1])} - the return is not reading the policy")


def test_the_return_is_the_places_and_not_the_clock():
    """Two withdrawn rewards, and the guard that outlives both.

    The first was elapsed time, and the defect was that a timed race takes the
    same time however it is driven, so the return was `duration_s /
    base_pace_s` for every policy. The second was laps, and the defect was
    that caution compression severs laps from the score. The clock limb below
    is what caught the first and it is kept unchanged - it is still true, and
    it is what makes "the return is not the clock" worth asserting at all.

    What has changed is the other limb. It used to require the return to track
    laps at a correlation above 0.99, which was a tautology when the return
    *was* laps and is simply false now. In its place: the return is exactly
    the places the car gained, and it is emphatically not the lap count, which
    is the shape the withdrawn reward had.
    """
    e = env()
    runs = []
    for policy in (lambda obs, mask: STAY if mask[STAY] else FULL_TYRES,
                   lambda obs, mask: FULL_TYRES if mask[FULL_TYRES] else STAY,
                   lambda obs, mask: FLAG_KEEP if mask[FLAG_KEEP] else STAY):
        fresh = env()
        _obs, first = fresh.reset(seed=SEEDS[0])
        start = first["class_pos"]
        steps, total, info = 0, 0.0, first
        while True:
            _obs, r, term, trunc, info = fresh.step(
                policy(_obs, info["action_mask"]))
            steps += 1
            total += r
            if term or trunc:
                break
        runs.append((steps, total, info["classification"]["race_time_s"],
                     start, info["class_pos"]))

    laps = np.array([r[0] for r in runs], dtype=float)
    returns = np.array([r[1] for r in runs], dtype=float)
    clock = np.array([r[2] for r in runs], dtype=float)

    assert laps.std() > 1.0, "the three policies drove the same race"
    assert clock.std() / clock.mean() < 0.01, (
        "race time varied, so the first withdrawn reward would have had "
        "signal after all and this test is not measuring what it claims")

    for steps, total, _clock, start, finish in runs:
        assert total == pytest.approx(start - finish, abs=1e-9)

    # And not the lap count. A return in the tens or hundreds is the withdrawn
    # reward come back; a place delta lives in single figures.
    assert not np.allclose(returns, laps - 1.0)
    assert (np.abs(returns) <= laps.min()).all()


def test_the_episode_ends_once_and_reports_the_classification():
    e = env()
    steps, _rewards, info = drive(
        e, lambda obs, mask: STAY if mask[STAY] else FULL_TYRES, seed=SEEDS[0])
    assert steps > 50
    row = info["classification"]
    assert row["laps"] > 0 and row["class_pos"] >= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
