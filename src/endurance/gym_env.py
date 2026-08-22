"""The engine, adapted to the standard RL API. Nothing else lives here.

**Zero physics.** Every lap time, fuel burn, caution and gap is computed by
the engine and read here. If this file ever computes a quantity the engine
could have been asked for, the architecture has failed: there is one
simulator in this project and it is `engine.py`.

How the agent gets a turn
-------------------------
`RaceEngine.run_stream` is a generator that suspends at the focal car's
decisions and resumes on `send`. `step` therefore drives one race forward to
the focal car's next crossing and hands back what it finds there.

The agent is not privileged
---------------------------
It is asked the same question at the same moment through the same
`(CarState, RaceState) -> PitDecision` interface the roster uses, and the
engine cannot tell it apart from a human strategy. `_must_pit` is untouched.
What this file adds is a **mask**: the forced action is removed from the
agent's choices rather than its answer being taken and discarded, so the
policy is never trained on a decision it does not own. That is a training
convenience and nothing more, which is why `PolicyStrategy` at the foot of
this module - the path `harness.compare_roster` scores through - has none.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .assets import BackgroundField, SeedBank
from .engine import CarState, PitDecision, RaceEngine, RaceState
from .harness import DEFAULT_PACE_RANK, focal_car, headline_class
from .strategies import fuel_to_the_flag

# The five actions span exactly what the roster can express and no more.
# There is no refuel *level*: a level is a number, and a number the agent
# chooses is a tuning surface the human strategies were forbidden.
STAY, FULL_TYRES, FULL_KEEP, FLAG_TYRES, FLAG_KEEP = range(5)
N_ACTIONS = 5

OBS_ROWS = ("race_progress", "fuel", "tyre_age", "gap_ahead", "gap_behind",
            "under_caution", "stint_laps", "laps_down", "pit_lane_open",
            "class_position")
N_OBS = len(OBS_ROWS)

STINT_SCALE = 40.0        # laps, per the blueprint's observation table
LAPS_DOWN_SCALE = 3.0

# What the reward is, as a name rather than a sentence somebody retypes.
# `train.py` records this on every policy card. At 03b the card carried a
# string literal instead, and it would have gone on claiming the superseded
# reward after this amendment landed - a provenance file that lies is worse
# than one that is absent, because nobody checks it. Amended twice now, which
# is the argument for the constant.
REWARD = "places gained in class, credited as they change"


# ----------------------------------------------------------------------
# The three pure functions the env is built from
# ----------------------------------------------------------------------
def gap_scale_s(cls) -> float:
    """What one unit of gap means, in seconds.

    The blueprint drafted a flat 120 s. Over twenty stand-in races the clip
    never binds - the widest gap is 92 s - and the median is 3.0 s, so /120
    puts the median at 0.03 and the ninety-ninth percentile at 0.36: two
    thirds of the row carries one observation in a hundred. On
    `pit_time_mean_s` those are 0.07 and 0.93, and 1.0 means a stop's worth
    of gap, which is the unit the decision is taken in. It is also a dial
    rather than a constant somebody chose. Supersedes the blueprint's table.
    """
    return max(cls.pit_time_mean_s, 1e-6)


def observe(car: CarState, state: RaceState) -> np.ndarray:
    """The ten rows, every one of them read from the engine.

    `pit_lane_open` is the ninth, on 02c's evidence: a gambler that cannot
    see the lane is not gambling. A `None` gap maps to 1.0, where the clip
    sends a very large one too: both say the same thing, which is that nobody
    is there.

    `class_position` is the tenth, added with the position reward and for the
    same reason. **The reward is a change in class position and nothing in the
    first nine rows tracked it.** Measured over 2,271 IMSA crossings,
    `laps_down` - the row that was supposed to carry position - has a standard
    deviation of 0.075 and correlates 0.091 with the car's actual class
    position; it is clipped at three laps and a front-runner sits at zero all
    race. The best proxy was `gap_ahead` at 0.52, which is local and clipped
    at one stop's worth of gap.

    That left the value function estimating "how many places will I gain from
    here" without being told where here is, so the baseline could not reduce
    any variance and every advantage was raw return noise. The IMSA policy's
    action distribution after 500,000 steps was 0.15 to 0.25 across all five
    actions - flat, with the argmax of the flatness picking a stop on every
    lap. `wave_eligible` was the tenth row this project expected to need and
    it is still absent, for the reason recorded at 03a.

    0.0 is leading the class and 1.0 is last, so the row reads the way the
    reward does: lower is better.
    """
    cls = state.config.class_by_name(car.class_name)
    scale = gap_scale_s(cls)

    def gap(v: float | None) -> float:
        return 1.0 if v is None else float(np.clip(v / scale, 0.0, 1.0))

    return np.array([
        state.race_progress,
        np.clip(car.fuel, 0.0, 1.0),
        np.clip(car.tyre_age / max(cls.tyre_life_laps, 1e-6), 0.0, 1.0),
        gap(state.gap_ahead_s(car)),
        gap(state.gap_behind_s(car)),
        1.0 if state.under_caution else 0.0,
        np.clip(car.stint_laps / STINT_SCALE, 0.0, 1.0),
        np.clip(state.laps_down(car) / LAPS_DOWN_SCALE, 0.0, 1.0),
        1.0 if state.pit_lane_open else 0.0,
        (state.class_position(car) - 1) / max(cls.n_cars - 1, 1),
    ], dtype=np.float32)


def action_mask(state: RaceState, forced: str) -> np.ndarray:
    """Which actions are available, given what the rules have already decided.

    Each rule restates something the engine would have done anyway. A
    forced stop removes *staying out*, which the engine would have
    overridden. Tyre life also removes the keep-tyre actions, because
    `_apply_pit` fits tyres on that reason whatever the decision said. A
    shut lane removes every stop unless one is forced, in which case the
    engine takes it and records `lane_closed_stop`.

    Never all-false; the assertion says so, because an empty mask hangs.
    """
    mask = np.ones(N_ACTIONS, dtype=bool)
    if forced:
        mask[STAY] = False
        if forced == "tyres done":
            mask[[FULL_KEEP, FLAG_KEEP]] = False
    elif not state.pit_lane_open:
        mask[[FULL_TYRES, FULL_KEEP, FLAG_TYRES, FLAG_KEEP]] = False
    assert mask.any(), "empty action mask"
    return mask


def to_decision(action: int, car: CarState, state: RaceState) -> PitDecision:
    """One action, as the `PitDecision` any strategy would have returned."""
    if action == STAY:
        return PitDecision(pit=False, reason="agent: stay out")
    cls = state.config.class_by_name(car.class_name)
    to_flag = action in (FLAG_TYRES, FLAG_KEEP)
    refuel_to = fuel_to_the_flag(car, state, cls) if to_flag else 1.0
    tyres = action in (FULL_TYRES, FLAG_TYRES)
    return PitDecision(pit=True, refuel_to=refuel_to, change_tyres=tyres,
                       reason=f"agent: {'flag' if to_flag else 'full'}"
                              f"{'+tyres' if tyres else ''}")


# ----------------------------------------------------------------------
# The environment
# ----------------------------------------------------------------------
class EnduranceEnv(gym.Env):
    """One focal car in one race, stepped at its own crossings.

    A step is one lap of the focal car, not one of the race, and its length
    in race time varies - a stop or a caution makes a long one. **The reward
    is the class positions gained on that lap**, credited as they change.
    Nothing else.

    Amended twice. The original was elapsed time negated; 03b replaced it
    with one lap per completed lap; this replaces that. Both superseded
    versions are kept below, because each was wrong in a way worth knowing.

    **Why not negative time.** Summed over an episode those elapsed times are
    the length of the race, and this is a *timed* race, so the return was
    `duration_s / base_pace_s` whatever the policy did. Over 2,664 training
    episodes it was -219.85 +- 0.42 while laps ran 167 to 206, correlated
    -0.11 with them; 500,000 steps moved it by 0.016%.

    **Why not laps.** Laps looked like the fix, and the argument for it was
    that class position derives from laps and then time, so laps sit near the
    score. Measured on the calibrated dials, they do not. A car that stops
    forty extra times at Daytona spends 1,072 s more in the pits and loses
    *zero* laps: caution compression hands ~70% of that time back, because a
    car with a gap ahead runs shorter caution laps until it closes. The
    refund tracks the caution rate - 70% at 0.355, 35% at 0.089, 8% at 0.018 -
    so the proxy detaches from the score exactly where there is most yellow.
    The IMSA policy took 65 stops against the null's 24.5 and was scored, in
    laps, as having lost nothing at all, while its class position fell by one
    and it lost in 52.5% of races. In WEC the same 51 extra stops did cost 9
    laps, but 9 laps is 1.3 sd of the seed-to-seed spread there and would be
    0.27 sd at Daytona - so the gradient is either absent or drowned.

    **So the reward is the score.** Decision 1 makes class position the
    primary score; this credits it directly rather than through a proxy that
    the caution model can sever. Per lap: the position before the lap minus
    the position after, so gaining a place is +1. At the flag the last credit
    is taken against `classification()`, which is the number every table in
    this project reports.

    **It is dense and it is still the terminal score.** The per-lap deltas
    telescope: summed over an episode they are `start_position - final
    position`, and the starting position is fixed by the pace-rank draw,
    which is independent of strategy. So the return differs from crediting
    only the final position by a per-episode constant, and a constant is the
    same policy gradient - the identical argument amendment 8 used, now
    pointing somewhere else.

    A stop needs no term of its own. It costs whatever it costs in places,
    which is the thing being scored, and where the field hands the time back
    it genuinely was cheap.

    **The observation needed a tenth row and now has one.** The first version
    of this reward shipped with the nine rows unchanged, on the argument that
    `laps_down` and the two gaps determine a position change locally. Measured,
    they do not: `laps_down` correlates 0.091 with class position. See
    `observe` for what that did to the value function.

    Section 7A's DNF term is dropped: the engine models no retirement and
    inventing one here would be physics.
    """

    metadata = {"render_modes": []}

    def __init__(self, config, field: BackgroundField, bank: SeedBank,
                 class_name: str | None = None,
                 pace_rank: int = DEFAULT_PACE_RANK,
                 allow_held_out: bool = False):
        self.config = config
        self.field = field
        self.bank = bank
        self.class_name = class_name or headline_class(config)
        self.pace_rank = pace_rank
        self.allow_held_out = allow_held_out

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, (N_OBS,), dtype=np.float32)

        self._stream = None
        self._pending = None
        self.focal: str | None = None
        self.result = None

    # -- seeds -----------------------------------------------------------
    def _pick_seed(self, seed: int | None) -> int:
        """Training draws from the headline bank; held-out seeds are refused.

        Refused here rather than left to the caller: the held-out fifty exist
        to answer whether a design chosen on the headline races generalises,
        and a training loop that touches them has destroyed the only set that
        could answer it.
        """
        if seed is None:
            return int(self.np_random.choice(self.bank.headline))
        if seed in set(self.bank.held_out) and not self.allow_held_out:
            raise ValueError(f"seed {seed} is in the held-out bank; pass "
                             f"allow_held_out=True to evaluate on it")
        return int(seed)

    # -- the gym API -----------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self._stream is not None:
            self._stream.close()          # a half-run race is disposable

        race_seed = self._pick_seed(seed)
        self.focal = focal_car(self.config, race_seed, self.class_name,
                               self.pace_rank)
        engine = RaceEngine(self.config, seed=race_seed)
        self._stream = engine.run_stream(self.field.resolve(focal=self.focal),
                                         focal=self.focal)
        self.result = None
        self._pending = next(self._stream)
        return self._observation(), self._info(race_seed)

    def step(self, action: int):
        car, state, _forced, _lane = self._pending
        before_t = state.t
        before_pos = state.class_position(car)
        decision = to_decision(int(action), car, state)

        try:
            self._pending = self._stream.send(decision)
        except StopIteration as done:
            self.result = done.value
            row = self.result.classification().set_index("car_id").loc[self.focal]
            # The last credit is taken against `classification()` rather than
            # against a live state, so an episode's rewards sum to exactly the
            # places gained between the start and the finishing position every
            # table in this project reports. The live rule and the scored rule
            # are the same rule, so this is a join rather than a correction.
            reward = float(before_pos - int(row["class_pos"]))
            info = {"classification": row.to_dict(), "result": self.result,
                    "class_pos": int(row["class_pos"])}
            return (self._observation(), reward, True, False, info)

        # Places gained on this lap. Positive is a place taken. See the class
        # docstring for why laps, which this replaced, were severed from the
        # score by caution compression.
        car_after, state_after, _f, _l = self._pending
        after_pos = state_after.class_position(car_after)
        reward = float(before_pos - after_pos)

        info = self._info()
        info["step_s"] = state_after.t - before_t        # diagnostic, not scored
        return (self._observation(), reward, False, False, info)

    def action_masks(self) -> np.ndarray:
        """The mask under the name `sb3-contrib` looks for.

        No SB3 import and no new logic - `action_mask` computed against the
        pending decision, which `_info` already publishes. It is here so a
        training script needs neither a wrapper nor a private attribute.
        """
        _car, state, forced, _lane = self._pending
        return action_mask(state, forced)

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    # -- helpers ---------------------------------------------------------
    def _observation(self) -> np.ndarray:
        car, state, _forced, _lane = self._pending
        return observe(car, state)

    def _info(self, race_seed: int | None = None) -> dict:
        car, state, forced, lane = self._pending
        info = {"action_mask": action_mask(state, forced), "forced": forced,
                "lane_reason": lane.reason, "lap": car.laps_done,
                "class_pos": state.class_position(car), "focal": self.focal}
        if race_seed is not None:
            info["seed"] = race_seed
        return info


# ----------------------------------------------------------------------
# The evaluation path
# ----------------------------------------------------------------------
class PolicyStrategy:
    """A trained policy, wearing the ordinary strategy interface.

    How the agent is scored: inserted into `ROSTER` as a sixth member and
    run through `harness.compare_roster` on the same banks, field and pace
    rank as the five humans. Anything else is a second evaluation path that
    can differ from the roster's, which decision 6 exists to prevent. No mask
    here: the engine's override holds instead, as it does for the humans, so
    an agent asking to stay out on an empty tank is made to stop and scored
    on that, which is the honest number.
    """

    def __init__(self, predict, deterministic: bool = True):
        self.predict = predict
        self.deterministic = deterministic

    def __call__(self, car: CarState, state: RaceState) -> PitDecision:
        action, _ = self.predict(observe(car, state),
                                 deterministic=self.deterministic)
        return to_decision(int(action), car, state)
