"""Stepping one race for the app, and the only place the app does it.

**Presentation only.** Nothing here computes a lap time, a fuel burn, a
caution or a gap. Every quantity on the screen is read off `CarState`,
`RaceState` and `LaneStatus` exactly as the engine produced them. If this
file ever computes a quantity the engine could have been asked for, 04 has
broken the rule 03a was built to keep.

**Not a second race loop.** `RaceEngine.run_stream` is the loop; this drives
it. The whole of the stepping is `_advance`, which is eleven lines and does
nothing but choose which decision to `send`.

The log is the race; the generator is a cache
---------------------------------------------
A suspended generator cannot be serialised, does not survive a Streamlit hot
reload, and cannot be rewound. So the thing this class keeps is a **decision
log** - a map of lap number to the action a human took on that lap - plus
the seed, the seat and the dials. That tuple reproduces the race exactly,
because the engine draws every random number from the seed before the race
starts (invariant 1) and the seat is deterministic given the log.

The live generator is held beside it as an optimisation for the common case,
which is stepping forwards. Any seek backwards, and any rerun that finds the
generator missing, replays the log from lap zero. One race is under a second,
so the replay is not felt; what it buys is that the app's state is JSON and
that undo is free.

**The correctness of that optimisation is exactly gate A**: a race driven
through this controller must reproduce `harness.run_focal` bit for bit, and
a race that was seeked through must equal the same race driven straight.

The seat, and the override
--------------------------
The seat is whoever is taking the focal car's decisions - a roster human, or
the trained policy through `policy.load_policy`. A human override replaces
that seat's answer on one lap, and is expressed as one of `gym_env`'s five
actions so that "what the human did" and "what the policy would have done"
are the same kind of object and can be put side by side. `to_decision` turns
it into a `PitDecision`; there is no second way of doing that.

**The override is unmasked, and that is deliberate.** `PolicyStrategy`
carries no mask either, because the engine's forced-stop rule is what holds
the agent to the rules exactly as it holds the five humans. So the panel can
correctly report that the agent would have stayed out on an empty tank. That
needs a line of text beside it, not a fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Callable

import numpy as np

from endurance.assets import BackgroundField, dials_fingerprint
from endurance.engine import (CarState, PitDecision, RaceEngine, RaceResult,
                              RaceState)
from endurance.gym_env import (N_ACTIONS, OBS_ROWS, action_mask, observe,
                               to_decision)
from endurance.harness import DEFAULT_PACE_RANK, focal_car, headline_class
from endurance.pitstop import LaneStatus
from endurance.strategies import ROSTER, _fuel_window_reached

# Imported rather than restated. The roster's notion of "the fuel window has
# opened" is a dial comparison living in one place, and a pause trigger that
# wrote its own would be a second definition of the moment the app claims to
# be pausing on - the same failure as a second simulator, one storey down.

ACTION_NAMES = ("stay out", "fill up, new tyres", "fill up, same tyres",
                "fuel to the flag, new tyres", "fuel to the flag, same tyres")
assert len(ACTION_NAMES) == N_ACTIONS


# ----------------------------------------------------------------------
# What the app renders at a pause
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Pause:
    """Why the run stopped here. `kind` is for the code, `detail` for the page."""

    kind: str
    detail: str = ""


@dataclass
class Frame:
    """One crossing of the line by the focal car, as the page sees it.

    Everything on it was read from the engine. `obs` and `mask` are the
    agent's own view of this moment, built by `gym_env` rather than by
    anything here, so the explainability panel is showing the policy what it
    would actually have been shown.
    """

    lap: int
    t: float
    car: CarState
    state: RaceState
    forced: str
    lane: LaneStatus
    obs: np.ndarray
    mask: np.ndarray
    pauses: tuple[Pause, ...] = ()

    @property
    def observation(self) -> dict[str, float]:
        """The ten rows under their names, for the panel."""
        return dict(zip(OBS_ROWS, (float(v) for v in self.obs)))


# ----------------------------------------------------------------------
# Pause triggers
# ----------------------------------------------------------------------
def triggers(frame_car: CarState, state: RaceState, forced: str,
             lane: LaneStatus, previous: Frame | None) -> tuple[Pause, ...]:
    """The three the blueprint names, each read rather than approximated.

    A caution *called* and a pit window *opening* are both transitions, so
    they need the previous frame; on the first frame there is no transition
    to see and only the standing conditions can fire. `previous` is the
    focal car's last crossing rather than the last lap of the race, which is
    the only granularity this stage has - a caution that starts and ends
    between two crossings is not shown, and cannot be.
    """
    out: list[Pause] = []
    cls = state.config.class_by_name(frame_car.class_name)

    was_caution = previous.state.under_caution if previous else False
    if state.under_caution and not was_caution:
        out.append(Pause("caution", "a full-course yellow is out"))

    was_open = previous.lane.open if previous else True
    if lane.open and not was_open:
        out.append(Pause("pit_window", "the pit lane has opened to this class"))
    elif not lane.open and was_open:
        out.append(Pause("lane_shut", lane.reason))

    if forced:
        out.append(Pause("forced", f"the rules are taking this one: {forced}"))
    elif _fuel_window_reached(frame_car, cls):
        out.append(Pause("fuel", "the car is inside its fuel window"))

    return tuple(out)


# ----------------------------------------------------------------------
# The controller
# ----------------------------------------------------------------------
@dataclass
class RaceController:
    """One race, stepped at the focal car's crossings.

    `seat` is a factory rather than a strategy, matching `compare_roster`'s
    convention: a replay builds a fresh one, so a seat that happens to carry
    per-race state cannot leak it across a seek.
    """

    config: object
    field: BackgroundField
    seed: int
    seat: Callable[[], Callable[[CarState, RaceState], PitDecision]]
    seat_name: str = "fuel_window"
    class_name: str | None = None
    pace_rank: int = DEFAULT_PACE_RANK
    log: dict[int, int] = dc_field(default_factory=dict)

    focal: str = ""
    frame: Frame | None = None
    result: RaceResult | None = None
    _stream: object | None = None
    _seat: object | None = None
    _previous: Frame | None = None

    def __post_init__(self) -> None:
        self.class_name = self.class_name or headline_class(self.config)
        self.focal = focal_car(self.config, self.seed, self.class_name,
                               self.pace_rank)
        self.reset()

    # -- the race --------------------------------------------------------
    def reset(self) -> None:
        """Back to lap zero, keeping the log. Replay starts here."""
        self.close()
        self._seat = self.seat()
        engine = RaceEngine(self.config, seed=self.seed)
        self._stream = engine.run_stream(self.field.resolve(focal=self.focal),
                                         focal=self.focal)
        self.result = None
        self._previous = None
        self.frame = self._frame(next(self._stream))

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    @property
    def finished(self) -> bool:
        return self.result is not None

    @property
    def lap(self) -> int:
        return self.frame.lap if self.frame else 0

    # -- stepping --------------------------------------------------------
    def step(self, action: int | None = None) -> Frame | None:
        """One crossing forward. `action` overrides the seat on this lap.

        The override is written to the log before it is taken, so the same
        race comes back out of a replay. Returns the new frame, or `None` at
        the flag.
        """
        if self.finished:
            return None
        if action is not None:
            self.log[self.lap] = int(action)
        return self._advance()

    def run(self, max_laps: int = 10_000) -> Frame | None:
        """Forward until something is worth stopping for, or the flag.

        `max_laps` is a guard rather than a feature: an app that has lost
        its trigger conditions should stop rather than run a whole race
        under a spinner.
        """
        for _ in range(max_laps):
            frame = self._advance()
            if frame is None or frame.pauses:
                return frame
        return self.frame

    def finish(self) -> RaceResult:
        """Drive to the flag and hand back the result the charts want."""
        while not self.finished:
            self._advance()
        return self.result

    def seek(self, lap: int) -> Frame | None:
        """Any lap, forwards or back, by replaying the log.

        Backwards is a replay from zero rather than a rewind, because there
        is nothing to rewind: the generator has consumed its noise streams.
        Deterministic, so the race that comes back is the same one.
        """
        if lap < self.lap or self._stream is None:
            self.reset()
        while self.lap < lap and not self.finished:
            self._advance()
        return self.frame

    def _advance(self) -> Frame | None:
        """Send one decision. The whole of the stepping.

        The seat is asked once per crossing and its answer is used or
        discarded here, never both. A strategy carrying state must not see
        a lap twice because the page rendered it twice.
        """
        if self.finished:
            return None
        car, state = self.frame.car, self.frame.state

        if self.lap in self.log:
            decision = to_decision(self.log[self.lap], car, state)
        else:
            decision = self._seat(car, state)
        if decision is None:
            raise ValueError(f"no decision from seat {self.seat_name!r}")

        self._previous = self.frame
        try:
            self.frame = self._frame(self._stream.send(decision))
        except StopIteration as done:
            self.result = done.value
            self._stream = None
            return None
        return self.frame

    def _frame(self, yielded) -> Frame:
        car, state, forced, lane = yielded
        return Frame(lap=car.laps_done, t=state.t, car=car, state=state,
                     forced=forced, lane=lane,
                     obs=observe(car, state),
                     mask=action_mask(state, forced),
                     pauses=triggers(car, state, forced, lane, self._previous))

    # -- what the seat would have done, for the override panel -----------
    def seat_decision(self) -> PitDecision | None:
        """The seat's answer at the current frame, taken without advancing.

        Only safe to call on a stateless seat, which `PolicyStrategy` is and
        the roster's five are not guaranteed to be. So the panel calls this
        for the *policy* - which is the comparison 04 promises - and never
        for a human seat mid-race.
        """
        if self.frame is None or self.finished:
            return None
        return self._seat(self.frame.car, self.frame.state)

    # -- state that survives a rerun -------------------------------------
    def to_dict(self) -> dict:
        """Everything needed to rebuild this race, and nothing that cannot
        be written to JSON. The dials fingerprint travels with it so a
        session restored against recalibrated dials is caught rather than
        silently continued."""
        return {"series_code": self.config.series_code,
                "dials_fingerprint": dials_fingerprint(self.config),
                "seed": self.seed, "focal": self.focal,
                "seat_name": self.seat_name, "class_name": self.class_name,
                "pace_rank": self.pace_rank,
                "log": {str(k): v for k, v in self.log.items()},
                "lap": self.lap}

    @classmethod
    def from_dict(cls, d: dict, config, field: BackgroundField,
                  seat: Callable) -> "RaceController":
        actual = dials_fingerprint(config)
        if actual != d.get("dials_fingerprint"):
            raise ValueError(
                f"this session was stepping dials {d.get('dials_fingerprint')!r} "
                f"and the config now loaded is {actual!r}. Same seed, "
                f"different race - start a new one.")
        ctrl = cls(config=config, field=field, seed=int(d["seed"]), seat=seat,
                   seat_name=d["seat_name"], class_name=d["class_name"],
                   pace_rank=int(d["pace_rank"]),
                   log={int(k): int(v) for k, v in d["log"].items()})
        ctrl.seek(int(d["lap"]))
        return ctrl


def roster_seat(name: str) -> Callable:
    """A roster member as a seat factory, by name."""
    if name not in ROSTER:
        raise KeyError(f"{name!r} is not in the roster: {sorted(ROSTER)}")
    return ROSTER[name]


def policy_seat(strategy) -> Callable:
    """A loaded policy as a seat factory.

    Shared rather than reloaded, for `policy.agent_roster`'s reason: the
    checkpoint is expensive to read and `PolicyStrategy` holds no per-race
    state, so one instance is both correct and reproducible.
    """
    return lambda: strategy
