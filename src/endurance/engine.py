"""The race engine: a whole multi-class field, lap by lap, in race time.

Design decisions worth knowing before reading the code
------------------------------------------------------

**The whole field is simulated, and running order falls out of cumulative
race time.** There is no separate model of "position" - a car is ahead
because it has completed more laps, or the same lap sooner. That is what
makes a live race view honest rather than decorative, and it is what lets
traffic be an emergent consequence of where cars actually are.

**Cars advance on an event queue, not a shared lap counter.** A GTP car and
an LMP2 car do not complete laps at the same moment, so a global "for each
lap" loop would quietly desynchronise the field. Instead the engine always
advances whichever car finishes its current lap soonest, which keeps every
car's position well-defined at every moment of race time.

**All randomness is drawn in advance, at the start of the race.** The
caution timeline in particular is generated up front rather than sampled
lap by lap. This costs nothing and buys a great deal: any two strategies
can be compared against *exactly the same race*, so a difference between
them is a difference in the strategy and not in the luck. That property is
what makes the strategy comparison in 02 and the RL evaluation in 03 fair,
and it is worth preserving in anything built on top of this.

**There is no overtaking model.** Cars pass by being faster over a lap.
Traffic is modelled as time lost when a car is close behind slower cars,
which is the part of the effect that lap-time data can speak to. A genuine
pass-probability model would need parameters no timing sheet contains.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .params import RaceConfig
from .caution import (
    CautionRules,
    compressed_lap_time,
    wave_eligible,
    wave_lap_time,
)
from .pitstop import PitRules, lane_status, stop_cost, transit_s


# ----------------------------------------------------------------------
# Which engine you are running
# ----------------------------------------------------------------------
# Stream identifiers. Every generator in the engine is keyed off the seed
# and one of these, so no two concerns can ever consume from the same
# stream by accident.
_STREAM_CAUTION = 0
_STREAM_FIELD = 1
_STREAM_LAP_NOISE = 2
_STREAM_PIT_COST = 3


@dataclass(frozen=True)
class Compat:
    """Which behaviours are 01's and which are 02a's.

    Each flag names one 02a change and can be switched off independently.
    They are deliberately not one master switch: the notebook has to turn
    the changes on one at a time to show what each did, which is the same
    argument that kept the caution and field streams separate.

    A flag whose change has not landed yet defaults to the legacy
    behaviour, because that is the only behaviour the engine has. Setting
    such a flag to False raises rather than silently doing nothing - a
    switch that quietly has no effect is how a stage becomes a different
    stage. As each change lands, its default flips and its guard goes.
    """

    legacy_cautions: bool = False    # merging draw, calibrated in laps
    split_streams: bool = True       # caution and field on their own streams
    legacy_noise: bool = False       # one shared rng, consumed in queue order
    legacy_pit: bool = False         # single pit_time_mean_s, lane always open
    legacy_caution_pace: bool = False  # no compression, no wave-arounds
    legacy_traffic: bool = False     # blockers judged on base pace alone

    # Every 02a change has landed, so nothing is guarded here any more. The
    # mechanism stays: the next stage that adds a switch adds it to this map
    # so a flag can never be set for a change that does not exist yet.
    _NOT_YET = {}          # name -> the change it is waiting on

    def __post_init__(self) -> None:
        for flag, change in self._NOT_YET.items():
            if not getattr(self, flag):
                raise NotImplementedError(
                    f"{flag}=False needs {change}, which has not landed yet")

    @classmethod
    def v01(cls) -> "Compat":
        """The engine notebook 01 validated. The regression gate runs on this."""
        return cls(legacy_cautions=True, split_streams=False, legacy_noise=True,
                   legacy_pit=True, legacy_caution_pace=True, legacy_traffic=True)


class NoiseStream:
    """One car's own supply of standard normals, indexed rather than consumed.

    The point of the index is that it does not move. Lap noise is asked for
    by lap number and pit noise by stop number, so a car that pits earlier
    reads a different *entry* but never shifts anyone else's - which is what
    makes two strategies comparable on one seed.

    Values are standard normal and scaled at the point of use, so twisting
    `lap_noise_s` or `pit_time_std_s` changes the size of the noise without
    redrawing it. A sweep of those dials is therefore paired too.
    """

    def __init__(self, seed: int, stream_id: int, car_idx: int, size: int,
                 block: int = 256):
        self._rng = np.random.default_rng([seed, stream_id, car_idx])
        self._block = max(int(block), 1)
        self._z = self._rng.standard_normal(max(int(size), self._block))

    def __getitem__(self, i: int) -> float:
        # Growing draws the next values from the same generator, so a stream
        # that had to grow holds exactly what a longer one would have held.
        while i >= len(self._z):
            self._z = np.concatenate([self._z, self._rng.standard_normal(self._block)])
        return float(self._z[i])

    def __len__(self) -> int:
        return len(self._z)


# ----------------------------------------------------------------------
# Cautions, drawn in advance
# ----------------------------------------------------------------------
@dataclass
class CautionTimeline:
    """Every caution period of the race, fixed before a wheel is turned."""

    periods: list[tuple[float, float]] = field(default_factory=list)

    def is_caution(self, t: float) -> bool:
        for start, end in self.periods:
            if start <= t < end:
                return True
            if start > t:
                break
        return False

    def total_caution_s(self) -> float:
        return sum(end - start for start, end in self.periods)

    @classmethod
    def draw(cls, duration_s: float, caution_rate: float, mean_dur_s: float,
             rng: np.random.Generator, legacy: bool = False) -> "CautionTimeline":
        """Draw a caution timeline matching a target share and mean length.

        The race alternates between green and caution: green gaps are
        exponential, caution episodes are exponential with the calibrated
        mean, and a caution can only begin when none is running. Episodes
        therefore cannot overlap, so nothing has to be merged or rejected.

        Two properties follow, and both are load-bearing later. The realised
        caution share is `mean_dur_s / (mean_dur_s + mean_gap_s)` exactly, so
        the mean gap is solved for rather than tuned. And episode lengths
        stay exactly exponential, so the remaining duration of an ongoing
        caution is memoryless - which is what makes the causal benchmark in
        02b exactly computable rather than something to approximate.

        The old draw scattered uniform starts and merged the overlaps. The
        union of overlapping exponentials is not exponential, so the merge
        destroyed the memorylessness, and it destroyed caution time too: the
        overlap that got collapsed was time the race never ran under caution,
        leaving the realised share well short of the calibrated one.

        `legacy=True` restores that behaviour so 01's numbers can be
        reproduced before this change is turned on.
        """
        if caution_rate <= 0 or mean_dur_s <= 0:
            return cls([])
        if legacy:
            return cls._draw_legacy(duration_s, caution_rate, mean_dur_s, rng)
        if caution_rate >= 1.0:
            return cls([(0.0, duration_s)])

        mean_gap_s = mean_dur_s * (1.0 - caution_rate) / caution_rate

        periods: list[tuple[float, float]] = []
        t = 0.0
        # Start the process in its stationary state: a race is as likely to
        # open under caution as any other moment is to be under caution.
        # Without this the timeline always begins green and the realised
        # share comes in about one percentage point light.
        if rng.random() < caution_rate:
            end = min(rng.exponential(mean_dur_s), duration_s)
            periods.append((0.0, end))
            t = end

        while True:
            t += rng.exponential(mean_gap_s)
            if t >= duration_s:
                break
            end = min(t + rng.exponential(mean_dur_s), duration_s)
            periods.append((t, end))
            t = end

        return cls(periods)

    @classmethod
    def _draw_legacy(cls, duration_s: float, caution_rate: float,
                     mean_dur_s: float, rng: np.random.Generator) -> "CautionTimeline":
        """Notebook 01's caution draw, kept verbatim for the regression gate."""
        expected_episodes = (caution_rate * duration_s) / mean_dur_s
        n = rng.poisson(max(expected_episodes, 0.0))
        if n == 0:
            return cls([])

        starts = np.sort(rng.uniform(0, duration_s, size=n))
        lengths = rng.exponential(mean_dur_s, size=n)

        merged: list[list[float]] = []
        for s, ln in zip(starts, lengths):
            e = min(s + ln, duration_s)
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return cls([(a, b) for a, b in merged])

    def merge_rate(self) -> float:
        """Share of adjacent episode pairs that touch - should be zero now.

        Kept as a diagnostic rather than deleted: reporting it once at the
        calibrated caution rate is what shows the old merge was doing real
        damage rather than tidying up a rare edge case.
        """
        if len(self.periods) < 2:
            return 0.0
        touching = sum(1 for (a, b), (c, _) in zip(self.periods, self.periods[1:])
                       if c <= b)
        return touching / (len(self.periods) - 1)


# ----------------------------------------------------------------------
# Car state
# ----------------------------------------------------------------------
@dataclass
class CarState:
    car_id: str
    class_name: str
    base_pace_s: float

    laps_done: int = 0
    race_time_s: float = 0.0
    tyre_age: int = 0
    fuel: float = 1.0
    stint_number: int = 0
    stint_laps: int = 0
    n_stops: int = 0
    driver_idx: int = 0
    driver_stint_s: float = 0.0

    lap_start_t: float = 0.0
    lap_expected_s: float = 0.0
    finished: bool = False

    def track_fraction(self, t: float) -> float:
        """How far round the lap this car is at time `t`, as 0.0 to 1.0."""
        if self.lap_expected_s <= 0:
            return 0.0
        return float(np.clip((t - self.lap_start_t) / self.lap_expected_s, 0.0, 1.0))


@dataclass
class PitDecision:
    """What a strategy asks for at the end of a lap."""

    pit: bool = False
    refuel_to: float = 1.0        # target fuel level, 0.0 to 1.0
    change_tyres: bool = True
    reason: str = ""


def _running_order(car: "CarState") -> tuple[int, float]:
    """The one sort key position is derived from, everywhere.

    Most laps completed, and among equals whoever got there soonest. Written
    once because four things read it - the class leader, the focal car's live
    position, `RaceResult.positions` and `classification` - and a project whose
    score is a position cannot afford two definitions of it.
    """
    return (-car.laps_done, car.race_time_s)


@dataclass
class RaceState:
    """The read-only view of the race a strategy gets to see."""

    t: float
    duration_s: float
    under_caution: bool
    cars: dict[str, CarState]
    config: RaceConfig
    # Whether this car's class may enter the pits right now. A strategy that
    # cannot see the lane cannot gamble on a caution, it can only be lucky.
    pit_lane_open: bool = True
    pit_lane_reason: str = ""

    def class_leader(self, class_name: str) -> CarState | None:
        """Whoever is leading that class right now, by the usual rule.

        Position is derived here as everywhere else: most laps completed,
        and among equals whoever got there soonest.
        """
        in_class = [c for c in self.cars.values() if c.class_name == class_name]
        if not in_class:
            return None
        return min(in_class, key=_running_order)

    def class_position(self, car: CarState) -> int:
        """Where this car stands in its class right now, one-based.

        The same rule as `class_leader`, `RaceResult.positions` and
        `classification` - most laps, then earliest - and it shares their
        sort key rather than restating it, so the four cannot drift apart.
        `class_leader` is this function's rank one, kept separate only
        because it is called every lap and a linear scan beats a sort.

        Added at the position reward. Nothing before it needed the focal
        car's own standing mid-race: `run_focal` scores at the flag and
        deliberately avoids `positions`, which sorts the whole field once
        per lap record. This sorts one class once per focal crossing, which
        is eleven to twenty-five cars a few hundred times a race.

        **Mid-race this wobbles, and that is the rule rather than a defect.**
        Cars in `self.cars` are each at their own last crossing, so a car that
        has just completed a lap outranks one nine tenths of the way round it -
        the trap `caution.wave_eligible` avoids by using fractional progress,
        and it avoids it because eligibility is a fact about the road while
        this is a fact about the timing screen. A screen ranks by laps
        completed and does show a place changing hands on a crossing. Two
        consequences worth knowing: a reward built on the *change* in this
        wobbles by a place either way lap to lap, and it sums exactly to the
        finishing position anyway, because the intermediate terms cancel and
        the final one is taken from `classification`.
        """
        in_class = sorted((c for c in self.cars.values()
                           if c.class_name == car.class_name),
                          key=_running_order)
        for i, other in enumerate(in_class, start=1):
            if other.car_id == car.car_id:
                return i
        raise KeyError(f"{car.car_id} is not in class {car.class_name!r}")

    def laps_down(self, car: CarState) -> int:
        """How many laps this car is behind its class leader, never negative.

        Lives here rather than in each strategy that wants it: the
        lap-down defender, the wave-around rule and the agent's observation
        all have to agree on what "a lap down" means, and they can only do
        that by asking the same function.
        """
        leader = self.class_leader(car.class_name)
        if leader is None:
            return 0
        return max(leader.laps_done - car.laps_done, 0)

    def _arrival_at_my_lap(self, car: CarState, other: CarState) -> float | None:
        """When `other` crossed, or will cross, the lap `car` has just finished.

        The projection section 5 requires, in the one place both the roster
        and the agent can reach it. At the line a car is on lap *N* and its
        slower class rivals are still on lap *N - 1*, so differencing two
        `race_time_s` values compares crossings of different laps and comes
        out negative for cars that are plainly behind.

        A car level on laps has already crossed, at `race_time_s`. One lap
        down is on that lap now and arrives at `lap_start_t + lap_expected_s`.
        One lap up crossed it when its current lap began, which is
        `lap_start_t` exactly. Anything further away cannot be resolved into
        a gap worth reading and is left out.
        """
        down = car.laps_done - other.laps_done
        if down == 0:
            return other.race_time_s
        if down == 1:
            return other.lap_start_t + other.lap_expected_s
        if down == -1:
            return other.lap_start_t
        return None

    def _gaps(self, car: CarState) -> tuple[float | None, float | None]:
        """Seconds to the nearest class rival ahead and behind, or None.

        Restricted to the car's own class on purpose: the score is class
        position, so a car of another class on the same piece of tarmac is
        traffic rather than a rival, and the engine already prices traffic.
        `None` means nobody is within a lap either side, which is a
        different statement from a large gap and is left for the caller to
        represent.
        """
        ahead = behind = None
        for other in self.cars.values():
            if other.finished or other.car_id == car.car_id:
                continue
            if other.class_name != car.class_name:
                continue
            arrives = self._arrival_at_my_lap(car, other)
            if arrives is None:
                continue
            if arrives < self.t:
                if ahead is None or arrives > ahead:
                    ahead = arrives
            elif arrives > self.t:
                if behind is None or arrives < behind:
                    behind = arrives
        return (None if ahead is None else self.t - ahead,
                None if behind is None else behind - self.t)

    def gap_ahead_s(self, car: CarState) -> float | None:
        return self._gaps(car)[0]

    def gap_behind_s(self, car: CarState) -> float | None:
        return self._gaps(car)[1]

    @property
    def time_remaining_s(self) -> float:
        return max(self.duration_s - self.t, 0.0)

    @property
    def race_progress(self) -> float:
        return min(self.t / self.duration_s, 1.0) if self.duration_s else 0.0


# ----------------------------------------------------------------------
# The engine
# ----------------------------------------------------------------------
class RaceEngine:
    """Run one race.

    `strategies` maps a car id to a callable taking (CarState, RaceState) and
    returning a PitDecision. Any car without a strategy uses `default_strategy`.
    """

    def __init__(self, config: RaceConfig, seed: int = 0,
                 legacy_cautions: bool | None = None,
                 split_streams: bool | None = None,
                 compat: Compat | None = None):
        """`compat=Compat.v01()` reproduces the engine notebook 01 validated.

        The switches inside `Compat` are separate on purpose. Under 01's
        single shared generator the caution draw and the field build consume
        from the same stream, so changing the caution model shifts every
        car's base pace as a side effect and the comparison stops being
        about cautions. With the streams split, the caution model is the
        only thing that moves, which is what makes "here is what the change
        did" a real claim.

        `legacy_cautions` and `split_streams` survive as keywords because
        01's cells and the existing tests pass them.
        """
        if compat is not None and (legacy_cautions is not None
                                   or split_streams is not None):
            raise TypeError("pass compat, or the individual keywords, not both")
        if compat is None:
            compat = Compat(
                legacy_cautions=bool(legacy_cautions),
                split_streams=True if split_streams is None else bool(split_streams),
            )

        self.config = config
        self.seed = seed
        self.compat = compat
        self.cars: dict[str, CarState] = {}
        self._car_pace: dict[str, float] = {}
        self._lap_noise: dict[str, NoiseStream] = {}
        self._pit_noise: dict[str, NoiseStream] = {}
        self._records: list[dict] = []
        self.rules = PitRules.for_series(config.series_code)
        self.caution_rules = CautionRules.for_series(config.series_code)
        self._reset()

    # Read-only conveniences, so call sites and tests written against the
    # old keywords keep reading.
    @property
    def legacy_cautions(self) -> bool:
        return self.compat.legacy_cautions

    @property
    def split_streams(self) -> bool:
        return self.compat.split_streams

    # -- setup ----------------------------------------------------------
    def _caution_lap_s(self) -> float:
        """How long one lap behind the safety car takes, for the leading class.

        The reopening rules are written in caution laps, so a lap length is
        needed to place them on the clock. The quickest class sets it, since
        it is the leader the safety car picks up.
        """
        return min(c.base_pace_s * c.caution_pace_multiplier
                   for c in self.config.classes)

    def _lane(self, class_name: str, t: float):
        """Lane status for one class at one moment, or always open in legacy."""
        if self.compat.legacy_pit:
            from .pitstop import LaneStatus
            return LaneStatus(open=True)
        cls = self.config.class_by_name(class_name)
        return lane_status(self.rules, self.cautions, t, class_name,
                           caution_lap_s=self._caution_lap_s(),
                           duration_s=self.config.duration_s,
                           open_delay_laps=cls.caution_pits_open_delay_laps)

    def _reset(self) -> None:
        """Put the generators back to the start of the seed and redraw.

        Called from both `__init__` and `run`, and the duplication is
        load-bearing rather than accidental: with the streams shared, the
        caution draw has to consume from `self.rng` *before* the field is
        built, or the entanglement that `split_streams=False` exists to
        reproduce disappears.
        """
        self._seed_streams()
        self.cautions = self._draw_cautions()
        # Wave-arounds are frozen at announcement, so each slot is resolved
        # once and remembered rather than recomputed per car.
        self._waves_pending: dict[tuple[int, int], set[str]] = {}
        self._waves_frozen: set[tuple[int, int]] = set()

    def _seed_streams(self) -> None:
        """One generator per concern, all keyed off the same seed.

        Lap noise and pit cost are no longer among them: they are per-car
        streams built in `_build_field`, because a generator shared across
        cars is consumed in event-queue order and so depends on when
        everyone else pitted.
        """
        self.rng = np.random.default_rng(self.seed)
        if self.compat.split_streams:
            self._caution_rng = np.random.default_rng([self.seed, _STREAM_CAUTION])
            self._field_rng = np.random.default_rng([self.seed, _STREAM_FIELD])
        else:
            self._caution_rng = self.rng
            self._field_rng = self.rng

    def _draw_cautions(self) -> CautionTimeline:
        rates = [c.caution_rate for c in self.config.classes]
        durs = [c.caution_mean_dur_s for c in self.config.classes]
        return CautionTimeline.draw(
            self.config.duration_s,
            float(np.mean(rates)) if rates else 0.0,
            float(np.mean(durs)) if durs else 0.0,
            self._caution_rng,
            legacy=self.legacy_cautions,
        )

    def _build_field(self) -> None:
        self.cars = {}
        for cls in self.config.classes:
            for i in range(cls.n_cars):
                car_id = f"{cls.class_name}-{i + 1:02d}"
                # Each car gets its own base pace, drawn once: real fields are
                # not made of identical cars, and pace spread is what creates
                # the closing speed that makes traffic bite.
                pace = cls.base_pace_s + self._field_rng.normal(0, cls.pace_spread_s)
                self.cars[car_id] = CarState(
                    car_id=car_id, class_name=cls.class_name, base_pace_s=pace
                )
                self._car_pace[car_id] = pace
                if not self.compat.legacy_noise:
                    self._make_streams(car_id, len(self.cars) - 1, cls, pace)

        # The traffic check runs once per car per lap, so for a 62-car 24-hour
        # race it is the hot loop of the whole engine. Car state is mirrored
        # into numpy arrays so that check can be a vectorised comparison
        # against the field rather than a Python loop over it.
        self._idx = {cid: i for i, cid in enumerate(self.cars)}
        n = len(self.cars)
        self._arr_pace = np.array([c.base_pace_s for c in self.cars.values()])
        # Degradation slope and tyre age per car, so that "slower than me" can
        # be asked of the car as it is now rather than as it started.
        self._arr_deg = np.array([
            self.config.class_by_name(c.class_name).deg_slope_s_per_lap
            for c in self.cars.values()])
        self._arr_tyre = np.zeros(n)
        self._arr_lap_start = np.zeros(n)
        self._arr_lap_expected = np.ones(n)
        self._arr_active = np.ones(n, dtype=bool)

    def _make_streams(self, car_id: str, car_idx: int, cls, pace: float) -> None:
        """Two streams per car: lap noise by lap, pit cost by stop.

        Two and not one. Sharing would re-couple them - one extra stop would
        shift every later lap-noise value - which is the defect being fixed,
        moved rather than removed.

        The sizes are only a hint. A car cannot lap faster than the floor
        `_next_lap` imposes, so this bound holds for the laps, and the fuel
        window bounds the stops; either way a stream that runs out grows
        from its own generator and holds what a longer one would have.
        """
        n_laps = int(self.config.duration_s / max(pace * 0.8, 1e-6)) + 8
        n_stops = int(n_laps * cls.fuel_per_lap) + 8
        self._lap_noise[car_id] = NoiseStream(
            self.seed, _STREAM_LAP_NOISE, car_idx, n_laps)
        self._pit_noise[car_id] = NoiseStream(
            self.seed, _STREAM_PIT_COST, car_idx, n_stops)

    # -- per-lap physics -------------------------------------------------
    def _traffic_penalty(self, car: CarState, t: float) -> tuple[float, int]:
        """Time lost to cars just ahead that are slower than this one *now*.

        Only cars *ahead* and *slower* count: those are the ones a car
        actually has to deal with. A faster car ahead is disappearing, and a
        slower car behind is someone else's problem.

        "Slower" is judged on current pace, degradation included. Under 01 it
        was judged on base pace alone, which made traffic a fixed property of
        the field: the same cars obstructed you on lap two and lap two
        hundred, and no stop you made could change it.
        """
        cls = self.config.class_by_name(car.class_name)
        if cls.traffic_penalty_s <= 0:
            return 0.0, 0

        i = self._idx[car.car_id]
        fracs = np.clip((t - self._arr_lap_start) / self._arr_lap_expected, 0.0, 1.0)
        gaps = (fracs - fracs[i]) % 1.0

        ahead = (gaps > 0.0) & (gaps <= cls.traffic_window_frac)
        if self.compat.legacy_traffic:
            slower = self._arr_pace > car.base_pace_s
        else:
            # Both sides of the comparison age. A car that has just taken
            # fresh tyres is quicker than it was, so more of the field counts
            # as an obstruction - which is what makes "I will come out into
            # traffic if I stop now" a thing a strategy can weigh at all.
            others = self._arr_pace + self._arr_deg * self._arr_tyre
            mine = car.base_pace_s + cls.deg_slope_s_per_lap * car.tyre_age
            slower = others > mine
        blockers = int(np.count_nonzero(ahead & slower & self._arr_active))

        return cls.traffic_penalty_s * blockers, blockers

    def _next_lap(self, car: CarState, t: float) -> dict:
        """Work out the lap this car is about to run, starting at time `t`."""
        cls = self.config.class_by_name(car.class_name)
        under_caution = self.cautions.is_caution(t)

        wave_by = False
        if under_caution and self.compat.legacy_caution_pace:
            lap_time = car.base_pace_s * cls.caution_pace_multiplier
            fuel_burn = cls.fuel_per_lap_caution
            tyre_wear = 0          # tyres do not age meaningfully at caution pace
            traffic_s, blockers = 0.0, 0
        elif under_caution:
            lap_time, wave_by = self._caution_lap(car, cls, t)
            fuel_burn = cls.fuel_per_lap_caution
            tyre_wear = 0
            traffic_s, blockers = 0.0, 0
        else:
            traffic_s, blockers = self._traffic_penalty(car, t)
            # Indexed by the lap about to be run, so this car's noise is a
            # function of (seed, car, lap) and of nothing anyone else did.
            if self.compat.legacy_noise:
                noise = self.rng.normal(0, cls.lap_noise_s)
            else:
                noise = self._lap_noise[car.car_id][car.laps_done] * cls.lap_noise_s
            lap_time = (
                car.base_pace_s
                + cls.deg_slope_s_per_lap * car.tyre_age
                + traffic_s
                + noise
            )
            fuel_burn = cls.fuel_per_lap
            tyre_wear = 1

        return {
            "lap_time": max(lap_time, car.base_pace_s * 0.8),
            "fuel_burn": fuel_burn,
            "tyre_wear": tyre_wear,
            "under_caution": under_caution,
            "traffic_s": traffic_s,
            "blockers": blockers,
            "wave_by": wave_by,
        }

    # -- behind the safety car -------------------------------------------
    def _caution_lap(self, car: CarState, cls, t: float) -> tuple[float, bool]:
        """This car's next lap behind the safety car, and whether it is a wave.

        Everyone runs the safety car's lap, not a multiple of their own pace:
        a caution that let a quick car pull away from a slow one would be
        doing the opposite of what a caution does.
        """
        sc_lap = self._caution_lap_s()
        floor = car.base_pace_s        # nobody laps quicker than they race

        if self._take_wave(car, t):
            # Floored at a car's spacing, not at racing pace: the lap being
            # credited is one the car does not drive, so pricing it at a
            # plausible lap time would quietly refuse to hand the lap back.
            back = max((c.race_time_s for c in self.cars.values()
                        if not c.finished), default=t)
            return wave_lap_time(t, back, cls.caution_queue_gap_s,
                                 cls.caution_queue_gap_s), True

        gap = self._queue_ahead(car, sc_lap)
        if gap is None:
            return sc_lap, False       # the leader runs the safety car's lap
        return compressed_lap_time(
            sc_lap, gap_s=gap, queue_gap_s=cls.caution_queue_gap_s,
            close_frac=cls.caution_close_frac, floor_s=floor), False

    def _queue_ahead(self, car: CarState, sc_lap_s: float):
        """The gap in seconds to the car directly ahead in the queue.

        The queue behind a safety car is the running order - both rulebooks
        have it pick up the overall leader, and everyone else forms up
        behind. So the order is taken from position, and the leader has
        nobody ahead of it and simply runs the safety car's lap.

        The gap is the difference between the two cars' last line crossings,
        wrapped into one lap. Wrapping is the whole trick: crossing times are
        read at one point on the track, so a car that crossed later in clock
        time can still be the one just ahead on the road, and without the
        wrap the leader would end up chasing the backmarker round.
        """
        my_key = (-car.laps_done, car.race_time_s)
        best, best_key = None, None
        for other in self.cars.values():
            if other.finished or other.car_id == car.car_id:
                continue
            key = (-other.laps_done, other.race_time_s)
            if key < my_key and (best_key is None or key > best_key):
                best, best_key = other, key
        if best is None:
            return None

        gap = (car.race_time_s - best.race_time_s) % sc_lap_s
        return gap if gap > 0.0 else sc_lap_s

    def _take_wave(self, car: CarState, t: float) -> bool:
        """Is this car owed a lap back right now?

        Eligibility is frozen when the wave is announced and consumed once
        per car per wave, so a car cannot be waved twice through the same
        one by crossing the line again.
        """
        if self.caution_rules.n_waves <= 0:
            return False
        episode = self._episode_index(t)
        if episode is None:
            return False
        start, end = self.cautions.periods[episode]
        cls_delay = min(c.caution_pits_open_delay_laps for c in self.config.classes)
        times = self.caution_rules.wave_times(
            start, end, self._caution_lap_s(), cls_delay)

        for slot, announced_at in enumerate(times):
            if t < announced_at:
                continue
            key = (episode, slot)
            if key not in self._waves_frozen:
                self._waves_frozen.add(key)
                self._waves_pending[key] = wave_eligible(
                    self.cars, self._progress(t))
            if car.car_id in self._waves_pending[key]:
                self._waves_pending[key].discard(car.car_id)
                return True
        return False

    def _progress(self, t: float) -> dict[str, float]:
        """Laps completed plus the fraction of the current lap, per active car.

        The same fractional-progress view the traffic penalty uses, so the
        engine has one idea of where a car is rather than two.
        """
        fracs = np.clip((t - self._arr_lap_start) / self._arr_lap_expected, 0.0, 1.0)
        return {c.car_id: c.laps_done + float(fracs[self._idx[c.car_id]])
                for c in self.cars.values() if not c.finished}

    def _episode_index(self, t: float) -> int | None:
        for k, (start, end) in enumerate(self.cautions.periods):
            if start <= t < end:
                return k
        return None

    def _must_pit(self, car: CarState, cls) -> str:
        """Reasons the rules or the car itself force a stop, regardless of strategy."""
        if car.fuel < cls.fuel_per_lap:
            return "out of fuel"
        if car.tyre_age >= cls.tyre_life_laps:
            return "tyres done"
        if car.driver_stint_s >= self.config.max_driver_stint_s:
            return "driver change"
        return ""

    def _apply_pit(self, car: CarState, decision: PitDecision, t: float,
                   forced_reason: str) -> float:
        cls = self.config.class_by_name(car.class_name)
        fuel_added = max(float(np.clip(decision.refuel_to, 0.0, 1.0)) - car.fuel, 0.0)
        change_tyres = bool(decision.change_tyres or forced_reason == "tyres done")
        under_caution = self.cautions.is_caution(t)
        legacy = self.compat.legacy_pit
        mean = stop_cost(cls, self.rules, fuel_added, change_tyres,
                         under_caution=under_caution and not legacy,
                         legacy=legacy)

        # Indexed by this car's stop number, so a strategy that stops once
        # more does not reprice anybody else's stops.
        if self.compat.legacy_noise:
            cost = self.rng.normal(mean, cls.pit_time_std_s)
        else:
            z = self._pit_noise[car.car_id][car.n_stops]
            cost = mean + cls.pit_time_std_s * z

        if legacy:
            # 01's engine: one flat mean, discounted whole. Left exactly as it
            # was - a compat flag that quietly acquired amendment 14 would stop
            # being a way of showing what 02a changed.
            if under_caution:
                cost *= (1.0 - cls.pit_caution_discount)
            cost = max(cost, 0.0)
        else:
            # Amendment 14's floor, applied after the noise. A discount cannot
            # put a stop below the lane transit, and neither can a draw from
            # the left tail: `pit_time_std_s` is two to five times its mean in
            # places, so that tail is not hypothetical.
            cost = max(cost, transit_s(cls))

        car.fuel = float(np.clip(decision.refuel_to, 0.0, 1.0))
        if change_tyres:
            car.tyre_age = 0
        if forced_reason == "driver change" or car.driver_stint_s >= self.config.max_driver_stint_s:
            car.driver_idx += 1
            car.driver_stint_s = 0.0
        car.stint_number += 1
        car.stint_laps = 0
        car.n_stops += 1
        return cost

    # -- main loop -------------------------------------------------------
    def run(self, strategies: dict | None = None, default_strategy=None) -> "RaceResult":
        """Run the race to the flag and return the result.

        A drain of `run_stream` rather than a loop of its own. There is one
        race loop in this project and this is not it - if `run` kept its own
        copy, the wrapper 03a builds would be running a second simulator
        that merely resembled this one.
        """
        stream = self.run_stream(strategies, default_strategy, focal=None)
        try:
            next(stream)
        except StopIteration as done:
            return done.value
        raise RuntimeError("run_stream yielded with no focal car")   # unreachable

    def run_stream(self, strategies: dict | None = None, default_strategy=None,
                   focal: str | None = None):
        """The race loop, suspending at `focal`'s decisions if one is named.

        Yields `(car, state, forced, lane)` each time the focal car reaches
        the line, and expects a `PitDecision` back through `send`. With no
        focal car it never yields and returns the result immediately, which
        is what `run` uses.

        Written as a generator rather than a callback or a worker thread so
        that a suspended race is still one single-threaded, deterministic
        object: `close()` disposes of a half-run episode, and a traceback
        out of a policy arrives here rather than across a thread boundary.

        The focal car is treated no differently from any other. It is asked
        the same question at the same moment through the same interface;
        the answer merely arrives from further away.
        """
        from .strategies import RunToFuelWindow

        strategies = strategies or {}
        default_strategy = default_strategy or RunToFuelWindow()

        self._reset()
        self._build_field()
        self._records = []

        duration = self.config.duration_s
        queue: list[tuple[float, str]] = []

        # Each car's in-flight lap is remembered so it can be recorded when
        # the car arrives at the line, rather than recomputed there.
        pending: dict[str, dict] = {}
        for car in self.cars.values():
            plan = self._next_lap(car, 0.0)
            pending[car.car_id] = plan
            self._set_lap(car, 0.0, plan["lap_time"])
            heapq.heappush(queue, (plan["lap_time"], car.car_id))

        while queue:
            t, car_id = heapq.heappop(queue)
            car = self.cars[car_id]
            if car.finished:
                continue

            plan = pending[car_id]

            # The car has just crossed the line at time t.
            car.laps_done += 1
            car.race_time_s = t
            car.tyre_age += plan["tyre_wear"]
            car.fuel = max(car.fuel - plan["fuel_burn"], 0.0)
            car.stint_laps += 1
            car.driver_stint_s += plan["lap_time"]

            record = {
                "t": t,
                "car_id": car_id,
                "class": car.class_name,
                "lap": car.laps_done,
                "lap_time": plan["lap_time"],
                "under_caution": plan["under_caution"],
                "traffic_s": plan["traffic_s"],
                "blockers": plan["blockers"],
                "tyre_age": car.tyre_age,
                "fuel": car.fuel,
                "stint_number": car.stint_number,
                "pitted": False,
                "pit_cost_s": 0.0,
                "wave_by": plan["wave_by"],
            }

            if t >= duration:
                car.finished = True
                self._arr_active[self._idx[car_id]] = False
                self._records.append(record)
                continue

            # Decide on a stop.
            cls = self.config.class_by_name(car.class_name)
            forced = self._must_pit(car, cls)
            lane = self._lane(car.class_name, t)
            state = RaceState(t=t, duration_s=duration,
                              under_caution=self.cautions.is_caution(t),
                              cars=self.cars, config=self.config,
                              pit_lane_open=lane.open, pit_lane_reason=lane.reason)
            if car_id == focal:
                decision = yield (car, state, forced, lane)
            else:
                strat = strategies.get(car_id, default_strategy)
                decision = strat(car, state)
            if decision is None:
                raise ValueError(f"no decision returned for {car_id}")

            if forced and not decision.pit:
                decision = PitDecision(pit=True, refuel_to=1.0, change_tyres=True,
                                       reason=forced)

            # A shut lane refuses a stop the car merely wanted. A car that
            # has to stop stops anyway and is recorded as having done so:
            # the model has no penalties, so the count of these is a
            # diagnostic - if it is not rare, something upstream is wrong.
            record["lane_closed_stop"] = False
            if decision.pit and not lane.open:
                if forced:
                    record["lane_closed_stop"] = True
                else:
                    decision = PitDecision(pit=False, reason=lane.reason)
                    record["stop_refused"] = lane.reason

            if decision.pit:
                cost = self._apply_pit(car, decision, t, forced)
                t += cost
                record["pitted"] = True
                record["pit_cost_s"] = cost
                record["pit_reason"] = decision.reason or forced
                car.race_time_s = t

            self._records.append(record)

            if t >= duration:
                car.finished = True
                self._arr_active[self._idx[car_id]] = False
                continue

            nxt = self._next_lap(car, t)
            pending[car_id] = nxt
            self._set_lap(car, t, nxt["lap_time"])
            heapq.heappush(queue, (t + nxt["lap_time"], car_id))

        return RaceResult(self.config, pd.DataFrame(self._records),
                          self.cautions, self.seed)

    def _set_lap(self, car: CarState, start_t: float, expected_s: float) -> None:
        """Put a car onto a new lap, keeping the state and the arrays in step."""
        car.lap_start_t = start_t
        car.lap_expected_s = expected_s
        i = self._idx[car.car_id]
        self._arr_lap_start[i] = start_t
        self._arr_lap_expected[i] = expected_s
        # Called after every crossing and every stop, so this is the one
        # place tyre age has to be mirrored into the arrays.
        self._arr_tyre[i] = car.tyre_age


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
class RaceResult:
    """Everything that happened, in a shape that plots and tables can use."""

    def __init__(self, config: RaceConfig, laps: pd.DataFrame,
                 cautions: CautionTimeline, seed: int):
        self.config = config
        self.laps = laps.sort_values("t").reset_index(drop=True)
        self.cautions = cautions
        self.seed = seed
        self._positions: pd.DataFrame | None = None

    # -- classification --------------------------------------------------
    def positions(self) -> pd.DataFrame:
        """Running order at every lap completion, overall and within class.

        Position is derived, never simulated: at each moment a car is ranked
        by laps completed, then by how early it got there.
        """
        if self._positions is not None:
            return self._positions

        laps = self.laps
        class_of = dict(zip(laps["car_id"], laps["class"]))

        laps_done: dict[str, int] = {}
        last_t: dict[str, float] = {}
        out_overall: list[int] = []
        out_class: list[int] = []

        for car_id, lap_no, t in zip(laps["car_id"], laps["lap"], laps["t"]):
            laps_done[car_id] = lap_no
            last_t[car_id] = t

            # `_running_order`'s rule, on the tallies this loop keeps rather
            # than on `CarState`. Same key; different thing holding it.
            ranked = sorted(laps_done, key=lambda c: (-laps_done[c], last_t[c]))
            out_overall.append(ranked.index(car_id) + 1)

            my_class = class_of[car_id]
            same_class = [c for c in ranked if class_of[c] == my_class]
            out_class.append(same_class.index(car_id) + 1)

        self._positions = laps.assign(position=out_overall, class_position=out_class)
        return self._positions

    # -- summaries --------------------------------------------------------
    def classification(self) -> pd.DataFrame:
        """Final result: laps completed, stops made, time lost where."""
        g = self.laps.groupby(["car_id", "class"], as_index=False).agg(
            laps=("lap", "max"),
            race_time_s=("t", "max"),
            stops=("pitted", "sum"),
            pit_time_s=("pit_cost_s", "sum"),
            traffic_time_s=("traffic_s", "sum"),
            caution_laps=("under_caution", "sum"),
            mean_green_lap_s=("lap_time", "mean"),
        )
        g = g.sort_values(["laps", "race_time_s"], ascending=[False, True])
        g["overall_pos"] = range(1, len(g) + 1)
        g["class_pos"] = g.groupby("class").cumcount() + 1
        return g.reset_index(drop=True)

    def summary(self) -> dict:
        c = self.classification()
        return {
            "seed": self.seed,
            "duration_h": round(self.config.duration_s / 3600, 2),
            "cars": len(c),
            "caution_periods": len(self.cautions.periods),
            "caution_share": round(
                self.cautions.total_caution_s() / self.config.duration_s, 3),
            "winner": c.iloc[0]["car_id"] if len(c) else None,
            "winning_laps": int(c.iloc[0]["laps"]) if len(c) else 0,
            "mean_stops": round(float(c["stops"].mean()), 1) if len(c) else 0.0,
        }

    def gap_to_leader(self, class_name: str | None = None) -> pd.DataFrame:
        """Gap in seconds to the class leader, per car per lap - the race story."""
        laps = self.laps
        if class_name:
            laps = laps[laps["class"] == class_name]
        wide = laps.pivot_table(index="lap", columns="car_id", values="t", aggfunc="min")
        leader = wide.min(axis=1)
        return wide.sub(leader, axis=0)


def run_race(config: RaceConfig, strategies: dict | None = None,
             default_strategy=None, seed: int = 0,
             legacy_cautions: bool | None = None,
             split_streams: bool | None = None,
             compat: Compat | None = None) -> RaceResult:
    """Convenience wrapper: build an engine, run it, hand back the result."""
    return RaceEngine(config, seed=seed, legacy_cautions=legacy_cautions,
                      split_streams=split_streams, compat=compat
                      ).run(strategies, default_strategy)
