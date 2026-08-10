"""The per-race reference: the best a plan could have done on this exact race.

This is a *reference*, not a strategy. Nothing here is ever handed to the
engine as a live decision-maker, nothing here appears in `BASELINES`, and
the agent never observes any of it. The one callable in this module,
`PlanRunner`, is a recording being played back: it holds a fixed list of
stops computed for one seed and reads nothing about the race it is in. Hand
it a different seed and it will run the wrong plan confidently, which is
exactly why it is not a strategy.

Two stages, per the 02 decision record
--------------------------------------
Stage one, `search_plans`: a dynamic program over stop plans minimising the
focal car's own race time against the frozen caution timeline. Stage two,
`rescore`: the top-*k* time-optimal plans replayed through the full engine
against the frozen rival field, ranked on class position. Position is not an
additive per-lap cost, so the DP cannot be run on it directly; the two
stages are how a position-valued reference stays tractable. Both stages are
02b's. `harness.py` is 02c's paired comparison and is a different thing.

Legality is stage two's job. Stage one cannot guarantee that a stop lands in
an open lane, because it places its stops against the arrival times of the
reduced race and the engine's differ. So `rescore` *discards* any plan the
engine had to refuse a stop for, and the benchmark is the best surviving
plan. A plan is legal when it has been run, not when it has been computed.

Why the DP is indexed by time and not by lap
--------------------------------------------
`engine._next_lap` asks `cautions.is_caution(t)` at the moment the lap
starts, and a caution lap burns less fuel and ages no tyres. Two plans
arrive at lap 400 minutes apart, so whether lap 400 is green is a property
of the plan and not of the lap number. Time therefore has to sit in the
state. That is affordable because the reachable set is narrow: at any lap
the achievable arrival times span a few stops' worth of seconds, so the DP
carries a sparse frontier keyed on
`(fuel, tyre age, stops, driver clock, time bucket)` and keeps the best few
labels in each.

What the reduced model leaves out, and why that is honest
--------------------------------------------------------
Two of the engine's lap-time terms cannot be computed for one car alone.
`_caution_lap` prices a caution lap off the gap to whoever is ahead in the
queue, and `_traffic_penalty` reads the whole field's current pace. Stage
one therefore runs a *reduced* race: pace, degradation, the car's own lap
noise, the regulation pit cost, and a caution lap at the safety car's own
lap time. It is exact when cautions and traffic are off - which is what the
brute-force gate checks, and what `test_dp_matches_the_engine_exactly`
checks against the engine itself - and approximate once they are on. Stage
two is what makes the answer honest, and it is a further argument for a
generous *k*: the ordering stage one produces is a good ordering, not a
true one.

What stage one is currently worth, measured rather than argued
--------------------------------------------------------------
On a six-hour IMSA-shaped race the reduced model's arrival times run ahead
of the engine's by 90 to 230 seconds and growing, and every plan it proposes
is refused a stop somewhere. Two omissions account for it, both
field-dependent and both larger than anything stage one is trying to
optimise:

* **Wave-arounds, about 800 seconds.** The focal car took ten wave-by
  credits at a mean of 80 s where this model charges a full safety-car lap
  of 160 s. A wave is a timing credit handed to a lapped car, so whether it
  arrives depends on the field and on the plan, and stage one cannot know.
* **Compression, about 185 seconds.** Caution laps averaged 156 s against
  the 160 s charged here, over 47 of them.

Until that is closed, `search_plans` orders plans on a quantity that differs
from the engine's by minutes, and the top-*k* it hands to `rescore` is not a
list worth re-scoring. This is recorded here rather than worked around.

The other thing this stage found
--------------------------------
Left free, the search buys laps by sitting in the pit lane while a caution
passes. `engine._next_lap` fixes a lap's character at the moment it starts,
so a stop that delays the next crossing past the end of a caution converts a
safety-car lap into a green one. On the gate config that is worth about
sixty seconds a time against roughly twenty-five seconds of stop, so the
unconstrained optimum takes sixteen stops where the rules require nine, runs
three fewer caution laps, and gains a lap.

Whether that is strategy or artefact is not this module's decision to take.
What this module does is refuse to hide it: `max_stops` caps the voluntary
stops the search may add above the forced minimum, it defaults to `None`
(no cap, the behaviour above), and `test_an_unconstrained_search_walks_into
_shut_lanes` holds the measurement so the cap can never look like a tuning
knob. Whatever cap is chosen is an assumption about the *search* and belongs
in the notebook's measured-versus-assumed table with everything else.

Foreknowledge
-------------
`clairvoyant=True` reads the frozen caution timeline, the car's own lap
noise stream and its pit-cost stream. It is an upper bound and is meant to
be: it knows the fourth stop will be a quick one. The causal reference,
which sees only what a strategist could see, lives alongside it and the gap
between the two is the value of foreknowledge - but with noise clairvoyance
in as well, that gap is two kinds of hindsight added together, so the
notebook reports a third row with the cautions known and the noise in
expectation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import Compat, PitDecision, RaceEngine, run_race
from .params import RaceConfig
from .pitstop import stop_cost


# The refuel targets the search is allowed to ask for. A grid rather than a
# continuum because the DP has to enumerate, and a coarse one because a
# splash is a strategic act rather than a fine adjustment. It is a property
# of the *search*, not of the model: widening it can only find better plans,
# so its sensitivity belongs next to k's in the notebook.
DEFAULT_FILL_LEVELS = (0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class Stop:
    """One stop, taken at the line at the end of `after_lap`."""

    after_lap: int
    refuel_to: float
    change_tyres: bool


@dataclass(frozen=True)
class Plan:
    """A finished plan and what the reduced race says it was worth.

    Ranked the way `RaceResult.classification` ranks: laps first, then time.
    `race_time_s` is the last line crossing, which is what `classification`
    reads, rather than the car's internal clock after a final stop.
    """

    stops: tuple[Stop, ...]
    laps: int
    race_time_s: float
    forced_closed_lane_stops: int = 0
    voluntary_closed_lane_stops: int = 0
    caution_stops: int = 0

    @property
    def family(self) -> tuple[int, int]:
        """What kind of plan this is, rather than how good stage one thinks it is.

        Two plans with the same number of stops, the same number of them
        taken under caution, are the same idea executed a lap apart. Two
        with different counts are different ideas. Stage one can tell those
        apart reliably; it cannot tell a lap apart, so the selection below
        picks across families rather than down the ordering.
        """
        return (len(self.stops), self.caution_stops)

    @property
    def sort_key(self) -> tuple[int, float]:
        return (-self.laps, self.race_time_s)

    def runner(self, defer: bool = True) -> "PlanRunner":
        return PlanRunner({s.after_lap: s for s in self.stops}, defer=defer)


@dataclass
class PlanRunner:
    """Plays back a fixed plan. Deliberately not a strategy.

    It cannot react, cannot see the race, and is correct only for the seed
    its plan was computed on. Kept out of `strategies.BASELINES` so it can
    never be picked up as a roster entry by accident.

    `defer` is the one thing it does that looks like a decision and is not.
    Stage one places its stops against the reduced race's arrival times, and
    the engine's differ by enough to land a stop in a window that has not
    opened yet - one that opens a caution lap later, or one that never opens
    at all. A crew that arrives to a shut lane comes round again; it does
    not abandon the stop. So a deferred stop is taken at the next crossing
    where the lane is open, which is the plan being executed rather than
    revised. Nothing is looked ahead at: `state.pit_lane_open` is the same
    thing a marshal shows the driver.

    `deferred` counts how often that happened, and it is the number to read
    rather than a legality flag. Deferring makes every plan executable, so
    a plan can no longer fail the lane gate - which means the gate stops
    being informative and this count takes over the job. If it is large,
    stage one is placing stops badly and the cost turns up in the position.
    """

    stops: dict[int, Stop]
    defer: bool = True
    pending: Stop | None = None
    deferred: int = 0
    _last_lap: int = -1

    def __call__(self, car, state) -> PitDecision:
        if car.laps_done < self._last_lap:      # a fresh race on a used runner
            self.pending, self.deferred = None, 0
        self._last_lap = car.laps_done

        stop = self.stops.get(car.laps_done)
        if stop is None and self.pending is not None and self.defer:
            if state.pit_lane_open:
                stop, self.pending = self.pending, None
            else:
                return PitDecision(pit=False)
        if stop is None:
            return PitDecision(pit=False)

        if self.defer and not state.pit_lane_open:
            self.pending = stop
            self.deferred += 1
            return PitDecision(pit=False)
        return PitDecision(pit=True, refuel_to=stop.refuel_to,
                           change_tyres=stop.change_tyres,
                           reason="benchmark plan")


# ----------------------------------------------------------------------
# What the focal car sees
# ----------------------------------------------------------------------
class FocalContext:
    """Everything the reduced race needs about one car on one seed.

    This is the only place that reaches into the engine's internals, and it
    does so deliberately: the per-car noise streams and the caution timeline
    are *read* from an engine built on the same seed rather than redrawn
    here. Redrawing would mean a second implementation of the seeding rules,
    which is the sort of duplication that agrees for a year and then quietly
    stops agreeing.
    """

    def __init__(self, config: RaceConfig, seed: int, car_id: str,
                 compat: Compat | None = None):
        engine = RaceEngine(config, seed=seed, compat=compat or Compat())
        engine._build_field()
        if car_id not in engine.cars:
            raise KeyError(f"no car {car_id!r} in this race")

        self.config = config
        self.seed = seed
        self.car_id = car_id
        self.engine = engine
        self.car = engine.cars[car_id]
        self.class_name = self.car.class_name
        self.cls = config.class_by_name(self.class_name)
        self.rules = engine.rules
        self.cautions = engine.cautions
        self.duration_s = config.duration_s
        self.base_pace_s = self.car.base_pace_s
        self.floor_s = self.base_pace_s * 0.8
        self.sc_lap_s = engine._caution_lap_s()
        # The two levels the reduced race is allowed to be wrong about, and
        # the two `anchor` measures off a reference run. Defaults are the
        # unanchored model: the safety car's own lap, and no traffic.
        self.caution_lap_s = self.sc_lap_s
        self.green_offset_s = 0.0
        self.anchor_report: dict | None = None
        self.max_driver_stint_s = config.max_driver_stint_s
        self._lap_z = engine._lap_noise.get(car_id)
        self._pit_z = engine._pit_noise.get(car_id)

    # -- the two random streams, or zero when they are not being read ----
    def lap_noise(self, lap_index: int, clairvoyant: bool) -> float:
        if not clairvoyant or self._lap_z is None:
            return 0.0
        return self._lap_z[lap_index] * self.cls.lap_noise_s

    def pit_noise(self, stop_index: int, clairvoyant: bool) -> float:
        if not clairvoyant or self._pit_z is None:
            return 0.0
        return self._pit_z[stop_index] * self.cls.pit_time_std_s

    def lane_open(self, t: float) -> bool:
        """Delegated to `pitstop.lane_status` through the engine.

        Gate two exists because the search must exclude closed windows in
        both series. Asking the engine means the benchmark and the race
        cannot disagree about when the lane is shut.
        """
        return self.engine._lane(self.class_name, t).open

    # -- one lap of the reduced race -------------------------------------
    def lap(self, laps_done: int, tyre_age: int, t: float,
            clairvoyant: bool) -> tuple[float, float, int]:
        """Lap time, fuel burnt and tyre wear for the lap starting at `t`."""
        cls = self.cls
        if self.cautions.is_caution(t):
            # Anchored to what this car's caution laps actually cost when the
            # race was run, because compression and wave credits are
            # field-dependent and no single-car model can derive them.
            return max(self.caution_lap_s, self.floor_s), cls.fuel_per_lap_caution, 0
        lap_s = (self.base_pace_s
                 + cls.deg_slope_s_per_lap * tyre_age
                 + self.lap_noise(laps_done, clairvoyant)
                 + self.green_offset_s)
        return max(lap_s, self.floor_s), cls.fuel_per_lap, 1

    def forced_reason(self, fuel: float, tyre_age: int, driver_s: float) -> str:
        """`engine._must_pit`, on the reduced state. Same order, same tests."""
        cls = self.cls
        if fuel < cls.fuel_per_lap:
            return "out of fuel"
        if tyre_age >= cls.tyre_life_laps:
            return "tyres done"
        if driver_s >= self.max_driver_stint_s:
            return "driver change"
        return ""

    def pit_cost(self, fuel: float, refuel_to: float, change_tyres: bool,
                 n_stops: int, t: float, clairvoyant: bool) -> float:
        """`engine._apply_pit`, arithmetic for arithmetic."""
        fuel_added = max(min(max(refuel_to, 0.0), 1.0) - fuel, 0.0)
        mean = stop_cost(self.cls, self.rules, fuel_added, change_tyres)
        cost = mean + self.pit_noise(n_stops, clairvoyant)
        if self.cautions.is_caution(t):
            cost *= (1.0 - self.cls.pit_caution_discount)
        return max(cost, 0.0)


# ----------------------------------------------------------------------
# The search
# ----------------------------------------------------------------------
@dataclass
class _Label:
    """One reachable way of being partway through the race."""

    t: float
    fuel: float
    tyre_age: int
    n_stops: int
    driver_s: float
    stops: tuple[Stop, ...]
    forced_closed: int
    voluntary_closed: int
    caution_stops: int = 0


@dataclass
class SearchReport:
    """What the search did, so a plan is never read without its provenance."""

    plans: list[Plan]
    laps_expanded: int
    labels_expanded: int
    peak_frontier: int
    pruned_by_cap: bool

    def best(self) -> Plan | None:
        return self.plans[0] if self.plans else None


def _actions(ctx: FocalContext, forced: str, lane_open: bool, fuel: float,
             fill_levels) -> list[tuple[bool, float, bool]]:
    """Every decision available at this line crossing.

    Returned as `(pit, refuel_to, change_tyres)`. Two constraints are the
    engine's rather than the search's: a stop the car merely wants is
    refused at a shut lane, and `_apply_pit` *sets* fuel to `refuel_to`, so
    asking for less than the car already has throws fuel away. Neither is
    worth letting the DP discover.
    """
    if forced:
        tyres_options = (True,) if forced == "tyres done" else (True, False)
        out = []
        for level in fill_levels:
            if level < fuel:
                continue
            for tyres in tyres_options:
                out.append((True, level, tyres))
        if not out:                      # tank fuller than every grid level
            out.append((True, 1.0, forced == "tyres done"))
        return out

    out: list[tuple[bool, float, bool]] = [(False, 0.0, False)]
    if not lane_open:
        return out
    for level in fill_levels:
        if level < fuel:
            continue
        for tyres in _tyre_options(ctx, level - fuel):
            # A stop that neither fuels nor changes tyres is a lap of the
            # pit lane for nothing.
            if level <= fuel and not tyres:
                continue
            out.append((True, level, tyres))
    return out


def _tyre_options(ctx: FocalContext, fuel_added: float) -> tuple[bool, ...]:
    """Both, unless the tyres are free - in which case only fresh ones.

    Under IMSA's four-over-the-wall rule the tyre change happens while fuel
    goes in, so on any stop where the refuel is the longer job the tyres
    cost nothing and declining them is strictly worse. Dropping the choice
    there is exact rather than a heuristic, and it halves the branching and
    collapses the tyre-age dimension of the state. Under WEC the jobs are
    sequential and this never fires, which is the rulebook difference doing
    real work rather than decorating a comment.
    """
    with_tyres = stop_cost(ctx.cls, ctx.rules, max(fuel_added, 0.0), True)
    without = stop_cost(ctx.cls, ctx.rules, max(fuel_added, 0.0), False)
    return (True,) if with_tyres <= without + 1e-12 else (True, False)


def search_plans(config: RaceConfig, seed: int, car_id: str, *,
                 clairvoyant: bool = True,
                 k: int = 20,
                 fill_levels: tuple[float, ...] = DEFAULT_FILL_LEVELS,
                 labels_per_state: int = 1,
                 time_bucket_s: float | None = None,
                 driver_bucket_s: float | None = None,
                 fuel_quantum: float = 0.10,
                 per_family: int = 2,
                 max_stops: int | None = None,      # see the module docstring
                 max_frontier: int = 20000,
                 compat: Compat | None = None,
                 ctx: FocalContext | None = None) -> SearchReport:
    """Top-*k* stop plans for one car on one seed, on the reduced race.

    The defaults are coarse on purpose. A finer key does not buy a better
    answer here: at `labels_per_state=4, fuel_quantum=0.02` the frontier
    runs into `max_frontier` and gets truncated by arrival time, which is a
    greedy cut nobody chose, and the search takes three times as long to
    deliver it. The coarse settings complete without truncating. A search
    that finishes at low resolution is easier to defend than one that was
    cut off at high resolution.

    `time_bucket_s` is the one approximation worth knowing about. Two labels
    at the same lap with different arrival times face different caution
    futures, so time is part of the dominance key; bucketing it coarsens
    that. With no cautions on the timeline the future does not depend on
    arrival time at all, so the bucket is widened to the whole race and the
    result is exact - which is the case the brute-force gate covers.
    """
    ctx = ctx or FocalContext(config, seed, car_id, compat=compat)
    duration = ctx.duration_s
    time_key = _interval_key(ctx, time_bucket_s)
    # The driver clock matters only through whether it has reached the
    # limit, so it is keyed in eighths of that limit. At sixty seconds it
    # was a fine grid on the race clock in disguise, and it put the
    # frontier back where the interval key had just taken it from.
    if driver_bucket_s is None:
        driver_bucket_s = config.max_driver_stint_s / 8.0

    start = _Label(t=0.0, fuel=1.0, tyre_age=0, n_stops=0, driver_s=0.0,
                   stops=(), forced_closed=0, voluntary_closed=0)
    frontier: list[_Label] = [start]

    terminals: dict[tuple[Stop, ...], Plan] = {}
    laps_done = 0
    labels_expanded = 0
    peak = 1
    pruned = False

    while frontier:
        laps_done += 1
        nxt: dict[tuple, list[_Label]] = {}

        for label in frontier:
            labels_expanded += 1
            lap_s, burn, wear = ctx.lap(laps_done - 1, label.tyre_age,
                                        label.t, clairvoyant)
            t2 = label.t + lap_s
            fuel2 = max(label.fuel - burn, 0.0)
            tyre2 = label.tyre_age + wear
            driver2 = label.driver_s + lap_s

            if t2 >= duration:
                _record(terminals, label, laps_done, t2)
                continue

            forced = ctx.forced_reason(fuel2, tyre2, driver2)
            lane_open = ctx.lane_open(t2)

            can_stop = max_stops is None or label.n_stops < max_stops
            for pit, level, tyres in _actions(ctx, forced, lane_open, fuel2,
                                              fill_levels):
                if pit and not forced and not can_stop:
                    continue
                if not pit:
                    _offer(nxt, _Label(t2, fuel2, tyre2, label.n_stops, driver2,
                                       label.stops, label.forced_closed,
                                       label.voluntary_closed,
                                       label.caution_stops),
                           time_key, driver_bucket_s, labels_per_state,
                           ctx.cls.fuel_per_lap, fuel_quantum)
                    continue

                change = bool(tyres or forced == "tyres done")
                cost = ctx.pit_cost(fuel2, level, change, label.n_stops, t2,
                                    clairvoyant)
                t3 = t2 + cost
                # A stop that only pushes the car past the flag buys nothing
                # and would clutter the top-k with plans that tie.
                if t3 >= duration and not forced:
                    continue
                driver3 = 0.0 if driver2 >= ctx.max_driver_stint_s else driver2
                stops = label.stops + (Stop(laps_done, level, change),)
                forced_closed = label.forced_closed + int(
                    bool(forced) and not lane_open)
                voluntary_closed = label.voluntary_closed + int(
                    not forced and not lane_open)
                caution_stops = label.caution_stops + int(ctx.cautions.is_caution(t2))

                if t3 >= duration:
                    _record(terminals,
                            _Label(t3, level, 0 if change else tyre2,
                                   label.n_stops + 1, driver3, stops,
                                   forced_closed, voluntary_closed,
                                   caution_stops),
                            laps_done, t2)
                    continue

                _offer(nxt, _Label(t3, level, 0 if change else tyre2,
                                   label.n_stops + 1, driver3, stops,
                                   forced_closed, voluntary_closed,
                                   caution_stops),
                       time_key, driver_bucket_s, labels_per_state,
                       ctx.cls.fuel_per_lap, fuel_quantum)

        frontier = [lab for labels in nxt.values() for lab in labels]
        if len(frontier) > max_frontier:
            pruned = True
            frontier.sort(key=lambda lab: lab.t)
            frontier = frontier[:max_frontier]
        peak = max(peak, len(frontier))

    plans = _select(terminals.values(), k, per_family)
    return SearchReport(plans=plans, laps_expanded=laps_done,
                        labels_expanded=labels_expanded, peak_frontier=peak,
                        pruned_by_cap=pruned)


def _interval_key(ctx: FocalContext, sub_bucket_s: float | None):
    """Which caution interval a time falls in, optionally subdivided."""
    import bisect

    bounds = sorted({b for period in ctx.cautions.periods for b in period})

    def key(t: float):
        i = bisect.bisect_right(bounds, t)
        return i if sub_bucket_s is None else (i, int(t // sub_bucket_s))

    return key


def _select(plans, k: int, per_family: int) -> list[Plan]:
    """Take the best few of each family first, then fill from the ordering.

    Ranking by predicted time and taking the top *k* looks right and is
    close to useless here. On a six-hour race the top twenty plans differ by
    about six seconds of predicted time while the reduced race's own error
    is nearer thirty, and what decides the outcome is whether the final lap
    falls before or after the flag - a margin of seconds. So the ordering
    inside that band carries no information and deep *k* buys twenty
    spellings of one idea.

    Breadth is what stage two can use. Families are chosen first, so a
    twelve-stop plan and a nine-stop plan both get looked at, and only then
    is the remainder filled from the time ordering.
    """
    ordered = sorted(plans, key=lambda p: p.sort_key)
    chosen: list[Plan] = []
    seen: dict[tuple[int, int], int] = {}
    for plan in ordered:
        n = seen.get(plan.family, 0)
        if n < per_family:
            seen[plan.family] = n + 1
            chosen.append(plan)
        if len(chosen) >= k:
            return chosen
    picked = {p.stops for p in chosen}
    for plan in ordered:
        if len(chosen) >= k:
            break
        if plan.stops not in picked:
            chosen.append(plan)
    return chosen


def _record(terminals: dict, label: _Label, laps: int, crossing_t: float) -> None:
    """Finish a plan. `crossing_t` is what `classification` will read."""
    plan = Plan(stops=label.stops, laps=laps, race_time_s=crossing_t,
                forced_closed_lane_stops=label.forced_closed,
                voluntary_closed_lane_stops=label.voluntary_closed,
                caution_stops=label.caution_stops)
    best = terminals.get(plan.stops)
    if best is None or plan.sort_key < best.sort_key:
        terminals[plan.stops] = plan


def _offer(frontier: dict, label: _Label, time_key, 
           driver_bucket_s: float, labels_per_state: int,
           fuel_per_lap: float, fuel_quantum: float) -> None:
    """Add a label to the frontier, keeping the best few of its kind.

    The key is everything the future depends on: how much fuel and tyre the
    car has, which pit-noise draw its next stop will read, how close the
    driver is to a mandatory change, and roughly when it is. Labels sharing
    a key are ordered by time, and only the leading few survive - which is
    where the *k*-best plans come from rather than a second pass.

    Exact fuel stays on the label because the engine's arithmetic needs it,
    but the key is coarse in two deliberate components: whole laps of fuel
    remaining, which is what decides when a stop is forced, and a coarse
    level, which is what decides the next fill's cost. Keying on the raw
    float instead makes every rounding difference a separate state and the
    frontier grows until the cap truncates it.

    `time_key` is the same argument applied to the clock. What matters about
    when a car is somewhere is which side of a caution boundary it is on -
    between two boundaries the flag sequence ahead is fixed and arriving
    sooner is better - so the clock is keyed by caution interval rather than
    by a fixed number of seconds. On a six-hour race that is about
    twenty-five buckets where a sixty-second grid was three hundred and
    sixty, and it is the more faithful of the two.
    """
    key = (int(label.fuel / fuel_per_lap), int(label.fuel / fuel_quantum),
           label.tyre_age, label.n_stops,
           int(label.driver_s // driver_bucket_s),
           time_key(label.t))
    bucket = frontier.setdefault(key, [])
    if len(bucket) < labels_per_state:
        bucket.append(label)
        bucket.sort(key=lambda lab: lab.t)
        return
    if label.t < bucket[-1].t:
        bucket[-1] = label
        bucket.sort(key=lambda lab: lab.t)


def anchor(ctx: FocalContext, background=None, plan: "Plan | None" = None,
           compat: Compat | None = None) -> dict:
    """Measure the two levels the reduced race cannot derive, and set them.

    Same convention `pitstop.py` works to: the level is measured and only
    the shape is modelled. A caution lap's length depends on the gap to the
    car ahead and on whether a wave credit arrives, both of which are
    properties of the field; a green lap carries a traffic penalty that is
    the same. Neither is recoverable from one car, so both are read off one
    reference run of the race in question rather than guessed at.

    `plan` is the focal car's reference behaviour. Starting from
    `forced_only_plan` and re-anchoring on the plan the search then returns
    is a two-pass fixed point, which is what `build_benchmark` does.
    """
    from .strategies import RunToFuelWindow

    background = background or RunToFuelWindow()
    focal = plan.runner() if plan is not None else background
    result = run_race(ctx.config, strategies={ctx.car_id: focal},
                      default_strategy=background, seed=ctx.seed, compat=compat)
    mine = result.laps[result.laps["car_id"] == ctx.car_id]

    caution = mine[mine["under_caution"]]
    green = mine[~mine["under_caution"]]

    # Green: the residual against what the reduced race would have charged.
    # Taken as a residual rather than as `traffic_s` so that anything else
    # the engine adds to a green lap is caught too, rather than only the
    # term that was remembered.
    residuals = []
    for row in green.itertuples():
        predicted = (ctx.base_pace_s
                     + ctx.cls.deg_slope_s_per_lap * row.tyre_age
                     + ctx.lap_noise(int(row.lap) - 1, True))
        residuals.append(row.lap_time - max(predicted, ctx.floor_s))

    report = {
        "caution_laps": int(len(caution)),
        "wave_laps": int(mine["wave_by"].sum()) if "wave_by" in mine else 0,
        "caution_lap_s": float(caution["lap_time"].mean()) if len(caution) else None,
        "sc_lap_s": ctx.sc_lap_s,
        "green_laps": int(len(green)),
        "green_offset_s": float(sum(residuals) / len(residuals)) if residuals else 0.0,
    }
    if report["caution_lap_s"] is not None:
        ctx.caution_lap_s = report["caution_lap_s"]
    ctx.green_offset_s = report["green_offset_s"]
    ctx.anchor_report = report
    return report


def forced_only_plan(ctx: FocalContext, clairvoyant: bool = True) -> Plan:
    """The plan that stops only when the rules or the tank make it stop.

    Not a strategy either - it is the floor the search has to beat, and the
    number of stops in it is what `max_stops` is counted from. It is also
    the cheapest possible check that the reduced race is running at all.
    """
    t = 0.0
    fuel, tyre, stops, driver, laps = 1.0, 0, 0, 0.0, 0
    taken: list[Stop] = []
    forced_closed = 0
    while True:
        lap_s, burn, wear = ctx.lap(laps, tyre, t, clairvoyant)
        t += lap_s
        laps += 1
        fuel = max(fuel - burn, 0.0)
        tyre += wear
        driver += lap_s
        if t >= ctx.duration_s:
            return Plan(tuple(taken), laps, t, forced_closed, 0)
        reason = ctx.forced_reason(fuel, tyre, driver)
        if not reason:
            continue
        forced_closed += int(not ctx.lane_open(t))
        t += ctx.pit_cost(fuel, 1.0, True, stops, t, clairvoyant)
        taken.append(Stop(laps, 1.0, True))
        fuel, tyre, stops = 1.0, 0, stops + 1
        if driver >= ctx.max_driver_stint_s:
            driver = 0.0
        if t >= ctx.duration_s:
            return Plan(tuple(taken), laps, t, forced_closed, 0)


# ----------------------------------------------------------------------
# Stage two: what the plan is worth against the real field
# ----------------------------------------------------------------------
@dataclass
class ScoredPlan:
    """One plan, run for real. `legal` is the gate, not a warning."""

    plan: Plan
    class_pos: int
    overall_pos: int
    laps: int
    race_time_s: float
    stops: int
    refused_stops: int
    lane_closed_stops: int
    deferred_stops: int = 0

    @property
    def legal(self) -> bool:
        return self.refused_stops == 0

    @property
    def sort_key(self) -> tuple[int, float]:
        return (self.class_pos, self.race_time_s)


@dataclass
class RescoreReport:
    """The re-scored field of candidates, and what happened to the rest.

    `discarded` is not bookkeeping. If it is most of `scored`, stage one is
    proposing plans the race will not accept and the search wants looking
    at rather than the number wants reporting.
    """

    scored: list[ScoredPlan]
    discarded: int

    @property
    def legal(self) -> list[ScoredPlan]:
        return sorted([s for s in self.scored if s.legal], key=lambda s: s.sort_key)

    def best(self) -> ScoredPlan | None:
        legal = self.legal
        return legal[0] if legal else None

    def time_optimal(self) -> ScoredPlan | None:
        """The plan stage one liked best, whatever stage two made of it.

        Reported alongside `best()` because the distance between the two is
        the whole reason stage two exists: it is compression turning a
        quick plan into a worse-placed one.
        """
        return self.scored[0] if self.scored else None


def rescore(config: RaceConfig, seed: int, car_id: str, plans,
            background=None, compat: Compat | None = None) -> RescoreReport:
    """Run each plan through the full engine and rank the survivors on position.

    The focal car plays its plan back; every other car runs the frozen
    background strategy, unchanged from plan to plan, so the only thing
    moving between these races is the focal car's stops. That is the same
    pairing argument the seed bank rests on, applied within one seed.
    """
    from .strategies import RunToFuelWindow

    background = background or RunToFuelWindow()
    scored: list[ScoredPlan] = []
    discarded = 0

    for plan in plans:
        runner = plan.runner()
        result = run_race(config, strategies={car_id: runner},
                          default_strategy=background, seed=seed, compat=compat)
        mine = result.laps[result.laps["car_id"] == car_id]
        refused = (int(mine["stop_refused"].notna().sum())
                   if "stop_refused" in mine.columns else 0)
        closed = (int(mine["lane_closed_stop"].sum())
                  if "lane_closed_stop" in mine.columns else 0)
        row = result.classification().set_index("car_id").loc[car_id]
        entry = ScoredPlan(plan=plan, class_pos=int(row["class_pos"]),
                           overall_pos=int(row["overall_pos"]),
                           laps=int(row["laps"]),
                           race_time_s=float(row["race_time_s"]),
                           stops=int(row["stops"]), refused_stops=refused,
                           lane_closed_stops=closed,
                           deferred_stops=runner.deferred)
        scored.append(entry)
        discarded += int(not entry.legal)

    return RescoreReport(scored=scored, discarded=discarded)


@dataclass
class BenchmarkResult:
    """The reference for one race, and everything needed to distrust it."""

    best: ScoredPlan | None
    rescored: RescoreReport
    anchor_passes: list[dict]
    search: SearchReport
    reference: ScoredPlan
    winner_rank: int | None

    @property
    def beat_the_reference(self) -> bool:
        if self.best is None:
            return False
        return self.best.sort_key <= self.reference.sort_key


def build_benchmark(config: RaceConfig, seed: int, car_id: str, *,
                    clairvoyant: bool = True, k: int = 20,
                    background=None, compat: Compat | None = None,
                    passes: int = 2, **search_kw) -> BenchmarkResult:
    """Anchor, search, re-anchor on what the search found, search again, score.

    Two passes because the anchor is measured off a reference run, and the
    plan the search returns is not the plan that reference run used - so the
    caution laps it will actually meet are not quite the ones that were
    measured. Re-anchoring on the search's own answer closes most of that.
    A third pass has not been worth its runtime on anything tried so far.

    The forced-only plan is always scored alongside the top *k*. It is a
    feasible plan, so a reference that loses to it is not a reference; and
    where it wins, `winner_rank` says how far down the time ordering the
    position-optimal plan actually sat, which is the statistic that sizes
    *k* rather than a number chosen in advance.
    """
    ctx = FocalContext(config, seed, car_id, compat=compat)
    reference_plan = forced_only_plan(ctx, clairvoyant)
    reports = [anchor(ctx, background=background, plan=reference_plan, compat=compat)]

    search = search_plans(config, seed, car_id, clairvoyant=clairvoyant, k=k,
                          ctx=ctx, **search_kw)
    for _ in range(max(passes - 1, 0)):
        if not search.plans:
            break
        reports.append(anchor(ctx, background=background, plan=search.plans[0],
                              compat=compat))
        search = search_plans(config, seed, car_id, clairvoyant=clairvoyant,
                              k=k, ctx=ctx, **search_kw)

    candidates = list(search.plans)
    if reference_plan.stops not in {p.stops for p in candidates}:
        candidates.append(reference_plan)
    scored = rescore(config, seed, car_id, candidates, background=background,
                     compat=compat)

    reference = next(e for e in scored.scored if e.plan.stops == reference_plan.stops)
    best = scored.best()
    rank = None
    if best is not None:
        order = [p.stops for p in candidates]
        rank = order.index(best.plan.stops) + 1
    return BenchmarkResult(best=best, rescored=scored, anchor_passes=reports,
                           search=search, reference=reference, winner_rank=rank)


# ----------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------
def brute_force(config: RaceConfig, seed: int, car_id: str, *,
                clairvoyant: bool = True,
                k: int = 20,
                fill_levels: tuple[float, ...] = DEFAULT_FILL_LEVELS,
                max_stops: int = 4,
                compat: Compat | None = None,
                ctx: FocalContext | None = None) -> list[Plan]:
    """Every plan, enumerated, ranked. Only tractable on a short race.

    This exists to be disagreed with. The 02 decision record is explicit
    that the same gate caught a benchmark in the F1 work which passed a
    total-time comparison while reconstructing a suboptimal plan - so the
    test compares the plans, not only their times.
    """
    ctx = ctx or FocalContext(config, seed, car_id, compat=compat)
    duration = ctx.duration_s
    out: dict[tuple[Stop, ...], Plan] = {}

    def walk(label: _Label, laps_done: int) -> None:
        laps_done += 1
        lap_s, burn, wear = ctx.lap(laps_done - 1, label.tyre_age, label.t,
                                    clairvoyant)
        t2 = label.t + lap_s
        fuel2 = max(label.fuel - burn, 0.0)
        tyre2 = label.tyre_age + wear
        driver2 = label.driver_s + lap_s

        if t2 >= duration:
            _record(out, label, laps_done, t2)
            return

        forced = ctx.forced_reason(fuel2, tyre2, driver2)
        lane_open = ctx.lane_open(t2)
        for pit, level, tyres in _actions(ctx, forced, lane_open, fuel2,
                                          fill_levels):
            if not pit:
                walk(_Label(t2, fuel2, tyre2, label.n_stops, driver2,
                            label.stops, label.forced_closed,
                            label.voluntary_closed), laps_done)
                continue
            if label.n_stops >= max_stops:
                continue
            change = bool(tyres or forced == "tyres done")
            cost = ctx.pit_cost(fuel2, level, change, label.n_stops, t2,
                                clairvoyant)
            t3 = t2 + cost
            if t3 >= duration and not forced:
                continue
            driver3 = 0.0 if driver2 >= ctx.max_driver_stint_s else driver2
            stops = label.stops + (Stop(laps_done, level, change),)
            nxt = _Label(t3, level, 0 if change else tyre2, label.n_stops + 1,
                         driver3, stops,
                         label.forced_closed + int(bool(forced) and not lane_open),
                         label.voluntary_closed + int(not forced and not lane_open))
            if t3 >= duration:
                _record(out, nxt, laps_done, t2)
                continue
            walk(nxt, laps_done)

    walk(_Label(0.0, 1.0, 0, 0, 0.0, (), 0, 0), 0)
    return sorted(out.values(), key=lambda p: p.sort_key)[:k]
