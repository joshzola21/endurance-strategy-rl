"""How a car decides when to stop.

A strategy is just a callable: given this car's state and the state of the
race, return a PitDecision. That is the whole interface, and it is
deliberately the same interface the RL agent will implement later - the
agent is not a special case bolted on, it is another strategy the engine
cannot tell apart from a human one.

Two sets live here and they are not the same set.

`BASELINES` is **background material**. It is what `assets.freeze_background`
is allowed to hand to the rest of the field, and its members carry chosen
constants - `min_fuel_used=0.5`, `stint_laps=30` - which is exactly what
decision 9 rules out of the roster. They stay because decision 2's mixed
background sweep needs something to mix: a field all running one strategy
pits in lockstep, every car carries the same tyre age, and 02a's traffic
correction cancels out of the comparison entirely.

`ROSTER` is **decision 9's five**, the ones an agent is measured against.
Parameter-free, meaning each derives its numbers from the dials rather than
being handed them. Nothing in here may be constructed with a threshold.

The two are kept apart in code rather than by convention because
`freeze_background` looks names up in `BASELINES`, so a roster strategy that
drifted into that mapping would silently become part of the field it is
supposed to be measured against.

What the roster may not see
---------------------------
02b's `anchor` - the measured caution-lap level - and anything else read off
a completed run. It is a per-seed quantity measured from a reference race,
and a strategy that consults it is being tuned per race, which this stage's
boundary constraint forbids. Everything below reads `ClassDials`,
`CarState`, `RaceState` and `pitstop.stop_cost`, and nothing else.

The consequence is deliberate and is a result rather than a limitation: a
strategist works from a model of a caution lap that the field does not
honour, and the size of that error is part of the gap between the roster and
the benchmark. It gets reported, not corrected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .engine import CarState, PitDecision, RaceState
from .pitstop import PitRules, stop_cost


# ----------------------------------------------------------------------
# Shared arithmetic
# ----------------------------------------------------------------------
# A stop is "voluntary" when nothing forces it. The defenders may decline
# only these: declining past the fuel window does not avoid the stop, it
# hands the decision to `_must_pit`, which takes a full service through a
# shut lane if it has to and records `lane_closed_stop`. A strategy that
# defends its way into that has not defended anything.
#
# One green lap in hand, which is where the existing baselines already sit.
# Not a tuned margin - it is the smallest quantity that is certainly enough,
# expressed in the dial that decides it.
def _fuel_window_reached(car: CarState, cls) -> bool:
    return car.fuel < cls.fuel_per_lap * 1.5


def _can_run_another_lap(car: CarState, state: RaceState, cls) -> bool:
    """Is there fuel to defer this stop by one lap?

    The engine forces a stop below one lap's burn, so a defence is only a
    decision while the car is above that line - below it, declining does not
    avoid the stop, it converts a voluntary one into a forced one taken
    through whatever the lane happens to be doing.

    This is the whole width of a defender's discretion, and it is narrow by
    construction: the fuel window opens at a lap and a half in hand and the
    forced point is at one lap, so a defence buys one more lap and then the
    rules take the decision back. Under caution the burn is lower and the
    lap is worth more, which is the right way round.
    """
    burn = cls.fuel_per_lap_caution if state.under_caution else cls.fuel_per_lap
    return car.fuel >= burn


def _green_lap_s(car: CarState, cls) -> float:
    """This car's green lap, as a strategist would estimate it.

    Base pace plus the degradation already on the tyres. Lap noise and
    traffic are not estimable in advance and are left out, which is the
    point: this is what the strategy believes, not what the engine will do.
    """
    return car.base_pace_s + cls.deg_slope_s_per_lap * car.tyre_age


def _caution_lap_s(car: CarState, cls) -> float:
    return car.base_pace_s * cls.caution_pace_multiplier


def _remaining(car: CarState, state: RaceState, cls) -> tuple[float, float]:
    """Laps left to the flag, and the fuel they will take.

    Both are expectations over the caution pattern rather than the realised
    timeline, because a strategy cannot see the timeline. `caution_rate` is
    a share of *time*, so the remaining time splits into a caution part and
    a green part first, and each converts to laps at its own lap length.

    This is the whole of what decision 9 item 2 means by "the threshold
    falls out of `fuel_per_lap` and the time remaining": no measured level
    enters, only dials the calibration produced and the clock.
    """
    left = state.time_remaining_s
    if left <= 0.0:
        return 0.0, 0.0

    rate = min(max(cls.caution_rate, 0.0), 1.0)
    green_lap = max(_green_lap_s(car, cls), 1e-6)
    caution_lap = max(_caution_lap_s(car, cls), 1e-6)

    green_laps = (1.0 - rate) * left / green_lap
    caution_laps = rate * left / caution_lap

    fuel = green_laps * cls.fuel_per_lap + caution_laps * cls.fuel_per_lap_caution
    return green_laps + caution_laps, fuel


def _refills_needed(fuel_aboard: float, fuel_required: float) -> int:
    """Full tanks still to be taken on to cover `fuel_required`.

    A tank is 1.0 by construction, so this is a count of stops and not a
    volume. Returned as an integer because that is what the quantity is;
    the callers that care about how near the boundary sits ask for the
    margin separately.
    """
    shortfall = fuel_required - fuel_aboard
    if shortfall <= 0.0:
        return 0
    return int(math.ceil(shortfall - 1e-9))


def _stop_seconds(car: CarState, state: RaceState, cls,
                  refuel_to: float = 1.0, change_tyres: bool = True) -> float:
    """What this stop is expected to cost, before noise.

    Priced through `pitstop.stop_cost` so the two series' sequencing rules
    reach the strategies rather than only the engine, and discounted under
    caution by the assumed dial. The discount is an assumption and gets
    swept; a defender's threshold moves with it, which is a dependence to
    label rather than to hide.
    """
    rules = PitRules.for_series(state.config.series_code)
    fuel_added = max(refuel_to - car.fuel, 0.0)
    cost = stop_cost(cls, rules, fuel_added, change_tyres)
    if state.under_caution:
        cost *= (1.0 - cls.pit_caution_discount)
    return cost


def fuel_to_the_flag(car: CarState, state: RaceState, cls) -> float:
    """The refuel level `SplashAndDashPlanner` would ask for, shared with the agent.

    Laps left convert to fuel left at the dials' burn rates, plus one green
    lap in hand for the same reason `_can_run_another_lap` uses that margin -
    the estimate is an expectation over the caution pattern, and coming up
    short costs an entire extra stop. Never below what is already aboard:
    `_apply_pit` sets fuel *to* this value, so a low ask would throw fuel
    away rather than decline to add it.
    """
    _laps_left, required = _remaining(car, state, cls)
    target = min(1.0, required + cls.fuel_per_lap)
    return max(target, car.fuel)


def _would_be_passed(car: CarState, state: RaceState, cost_s: float) -> CarState | None:
    """The nearest car in this class that a stop of `cost_s` would let past.

    Asked through the engine's own position rule rather than by measuring a
    gap, because at the decision point a gap cannot be measured. The car is
    at the line on lap *N* and every slower car in its class is still on lap
    *N - 1*, so "the car behind on the same lap" mostly does not exist, and
    where it does the difference of two `race_time_s` values is comparing
    crossings of different laps. Doing it that way fires the defence on a
    negative number about a quarter of the time and looks like a working
    strategy.

    Instead every rival is projected forward to *this* car's lap count -
    a car one lap down arrives at `lap_start_t + lap_expected_s`, which
    `CarState` already carries - and the comparison is between arrival
    times at the same lap, which is what position means here. A rival more
    than a lap down cannot be let past by one stop and is skipped.

    The projection is a snapshot: the rivals' times advance too. That is
    what a strategist has on the pit wall, and it is right to first order
    because a car a lap down arrives after this one either way.
    """
    mine_after = state.t + cost_s
    nearest, nearest_t = None, None

    for other in state.cars.values():
        if other.finished or other.car_id == car.car_id:
            continue
        if other.class_name != car.class_name:
            continue

        down = car.laps_done - other.laps_done
        if down == 0:
            arrives = other.race_time_s
        elif down == 1:
            arrives = other.lap_start_t + other.lap_expected_s
        else:
            continue                      # ahead, or too far back to matter

        # Behind now, ahead after the stop: exactly the car being defended
        # against. Everyone else is either already past or cannot get past.
        if state.t < arrives < mine_after:
            if nearest_t is None or arrives < nearest_t:
                nearest, nearest_t = other, arrives

    return nearest


# ----------------------------------------------------------------------
# Background material - NOT the roster
# ----------------------------------------------------------------------
@dataclass
class RunToFuelWindow:
    """Stop when the fuel runs out, and not before.

    The simplest thing that is not stupid, and a genuine baseline: in a
    fuel-limited formula, running the tank dry every stint is close to what
    a team does when nothing unusual is happening. It ignores cautions
    entirely, which is exactly the weakness the next strategy exploits.

    It is also the one member of both sets. As background it is what decision
    10 measures every paired delta from; as the roster's null it is the same
    idea, which is deliberate - the field a strategy is measured against and
    the plan it is measured against should not be two different things.
    """

    def __call__(self, car: CarState, state: RaceState) -> PitDecision:
        cls = state.config.class_by_name(car.class_name)
        if _fuel_window_reached(car, cls):
            return PitDecision(pit=True, refuel_to=1.0, change_tyres=True,
                               reason="fuel window")
        return PitDecision(pit=False)


@dataclass
class OpportunistUnderCaution:
    """Take a cheap stop if a caution appears late enough in the stint.

    Background material only. `min_fuel_used` is a chosen constant, which is
    what keeps this out of the roster - the caution gambler below is the
    parameter-free strategy that asks the same question properly.

    How much cheaper a caution stop really is is the `pit_caution_discount`
    assumption in the dials - not measurable from lap data. So this strategy
    beating the fuel-window baseline is partly a statement about that
    assumption, and the size of its advantage should always be read against
    a sweep of that value rather than taken at face value.
    """

    min_fuel_used: float = 0.5

    def __call__(self, car: CarState, state: RaceState) -> PitDecision:
        cls = state.config.class_by_name(car.class_name)

        if _fuel_window_reached(car, cls):
            return PitDecision(pit=True, refuel_to=1.0, change_tyres=True,
                               reason="fuel window")

        if state.under_caution and (1.0 - car.fuel) >= self.min_fuel_used:
            return PitDecision(pit=True, refuel_to=1.0, change_tyres=True,
                               reason="caution opportunity")

        return PitDecision(pit=False)


@dataclass
class FixedLapStint:
    """Stop every `stint_laps` laps, come what may.

    Deliberately inflexible, and background material for the same reason as
    the opportunist: the stint length is handed to it. Its job is to show
    how much value reacting to circumstances actually adds, and to give the
    mixed background field something that pits out of phase with everything
    else.
    """

    stint_laps: int = 30

    def __call__(self, car: CarState, state: RaceState) -> PitDecision:
        cls = state.config.class_by_name(car.class_name)
        if _fuel_window_reached(car, cls):
            return PitDecision(pit=True, refuel_to=1.0, change_tyres=True,
                               reason="fuel window")
        if car.stint_laps >= self.stint_laps:
            return PitDecision(pit=True, refuel_to=1.0, change_tyres=True,
                               reason="fixed stint")
        return PitDecision(pit=False)


# ----------------------------------------------------------------------
# The roster - decision 9's five
# ----------------------------------------------------------------------
@dataclass
class CautionGambler:
    """Take a caution stop only when it is free in stops.

    The threshold is a *count*, not a price, and that distinction is the
    reason this strategy needs nothing 02b measured. A caution stop is
    cheap, but taking one with three quarters of a tank aboard throws that
    fuel away, and fuel thrown away is a stop added to the race. So the
    question is not "how much does this stop save" - which needs the
    anchored caution level - but "does taking it now change how many more
    times I have to stop before the flag", which is arithmetic on
    `fuel_per_lap`, `caution_rate` and the clock.

    Stop now and the tank comes back to full: refills still needed after
    this one are `_refills_needed(1.0, required)`, and the total is one
    more than that. Wait, and it is `_refills_needed(fuel, required)`. If
    those two are equal the stop is free and worth taking under yellow.

    `margin` is reported rather than acted on. The comparison turns on a
    ceiling, so the answer can flip on a hundredth of a tank, and 02b's
    third finding is that a ranking whose spread is smaller than its error
    is noise. Carrying the distance to the boundary into the reason string
    means a sweep can show how often the decision sat on a knife edge
    instead of the notebook assuming it never did.

    **The lane is the risk.** Waiting for a caution and arriving at a shut
    lane loses more than the stop would have saved, and under IMSA's Short
    FCY the lane never opens at all. That is checked here rather than left
    to the engine's refusal, because a strategy that cannot see the lane is
    not gambling, it is being lucky.
    """

    def __call__(self, car: CarState, state: RaceState) -> PitDecision:
        cls = state.config.class_by_name(car.class_name)

        if _fuel_window_reached(car, cls):
            return PitDecision(pit=True, refuel_to=1.0, change_tyres=True,
                               reason="fuel window")

        if not (state.under_caution and state.pit_lane_open):
            return PitDecision(pit=False)

        _, required = _remaining(car, state, cls)
        if _refills_needed(1.0, required) + 1 <= _refills_needed(car.fuel, required):
            margin = (required - car.fuel) - math.floor(required - car.fuel)
            return PitDecision(pit=True, refuel_to=1.0, change_tyres=True,
                               reason=f"free caution stop (margin {margin:.3f})")

        return PitDecision(pit=False)


@dataclass
class TrackPositionDefender:
    """Refuse a voluntary stop that would concede a place.

    The other of the two arguments on any pit wall. The rival is whoever is
    nearest behind in the class order at the moment of the decision, found
    the same way the classification is found, so it changes through the race
    and is never a named car.

    A stop costs `stop_cost` seconds. If the car behind is closer than that,
    the stop hands it the place. This strategy takes the worse fuel window
    instead - and pays for it, because the stop it declines has to be taken
    later on a lap that is likely to be greener and therefore dearer.

    Field compression is what makes this different from the lap-down
    defender rather than a restatement of it: without it a caution costs
    nothing positionally and both strategies reduce to guarding the same
    gap. If compression is ever backed out, decision 9 says merge them.

    **The rival, settled at 02c.** Decision 9 item 3 said "whoever is
    directly ahead on the road", which taken literally is a car a stop
    cannot drop you behind. The two readings that do describe a decision -
    the nearest car a stop would concede a place to, and the car you would
    emerge behind - are the same car, because you come out behind a rival
    exactly when its arrival is nearer than the stop is long. That is the
    rule `_would_be_passed` implements, stated from the side that makes it
    computable.
    """

    def __call__(self, car: CarState, state: RaceState) -> PitDecision:
        cls = state.config.class_by_name(car.class_name)
        if not _fuel_window_reached(car, cls):
            return PitDecision(pit=False)

        if _can_run_another_lap(car, state, cls):
            cost_s = _stop_seconds(car, state, cls)
            if _would_be_passed(car, state, cost_s) is not None:
                return PitDecision(pit=False, reason="defending track position")

        return PitDecision(pit=True, refuel_to=1.0, change_tyres=True,
                           reason="fuel window")


@dataclass
class SplashAndDashPlanner:
    """Take on the fuel the race still needs, and not a drop more.

    Works backwards from the flag. Laps left convert to fuel left at the
    dials' burn rates, and the final stop asks for exactly that instead of a
    full tank. The saving is real rather than bookkeeping: `pitstop.py`
    charges for fuel actually taken, so a short fill is a short stop, and
    01's engine could not represent this at all.

    Tyres come off the same calculation. If what is left of their life
    covers what is left of the race, the stop skips them and saves the tyre
    job as well - which under WEC's sequential rules (art. 12) is worth more
    than under IMSA's overlapping ones (art. 34.1.1), so the two series
    should not gain the same amount here.

    Two guards, both of them arithmetic rather than judgement:

    * **Never ask for less than is aboard.** `_apply_pit` sets fuel *to*
      `refuel_to`, so a low ask throws fuel away rather than declining to
      add it. `test_benchmark.py` asserts the search cannot find that as an
      optimisation and the same applies here.
    * **A lap in hand.** The estimate is an expectation over the caution
      pattern, and a splash that comes up one lap short costs an entire
      extra stop. The margin is one green lap of fuel - not a chosen
      number, the smallest quantity that is certainly enough, in the dial
      that decides it.
    """

    def __call__(self, car: CarState, state: RaceState) -> PitDecision:
        cls = state.config.class_by_name(car.class_name)
        if not _fuel_window_reached(car, cls):
            return PitDecision(pit=False)

        laps_left, _required = _remaining(car, state, cls)
        target = fuel_to_the_flag(car, state, cls)

        keep_tyres = (car.tyre_age + laps_left) < cls.tyre_life_laps
        short = target < 1.0 - 1e-9

        if short or keep_tyres:
            return PitDecision(pit=True, refuel_to=target,
                               change_tyres=not keep_tyres,
                               reason=f"splash to {target:.2f}")
        return PitDecision(pit=True, refuel_to=1.0, change_tyres=True,
                           reason="fuel window")


@dataclass
class LapDownDefender:
    """Protect the lap, and then protect the credit that gives it back.

    Two clauses, read in order.

    **One: do not concede the lap.** A voluntary stop costs seconds, and
    seconds convert to a share of a lap at the class leader's pace. If the
    deficit is already close enough that the stop would push it past a whole
    lap, the stop is declined. Only voluntary stops - declining past the
    fuel window concedes the lap anyway and adds a forced stop through a
    possibly shut lane.

    **Two: once lapped, stay out under caution.** This is decision 9 item 5
    as amended at 02c, and it is a standing condition rather than a
    threshold. The strategy has no action that takes a wave-around -
    `_take_wave` is engine-internal and unconditional on the frozen eligible
    set - so the only lever is not being outside that set when a wave is
    announced.

    The reason it is a standing condition and not a comparison comes out of
    the arithmetic. Behind the safety car every car runs the same lap, so
    two cars' positions within the lap keep a fixed offset, and eligibility
    - being further round the lap than the class leader - holds for a share
    of each caution lap equal to that offset. A stop delays the next lap
    start by its own cost and so reduces the offset by `cost / caution_lap`.
    **Every voluntary stop therefore costs eligibility; none is free.**
    There is no threshold to compute, and the announcement time that would
    let the strategy find one is not in `RaceState` anyway.

    The two clauses pull opposite ways, because being lapped is the
    precondition for the credit clause two protects. That is intended: clause
    one tries to avoid the state, clause two governs conduct once clause one
    has already failed. Do not resolve the tension by deleting one.

    **Where it is inert.** The Pass-Around is frozen a full caution lap
    before the lane opens to anybody, in both series, so it cannot be
    forfeited by a decision. Only IMSA's Final Wave-By is reachable, and WEC
    runs one wave - so clause two never fires in WEC and the strategy
    reduces to clause one there. That is a rulebook consequence to report,
    not an asymmetry to engineer away.
    """

    def __call__(self, car: CarState, state: RaceState) -> PitDecision:
        cls = state.config.class_by_name(car.class_name)
        if not _fuel_window_reached(car, cls):
            return PitDecision(pit=False)
        if not _can_run_another_lap(car, state, cls):
            return PitDecision(pit=True, refuel_to=1.0, change_tyres=True,
                               reason="fuel window")

        # Clause two first: it is the cheaper test and the stronger claim.
        if state.under_caution and state.laps_down(car) >= 1:
            return PitDecision(pit=False, reason="holding wave eligibility")

        leader = state.class_leader(car.class_name)
        if leader is None or leader.car_id == car.car_id:
            return PitDecision(pit=True, refuel_to=1.0, change_tyres=True,
                               reason="fuel window")

        # The car is at the line, so its own progress is exactly its lap
        # count - reading its lap window instead would saturate the clip and
        # put it a full lap further round than it is, which is the defect
        # that produced ninety-nine wave-arounds in one race.
        deficit = (leader.laps_done + leader.track_fraction(state.t)) - car.laps_done
        leader_lap_s = max(
            _caution_lap_s(leader, cls) if state.under_caution
            else _green_lap_s(leader, cls), 1e-6)
        cost_in_laps = _stop_seconds(car, state, cls) / leader_lap_s

        if deficit < 1.0 <= deficit + cost_in_laps:
            return PitDecision(pit=False, reason="defending the lap")

        return PitDecision(pit=True, refuel_to=1.0, change_tyres=True,
                           reason="fuel window")


# ----------------------------------------------------------------------
# The two mappings
# ----------------------------------------------------------------------
# What `assets.freeze_background` may give the field. Members carry chosen
# constants and are not measured against the agent.
BASELINES = {
    "fuel_window": RunToFuelWindow,
    "caution_opportunist": OpportunistUnderCaution,
    "fixed_stint": FixedLapStint,
}

# Decision 9's five. Every member takes no constructor arguments, which is
# the shape parameter-freeness has to have: a strategy that can be tuned has
# somewhere to put the tuning.
ROSTER = {
    "fuel_window": RunToFuelWindow,
    "caution_gambler": CautionGambler,
    "track_position": TrackPositionDefender,
    "splash_and_dash": SplashAndDashPlanner,
    "lap_down": LapDownDefender,
}


def assign_strategy(config, strategy, class_name: str | None = None) -> dict:
    """Give every car (optionally, every car of one class) the same strategy.

    The whole-class diagnostic decision 8 keeps: run the identical race
    twice, changing only which strategy one class is using, and the
    difference is the strategy - because the caution timeline was drawn in
    advance and does not depend on what anyone does. It answers "what if
    everyone did this", which is a different question from the focal-car
    comparison and is reported as one line rather than as a headline.
    """
    out = {}
    for cls in config.classes:
        if class_name and cls.class_name != class_name:
            continue
        for i in range(cls.n_cars):
            out[f"{cls.class_name}-{i + 1:02d}"] = strategy
    return out
