"""The causal reference: the best a strategist could have done without foresight.

The clairvoyant reference in `benchmark.py` reads the frozen timeline and is
an upper bound. This is the other end of the pair, and it is the one 03
scores against - scoring a policy against the clairvoyant plan alone would
penalise it for failing to predict the future, which is not a failing.

Why this is exactly computable
------------------------------
Decision 17 draws caution episodes non-overlapping with `rng.exponential`
durations, and `test_episode_lengths_stay_exponential` holds the coefficient
of variation at one. Green gaps are exponential too. So the flag is a
two-state continuous-time Markov chain, and the probability of being under
caution a given interval later has a closed form rather than needing
simulation. That is what makes a causal optimum computable at all, and it is
why the merge in the old draw had to go: the union of overlapping
exponentials is not exponential, and none of this would have held.

What the seed does and does not reach
------------------------------------
The causal problem has no realised timeline in it, so the policy does not
depend on when the cautions fell. It does still depend on the seed, through
exactly three scalars: the focal car's drawn base pace, and the two levels
`benchmark.anchor` measures off a reference run. That is worth stating
precisely rather than claiming a saving that is not there - and
`test_the_policy_depends_on_the_seed_only_through_three_numbers` holds it,
so if the solver ever picks up something else from the draw it is a failing
test rather than a quiet cost. Holding those three fixed, one solve serves
every seed.

What the state leaves out, deliberately
---------------------------------------
Four reductions, all declared rather than discovered later:

* **Lane status needs the caution's age**, because both rulebooks reopen the
  lane a set number of caution laps after the field forms up. So a caution
  carries an age counter rather than being one state, capped at the point
  where every class has been released.
* **A Short FCY is a separate phase.** Whether a caution ever opens depends
  on when it started relative to the race, which the clock already carries;
  the restart-proximity limb of art. 46.3.3 depends on the previous
  episode's end, which the state does not carry, so that limb is not
  modelled and the causal reference is slightly optimistic about late
  cautions. Named here because it is the one place this model knows less
  than `pitstop.lane_status` does.
* **The driver clock is treated as deterministic in elapsed race time.** It
  accumulates lap time and resets at the first stop past the limit, so its
  crossings sit at roughly fixed points of the race whatever the plan.
* **Tyre age is bucketed**, because with fuel binding at half the tyre life
  it never forces a stop and only enters through degradation.

The rollout uses the realised timeline - that is the race happening to the
car - but every decision reads only the current state, so nothing about the
future leaks in. `test_causal_never_reads_the_future` is what holds that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .benchmark import FocalContext, Plan, Stop
from .pitstop import stop_cost


# Phases the flag can be in, as far as a decision is concerned.
GREEN = 0
NEVER_OPENS = 1          # a caution that will not offer a stop at all
CAUTION_BASE = 2         # CAUTION_BASE + age, in caution laps since the call


@dataclass
class CausalPolicy:
    """A decision rule over the reduced state, plus how it was built."""

    value: np.ndarray             # [time bucket, phase, fuel, tyre]
    action: np.ndarray            # index into `fill_levels`, or -1 for no stop
    fill_levels: tuple[float, ...]
    bucket_s: float
    fuel_edges: np.ndarray
    tyre_edges: np.ndarray
    max_age: int
    shape: dict

    def n_phases(self) -> int:
        return CAUTION_BASE + self.max_age + 1


def hazards(cls_dials) -> tuple[float, float]:
    """Green-to-caution and caution-to-green rates, from the calibrated dials.

    `caution_rate` is the share of race time under caution, which for an
    alternating process is `mean_dur / (mean_dur + mean_gap)`. Inverting that
    gives the mean green gap, and the two rates follow. Nothing here is a new
    assumption: both numbers come from `calibrate.calibrate_cautions`.
    """
    rate = min(max(cls_dials.caution_rate, 1e-6), 0.95)
    mean_dur = max(cls_dials.caution_mean_dur_s, 1e-6)
    mean_gap = mean_dur * (1.0 - rate) / rate
    return 1.0 / mean_gap, 1.0 / mean_dur


def p_caution_starts(lam_green: float, d: float) -> float:
    """Probability a caution begins during an interval of length `d`."""
    return -math.expm1(-lam_green * max(d, 0.0))


def p_caution_ends(lam_caution: float, d: float) -> float:
    """Probability the current caution ends during `d`. Memoryless: its age
    does not enter, which is the whole reason this is closed-form."""
    return -math.expm1(-lam_caution * max(d, 0.0))


def _bucket(value: float, edges: np.ndarray) -> int:
    """Nearest representative, not the one below.

    Flooring looks harmless and is not: with a grid that stopped at 0.9 a
    full tank read as nine tenths, the policy believed it was a lap and a
    half short of where it was, and stopped early every stint. The grid
    includes both ends and the lookup rounds to the nearest.
    """
    return int(np.argmin(np.abs(edges - value)))


def solve_policy(ctx: FocalContext, *,
                 fill_levels: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
                 bucket_s: float = 30.0,
                 fuel_step: float | None = None,
                 tyre_bucket_laps: int = 12,
                 max_age: int = 4) -> CausalPolicy:
    """Backward induction over the reduced state. Once per race shape.

    The value carried is expected laps still to come. Position is not
    additive so it cannot be optimised here any more than it can in stage
    one; the causal plan is re-scored on position by `benchmark.rescore`
    like every other candidate.
    """
    cls = ctx.cls
    duration = ctx.duration_s
    lam_g, lam_c = hazards(cls)

    # The fuel grid has to be fine enough that burning a lap's worth moves
    # the car off its bucket. It is not a resolution preference: with a step
    # wider than the burn, nearest-rounding returns the car to the bucket it
    # started in, fuel never falls, the value function goes flat in fuel and
    # the policy - correctly, given what it has been told - stops as cheaply
    # and as often as it can. The caution burn is the smaller of the two, so
    # it sets the step.
    if fuel_step is None:
        fuel_step = cls.fuel_per_lap_caution
    if fuel_step > cls.fuel_per_lap_caution + 1e-12:
        raise ValueError(
            f"fuel_step {fuel_step:.4f} is coarser than a caution lap's burn "
            f"{cls.fuel_per_lap_caution:.4f}; the policy would not see fuel "
            f"being used")
    n_fuel = int(math.ceil(1.0 / fuel_step)) + 1
    n_time = int(math.ceil(duration / bucket_s)) + 1
    fuel_edges = np.linspace(0.0, 1.0, n_fuel)
    n_tyre = int(math.ceil(cls.tyre_life_laps / tyre_bucket_laps)) + 1
    tyre_edges = np.arange(n_tyre) * tyre_bucket_laps
    n_phase = CAUTION_BASE + max_age + 1

    value = np.zeros((n_time, n_phase, n_fuel, n_tyre))
    action = np.full((n_time, n_phase, n_fuel, n_tyre), -1, dtype=np.int8)

    open_delay = cls.caution_pits_open_delay_laps
    stagger = ctx.rules.stagger_laps[ctx.rules.group_index(ctx.class_name)]
    opens_after = open_delay + stagger

    def lane_open(phase: int, t: float) -> bool:
        if phase == GREEN:
            return True
        if phase == NEVER_OPENS:
            return False
        return (phase - CAUTION_BASE) >= opens_after

    def lap_time(phase: int, tyre_age: float) -> float:
        if phase == GREEN:
            return max(ctx.base_pace_s + cls.deg_slope_s_per_lap * tyre_age
                       + ctx.green_offset_s, ctx.floor_s)
        return max(ctx.caution_lap_s, ctx.floor_s)

    def lookup(t: float, phase: int, fuel: float, tyre: float) -> float:
        """Value at an arbitrary point, interpolated on the clock only."""
        if t >= duration:
            return 0.0
        x = t / bucket_s
        lo = int(math.floor(x))
        hi = min(lo + 1, n_time - 1)
        w = x - lo
        fi = _bucket(fuel, fuel_edges)
        ti = _bucket(tyre, tyre_edges)
        return (1.0 - w) * value[lo, phase, fi, ti] + w * value[hi, phase, fi, ti]

    def phase_after(phase: int, d: float, t_arrival: float) -> list[tuple[float, int]]:
        """Where the flag may be `d` seconds later, with probabilities."""
        if phase == GREEN:
            p = p_caution_starts(lam_g, d)
            started = (NEVER_OPENS if _never_opens_at(ctx, t_arrival)
                       else CAUTION_BASE)
            return [(1.0 - p, GREEN), (p, started)]
        p = p_caution_ends(lam_c, d)
        if phase == NEVER_OPENS:
            return [(p, GREEN), (1.0 - p, NEVER_OPENS)]
        age = min(phase - CAUTION_BASE + 1, max_age)
        return [(p, GREEN), (1.0 - p, CAUTION_BASE + age)]

    # --- backward over the clock ----------------------------------------
    for i in range(n_time - 2, -1, -1):
        t = i * bucket_s
        for phase in range(n_phase):
            open_now = lane_open(phase, t)
            for fi in range(n_fuel):
                fuel = fuel_edges[fi]
                for ti in range(n_tyre):
                    tyre = float(tyre_edges[ti])
                    best, best_a = -1.0, -1

                    forced = (fuel < cls.fuel_per_lap
                              or tyre >= cls.tyre_life_laps)
                    options: list[int] = []
                    if not forced:
                        options.append(-1)
                    if open_now or forced:
                        options += [a for a, lv in enumerate(fill_levels)
                                    if lv >= fuel]
                    if not options:
                        options = [len(fill_levels) - 1]

                    for a in options:
                        t2, f2, y2 = t, fuel, tyre
                        if a >= 0:
                            level = fill_levels[a]
                            cost = _causal_pit_cost(ctx, fuel, level, phase)
                            t2, f2, y2 = t + cost, level, 0.0
                            if t2 >= duration:
                                continue
                        lap = lap_time(phase, y2)
                        burn = (cls.fuel_per_lap if phase == GREEN
                                else cls.fuel_per_lap_caution)
                        wear = 1.0 if phase == GREEN else 0.0
                        t3 = t2 + lap
                        # The lap counts even when it ends past the flag.
                        total = 1.0
                        if t3 < duration:
                            for p, nxt in phase_after(phase, t3 - t, t3):
                                if p <= 0.0:
                                    continue
                                total += p * lookup(t3, nxt, max(f2 - burn, 0.0),
                                                    y2 + wear)
                        # Laps first, then finish sooner. Without the second
                        # term the policy is indifferent between two plans on
                        # the same lap count and can pick the slower, which
                        # loses races decided on time - decision 1's
                        # low-variance diagnostic, and the tie-break
                        # `classification` actually applies.
                        total -= 1e-6 * (t3 - t)
                        if total > best:
                            best, best_a = total, a

                    value[i, phase, fi, ti] = best
                    action[i, phase, fi, ti] = best_a

    return CausalPolicy(value=value, action=action, fill_levels=fill_levels,
                        bucket_s=bucket_s, fuel_edges=fuel_edges,
                        tyre_edges=tyre_edges, max_age=max_age,
                        shape={"duration_s": duration, "class": ctx.class_name,
                               "series": ctx.rules.series_code,
                               "lam_green": lam_g, "lam_caution": lam_c,
                               "opens_after_laps": opens_after})


def _causal_pit_cost(ctx: FocalContext, fuel: float, level: float,
                     phase: int) -> float:
    """What a stop costs, from the phase rather than from the timeline.

    `FocalContext.pit_cost` asks `cautions.is_caution(t)` whether to apply
    the caution discount, which is the right question for a plan being
    priced against a race that has happened and the wrong one here: it reads
    the realised timeline and would let the policy know where the cautions
    fell. The phase already says whether the car is under caution, so it is
    what the discount hangs on.
    """
    mean = stop_cost(ctx.cls, ctx.rules, max(level - fuel, 0.0), True)
    if phase != GREEN:
        mean *= (1.0 - ctx.cls.pit_caution_discount)
    return max(mean, 0.0)


def _never_opens_at(ctx: FocalContext, t: float) -> bool:
    """The two limbs of art. 46.3.3 the clock alone can answer."""
    w = ctx.rules.never_opens_window_s
    if w <= 0.0:
        return False
    return t < w or t > ctx.duration_s - w


def observed_phase(ctx: FocalContext, t: float, max_age: int) -> int:
    """Which phase the car is in, from what it can see right now.

    Everything read here is available at time `t`: the flag, how long this
    caution has been running, and whether the lane will open at all - the
    last from `pitstop.lane_status`, whose Short FCY test looks only at the
    caution's start, the race length and the *previous* episode's end.
    """
    episode = None
    for start, end in ctx.cautions.periods:
        if start <= t < end:
            episode = (start, end)
            break
    if episode is None:
        return GREEN

    status = ctx.engine._lane(ctx.class_name, t)
    if not status.open and status.opens_at_s is None and "short" in status.reason:
        return NEVER_OPENS
    age = int((t - episode[0]) // max(ctx.caution_lap_s, 1e-6))
    return CAUTION_BASE + min(age, max_age)


def causal_plan(ctx: FocalContext, policy: CausalPolicy,
                trace: list | None = None) -> Plan:
    """Roll the policy forward against the race that actually happened.

    The timeline is realised because the race is realised - the cautions
    fall when they fall. What is never read is anything past `t`: the
    decision at each line crossing is a table lookup on the current state.
    """
    cls = ctx.cls
    duration = ctx.duration_s
    t, fuel, tyre, driver = 0.0, 1.0, 0, 0.0
    laps, n_stops, caution_stops = 0, 0, 0
    stops: list[Stop] = []
    forced_closed = 0

    while True:
        phase = observed_phase(ctx, t, policy.max_age)
        lap_s, burn, wear = ctx.lap(laps, tyre, t, True)
        t += lap_s
        laps += 1
        fuel = max(fuel - burn, 0.0)
        tyre += wear
        driver += lap_s
        if t >= duration:
            return Plan(tuple(stops), laps, t, forced_closed, 0,
                        caution_stops=caution_stops)

        reason = ctx.forced_reason(fuel, tyre, driver)
        phase = observed_phase(ctx, t, policy.max_age)
        i = min(int(t / policy.bucket_s), policy.value.shape[0] - 1)
        fi = _bucket(fuel, policy.fuel_edges)
        ti = _bucket(float(tyre), policy.tyre_edges)
        a = int(policy.action[i, phase, fi, ti])

        wants = a >= 0 or bool(reason)
        if not wants:
            continue
        if not reason and not ctx.lane_open(t):
            continue                      # the lane is shut; come round again
        level = policy.fill_levels[a] if a >= 0 else 1.0
        level = max(level, fuel)
        forced_closed += int(bool(reason) and not ctx.lane_open(t))
        caution_stops += int(ctx.cautions.is_caution(t))
        if trace is not None:
            trace.append((laps, t))
        t += ctx.pit_cost(fuel, level, True, n_stops, t, True)
        stops.append(Stop(laps, level, True))
        fuel, tyre, n_stops = level, 0, n_stops + 1
        if driver >= ctx.max_driver_stint_s:
            driver = 0.0
        if t >= duration:
            return Plan(tuple(stops), laps, t, forced_closed, 0,
                        caution_stops=caution_stops)
