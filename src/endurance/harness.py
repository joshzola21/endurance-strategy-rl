"""The paired comparison, and the one gate it has to pass.

Decision 10: the headline statistic is the **paired class-position delta
against the fuel-window baseline on the same race**, reported as a
distribution rather than a mean. Everything in this module exists to make
that sentence true, and most of it is about the word *same*.

Why pairing is nearly free here, and what it costs if it is broken
------------------------------------------------------------------
The caution timeline is drawn before the race and does not depend on what
anyone does, and since 02a lap noise and pit cost come from per-car streams
indexed by lap and by stop. So one seed is one race, and running two
strategies on it changes the strategy and nothing else. That is common
random numbers without any of the usual bookkeeping - but it holds only if
everything *except* the focal strategy is held fixed, which is what the
three rules below are for and why the gate at the foot of this module is the
stage's verification gate.

1. **The focal car is chosen from the seed, never from the arm.** The pace
   draw is independent of strategy, so pace rank picks the same car id in
   both arms. A focal car chosen any other way - by id, by finishing
   position, by anything downstream of the race - breaks the pairing while
   leaving every number looking plausible.
2. **The background field is the frozen artefact**, resolved once per race
   and given to both arms.
3. **The null arm is run once per race and shared across the roster.** Not
   only for the runtime: recomputing it per strategy invites it to be
   computed slightly differently for one of them.

Two tables, never one
---------------------
Decision 10's budget is per strategy *per series*, and 02c reads that
strictly. The rulebooks differ in ways that make a lever live in one series
and dead in the other - the lap-down defender's wave clause cannot fire in
WEC at all - so a pooled row would report the mean of a measurement and a
non-measurement. `summarise` therefore always groups by series and there is
no option to turn that off.

What the score is
-----------------
Class finishing position is the primary score and race time the diagnostic,
both reported side by side per decision 1. A caveat on the second, stated
here rather than found later: in a *timed* race every car's `race_time_s`
lands within a lap of the flag, so time deltas are dominated by where the
final crossing fell rather than by pace. `d_laps` is the diagnostic that
carries the information; `d_race_time_s` is reported beside it because
decision 1 asks for it, not because it is the better number.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .assets import BackgroundField, SeedBank, dials_fingerprint
from .engine import RaceEngine, run_race
from .params import RaceConfig, scale_dials, set_dials
from .strategies import ROSTER

# Decision 8's slot, read as 02b's decision A: fifth-fastest drawn base pace
# in the headline class. `_build_field` starts every car at t = 0 on lap 1
# and there is no grid, so "P5" has no other referent - and fixing the pace
# rank holds the focal car's competitive position constant across the bank,
# which a fixed car id would not.
DEFAULT_PACE_RANK = 5


def headline_class(config: RaceConfig) -> str:
    """The quickest class in the race, which is the one being reported on.

    Taken from base pace rather than from a name list, so a recalibration
    that renames a class does not silently start measuring a different one.
    """
    return min(config.classes, key=lambda c: c.base_pace_s).class_name


def focal_car(config: RaceConfig, seed: int, class_name: str | None = None,
              pace_rank: int = DEFAULT_PACE_RANK) -> str:
    """Which car is the focal car on this seed, by drawn pace rank.

    Built from `_build_field` alone - no race is run - because the field
    draw is what decides it and running a race to find out would make the
    answer depend on the strategy. Costs under two milliseconds, which
    matters at 200 seeds times two arms times two series.
    """
    class_name = class_name or headline_class(config)
    engine = RaceEngine(config, seed=seed)
    engine._build_field()

    in_class = [(car.base_pace_s, car_id) for car_id, car in engine.cars.items()
                if car.class_name == class_name]
    if not in_class:
        raise KeyError(f"no class {class_name!r} in race {config.name!r}")
    if not 1 <= pace_rank <= len(in_class):
        raise ValueError(f"pace_rank {pace_rank} outside 1..{len(in_class)}")

    return sorted(in_class)[pace_rank - 1][1]


# ----------------------------------------------------------------------
# One race
# ----------------------------------------------------------------------
_SCORE_COLUMNS = ("class_pos", "overall_pos", "laps", "race_time_s",
                  "stops", "pit_time_s", "traffic_time_s", "caution_laps")


def run_focal(config: RaceConfig, seed: int, focal: str, strategy,
              field: BackgroundField) -> dict:
    """Run one race with `strategy` in the focal seat and score that car.

    `classification()` rather than `positions()` on purpose: the latter
    sorts the whole field once per lap record, which across 200 seeds
    dominates the runtime of the entire comparison and answers a question
    nobody is asking here.
    """
    strategies = field.resolve(focal=focal)
    strategies[focal] = strategy
    result = run_race(config, strategies=strategies, seed=seed)

    row = result.classification().set_index("car_id").loc[focal]
    out = {"seed": seed, "focal": focal}
    out.update({c: row[c] for c in _SCORE_COLUMNS})
    return out


class NullRuns:
    """The fuel-window arm, run once per race and reused across the roster.

    Keyed on the dials as well as the seed. A seed is only a race given a
    set of dials, so a cache that ignored them would happily serve a sweep
    point the null from a different one - which is the failure mode
    `dials_fingerprint` exists to catch, applied to the thing most likely to
    outlive a config change in memory.
    """

    def __init__(self) -> None:
        self._runs: dict[tuple, dict] = {}

    def get(self, config: RaceConfig, seed: int, focal: str,
            field: BackgroundField, null_strategy) -> dict:
        key = (dials_fingerprint(config), seed, focal)
        if key not in self._runs:
            self._runs[key] = run_focal(config, seed, focal, null_strategy(),
                                        field)
        return self._runs[key]


# ----------------------------------------------------------------------
# Running the races, serially or in a pool
# ----------------------------------------------------------------------
# **The rows are built in exactly one place.** These two functions return the
# same thing - one entry per seed, holding the focal car and every arm's scored
# race - and `compare_roster` turns that into rows afterwards. A parallel path
# that built its own rows would be a second comparison, which is decision 6's
# failure, and the delta signs are the single easiest thing in this project to
# get quietly backwards.
def _run_seed_serial(config, seeds, field, roster, class_name, pace_rank,
                     null_name, nulls):
    out = []
    for seed in seeds:
        focal = focal_car(config, seed, class_name, pace_rank)
        null = nulls.get(config, seed, focal, field, roster[null_name])
        runs = {null_name: null}
        for name, strategy in roster.items():
            if name != null_name:
                runs[name] = run_focal(config, seed, focal, strategy(), field)
        out.append((seed, focal, runs))
    return out


def _one_seed(payload):
    """One seed's whole roster, in a worker process.

    The roster arrives as **names** rather than callables. `ROSTER`'s values
    are classes and would pickle, but `agent_roster` hands over a closure round
    a live onnxruntime session and that would not - so the parallel path takes
    names, rebuilds from `ROSTER` here, and `compare_roster` refuses to use it
    for anything `ROSTER` does not contain.

    A fresh `NullRuns` per worker is correct rather than wasteful: the cache
    exists to share the null across a seed's arms, and every arm of a seed is
    in the same call.
    """
    config, seed, field, names, class_name, pace_rank, null_name = payload
    focal = focal_car(config, seed, class_name, pace_rank)
    null = run_focal(config, seed, focal, ROSTER[null_name](), field)
    runs = {null_name: null}
    for name in names:
        if name != null_name:
            runs[name] = run_focal(config, seed, focal, ROSTER[name](), field)
    return seed, focal, runs


def _run_seeds_parallel(config, seeds, field, roster, class_name, pace_rank,
                        null_name, n_workers):
    """The same races, across processes, in seed order.

    Results are reassembled in the order `seeds` was given rather than the
    order they complete, so the frame does not depend on scheduling. Each race
    is drawn entirely from its own seed, so nothing crosses between workers and
    the output is identical to the serial path by construction - which the
    gate at the foot of this module asserts rather than assumes.
    """
    from concurrent.futures import ProcessPoolExecutor

    names = tuple(roster)
    payloads = [(config, seed, field, names, class_name, pace_rank, null_name)
                for seed in seeds]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(_one_seed, payloads, chunksize=1))


# ----------------------------------------------------------------------
# The comparison
# ----------------------------------------------------------------------
@dataclass
class Comparison:
    """Every paired race, long, plus what it was run against.

    Held as one long frame rather than a table of summaries because decision
    10 asks for a distribution: the summary is a view of this, and anything
    that wants a different view - a sweep, a quantile, the share of races
    where nothing moved - reads the same rows rather than a second run.
    """

    rows: pd.DataFrame
    provenance: dict

    def summarise(self) -> pd.DataFrame:
        return summarise(self.rows)


def compare_roster(config: RaceConfig, seeds: list[int], field: BackgroundField,
                   roster: dict | None = None,
                   class_name: str | None = None,
                   pace_rank: int = DEFAULT_PACE_RANK,
                   null_name: str = "fuel_window",
                   nulls: NullRuns | None = None,
                   n_workers: int = 1) -> Comparison:
    """Score every roster strategy against the null, race by race.

    One row per (strategy, seed). The delta columns are signed so that
    **positive is better in every one of them** - a position delta of +1
    means a place gained, which is a *lower* `class_pos`, and getting that
    sign wrong is the single easiest way to publish a table that says the
    opposite of what happened.

    `n_workers` above 1 runs the seeds across processes. It changes how long
    this takes and nothing else: the races are drawn entirely from their own
    seeds, the results are reassembled in the order `seeds` was given, and the
    rows below are built by the same code either way.

    Refused rather than silently serialised when the roster holds anything
    `ROSTER` does not - which today means the agent, whose strategy closes over
    an onnxruntime session that cannot cross a process boundary. A parallel
    request that quietly ran serially would misreport the runtime, and a
    runtime is what somebody is measuring when they ask for it.
    """
    roster = roster or ROSTER
    if null_name not in roster:
        raise KeyError(f"null {null_name!r} is not in the roster")
    nulls = nulls if nulls is not None else NullRuns()
    class_name = class_name or headline_class(config)

    if n_workers > 1:
        outside = [n for n in roster if n not in ROSTER]
        if outside:
            raise ValueError(
                f"cannot run {outside} across processes: the parallel path "
                f"rebuilds strategies from ROSTER by name, and these are not "
                f"in it. Run this roster with n_workers=1.")
        races = _run_seeds_parallel(config, seeds, field, roster, class_name,
                                    pace_rank, null_name, n_workers)
    else:
        races = _run_seed_serial(config, seeds, field, roster, class_name,
                                 pace_rank, null_name, nulls)

    rows = []
    for seed, focal, runs in races:
        null = runs[null_name]
        for name in roster:
            run = runs[name]
            rows.append({
                "series": config.series_code,
                "race": config.name,
                "strategy": name,
                "class": class_name,
                "pace_rank": pace_rank,
                **run,
                "d_class_pos": null["class_pos"] - run["class_pos"],
                "d_laps": run["laps"] - null["laps"],
                "d_race_time_s": null["race_time_s"] - run["race_time_s"],
                "d_stops": run["stops"] - null["stops"],
                "d_pit_time_s": null["pit_time_s"] - run["pit_time_s"],
            })

    return Comparison(
        rows=pd.DataFrame(rows),
        provenance={
            "race": config.name,
            "series_code": config.series_code,
            "dials_fingerprint": dials_fingerprint(config),
            "n_seeds": len(seeds),
            "class": class_name,
            "pace_rank": pace_rank,
            "null": null_name,
            "background": field.provenance.get("uniform_strategy"),
            "n_workers": n_workers,
        },
    )


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    """Decision 10's headline, as a distribution rather than a mean.

    "Gains a place in 40% of races, loses one in 12%" is both a stronger
    claim than a mean of 4.7 and a legible one, and it survives the long
    tails a position delta has when a strategy occasionally throws a race
    away. The mean is deliberately absent: on a bounded, discrete,
    heavy-tailed quantity it is the statistic most likely to be quoted and
    least likely to mean anything.

    Always grouped by series. There is no argument to pool them.
    """
    def one(g: pd.DataFrame) -> pd.Series:
        d = g["d_class_pos"]
        return pd.Series({
            "n": len(g),
            "gained": float((d > 0).mean()),
            "level": float((d == 0).mean()),
            "lost": float((d < 0).mean()),
            "median_d_pos": float(d.median()),
            "p10_d_pos": float(d.quantile(0.10)),
            "p90_d_pos": float(d.quantile(0.90)),
            "median_d_laps": float(g["d_laps"].median()),
            "median_d_pit_s": float(g["d_pit_time_s"].median()),
            "median_stops": float(g["stops"].median()),
        })

    return (rows.groupby(["series", "strategy"], sort=False)
                .apply(one, include_groups=False)
                .reset_index())


# ----------------------------------------------------------------------
# Sweeps
# ----------------------------------------------------------------------
def sweep_dial(config: RaceConfig, seeds: list[int], field: BackgroundField,
               dial: str, multipliers: tuple[float, ...] | None = None,
               *, values: tuple[float, ...] | None = None,
               **kw) -> pd.DataFrame:
    """Decision 11's one-at-a-time sweep, on the sweep bank.

    The bank passed in should be the sweep fifty, which are the *first fifty
    of the headline two hundred* rather than a separate draw: a sweep asks
    how a claim moves as a dial moves, and the cleanest version of that
    compares against the same races the claim was made on.

    Each point gets its own `NullRuns`, because moving a dial changes the
    race and therefore changes the null. Sharing one cache across points
    would serve every point the baseline from the first, which would look
    like a beautifully smooth response curve.

    **`multipliers` or `values`, exactly one.** Multiplying is the default and
    is what every published sweep used, because a multiplier is comparable
    across classes whose values differ. It cannot move a dial that sits at
    zero, though, and `pit_transit_caution_discount` does - so a sweep of that
    dial passes `values` and goes through `set_dials`. Both paths write both
    columns: `value` is the dial as the race saw it, and `multiplier` is empty
    on the value path rather than back-computed, because dividing by a zero
    default would invent a number.
    """
    if (multipliers is None) == (values is None):
        raise ValueError("pass multipliers or values, and not both: a sweep "
                         "point is either a factor on what the dial is or a "
                         "number to put there, and the two are different "
                         "questions about the same dial")

    points = multipliers if multipliers is not None else values
    frames = []
    for point in points:
        if multipliers is not None:
            moved = scale_dials(config, **{dial: point})
        else:
            moved = set_dials(config, **{dial: point})
        comparison = compare_roster(moved, seeds, field, nulls=NullRuns(), **kw)
        out = comparison.summarise()
        out.insert(0, "value", getattr(moved.classes[0], dial))
        out.insert(0, "multiplier", point if multipliers is not None else None)
        out.insert(0, "dial", dial)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def sweep_grid(config: RaceConfig, seeds: list[int], field: BackgroundField,
               dial_a: str, values_a: tuple[float, ...],
               dial_b: str, values_b: tuple[float, ...],
               *, how: str = "scale", **kw) -> pd.DataFrame:
    """Decision 11's 2-D grid, `pit_caution_discount` x caution rate by default.

    Those two interact by construction - how often a caution arrives and how
    much a caution stop saves - so a pair of one-at-a-time sweeps cannot say
    what the grid says.

    `how="scale"` reads `values_a` and `values_b` as multipliers, which is what
    they have always been and what the published grid used. `how="set"` reads
    them as the values themselves, for the same reason `sweep_dial` gained the
    option: a multiplier cannot move a dial that sits at zero. The axis columns
    hold whichever was passed, so a grid says what it swept.
    """
    if how not in ("scale", "set"):
        raise ValueError(f"how must be 'scale' or 'set', not {how!r}")
    move = scale_dials if how == "scale" else set_dials

    frames = []
    for a in values_a:
        for b in values_b:
            scaled = move(config, **{dial_a: a, dial_b: b})
            out = compare_roster(scaled, seeds, field, nulls=NullRuns(),
                                 **kw).summarise()
            out.insert(0, dial_b, b)
            out.insert(0, dial_a, a)
            frames.append(out)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# Slot rotation
# ----------------------------------------------------------------------
def rotate_pace_rank(config: RaceConfig, seeds: list[int],
                     field: BackgroundField, **kw) -> pd.DataFrame:
    """Decision 8's rotation, as pace rank rather than starting slot.

    Answers whether a strategy's value depends on where in the class you
    start. Reported as one figure, not as a table per rank.
    """
    cls = config.class_by_name(kw.get("class_name") or headline_class(config))
    frames = []
    for rank in range(1, cls.n_cars + 1):
        out = compare_roster(config, seeds, field, nulls=NullRuns(),
                             pace_rank=rank, **kw).summarise()
        out.insert(0, "pace_rank", rank)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# The verification gate
# ----------------------------------------------------------------------
def null_is_the_null(config: RaceConfig, seeds: list[int],
                     field: BackgroundField,
                     class_name: str | None = None,
                     pace_rank: int = DEFAULT_PACE_RANK,
                     null_strategy=None) -> pd.DataFrame:
    """02c's verification gate. Raises if the null arm is not the null.

    With the focal car on `RunToFuelWindow` and the field on the frozen
    background, the harness must reproduce the focal car's classification
    row **bit for bit against a bare `run_race` call** on the same seed.

    Against a bare call rather than against the harness's own null arm. A
    gate that compares the code under test with itself passes whatever has
    broken - 02a's reference numbers were captured from the engine before it
    was touched for exactly this reason, and the same argument applies here.

    It gates the apparatus rather than the roster, which is the right target:
    every paired delta this stage reports is measured *from* this run, so a
    null that is not the null puts a systematic offset under every number in
    02c and everything 03b inherits. It also re-asserts 02a's central
    property at the one place 02c could break it - the focal car's noise
    streams must not depend on which strategy occupies the seat - and it
    catches the plumbing faults that produce plausible numbers rather than
    errors: a mis-set focal id, a background field missing a car, a compat
    flag differing between arms, dials scaled in one arm and not the other,
    the rotation indexed off by one.

    **Its limit, stated rather than discovered.** A `fuel_window` focal car
    against a `fuel_window` background resolves to the same strategy map
    either way, so this ought to pass by construction and it says nothing
    about the four strategies that are not the null.
    """
    null_strategy = null_strategy or ROSTER["fuel_window"]
    class_name = class_name or headline_class(config)

    rows = []
    for seed in seeds:
        focal = focal_car(config, seed, class_name, pace_rank)
        through_harness = run_focal(config, seed, focal, null_strategy(), field)

        bare = run_race(config, strategies=field.resolve(), seed=seed)
        bare_row = bare.classification().set_index("car_id").loc[focal]

        mismatched = [c for c in _SCORE_COLUMNS
                      if through_harness[c] != bare_row[c]]
        if mismatched:
            raise AssertionError(
                f"the null is not the null on seed {seed}, car {focal}: "
                + ", ".join(f"{c} {through_harness[c]!r} != {bare_row[c]!r}"
                            for c in mismatched))
        rows.append({"seed": seed, "focal": focal,
                     **{c: bare_row[c] for c in _SCORE_COLUMNS}})

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# The benchmark gap, when there is a benchmark to read
# ----------------------------------------------------------------------
def attach_benchmark(rows: pd.DataFrame, benchmark: dict | None) -> pd.DataFrame:
    """Join 02b's per-seed reference onto the paired rows, if it exists.

    `benchmark` maps a seed to a dict carrying at least `class_pos` and
    `laps`. Kept as a join onto a cache rather than a call into
    `benchmark.py` because stage one costs 50-130 s a seed and the notebook
    is the argument, not the run.

    Returns the frame untouched when there is no cache, so every plot and
    table downstream works before the benchmark run finishes and gains a
    column when it lands.
    """
    if not benchmark:
        return rows

    ref = pd.DataFrame([{"seed": s, "bench_class_pos": v["class_pos"],
                         "bench_laps": v["laps"]}
                        for s, v in benchmark.items()])
    out = rows.merge(ref, on="seed", how="left")
    out["gap_to_benchmark_pos"] = out["class_pos"] - out["bench_class_pos"]
    out["gap_to_benchmark_laps"] = out["bench_laps"] - out["laps"]
    return out


def load_bank(path, series_code: str | None = None) -> SeedBank:
    """Read a seed bank and refuse one drawn for a different series."""
    bank = SeedBank.load(path)
    if series_code and bank.series_code != series_code:
        raise ValueError(f"bank is for {bank.series_code!r}, "
                         f"asked for {series_code!r}")
    return bank
