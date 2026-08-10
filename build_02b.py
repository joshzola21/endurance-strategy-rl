"""The 02b notebook builder.

Paste `build_02b` into `build_nb.py` alongside `build_01` and `build_02a`,
and add one line to `TARGETS`:

    "02b": ("02b_benchmark.ipynb", build_02b),

Kept in a separate file here only because reproducing all fourteen hundred
lines of `build_nb.py` to add one function would be a worse diff.
"""

import json
import sys
from pathlib import Path


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def build_02b():
    cells = []

    cells.append(md("""# The per-race benchmark (02b)

This stage builds the reference every later number is read against: for one
race, the best a plan could have done. Two of them, in fact - a
**clairvoyant** one that reads the caution timeline, and a **causal** one
that sees only what a strategist could see. The gap between them is the
value of foreknowledge, and the causal one is what 03 scores against,
because marking a policy down for failing to predict the future is not a
measurement of the policy.

**The benchmark is a reference, not a strategy.** It is never handed to the
engine as a live decision-maker, it never appears in `BASELINES`, and the
agent never observes it. The one callable involved, `PlanRunner`, is a
recording being played back.

**Two stages, because position is not additive.** Stage one is a dynamic
program over stop plans minimising the focal car's own race time against the
frozen timeline. Stage two re-scores the surviving candidates through the
full engine against the frozen rival field and ranks them on class
position. A DP cannot be run on position directly; this is how a
position-valued reference stays tractable.

**Two gates, neither optional.** The DP must match brute force in the
no-caution limit - on the reconstructed plan, not only the total, because in
the F1 work the same gate caught a benchmark that passed the total-time
check while rebuilding a suboptimal plan. And no plan may stop through a
closed pit lane, in either series, using `pitstop.lane_status` rather than a
reimplementation of it.

**What this stage found that no decision document contains** is collected at
the end rather than buried. The short version: an engine defect that was
handing wave-arounds to the entire field; a reduced model whose error was
dominated by two field-dependent terms rather than by anything it modelled
badly; a top-*k* selection that was answering the wrong question; and a
causal policy that was reading the future through the pit-stop cost."""))

    # ------------------------------------------------------------------
    cells.append(md("""### Setup

The frozen dials are loaded if they are there. If they are not, a clearly
labelled stand-in is used instead and every number below is a number about
the stand-in - which is stated rather than assumed, because this notebook
has to run in the replica project `tests/run_notebook.py` builds, where the
real calibration does not exist.

The race here is three hours rather than twenty-four. The headline
artefacts are produced by a script over the full seed bank; this notebook is
the argument, not the run."""))

    cells.append(code('''import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent / "src"))

from endurance import ClassDials, Compat, RaceConfig, run_race
from endurance import assets, benchmark, causal

DIALS = Path.cwd().parent / "data" / "processed" / "dials_imsa.json"

if DIALS.exists():
    CONFIG = RaceConfig.load(DIALS)
    CONFIG.duration_s = 3 * 3600.0
    PROVENANCE = f"frozen dials from {DIALS.name}"
else:
    CONFIG = RaceConfig(
        name="stand-in", series_code="imsa", duration_s=3 * 3600.0,
        classes=[ClassDials(
            series_code="imsa", class_name="GTP", base_pace_s=100.0,
            deg_slope_s_per_lap=0.012, pace_spread_s=0.5, lap_noise_s=0.35,
            caution_rate=0.30, caution_mean_dur_s=600.0, green_stint_laps=30.0,
            fuel_per_lap=1 / 30, fuel_per_lap_caution=0.6 / 30,
            tyre_life_laps=60.0, pit_time_mean_s=45.0, pit_time_std_s=3.0,
            n_cars=10, traffic_penalty_s=0.8)])
    PROVENANCE = "STAND-IN DIALS - every number below is about the stand-in"

FOCAL = "GTP-05"          # decision 8: fifth-fastest of the headline class
FILLS = (0.25, 0.5, 0.75, 1.0)

print(PROVENANCE)
print(f"{CONFIG.name}: {CONFIG.duration_s / 3600:.0f}h, "
      f"{CONFIG.total_cars} cars, series {CONFIG.series_code}")'''))

    cells.append(md("""**A note on the focal car.** Decision 8 puts it at P5 of the headline
class, which has no referent in this engine: `_build_field` starts every car
at *t* = 0 on lap 1 and there is no grid. So P5 is read here as *fifth-
fastest drawn base pace in the class on that seed*, which is the property
decision 8 is reaching for - it holds the focal car's competitive position
constant across the bank, which a fixed car id would not."""))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 1 - the reduced race, and what it cannot know

Stage one runs a *reduced* race: one car, pace, degradation, its own lap
noise, the regulation pit cost. Two of the engine's lap-time terms are
missing from it and cannot be added, because neither is computable for one
car alone. `_caution_lap` prices a caution lap off the gap to whoever is
ahead in the queue, and `_traffic_penalty` reads the whole field's current
pace.

That is not a shortcut. It is the reason there are two stages.

The consequence is measurable, so it is measured rather than argued
about."""))

    cells.append(code('''ctx = benchmark.FocalContext(CONFIG, seed=11, car_id=FOCAL)
reference = benchmark.forced_only_plan(ctx)

before = dict(caution_lap_s=ctx.caution_lap_s, green_offset_s=ctx.green_offset_s)
report = benchmark.anchor(ctx, plan=reference)

print(f"unanchored, the reduced race charges the safety car's own lap: "
      f"{before['caution_lap_s']:.1f}s")
print(f"measured off a reference run, this car's caution laps cost:      "
      f"{report['caution_lap_s']:.1f}s  "
      f"({report['caution_laps']} laps, {report['wave_laps']} of them wave credits)")
print(f"green laps carry a residual of {report['green_offset_s']:+.2f}s "
      f"over {report['green_laps']} laps")'''))

    cells.append(md("""The anchor follows the convention `pitstop.py` set: **the level is measured
and only the shape is modelled**. A caution lap's length depends on the gap
to the car ahead and on whether a wave credit arrives, both properties of
the field; a green lap carries a traffic penalty that is the same. So both
are read off one reference run of the race in question rather than guessed
at.

Without it, arrival times ran ahead of the engine's by 90 to 230 seconds and
growing, and every plan stage one proposed was refused a stop somewhere."""))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 2 - gate one: the DP is the brute force

With cautions and traffic switched off, the reduced race *is* the real one,
so the two have to agree exactly - and on the plan, not only on the total.

The comparison is run on a race short enough to enumerate exhaustively. That
is the only way this gate means anything: brute force on a three-hour race
is not available, and a gate you cannot run is a comment."""))

    cells.append(code('''gate = RaceConfig(
    name="gate", series_code="imsa", duration_s=1500.0,
    classes=[ClassDials(
        series_code="imsa", class_name="GTP", base_pace_s=100.0,
        deg_slope_s_per_lap=0.02, pace_spread_s=0.8, lap_noise_s=0.3,
        caution_rate=0.0, caution_mean_dur_s=400.0, green_stint_laps=6.0,
        fuel_per_lap=1 / 6, fuel_per_lap_caution=0.6 / 6, tyre_life_laps=12.0,
        pit_time_mean_s=40.0, pit_time_std_s=2.0, n_cars=4,
        traffic_penalty_s=0.0)])

rows = []
for seed in range(5):
    kw = dict(clairvoyant=True, fill_levels=(0.5, 1.0), k=12)
    dp = benchmark.search_plans(gate, seed, "GTP-02", labels_per_state=12,
                                fuel_quantum=0.01, per_family=12, **kw)
    bf = benchmark.brute_force(gate, seed, "GTP-02", max_stops=4, **kw)
    optima = {p.stops for p in bf if p.sort_key == bf[0].sort_key}
    rows.append(dict(seed=seed, dp_laps=dp.best().laps, bf_laps=bf[0].laps,
                     time_agrees=abs(dp.best().race_time_s - bf[0].race_time_s) < 1e-9,
                     plan_agrees=dp.best().stops in optima,
                     plans_enumerated=len(bf)))

print(pd.DataFrame(rows).to_string(index=False))'''))

    cells.append(md("""### And the same DP replayed through the engine

Agreeing with a brute force that shares its arithmetic is a weaker claim
than it looks - both could be wrong together. So the plan is also handed
back to the engine on a config where everything stage one omits is switched
off. If the reduced race's arithmetic *is* the engine's, the predicted laps
and race time come back to the microsecond."""))

    cells.append(code('''rows = []
for seed in range(3):
    plan = benchmark.search_plans(gate, seed, "GTP-02", clairvoyant=True,
                                  fill_levels=(0.5, 1.0), k=1).best()
    result = run_race(gate, strategies={"GTP-02": plan.runner()}, seed=seed)
    row = result.classification().set_index("car_id").loc["GTP-02"]
    rows.append(dict(seed=seed, predicted_laps=plan.laps, engine_laps=int(row["laps"]),
                     residual_s=float(row["race_time_s"]) - plan.race_time_s))

print(pd.DataFrame(rows).to_string(index=False))'''))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 3 - gate two: the lane

The no-caution gate structurally cannot provide this one: it removes the
windows in question. So the lane is checked separately, in both series and
on a staged class as well as a leading one - IMSA releases prototypes before
GTs (art. 46.3.1) and WEC releases everyone together (art. 14.6.5), so a
search that had quietly reimplemented the rule would pass one and fail the
other.

**Legality turned out to belong to stage two, not stage one.** The search
places its stops against the *reduced* race's arrival times and the
engine's differ, so a stop that was legal when it was computed can land in a
window that opens a caution lap later, or one that never opens at all.

The answer is not a cleverer search. A crew that arrives to a shut lane
comes round again; it does not abandon the stop. `PlanRunner` defers, taking
the stop at the next crossing where the lane is open, reading nothing but
`state.pit_lane_open` - the same thing a marshal shows the driver."""))

    cells.append(code('''lane_cfg = RaceConfig(
    name="lane", series_code="imsa", duration_s=7200.0,
    classes=[ClassDials(
        series_code="imsa", class_name="GTP", base_pace_s=100.0,
        deg_slope_s_per_lap=0.02, pace_spread_s=0.8, lap_noise_s=0.3,
        caution_rate=0.35, caution_mean_dur_s=420.0, green_stint_laps=6.0,
        fuel_per_lap=1 / 6, fuel_per_lap_caution=0.6 / 6, tyre_life_laps=12.0,
        pit_time_mean_s=40.0, pit_time_std_s=2.0, n_cars=4,
        traffic_penalty_s=0.0)])

plans = benchmark.search_plans(lane_cfg, 5, "GTP-02", clairvoyant=True,
                               fill_levels=(0.5, 1.0), k=5).plans

def refusals(plan, defer):
    result = run_race(lane_cfg, strategies={"GTP-02": plan.runner(defer=defer)},
                      seed=5)
    mine = result.laps[result.laps["car_id"] == "GTP-02"]
    return int(mine["stop_refused"].notna().sum()) if "stop_refused" in mine else 0

print(pd.DataFrame([
    dict(plan=i, stops=len(p.stops),
         refused_rigid=refusals(p, False), refused_deferring=refusals(p, True))
    for i, p in enumerate(plans)]).to_string(index=False))'''))

    cells.append(md("""Deferring makes every plan runnable, which means a plan can no longer
*fail* the lane gate - so the gate stops being the informative thing and the
deferral count takes over its job. If deferrals are common, stage one is
placing stops badly, and the cost turns up in the position rather than in a
flag."""))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 4 - stage two, and why top-*k* by time was the wrong question

Stage two replays each candidate through the full engine against the frozen
background field and ranks on class position. Decision 2's focal-car
arrangement means the only thing moving between these races is the focal
car's stops - though not quite: through traffic, the focal car's stops
perturb its rivals too, which is 02a's traffic correction doing exactly what
it was built to do.

Ranking the candidates by predicted time and taking the top *k* is the
obvious thing and is close to useless. On a six-hour race the top twenty
plans differ by about **six seconds** of predicted time while the reduced
race's own error is nearer **thirty**, and what decides the outcome is
whether the final lap falls before or after the flag - a margin of seconds.
The ordering inside that band carries no information.

So candidates are selected across **families** - plans grouped by stop count
and by how many of those stops fall under caution - and only then filled
from the time ordering. Breadth is what stage two can use; depth buys twenty
spellings of one idea."""))

    cells.append(code('''result = benchmark.build_benchmark(
    CONFIG, seed=11, car_id=FOCAL, k=20, fill_levels=FILLS, passes=1,
    labels_per_state=1, fuel_quantum=0.10)

print(f"candidates scored : {len(result.rescored.scored)}")
print(f"families covered  : {len({p.family for p in result.search.plans})}")
print(f"benchmark         : P{result.best.class_pos}, {result.best.laps} laps, "
      f"{len(result.best.plan.stops)} stops, {result.best.deferred_stops} deferred")
print(f"forced-only ref   : P{result.reference.class_pos}, {result.reference.laps} laps")
print(f"winner sat at rank {result.winner_rank} of {len(result.rescored.scored)} "
      f"in the time ordering")

table = pd.DataFrame([
    dict(stops=len(e.plan.stops), caution_stops=e.plan.caution_stops,
         predicted_t=round(e.plan.race_time_s, 1), engine_t=round(e.race_time_s, 1),
         laps=e.laps, class_pos=e.class_pos, deferred=e.deferred_stops)
    for e in result.rescored.scored]).sort_values("class_pos")
print()
print(table.head(10).to_string(index=False))'''))

    cells.append(md("""The forced-only plan is always scored alongside the candidates. It is
feasible, so a reference that loses to it is not a reference - and where it
wins, `winner_rank` says how far down the ordering the position-optimal plan
actually sat, which is the statistic that sizes *k* rather than a number
chosen in advance."""))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 5 - is the reference stable?

A benchmark whose answer moves when a pruning parameter moves is not yet a
benchmark. Two search resolutions are compared on the same seeds: a coarse
one that completes without truncating its frontier, and a fine one that does
not.

The claim being tested is *not* that individual races agree. They cannot -
the flag boundary is a knife edge that no model refinement resolves at six
hours' range. It is that the **distribution** of the position delta agrees,
which is what decision 10 commits to reporting anyway."""))

    cells.append(code('''rows = []
for seed in (11, 12):          # two here; the full bank is a script's job
    for name, kw in (("coarse", dict(labels_per_state=1, fuel_quantum=0.10)),
                     ("fine", dict(labels_per_state=4, fuel_quantum=0.02))):
        r = benchmark.build_benchmark(CONFIG, seed, FOCAL, k=20,
                                      fill_levels=FILLS, passes=1, **kw)
        rows.append(dict(seed=seed, search=name,
                         delta=r.reference.class_pos - r.best.class_pos,
                         rank=r.winner_rank, capped=r.search.pruned_by_cap))

stability = pd.DataFrame(rows).pivot(index="seed", columns="search", values="delta")
stability["agree_within"] = (stability["coarse"] - stability["fine"]).abs()
print(stability.to_string())'''))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 6 - the causal reference

The clairvoyant reference is an upper bound and is meant to be: it knows
which stop will be a quick one. The causal one is the other end of the pair.

**It is exactly computable, and decision 17 is why.** Caution episodes are
drawn non-overlapping with exponential durations and green gaps are
exponential too, so the flag is a two-state continuous-time Markov chain and
the probability of being under caution a given interval later has a closed
form. The merge in the old draw would have destroyed that - the union of
overlapping exponentials is not exponential - which is the concrete cost the
decision avoided.

Backward induction runs over `(clock, phase, fuel, tyre age)`. Phase carries
the caution's **age in caution laps**, because both rulebooks reopen the
lane a set number of laps after the field forms up, so a single "under
caution" state cannot answer whether a stop is available.

The policy is then rolled forward against the realised timeline - the
cautions fall when they fall - but every decision is a table lookup on the
current state. Nothing past *t* is read."""))

    cells.append(code('''ctx = benchmark.FocalContext(CONFIG, seed=11, car_id=FOCAL)
benchmark.anchor(ctx, plan=benchmark.forced_only_plan(ctx))

lam_green, lam_caution = causal.hazards(CONFIG.classes[0])
print(f"green -> caution: one every {1 / lam_green / 60:.1f} min")
print(f"caution -> green: mean {1 / lam_caution / 60:.1f} min, memoryless")

policy = causal.solve_policy(ctx, bucket_s=60.0)
print(f"policy grid {policy.value.shape}, expected laps from the flag fall: "
      f"{policy.value[0, 0, -1, 0]:.2f}")

rows = []
for seed in (11, 12, 13):      # three here; the full bank is a script's job
    c = benchmark.FocalContext(CONFIG, seed, FOCAL)
    benchmark.anchor(c, plan=benchmark.forced_only_plan(c))
    pol = causal.solve_policy(c, bucket_s=60.0)
    causal_p = causal.causal_plan(c, pol)
    clair = benchmark.build_benchmark(CONFIG, seed, FOCAL, k=20,
                                      fill_levels=FILLS, passes=1).best
    scored = benchmark.rescore(CONFIG, seed, FOCAL,
                               [causal_p, benchmark.forced_only_plan(c)])
    rows.append(dict(seed=seed,
                     causal_pos=scored.scored[0].class_pos,
                     clairvoyant_pos=clair.class_pos,
                     reference_pos=scored.scored[1].class_pos,
                     foreknowledge=scored.scored[0].class_pos - clair.class_pos))

print()
print(pd.DataFrame(rows).to_string(index=False))'''))

    cells.append(md("""The `foreknowledge` column is the thing this stage exists to produce: how
many places knowing the future is worth. **03 scores against the causal
column**, and the clairvoyant column is the ceiling neither a strategy nor
an agent can be expected to reach.

One limitation, stated rather than buried: the restart-proximity limb of
art. 46.3.3 depends on the previous episode's end, which the causal state
does not carry. So the causal reference is slightly optimistic about late
cautions - the one place it knows less than `pitstop.lane_status` does."""))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 7 - what ships

Decision 6 asks for artefacts and not just a notebook, because 03 has to
evaluate on the races 02 scored or the comparison is between two different
questions.

The **sweep fifty are the first fifty of the headline two hundred**, not a
separate draw: a sweep asks how a claim moves as a dial moves, and the
cleanest version of that compares against the same races the claim was made
on. The **held-out fifty are genuinely disjoint**, for the opposite reason -
they exist to answer whether a roster chosen on the headline races
generalises.

The background field is held as a per-car map even though every value is
identical today, because decision 2 makes the background an assumed
parameter that gets swept, and one sweep worth running is a *mixed* field:
02a's traffic correction only bites when the field is out of phase, and a
uniform background pits in lockstep and hides it entirely."""))

    cells.append(code('''bank = assets.draw_seed_bank(CONFIG)
field = assets.freeze_background(CONFIG)

print(f"headline {len(bank.headline)}, sweep {len(bank.sweep)} "
      f"(prefix: {bank.sweep == bank.headline[:len(bank.sweep)]}), "
      f"held out {len(bank.held_out)} "
      f"(disjoint: {not set(bank.held_out) & set(bank.headline)})")
print(f"dials fingerprint {bank.provenance['dials_fingerprint']}")
print(f"background field: {len(field.strategies)} cars, all "
      f"{field.provenance['uniform_strategy']}")

OUT = Path.cwd().parent / "data" / "processed"
if OUT.exists():
    bank.save(OUT / f"seed_bank_{CONFIG.series_code}.json")
    field.save(OUT / f"background_field_{CONFIG.series_code}.json")
    print(f"written to {OUT}")
else:
    print("no data/processed here - artefacts not written (replica project)")'''))

    # ------------------------------------------------------------------
    cells.append(md("""## Findings

Collected here rather than left in the parts that produced them.

**1. Wave-arounds were being handed to the entire field.** `_take_wave` is
called from `_next_lap`, which runs before `_set_lap` refreshes the car's
lap window, so the car at the line still carried its previous lap's window,
its progress fraction saturated at the clip, and it read a full lap further
round than it was. Inflated by a lap it became the apparent class leader,
and every car genuinely alongside it was declared lapped. Ninety-nine
wave-arounds in one six-hour race, on a field that was never lapped. Landed
as an 02a correction; waves fall to single figures.

The good news attached to it: 02a's headline compression claim **survives**.
Class spread with waves suppressed is identical to spread with them on, seed
for seed. Compression really is what closes the field. But
`test_the_field_bunches_up_instead_of_spreading_out` was passing on the
defect - on the fixed engine the direction still holds on the mean and its
per-episode hit-rate threshold does not, so that test wants re-stating on
the mean.

**2. The reduced race's error was dominated by what it could not model, not
by what it modelled badly.** Wave credits were worth about 800 seconds over
six hours and compression about 185, against a green-lap residual of five.
Both are field-dependent. The fix was to measure the level rather than
improve the model, which is the convention `pitstop.py` already set.

**3. Left free, the search buys laps by sitting in the pit lane.** A lap's
character is fixed when it starts, so a stop that delays the next crossing
past the end of a caution converts a safety-car lap into a green one - worth
about sixty seconds for twenty-five seconds of stop. The unconstrained
optimum took sixteen stops where the rules required nine. It is bounded by
the field rather than by the search, so `max_stops` is a budget on the
search and not a claim about strategy, and it is labelled as such.

**4. Top-*k* by predicted time was answering the wrong question**, and this
is the one most likely to matter elsewhere. When a model's error exceeds the
spread of the candidates it is ranking, its ordering is noise and deeper *k*
does not help - breadth does. Any later stage that ranks candidates on a
modelled quantity should check that spread against that error before
trusting the order.

**5. The causal policy was reading the future through the pit-stop cost.**
`FocalContext.pit_cost` asks the realised timeline whether the caution
discount applies, which is the right question for pricing a plan against a
race that has happened and the wrong one inside a policy that is not allowed
to know. Caught by the test asserting the policy depends on the seed only
through three named scalars - not by reading the code, which had been read
several times.

**6. Runtime is all search.** Re-scoring twenty candidates through the full
engine costs about a second; the search costs fifty to a hundred. Coarsening
the state key made it three times faster *and* stopped the frontier being
truncated at its cap, which is why the coarse settings are the defaults: a
search that finishes at low resolution is easier to defend than one cut off
at high resolution.

## What 02b does not do

- No position-valued DP. Position is not additive; that is what stage two is
  for, and it is a re-ranking rather than an optimisation.
- No repair of a plan beyond deferring a stop to the next open lane.
- The causal reference does not model the restart limb of art. 46.3.3.
- Dials still come from one race per series, and where the frozen dials are
  absent every number above is about the stand-in."""))

    return cells


def write(name, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent / "notebooks" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} - {len(cells)} cells")


if __name__ == "__main__":
    write("02b_benchmark.ipynb", build_02b())
