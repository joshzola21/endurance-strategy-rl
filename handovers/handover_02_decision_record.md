# Handover — the 02 decision record

Written so a fresh thread can pick this up cold. Read `PROJECT_BLUEPRINT.md`
first, then this, then the 02b handover, then the 03a handover, then the
notebook for the stage in hand.

Everything below was chosen deliberately in a decision quiz, in the same
spirit as the four decisions taken in the 01 thread. It should not be
silently reversed later.

**Amended seven times since it was written.** Decisions 17, 9 item 5 and 9
item 3 have been replaced, and 03a settled three questions this record left
open for it. The amendment log is at the foot of this document and the
superseded text is kept there rather than deleted.

---

## Where we are now

| Stage | Status |
|---|---|
| 00 — data recon | done; **calibration wrong, and now diagnosed** — see below |
| 01 — the race engine | done, Part 6 re-run outstanding |
| 02a — engine corrections | done, plus one correction landed from 02b |
| 02b — the per-race benchmark | done; per-seed artefacts still not produced |
| 02c — human-style strategies and the comparison | done |
| 03a — the gym wrapper | done |
| 03b — RL training | apparatus done and gated; **no agent result yet** — see amendment 8 |
| **04 — the Streamlit app** | **next** |

02 was split. The benchmark is what makes 03's numbers mean anything and is
the part most likely to expose an engine bug, so it went first. A
half-finished 02c blocks nothing; a half-finished 02b blocked everything.

**00's calibration is not merely unverified; it is wrong, and the fault is
locatable.** 03b read the frozen dials rather than assuming them. The base
paces survive — 97.9 s at Daytona and 209.4 s at Le Mans are both credible —
and everything that requires the query to be scoped to *one running of the
race* does not. `imsa.json` describes a 216-hour race with 49 GTD entries
carrying DPi, GTLM and GTDPRO in one field, which is nine editions of
Daytona pooled into one config. Green stints come out at 199 laps (5.4
hours, longer than `max_driver_stint_s`), because `stint_number` does not
reset across events. Every `pit_time_std_s` exceeds its mean by two to five
times, so the pit column is capturing something other than service time. And
HYPERCAR degradation is negative: the field-relative frame shrinks a slope
but cannot invert one. **The calibration queries are not scoped per event.**
That is a far more tractable problem than "the numbers look wrong", and it
should go in front of the re-run.

**01 is no longer frozen.** 02a changed the engine, so 01's Part 6
validation table must be re-run and its numbers will move slightly. Say so
in 01 rather than quietly regenerating it. 02a's own Part 6 comparison table
wants re-running too, against the engine as corrected by 02b.

---

## Three findings from reading the 01 code

These came out of reading `engine.py` against the 02 design, and two of them
changed the plan.

**1. The paired comparison was not actually paired.** `engine.py`'s
docstring says all randomness is drawn in advance. The caution timeline and
the per-car pace draw are, which is why
`test_caution_timeline_is_independent_of_strategy` passes. But lap noise
(`_next_lap`) and pit cost (`_apply_pit`) are drawn from the same `self.rng`
*during* the race, consumed in event-queue order. Change when one car pits
and every subsequent noise draw for every car shifts. At roughly a second of
lap noise over a few hundred laps, that is tens of seconds of accumulated
difference per car — the same order as the strategic effects being measured.
Fixed in 02a.

**2. Splash-and-dash currently has no value.** `_apply_pit` ignores
`refuel_to` and `change_tyres` entirely; a 30% splash costs exactly what a
full service costs. The strategy is not merely unimplemented, it is
unimplementable until the regulation pit layer exists. That promotes the pit
layer from an improvement to a hard prerequisite for the roster.

**3. Traffic cannot be exploited by stop timing.** `_traffic_penalty`
counts a car ahead as a blocker if its *base pace* is slower, ignoring tyre
age, so traffic is a static property of the field. "I will emerge into
traffic if I stop now" — one of the two main arguments on any pit wall —
cannot happen. Fixed in 02a.

---

## The decisions

### What is being measured

**1. Class finishing position is the primary score; race time is the
low-variance diagnostic. Both reported side by side.** The 01 engine
simulates the whole field and derives position from cumulative race time, so
position is representable here in a way it was not in the F1 thesis. Where
the two agree the claim is robust; where position moves and time does not,
that is a genuinely positional effect worth showing. Pre-registered now so
it cannot be chosen after seeing results.

**10. The headline statistic is the paired position delta against the fuel
baseline on the same race, reported as a distribution, not a mean.** Common
random numbers make the pairing free once 02a lands, and "gains a place in
40% of races, loses one in 12%" is both stronger and far more legible than a
mean of 4.7. Budget: **200 paired seeds** per strategy per series for
headline claims, **50** for sweeps, on a fixed bank saved to disk, with a
**disjoint 50-seed held-out bank** for anything the roster gets selected on.

*Clarified at 02c.* "Per strategy per series" means **two banks and two
tables, never a pooled row.** The series are not two samples of one
population: the rulebooks differ in ways that make a lever live in one and
dead in the other — see the scope note under decision 9 item 5 — so an
averaged row would report the mean of a measurement and a non-measurement.
`draw_seed_bank` should be given a **per-series `draw_seed`**. Its default
is shared, which would hand IMSA and WEC identical integer lists against
different dials; harmless in itself, but two tables of the same seed numbers
invite a reader to take them as paired, and they are not.

**14. A strategy that loses is a finding, not a bug — with one exception.**
If it loses in a way that violates a stated invariant (a stop under caution
costing more than the same stop under green, say), that is a bug and gets a
test. Everything else is reported, including plausible strategies that do
not work. This is what stops the roster being quietly tuned into agreeing
with expectations, and it follows the negative-results practice already
established in the methods draft.

### Who is being measured

**2. Focal car.** One car runs the strategy under test; the rest of the
field runs a fixed background strategy, frozen and saved to disk as a
project asset that 03 reuses unchanged. Clean attribution, and it matches
how the agent will sit in the field. The background choice is a new assumed
parameter: it goes in `ASSUMED_FIELDS` and gets swept.

**8. The focal car starts P5 of the headline class**, where the measurement
has the most resolution — P1 can only lose places, and a strung-out front
runner is insensitive to strategy. Starting-slot rotation across every car
in the class is reported as **one figure**, answering whether a strategy's
value depends on where you start.

Whole-class assignment (01's `assign_strategy`) survives as a one-line
diagnostic answering the different question "what if everyone did this".

### What the race is

**3. Fuel binds by default.** `tyre_life_laps` stays at twice the fuel
stint, which is the endurance norm and is honest to the data. Consequence:
**the tyre-saver and tyre-driven double-stint strategies are cut**, and the
notebook says why. A tyre-bound race is retained as a named regime in the
sweep, which should show the strategy ranking flipping when the binding
constraint changes.

**4 / 7. Pit stops are structured by the rulebook, in a new `pitstop.py`
layer**, not by a single `pit_time_mean_s`. Encode only the rules that
change the shape of a stop — minimum pit time, refuel and tyre-change
sequencing, refuel duration — and leave driver-time limits out for now. The
layer keeps the old single-number behaviour as a **legacy mode**, so 02a
must reproduce 01's numbers exactly with the layer off before it is turned
on. Then turn it on and show what changed; that comparison is itself a
result.

This is where the two series stop being one model with different constants,
which is what justifies simulating both. It is also the likeliest place the
legacy gate fails, since `calibrate.py` currently feeds a number where it
will need to feed a structure. That failing is informative.

**17. Caution episodes are drawn non-overlapping — by an alternating
renewal process, not by rejection.** *(Amended; the original text is in the
amendment log. Replacement text taken from
`02a_caution_verification.ipynb` Part 6, as the blueprint directed.)*

Durations are `rng.exponential`, which is memoryless, so the causal
benchmark is exactly computable. The merge of overlapping episodes in the
original draw destroys that, and the fix stands. But the reason originally
given for it does not, and the correct reason is a stronger argument.

The original claim was that rejection sampling — redraw the whole
configuration until nothing overlaps — reweights towards shorter length
vectors, so lengths stop being marginally exponential. **That diagnostic
does not fire.** Conditioned lengths have a coefficient of variation of 1.00
and pass a Kolmogorov–Smirnov test against an exponential comfortably. The
marginal shape survives the conditioning intact.

**What the conditioning destroys is the scale.** The mean episode falls from
3240 s to about 2300 s: still exponential, but a different exponential from
the one calibration asked for. Tested against the exponential the model
*claims* to draw rather than one fitted to its own output, it is rejected at
a p-value with 170 zeros in it. And the coverage consequence is worse than
the merge it was competing with — rejection sampling lands a 0.30 target at
0.214, a 29% shortfall, against the merging draw's 15%.

So: right decision, wrong reason. The engine implements an **alternating
renewal process with a stationary start**, which holds both the shape and
the scale. Report the merge rate at the calibrated caution rate once, as a
diagnostic showing what the old draw gave up. Anyone who checks the cv and
finds 1.00 should find this paragraph before concluding the supersession was
unjustified.

**Field compression under caution is in.** Cautions currently slow everyone
proportionally, so a caution costs nothing positionally and gains nothing.
With compression, a caution stop is cheap *in position* — which is the
effect `pit_caution_discount` exists as a proxy for and cannot represent.
This is the single highest-value addition available to the engine, and it
means the assumed dial stops carrying the entire caution story alone.

Wave-arounds come with it, which is what gives the lap-down strategy a real
lever: without them a lap can be lost but never regained.

### The roster

**9. Five strategies, all parameter-free.** No per-race tuning — the methods
draft rejects per-circuit tuning on the grounds it collapses a benchmark
into a second oracle, and the same argument applies here. Parameter-free
means each strategy *derives* its numbers from the dials rather than being
handed them:

1. **Fuel-window baseline.** Stop when the tank forces it. The null, and the
   reference for every paired delta.
2. **Caution gambler.** Pits under caution only when fuel used is enough
   that the stop would not be repeated within the same window — the
   threshold falls out of `fuel_per_lap` and the time remaining.
3. **Track-position defender.** *(Amended at 02c — see the amendment log.)*
   Refuses a voluntary stop that would let a rival past — one that is behind
   now and would be ahead once the stop is served — accepting a worse fuel
   window. The rival is positional, never a named car.

   *Why the wording changed.* "Whoever is directly ahead on the road" is a
   car a stop cannot drop you behind, so it does not name a decision. The
   two readings that do — the nearest car a stop would concede a place to,
   and the car you would emerge behind — turn out to be the same car,
   because you come out behind a rival exactly when its arrival is nearer
   than the stop is long. The amendment states it from the side that is
   computable, and is a clarification rather than a reversal.

   *And how it must be computed.* Through the engine's own position rule,
   not by differencing two `race_time_s` values. At the decision the car is
   at the line on lap *N* and every slower rival in its class is still on
   lap *N - 1*, so a same-lap car behind mostly does not exist, and where it
   does the difference compares crossings of different laps. Written that
   way the defence fires on a negative number about a quarter of the time
   and still looks like a working strategy. Rivals are projected to a common
   lap count instead — one lap down arrives at
   `lap_start_t + lap_expected_s`, which `CarState` already carries — and
   arrival times are compared at the same lap, which is what position means
   here.
4. **Splash-and-dash planner.** Works backwards from `duration_s` and the
   fuel window; the final stop's length falls out arithmetically. Depends
   entirely on `pitstop.py`.

   *Its arithmetic is shared from 03a.* The fill calculation was lifted out
   of the strategy into `strategies.fuel_to_the_flag`, because the agent's
   action space needs to reach the same fill and a second implementation of
   it would drift. `_remaining` stays private; one narrow function is
   exposed rather than the module's arithmetic wholesale, on the same
   argument that put `laps_down` on `RaceState`. The two guards travel with
   it — a lap of margin, and never asking for less than is aboard.
5. **Lap-down defender.** *(Amended at 02c — see the amendment log.)*
   Declines a voluntary stop that would concede a whole lap to the class
   leader; and, once already a lap or more down and under caution, declines
   **any** voluntary stop. Clause one's threshold derives from `stop_cost`
   and the class leader's lap time. Clause two has no threshold, for the
   reason below.

   The second clause is a **standing condition**, and *tightened at
   implementation* it is stronger than a threshold rather than a loose
   version of one. Behind the safety car every car runs the same lap, so two
   cars hold a fixed offset within that lap, and eligibility — being further
   round the lap than the class leader — holds for a share of each caution
   lap equal to that offset. A stop delays the next lap start by its own
   cost and so cuts the offset by `cost / caution_lap`. **Every voluntary
   stop therefore costs eligibility. None is free, so there is no boundary
   to test and nothing to compute.**

   It is also not conditioned on a wave being imminent, because a causal
   strategy cannot know that — the Pass-Around's time depends on the
   caution's start and the Final Wave-By's on its end, and neither is in
   `RaceState`. The cost of standing rather than timing is that a lapped car
   under caution defers a stop it could have taken, which is the trade this
   strategy exists to represent rather than an inefficiency in it.

   **There is no action that takes a wave-around**, and there never was one.
   `_take_wave` is engine-internal, is called once per car per lap, and is
   unconditional on the frozen eligible set. The only lever a strategy has
   is whether a stop has moved it out of that set, because a stop calls
   `_set_lap` with the post-stop time and so moves the car's track fraction
   backwards.

   *Why the two clauses do not contradict each other.* Being lapped is the
   precondition for the refund the second clause protects, so the first
   clause is working to avoid the state the second clause is for. The two
   are read in order: clause one tries to avoid the state, clause two
   governs conduct once clause one has already failed. Stated here so nobody
   later resolves the apparent tension by deleting one of them.

   *Considered and rejected.* A caution-age variant, carrying elapsed
   caution time as `causal.py`'s state does and defending eligibility only
   near the expected wave. Sharper, and rejected because it needs an
   expectation of when the wave falls, which the dials do not contain — in
   any form that would be a parameter, and decision 9's whole content is
   that these five are parameter-free. Revisit if the standing condition
   turns out to be defer-heavy enough to cost the strategy laps; that would
   be a measured reason rather than an argument.

   *Scope, recorded now rather than discovered in the results.* Wave one is
   announced at `start + open_delay_laps × caution_lap_s`, and the lane
   opens to the earliest class at
   `start + (open_delay_laps + min stagger) × caution_lap_s`. IMSA's minimum
   stagger is 1.0 lap and WEC's is 3.0. **The Pass-Around is therefore
   frozen a full caution lap before any car may take a voluntary stop, in
   both series, and cannot be forfeited by a decision.** Only IMSA's Final
   Wave-By (art. 46.4.1) is forfeitable at all, and **WEC runs one wave, so
   in WEC the second clause is inert** and the strategy reduces to its
   first. Its WEC row is not a broken result and should not be read as one.
   A forced stop — out of fuel under a shut lane — can still forfeit
   eligibility; that is the rules acting on the car rather than the
   strategy, and it belongs in the deferral-style diagnostics rather than in
   the strategy's score.

   *Measured at implementation, and worse than the a priori scope note.*
   Clause two fired in neither series, over sixty seeds a side and again
   over the full two hundred. WEC was expected. IMSA was not, and the reason
   is not the rulebook: 02a's
   compression takes the headline class from a ten-lap spread to about one,
   so the focal car is essentially never a lap down and the situation clause
   two responds to has been largely engineered out of the engine. The
   strategy came out level with the null in 97% of IMSA races and 90% of WEC
   ones — and at two hundred seeds it is 3% gained against 3% lost in IMSA
   and 7.5% against 8.5% in WEC, symmetric either side of zero, which is
   what inertness looks like rather than a small effect. What movement there
   was came from clause one. **This is
   evidence bearing on 02a's finding 1** — that compression's magnitude is
   unvalidated and may close the field too hard — and the two questions
   should be read together. Report it under decision 14; do not tune the
   strategy to make it visible.

**What the track-position defender did, measured — and a correction.** On
sixty seeds it looked as though the strategy lost in IMSA (33% gained
against 47% lost) and gained in WEC (50% against 25%), and that was written
down here as a result whose sign came from the rulebook. **At two hundred
seeds it does not survive.** IMSA is 37% against 34%, WEC 37% against 39.5%,
median zero in both and a tenth percentile three places down in both: a coin
flip either side, if anything marginally worse in WEC than in IMSA. The
direction reversed.

The strategy itself is still decision 14's case — a plausible idea that does
not work, reported rather than tuned — and the mechanism still holds: under
caution the field closes to the queue spacing, so nearly any stop lets
somebody past, and it declines about three quarters of caution stop
opportunities against two fifths under green. What does not hold is the
claim that the *sign* differs by series.

**And the lesson, which is 02b's third finding turned back on the reporting
rather than on the search.** Sixty seeds put a two-place gap between the
series on a statistic whose per-seed spread runs to three places. That is a
ranking read off a difference smaller than its own error, which 02b already
established is noise, and it went into two authoritative documents before
anybody re-ran it. **No roster claim goes in a document until the difference
has been checked against the spread, and the seed count goes in the sentence
alongside the number.** The one asymmetry that did survive to two hundred
seeds is the splash-and-dash planner's: 53.5% gained and 18.0 s saved a stop
in WEC against 34.5% and 10.7 s in IMSA, which is art. 12's sequential
refuelling doing what the rulebook says it should.

**A defender's discretion is one lap wide.** Measured at implementation and
recorded because it bounds what 3 and 5 can possibly be worth: the fuel
window opens at a lap and a half in hand and the engine forces a stop below
one lap, so declining buys exactly one more lap and then the rules take the
decision back. Under caution the burn is lower and the lap is worth more,
which is the right way round. This falls out of the existing convention
rather than being chosen, and neither defender should be read as having a
larger lever than that.

3 and 5 nearly collapsed into one — without field compression they pull the
same lever. Compression is what separates them, so if it is ever backed out,
merge them rather than shipping two rows that say the same thing. Under the
amended item 5 their referents differ as well, so no merge is in prospect:

| | Track-position defender | Lap-down defender |
|---|---|---|
| Referent | whoever is directly ahead on the road | the class leader |
| Granularity | sub-lap, in seconds | a whole lap, plus track fraction |
| Flag | any | the second clause, caution only |

**The anchor does not cross into the roster.** 02b's measured caution-lap
level stays in `benchmark.py` and no strategy may reach it. Three reasons,
settled at the opening of 02c because the caution gambler's threshold looked
as though it might need the figure:

* It is not reachable through the interface. A strategy is
  `(CarState, RaceState) -> PitDecision`; the anchor is read off a reference
  run of the race. Opening a channel to it means opening the same channel at
  03a, and an agent trained across seeds cannot carry a per-seed level.
* It is per-race tuning under another name, which this stage's boundary
  constraint forbids.
* The gambler's threshold does not want it. The condition is a *count* —
  would this stop be repeated inside the same window before the flag — not a
  price, and it derives from `time_remaining_s`, `base_pace_s`,
  `caution_rate`, `caution_pace_multiplier`, `fuel_per_lap` and
  `fuel_per_lap_caution`. The anchored level enters both branches of the
  comparison and largely cancels, because the caution timeline is drawn in
  advance and does not depend on strategy. It leaks in only through the
  integer boundary on laps remaining, so the threshold should not be a hard
  `ceil`: carry the margin and report its sensitivity.

The gap between the dial-derived caution lap and the anchored one is then a
quantity to *report*, not a defect to close. It is a component of the
strategy-to-benchmark gap, and 02c can decompose that gap into bad decisions
and a strategist working from a model of a caution lap the field does not
honour. Decision 14 material, and it belongs in the write-up.

**Lap-down status gets exposed properly.** It is computable today from
`RaceState.cars` and `CarState.laps_done`, but the logic belongs in one
place: add `RaceState.class_leader(class_name)` and
`RaceState.laps_down(car)` rather than repeating it per strategy.

*Proposed at 02c and withdrawn at implementation.* `RaceState.wave_eligible`
was to be added alongside these two. It is not needed: clause two of the
amended item 5 turns out to require only `laps_down(car) >= 1` and
`under_caution`, so the roster never asks the predicate and 02c adds nothing
to the engine. Whether 03a wants it for an observation row is an 03a
question.

*Answered at 03a: no, and conditionally.* The observation space has nine
rows and this is not one of them, on 02c's own evidence — under current
compression the focal car is almost never a lap down, so the predicate
would describe a situation the agent essentially never meets. The condition
is 02a's finding 1: if the re-run against real data shows compression
closing the field too hard, the situation returns and so does the row. It
is recorded that way rather than as a flat refusal, because the reason is a
measurement that may not hold.

**The hazard it was meant to contain is still there**, and is recorded as a
convention instead. At the moment a strategy is asked for a decision the car
has crossed the line and had `laps_done` incremented, but `_set_lap` has not
yet run, so its own `track_fraction(t)` is still measured against the lap it
has just *finished* and saturates at the clip. **A strategy must therefore
read its own progress as `laps_done` exactly and use `track_fraction` only
for other cars.** That is the same misreading that produced the engine's
wave-eligibility defect, it looks correct in review, and clause one of the
lap-down defender is currently the only place in the roster that touches
it.

*And it has a second half, which 03a had to build against.* The same moment
also makes a *gap* unreadable by differencing `race_time_s`, which is what
amendment 4 is about. 03a's observation space carries two gap rows, so the
projection now lives on `RaceState` as `gap_ahead_s` and `gap_behind_s`
rather than only inside `strategies._would_be_passed`. The strategy was
deliberately **not** rewired onto them: it answers a different question —
which rival a stop of a given length would let past — and refactoring it
would have moved 02c's published numbers for no gain.

### The benchmark

**5. Two-stage, per race.** Stage one: dynamic programming over stop laps
minimising the focal car's own race time against the frozen caution
timeline — cheap, and verifiable against brute force in the no-caution
limit. Stage two: re-score the top-*k* time-optimal plans through the full
engine against the frozen rival field, and take the best on position.

Position is not an additive per-lap cost, so DP is not valid on it directly.
The two stages keep tractability while still producing a position-valued
benchmark. **Field compression widens the gap between time-optimal and
position-optimal plans, so *k* needs to be generous** — check its
sensitivity rather than picking a number.

Both a **clairvoyant** version (knows when the cautions fell) and a
**causal** version (sees only what a strategist could see). The gap between
them is the value of foreknowledge, and the causal one is the reference for
03 — scoring against the clairvoyant benchmark alone would penalise a policy
for failing to predict the future.

The zero-noise regression gate is not optional. In the F1 thesis the same
gate caught a hindsight benchmark that passed the total-time check while
reconstructing a suboptimal plan.

### The sweeps

**11. One-at-a-time** around the default for `pit_caution_discount`,
`caution_pace_multiplier`, `traffic_penalty_s`, the background strategy and
the tyre-bound regime, **plus one 2-D grid of `pit_caution_discount` ×
caution rate**, because those two interact by construction and their
interaction is the whole caution story. Every claim in 02 is labelled
invariant or dependent. That labelling is what makes 03's results
admissible.

### What ships

**6. Artefacts, not just a notebook**, following the pattern 01 set when it
froze the dials to JSON: the seed bank, the held-out bank, the frozen
background field, the benchmark plans and the roster all written to disk. 03
must evaluate on the same pre-drawn races the human strategies were scored
on, or the comparison means nothing.

---

## Properties worth protecting

The three from 01 still stand — randomness drawn before the race, a strategy
is just a callable, fuel in normalised tank units. Add two:

**Noise is a pure function of the seed, not of the strategy.** After 02a,
lap noise for car *i* on lap *j* comes from a stream keyed by
`(seed, car_idx)` and consumed by lap index; pit cost from a stream keyed
the same way and consumed by stop index. A car that runs fewer laps stops
early on its own stream rather than shifting anyone else's. Guarded by a
test asserting identical per-car lap-noise sequences under two different
strategies on one seed. Do not consolidate these back into a single `rng`.

**Compression must not invent position.** 01's central claim is that
position is derived from cumulative race time and never simulated. Implement
compression as an adjustment to *lap times under caution* — the car at the
back of the queue runs a faster caution lap as it closes up — so that
running order still falls out of accumulated time. Do not reach in and
reorder cars directly. If a test has to be written for one thing in the
compression work, write it for this.

---

## Notebook 02c, six parts

*(Called "02" when this was written; the blueprint's stage map renames it
02c and nothing else about it changes.)*

1. Load the frozen dials; turn the regulation pit layer on; show the
   legacy-mode regression check passing.
2. The roster: each strategy in one sentence and one code cell.
3. One race, focal car at P5, all five strategies, same seed — the legible
   single case.
4. The paired harness: 200 seeds, delta distributions, the headline table.
   **Two tables, one per series**, per the clarification under decision 10.
5. Starting-slot rotation — does strategy value depend on where you start?
   Read as pace-rank rotation, per 02b's decision A.
6. Sweeps, each claim labelled invariant or dependent.

`viz.py` gains a paired-delta distribution plot, a strategy-versus-benchmark
gap plot and a sweep-response plot, all returning bare matplotlib figures so
the app can use them unchanged.

Use `classification()` in the harness, not `positions()` — the latter sorts
the whole field for every lap record and will dominate the runtime across
200 seeds.

---

## What 02 does not do, on purpose

- No overtaking model. Unchanged from 01, and unchanged for the same reason.
- No weather, no reliability, no driver skill.
- No driver-time regulations, so no driver-change strategy dimension.
- Cautions still arrive at a constant hazard rate. Making the rate vary over
  the race remains a good side quest, but it would cost the exact causal
  benchmark, so it is not free any more.
- Dials still come from one race per series.

---

## Conventions

Unchanged. British English, plain words except where a technical term is the
right one. Comments explain *why*, not *what*. New assumptions go in
`ASSUMED_FIELDS` — the background strategy is one, so it belongs there.
Notebooks are generated from `build_nb.py`; an edit made directly in Jupyter
will be lost.

---

## Amendment log

Kept rather than deleted, so no later thread re-litigates a settled decision
by accident. The blueprint's section 6 carries the same entries.

### Amendment 1 — decision 17, at 02a

Superseded text:

> **17. Caution episodes are drawn non-overlapping.** Durations are already
> `rng.exponential`, which is memoryless — so the causal benchmark is
> exactly computable, which is better than the 12B branch assumed. But the
> merge of overlapping episodes (`CautionTimeline.draw`, lines 87–90)
> destroys that: the union of overlapping exponentials is not exponential.
> Redraw starts until non-overlapping instead. Report the merge rate at the
> calibrated caution rate once, as a diagnostic showing what was given up.

Reason: the conclusion is right and the stated diagnostic is wrong.
Conditioned lengths stay exponential; what the conditioning destroys is the
scale, and rejection sampling's coverage error is roughly twice the merging
draw's. Evidence in `02a_caution_verification.ipynb` Part 6, over 1,500
seeds per process.

### Amendment 2 — decision 9 item 5, at 02c

Superseded text:

> 5. **Lap-down defender.** Declines a stop that would concede a lap to the
>    class leader, and takes the wave-around when one is available.

Revised at implementation; see amendment 3.

Reason: the second clause describes an action the engine does not offer.
`_take_wave` is engine-internal and unconditional on the frozen eligible
set, so a strategy cannot take a wave — it can only avoid being outside the
set when eligibility freezes. Replaced by a standing condition preserving
eligibility, with the scope limits recorded alongside it. Taken before the
roster was written, because a strategy built against the original wording
would have looked as though it worked.

### Amendment 3 — decision 9 item 5 again, at implementation

Amendment 2 was written from the rulebook and the engine's call graph.
Writing the strategy changed three things in it, all in the direction of
less machinery rather than more.

Superseded text, from amendment 2:

> ...declines a voluntary stop that would forfeit its wave-around
> eligibility by putting it behind the class leader on the road. Both
> thresholds derive from `stop_cost` and the class leader's lap time.

- **There is no threshold.** Under caution every car runs the same lap, so
  offsets within the lap are fixed and every stop cuts the offset by
  `cost / caution_lap`. Every voluntary stop costs eligibility; none is
  free. Clause two is `under_caution and laps_down >= 1`, and that is all.
- **`RaceState.wave_eligible` is withdrawn.** With no threshold to test the
  roster never asks the predicate, so 02c adds nothing to the engine. The
  at-the-line hazard it was meant to contain is recorded as a convention.
- **The scope note understated the problem.** Clause two fired in neither
  series, over sixty seeds and again over two hundred. The WEC half was
  predicted; the IMSA half is compression having closed the field so hard
  that the focal car is almost never a lap down.

### Amendment 4 — decision 9 item 3, at implementation

Superseded text:

> 3. **Track-position defender.** Refuses a stop that would drop it behind
>    whoever is directly ahead on the road at the decision, accepting a
>    worse fuel window. The rival is positional, never a named car.

Reason: as written the rival is a car a stop cannot drop you behind, so the
clause does not describe a decision. The two readings that do describe one
coincide, so the amendment names that single rival and says how it must be
found — by projecting rivals to a common lap count and comparing arrivals,
not by differencing crossing times at the line, which measures different
laps against each other and fires the defence on a negative number.

### Amendment 5 — the 03a verification gate, dropped rather than restated

This one amends the blueprint rather than this record, and is logged here
because 02c's handover proposed the replacement that also fails.

Superseded text, from the blueprint's 03a section:

> a policy that always returns "no pit" reproduces the fuel-window
> baseline's result exactly on the same seed

It does not. `RunToFuelWindow` stops at a lap and a half of fuel in hand and
`_must_pit` forces a stop below one lap, so a never-pit policy stops a lap
later and the two races diverge from there. 02c's handover proposed
restating the gate on **laps and stop count**, on the evidence of three
seeds where those agreed. Over twelve seeds they do not: laps differ on
four and stop count on two.

So there is no exact statement of it left to keep, and it is **dropped**.
Weakening a gate until it passes is worse than not having one, because it
leaves a document claiming a check that no longer checks anything. What
replaces it is the second candidate from the same handover, which is the
stronger test: a policy returning `RunToFuelWindow`'s *decision* must
reproduce that baseline bit for bit, with the agent's plumbing in the
middle. It exercises the wrapper rather than the engine's forced-stop path,
and the wrapper is the thing 03a can break.

### Amendment 6 — the observation space's gap normalisation

Superseded text, from the blueprint's 03a observation table:

> | Gap to car ahead | 0…120 s | clipped, /120 |
> | Gap to car behind | 0…120 s | clipped, /120 |

Measured over twenty stand-in races, ten seeds a side, the clip never binds
— the widest gap is about 92 s. The gaps are small and heavily skewed:
median about 3 s, so `/120` puts the median at 0.03 and the ninety-ninth
percentile at 0.36, and two thirds of the row carries one observation in a
hundred. Rescaled on `pit_time_mean_s` those become 0.07 and 0.93.

Both scales are skewed — that is a property of a compressed field rather
than of the divisor — but the second puts the resolution where the
observations are, 1.0 means *a stop's worth of gap*, which is the unit the
decision is taken in, and it is a dial rather than a constant somebody
chose. `viz.plot_gap_normalisation` draws the comparison.

**A caveat on the same page.** The other two chosen scales,
`stint_laps / 40` and `laps_down / 3`, have not been measured and remain
constants somebody picked. The argument that moved the gap rows applies to
them and has simply not been run yet.

### Amendment 7 — pace modes, settled

Superseded text, from the blueprint's section 7B:

> At 03a, decide once and for all — if pace modes go in, they go in for the
> humans too, or the comparison is rigged.

Settled at 03a: **they stay out, permanently.** They need an engine lever
that does not exist and a `PitDecision` field that does not exist; they
reopen the stint-length dimension that decision 3 closed by fixing fuel as
the binding constraint; and they would have to go into the human roster
too, which reopens decision 9's parameter-freeness. The question is closed
rather than deferred again.

### Amendment 8 — section 7A's reward, at 03b

Superseded text, from the blueprint's section 7A:

> **A. The reward is dense negative time; the score is position.** Training
> uses negative lap time plus pit penalty, because class position is sparse
> and nearly unlearnable as a reward signal. [...] 03a's step reward is the
> race time elapsed since the previous decision, negated and divided by the
> class's green lap, so a stop shows up as a longer step rather than as a
> constant to tune.

**The return this defines is a constant, and the stage trained on no signal
at all.** Summed over an episode, the elapsed times between decisions are
the length of the race. These are timed races. So the return is
`duration_s / base_pace_s` whatever the policy does, and the quantity that
actually varies at a fixed duration — laps completed — is invisible to it.

Measured three ways, independently:

| | |
|---|---|
| return over 2,664 training episodes | −219.85 ± 0.42 |
| laps completed, range | 167 to 206 |
| correlation(return, laps) | **−0.11** |
| movement over 500,000 steps | +0.034, or **0.016%** |
| `duration_s / base_pace_s` | 21600 / 97.5 = 221.5 |

And then, most starkly, in the evaluation table: the IMSA policy converged
on **157 stops in a 182-lap race** — pitting on 86% of its laps, 2,887 s in
the lane against the null's 276 s, twenty laps and four class positions
thrown away — and its race time came out at 21,648 s against the null's
21,657 s. It scored the same return as a sensible race, because there was no
gradient pointing away from it. PPO had nothing to descend and each series
drifted to an arbitrary corner: IMSA to "always pit", WEC to something
sitting near the baseline at 7.2 stops against the null's 6.6.

**Revised:** the reward credits **one lap per completed lap**, and nothing
else. Negative time is correct for a fixed *distance*, where finishing
sooner is winning; at a fixed *duration* the time is given. Laps are what
vary, and class position is derived from laps and then time, so the proxy
moves nearer the score as well as becoming non-degenerate. A stop still
costs: it makes that step long, consumes race time, and leaves fewer laps to
credit. Any affine combination with the old term is the same policy
gradient, since the old term is constant.

**`gamma = 1.0` made the cancellation exact rather than merely near-total**
and is kept: a fixed-length race scored at the flag should not tell the
agent that a place lost in hour six matters less than one lost in hour one.

**No test could have caught it, and now one can.** `test_gym_env.py` asked
whether a step's reward was negative and about a lap long, which is true of
a degenerate return as well as a useful one. The question it did not ask —
does the episode score change when the behaviour does — is now
`test_the_return_tells_two_policies_apart`. The lesson generalises past this
bug: **every reward assertion in the suite was about the shape of a step
rather than about the sum.**

### Amendment 9 — 03b's verification gate, supplied rather than dropped

Superseded text: none. The blueprint specified 03b with a boundary
constraint and no gate, which this record had already named, at 02c, as a
gate nobody can fail.

Two conditions, both with falsifiers, both passing on both series:

1. **The agent's path reproduces the same decision taken directly.** A
   `PolicyStrategy` returning a constant action must produce the
   classification a bare callable returning that decision produces, bit for
   bit, every action, both series. Asserted alongside: that `observe` was
   called, and that what it returned varied.
2. **The checkpoint is the policy that was measured.** The `.zip` reloads to
   itself and the `.onnx` export agrees with it on every observation a real
   race visits, not on uniform noise — a race visits a much narrower part of
   the box, and that is where a divergence would hide.

**This exposed a defect in 03a's replacement gate**, which is recorded and
not reopened. Amendment 5 says the agent's plumbing sits in the middle of
that gate — the observation built, the action chosen, the decision rebuilt.
It does not: the echo it compares is a bare callable touching neither
`observe` nor `to_decision` nor `PolicyStrategy`, so what it actually
asserts is that `run_focal` does not care whether a callable is a class
instance or a function. Amendment 5 stands; 03b's gate one closes the gap
rather than restating it.

### Amendment 10 — `pit_transit_frac`, and what it does to 02c's roster

Superseded text: none, which is the problem. `pit_transit_frac = 0.25` was
never decided; it is a default in `ClassDials` that every stage inherited,
and decision 2 says an assumed parameter gets swept rather than trusted.
It had not been swept.

**03b's agent found it.** Given a working reward the IMSA policy converged
on 164 stops in a 189-lap race at 12.8 s a stop, against the null's 41.9 s.
The arithmetic is the model's, not a bug: lane transit costs
`pit_time_mean_s * pit_transit_frac` = 47 × 0.25 = 11.75 s, and `stop_cost`
charges for the fuel actually taken, so a car stopping every lap is nearly
full and adds a thirtieth of a tank for 35.25 / 30 = 1.18 s. Total 12.93 s
against 12.85 observed.

**The sweep, on the sweep fifty, policy held fixed:**

| `pit_transit_frac` | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 |
|---|---|---|---|---|---|
| imsa `splash_and_dash` gained | 0.40 | 0.32 | 0.12 | 0.12 | **0.10** |
| wec `splash_and_dash` gained | 0.58 | 0.54 | 0.50 | 0.48 | **0.44** |
| imsa `track_position` | 0.32 | 0.40 | 0.40 | 0.44 | 0.38 |
| imsa `caution_gambler` | 0.54 | 0.44 | 0.52 | 0.58 | 0.38 |

**`splash_and_dash` falls at every step in both series and no other row
does.** At fifty seeds a share carries an interval near ±0.14, so a raw
swing proves nothing and `caution_gambler`'s 0.20 range is noise — it goes
down, up, up, down. Monotone movement across five points in two independent
series is a different kind of evidence. In IMSA the strategy goes from third
in the roster to next to last.

The mechanism is the strategy's own premise. Splash-and-dash exists to take
a **short fill cheaply**, and this dial is precisely what prices a short
fill. **02c's `splash_and_dash` rows are partly a statement about an
unmeasured assumption rather than about racing**, and should be quoted with
that attached.

**What the sweep cannot say.** The policy is frozen across the points, so
its stop *rate* is invariant by construction — 0.870, 0.868, 0.864, 0.860,
0.856 stops per lap in IMSA. Falling stop *counts* are only the race fitting
fewer laps. Whether a retrained agent would still exploit the dial needs a
retrain at each point, which is a different agent per point and a separate
decision. What is established is the cost: laps 189 → 168, and last place in
all fifty races at every point.

**The ceiling is structural.** A full service costs transit plus the larger
of the tyre and refuel jobs, which is what anchors it to the measured mean.
With `pit_tyre_frac` at 0.35 the sweep cannot pass 0.65 without breaking
that anchor. If a defensible value lies above 0.65, the finding is about the
*shape* of `stop_cost` — everything in it is a share of the measured mean,
so a large fixed per-stop overhead cannot be expressed at all — and that is
02a's territory rather than a dial to sweep.

**How to stop assuming it.** The pit lane speed limit is regulation and
identical in both series: 60 km/h, IMSA art. 32.3 and WEC art. 12.1.4.
Neither rulebook carries the pit lane length, which is a circuit fact, so
the regulations supply one of the two inputs and not both. The transit delta
is `lane_length / 16.67 m/s` minus the time to cover that distance at racing
pace, plus the deceleration and acceleration losses. **The better route is
the timing data**: a low quantile of `pit_time` for one properly scoped
event is a stop with no service in it, which is the transit delta measured
rather than derived — and it would move `pit_transit_frac` out of
`ASSUMED_FIELDS`. That waits on 00's re-run, because the current pit column
has a standard deviation five times its mean and a quantile of it would be
a quantile of noise.

---

### Amendment 11 — the dials, at the 00 re-run

**What it changes.** `data/processed/{series}.json`, refused at 03b, are now
real and gated. Two independent faults had made them describe an invented race,
and they composed.

**The scope.** `build_race_config` selected with an ILIKE pattern on `event`,
which carries a circuit name and no edition. Six Daytonas and three Le Mans
went into one config.

**The car identity.** `car` was parsed as an integer, so entries whose numbers
differ only by a leading zero merged — `#7` Toyota and `#007` Aston are both
Hypercar at Le Mans; `#21` and `#021` are both GTD at Daytona. Three of the
nine collisions found are inside one class, so a composite `(car, class)` key
would not have been enough.

**The arithmetic closes.** Le Mans as the old code pooled it: 24.09 + 48.12 +
48.17 = 120.38 h against the 120.36 in the file. Daytona: 4 x 48.04 + 24.02 =
216.2 h against 216.2. The incoming handover read 216.2 / 24 = 9.009 as nine
editions; the file holds six seasons, so that reading was not available. It was
an integer number of *car-races*, not of editions.

**Three further defects that scoping did not fix**, all corrected:
`stint_number` counts driver stints at two to three fuel stints apiece;
the pit column carries hour-long repairs inside a single race, so the dial is
now a median with the spread trimmed at 3x it; and Le Mans degradation is not
identifiable for a class that changes tyres at every stop, which is reported
through `deg_identified` rather than resolved.

**Scope is now `session_id`.** 1013 ids, none spanning an event, year, series
or session type. `(series, event, year)` is not sufficient — the Asian Le Mans
double-headers put two races under one key.

---

### Amendment 12 — 00's verification gate, supplied; condition two replaced

**Supplied.** 00 was specified with a boundary constraint and no gate, the same
shape 02c named at 03b and amendment 9 fixed there. Four conditions now, in
`src/endurance/gate00.py`, one of which is required to fail.

**Condition two was dropped and replaced, not restated.** As handed over it
asked for the simulated stint length and mean pit time to land within 10% of
the observed ones. Neither can be met, and the diagnostic located why — twice,
in opposite directions, and neither cause is a dial. At Le Mans the simulated
stint is 1.4 laps short of the tank in all three classes, which is
`RunToFuelWindow` stopping a lap and a half early per amendment 5. At Daytona
the file's cars average 26.4 laps between stops against a 30-lap tank despite a
caution share that should stretch stints by 11%, because real cars take fuel
under yellow with half a tank and the default field never does. The effect
scales with caution rate, which is why the two series miss in opposite
directions.

Both are the engine and the roster consuming the dials. The stage's boundary is
explicit that a dial the engine misuses is 02a's problem, so per amendment 5's
precedent the condition is dropped and replaced rather than widened until it
passes. What replaces it compares the calibration against the file — the stint
dial must lie between the median and the longest observed stint, the pit dial
inside the interquartile range of green stops — plus one one-sided engine
bound: the simulated winner must beat the real winner, having no damage,
penalties or unscheduled stops, by less than 10%.

**The replacement fails on the defect it was written for.** The driver-stint
dial it replaced came out at 58 to 73 laps against a longest observed stint of
36.

**Two things the gate does not detect, stated rather than discovered.** Grid
size does not catch pooling, because car numbers recur between editions — six
Daytonas carry 91 numbers, not 360. Nor does the pit dial, which became robust
to pooling the moment it became a median. Only quantities that count, sum or
sequence detect it, which is the same division the original diagnosis found.

---

### Amendment 13 — the calibration was nondeterministic

**The finding.** `calibrate_cautions` chose its reference car with
`GROUP BY car ORDER BY COUNT(*) DESC LIMIT 1`. At Le Mans many cars finish on
the same lap count, so `COUNT(*)` ties and DuckDB's parallel scan broke the tie
differently on each run. A different reference car gives a different flag
sequence, caution share and episode segmentation.

**How visible it was.** One notebook run produced three values for the same
quantity: 0.082 with 6 episodes in Part 1, 0.080 with 4 episodes in Part 2, and
0.0883 in the file it then wrote. Across runs, WEC passed through three
fingerprints and IMSA alternated between two.

**What it means for everything produced before the fix.** Every result in this
project up to that point was computed against dials that were an arbitrary draw
from a distribution nobody knew existed. Not wrong — unrepeatable.

**Fixed** by `ORDER BY COUNT(*) DESC, car ASC`, verified identical across five
separate processes.

**Residual, and recorded rather than fixed.** The caution dials are still
measured from one car, and two cars that both completed Daytona 2026 disagree
by 0.6% on the share. The tie-break makes the choice stable, not correct. The
durable fix is to measure the share from the field — caution lap-time over
total lap-time across all cars, episodes from the per-lap majority flag —
which is deterministic by construction and uses 36,756 lap records instead of
about 660. Deferred: it moves the dials again for 0.6%, on a quantity whose
edition-to-edition spread is 180%.

---

### Amendment 14 — the stop-cost floor, open

**The finding.** At Daytona the pit lane transit alone is
`89.81 x 0.25 = 22.45 s`; a top-up costs 24.70 s and a full service 89.81 s.
The trained agent's realised cost was **12.85 s a stop**, and it stopped on
85.8% of its decisions — about 0.9 stops a lap. A stop cannot cost less than
driving the length of the pit lane.

**The likely route** is `pit_caution_discount` at 0.4 applied to the whole
stop including transit, with 35.8% of the agent's stops taken under caution.
Whether a caution stop should discount transit is arguable — the field is slow
too, so the relative loss genuinely is smaller — but the result is a floor
violation and the policy optimised into it.

**What it is not.** It is not a demonstration that RL fails at endurance
strategy. The policy is behaving correctly given a cost function that is wrong.
This is reward hacking with a diagnosed mechanism, which is a more checkable
result than a policy that quietly wins.

**Status: open.** The fix belongs in `pitstop.py` — `stop_cost` must never
return less than `pit_time_mean_s * pit_transit_frac` at any flag — and
requires retraining both policies. Deferred on runtime grounds; see amendment
16. Until it lands, the agent result is quoted with its diagnosis attached and
never on its own.

**Superseded by this.** The claim that WEC's agent gains 62-70% of races and
beats most humans. That came from a policy trained against a config that was
subsequently refrozen, and did not survive. Both agents now sit at the random
floor: gained 0.000 and 0.005, median class position -6 and -12, against a
roster gaining 0.31 to 0.53.

---

### Amendment 15 — two assumed dials, measured

Both stay in `ASSUMED_FIELDS` and both stay swept. What changes is that the
assumption now has a measured counterpart printed beside it, and neither
counterpart agrees with it.

**`caution_pace_multiplier`**, assumed 1.6. Observed 1.87 in IMSA and 1.79 in
WEC, from the reference car's caution laps against its green ones. Notebook 00
Part 4 prints it.

**`pit_transit_frac`**, assumed 0.25, and the subject of amendment 10. A low
quantile of `pit_time` for one properly scoped race implies 0.51 to 0.66 in
IMSA. The WEC figures of 0.87 to 0.92 are **not** transit: LMP2's fifth
percentile of 81.8 s against a median of 88.6 is a tight distribution with no
service-free stops in it, so the estimator is measuring the low end of normal
stops. The measurement is therefore an upper bound, and the honest conclusion
is that 0.25 is too low rather than that 0.51 is right.

**A constraint amendment 10 did not name.** `pit_transit_frac + pit_tyre_frac`
must not exceed 1, and GTP's implied 0.66 plus the assumed 0.35 already breaks
it. Any promotion out of `ASSUMED_FIELDS` has to deal with that.

**What this does to amendment 10's finding.** 02c's `splash_and_dash` result
was computed at 0.25, the bottom of the sweep, while the measured value sits
near the top of the expressible range. 03b's sweep already says which way that
moves it. The finding stands and its provisionality is now quantified.

**A third assumed dial joins them.** `fuel_per_lap_caution`, at 0.6 of the
green rate, is doing real work and has never been swept: at Daytona's caution
share it turns a 30-lap tank into roughly 33 laps, and at Le Mans it changes
almost nothing. Its influence scales with `caution_rate`, which is the dial
every roster strategy is most sensitive to.

---

### Amendment 16 — the runtime constraint, recorded as a project fact

A full pass — 00, refreeze, train both series, evaluate, sweep, then 02b, 02c
and 03b — takes about five hours, of which **02c alone is 2 h 52 m**. That
number has silently set the terms of every decision taken in the last week,
including which defects were fixed and which were deferred, and it belongs in
the record rather than in anyone's memory.

**It appears to be avoidable.** One race is 0.72 s and the machine has 8 cores.
02c's work is roughly 200 seeds x 5 strategies x 2 series plus the dial
sweeps — order a thousand races, about 12 minutes of compute, run serially on
one core. The `fuel_window` null is re-simulated for every strategy and every
dial setting when it need only be computed once per seed.

**Therefore.** Parallelising over seeds and caching the null is the
highest-value work available in this project. It needs no recalibration, it can
be verified against the current table exactly — same seeds, same dials, same
numbers — and it converts every future pass from an evening into an hour.
Amendments 13 and 14 both become cheap the moment it lands, and both are
deferred until it does.

---

## The headline run, and what the engine was when it was made

Two hundred paired seeds per strategy per series, focal car at pace rank 5,
background frozen to `fuel_window`, six-hour stand-in dials. The
verification gate passed on both series before any of it was read, and the
`fuel_window` row is identically zero across all four delta columns.

| series | strategy | gained | level | lost | median Δpos | p10 | median Δpit s |
|---|---|---|---|---|---|---|---|
| imsa | caution_gambler | 0.500 | 0.290 | 0.210 | +0.5 | −2 | +11.9 |
| imsa | track_position | 0.370 | 0.290 | 0.340 | 0 | −3 | −1.9 |
| imsa | splash_and_dash | 0.345 | 0.565 | 0.090 | 0 | 0 | +10.7 |
| imsa | lap_down | 0.030 | 0.940 | 0.030 | 0 | 0 | 0 |
| wec | caution_gambler | 0.440 | 0.300 | 0.260 | 0 | −2 | 0.0 |
| wec | track_position | 0.370 | 0.235 | 0.395 | 0 | −3 | −0.5 |
| wec | splash_and_dash | 0.535 | 0.385 | 0.080 | +1 | 0 | +18.0 |
| wec | lap_down | 0.075 | 0.840 | 0.085 | 0 | 0 | 0 |

**The median lap delta is zero in every row.** Decision 1 split the score
into class position and a low-variance time diagnostic. In a compressed
field at six hours the diagnostic carries no information and position is
doing all the work, which is worth knowing before 03 chooses a reward.

**One new asymmetry.** The caution gambler saves 11.9 s a stop in IMSA and
nothing at all in WEC. WEC releases the whole field three laps after the
call (art. 14.6.5) rather than staggering by category (art. 46.3.1), so far
fewer caution stops are reachable at all.

**The engine these were run on.** The 02b wave-eligibility correction had to
be reconstructed at 02c, because the patch file was not to hand. The
reconstruction takes wave-arounds from 50-83 per six-hour race to 16-42,
with the residual in the class that genuinely carries lap spread, and 146 of
147 tests pass against it. The single failure is
`test_the_field_bunches_up_instead_of_spreading_out` on its per-episode
hit-rate assertion — which 02b records as having been passing on the defect
and ships a restated version of. Measured after the fix: hit rate 3/6, mean
spread 135.6 s against 142.1 s, so the direction holds and the per-episode
rate does not. **If the two patches differ, that is worth knowing before any
of the table above is quoted.**

**Answered at 03a, and it is the unpatched engine.** The tree handed to 03a
has no `at_line` anywhere in `engine.py`; `_progress` takes no such argument
and `_take_wave` calls it bare. Measured rather than inferred from the
substring: wave-arounds on seeds 3–6 of the six-hour IMSA stand-in come out
at 56, 71, 50 and 83, squarely inside the 50–83 band this record ascribes to
the *unpatched* engine and nowhere near the patched 16–42. The whole test
suite passing clean is corroborating rather than reassuring — 02b records
`test_the_field_bunches_up_instead_of_spreading_out` as passing on the
defect and failing against the fix.

**So the table above was produced on an engine the working tree does not
have.** Nothing in 03a depends on a wave count, so the wrapper is unaffected
and its own numbers are labelled as being against the unpatched engine. But
the roster comparison will move when the fix lands, and **the eight rows
above should be re-run at that point rather than carried forward.**

**Re-run at 03b, on the unpatched engine, at 200 seeds.** The agent has to
sit beside human rows produced on the engine it trained on, so the roster
was re-scored rather than the table above being carried forward. Compared
against the bootstrap interval rather than by eye:

| series | strategy | 02c gained | 03b gained | interval | inside? |
|---|---|---|---|---|---|
| imsa | caution_gambler | 0.500 | 0.440 | [0.370, 0.505] | yes |
| imsa | track_position | 0.370 | 0.330 | [0.270, 0.395] | yes |
| imsa | splash_and_dash | 0.345 | 0.405 | [0.340, 0.475] | yes |
| imsa | lap_down | 0.030 | 0.040 | [0.015, 0.070] | yes |
| wec | caution_gambler | 0.440 | 0.445 | [0.375, 0.515] | yes |
| wec | track_position | 0.370 | **0.480** | [0.410, 0.550] | **no** |
| wec | splash_and_dash | 0.535 | 0.570 | [0.500, 0.635] | yes |
| wec | lap_down | 0.075 | 0.050 | [0.020, 0.085] | yes |

One row moves beyond what the seed set explains: WEC `track_position` was
net-negative in 02c (lost 0.395 against gained 0.370) and is net-positive
here (0.300 against 0.480). IMSA's `track_position` changes sign the other
way but sits inside its interval, so that one is noise. The strategy most
exposed to the wave-eligibility difference is the one that moves, which is
what the engine explanation predicts.

**02c's eight rows must not be quoted beside anything from 03b onwards.**
The 03b table is the reference: same engine, same dials, same seeds as the
agent. Both should be re-run again when the wave fix lands.

**A sixth roster member perturbs nothing.** The five human rows are
byte-identical with the agent in the table and without it, which is the
property `compare_roster` has to have for any of this to be a comparison.

**`splash_and_dash`'s rows carry a caveat as of 03b.** See amendment 10:
both series' values fall monotonically as `pit_transit_frac` rises, so the
number above is partly about an assumption. The other three roster rows do
not move monotonically and are unaffected.

**The median position delta is uninformative at this compression.** Zero in
almost every row, with a bootstrap interval spanning a whole position. This
record's earlier note that the median *lap* delta is zero and position does
all the work now has a companion: at 200 seeds the median position delta is
zero too, and the gained/lost shares carry what information there is. **No
result in this project should be headlined on a median position delta**, the
agent's included.
