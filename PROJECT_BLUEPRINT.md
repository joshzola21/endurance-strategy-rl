# Endurance Racing RL Project — Blueprint

**Status: authoritative.** This document and the 02 decision record together
define the plan. Where anything else disagrees with them — an older draft, a
notebook comment, a half-remembered conversation in a previous thread — they
win. Where they disagree with each other, this document wins and the
decision record is amended.

**Read this first, then the 02 decision record, then the 02b handover, then
the 03a handover, then the notebook for the stage in hand.**

---

## 0. How to use this document

Every thread on this project opens by acknowledging this blueprint and works
from it. A suggested opening, adapted to whichever stage is in hand:

> This thread will focus on building the notebook and the subsequent `.py`
> files needed for **02b**. Please acknowledge the markdown in this
> project's context titled "Project Blueprint", and refer to it when
> building any work with me. If you need extra context, ask me for previous
> `.py` files, or quiz me where the blueprint is silent so we are both on
> the same page.

### What acknowledging means in practice

Before writing a line of code, the thread should:

1. **State the stage** it is on, together with that stage's boundary
   constraint and verification gate, taken from section 3. Getting these
   wrong is how a stage quietly becomes a different stage.
2. **Confirm its dependencies are done.** If a blocking stage is not
   complete, say so and stop rather than working around it.
3. **Ask for the files that stage needs** — the table below — rather than
   guessing at their contents or reconstructing them from memory.
4. **Quiz only where this document is silent.** Settled decisions are not
   reopened. If one looks wrong, say so plainly with reasons; do not work
   around it silently.

### Files to ask for, by stage

| Stage | Ask for |
|---|---|
| 00 | `laps.csv`, `drivers.csv`, `calibrate.py` |
| 01 | `params.py`, `calibrate.py`, `engine.py`, `strategies.py`, `viz.py`, `build_nb.py` |
| 02a | `engine.py`, `params.py`, `calibrate.py`, `strategies.py`, `tests/`, `build_nb.py`, both sporting regulations PDFs |
| 02b | `engine.py`, `params.py`, `pitstop.py`, `caution.py`, `strategies.py`, `tests/`, `build_nb.py`, the frozen dials JSON |
| 02c | the whole of `src/endurance/`, `tests/`, `build_nb.py`, the frozen dials JSON, the seed banks and frozen background field |
| 03a | `engine.py`, `strategies.py`, `harness.py`, `params.py`, `assets.py`, `pitstop.py`, `caution.py`, `__init__.py`, `tests/`, `build_nb.py`, `viz.py` |
| 03b | `gym_env.py`, `engine.py`, `strategies.py`, `harness.py`, `benchmark.py`, `assets.py`, `viz.py`, `tests/`, `build_nb.py`, the seed banks |
| 04 | `viz.py`, `gym_env.py`, the exported checkpoint |
| 05 | everything, plus the 01 and 02 handovers |

### Standing rules for every thread

- The invariants in section 5 are not negotiable and are not traded away for
  convenience, speed or a simpler implementation.
- Anything unverified is labelled unverified. The project has one large
  outstanding example, recorded under stage 00.
- British English, plain words. Comments explain *why*.
- A new assumption goes in `ASSUMED_FIELDS` the moment it is introduced.
- Tests accompany the change that needs them.
- Notebook edits go in `build_nb.py`, which takes a target: `build_nb.py`,
  `build_nb.py 01`, `build_nb.py 02a`, `build_nb.py 03a`. An edit made
  directly in Jupyter will be lost. `tests/run_notebook.py` takes the same
  targets and executes the notebooks against a replica project.
- **A gate that cannot be met is dropped, not weakened.** 03a set the
  precedent: a verification gate restated until it passes leaves a document
  claiming a check that checks nothing, which is worse than an honest gap.
  Drop it, say why in the amendment log, and replace it with something that
  holds if anything does.

### When to push back

If a request contradicts an invariant in section 5 or a stage's boundary
constraint, name the conflict and explain it before proceeding. That is the
whole reason this document exists — the failure mode is not disagreement,
it is a reasonable-sounding request that quietly undoes a decision taken
three threads ago.

### How this document changes

By explicit decision, recorded here, with the reasoning. A decision that
has been superseded moves to the table in section 6 rather than being
deleted, so no future thread re-litigates it by accident.

---

## 1. What the project is, and what it is not

An interactive endurance-strategy sandbox. Twist the real levers — cautions,
stint length, fuel, traffic — and watch a WEC or IMSA race unfold, with
human-style strategies and a reinforcement learning agent racing on the same
terms so you can see what each gets right and wrong.

**The point is not a champion agent.** It is fluency someone can click
through in two minutes and come away actually understanding how these races
are decided. Every scope decision resolves in favour of legibility over
sophistication.

**Scope boundary.** This is the portfolio sandbox. It is *not* the F1
explainable-RL dissertation, and the two must not merge. The sandbox borrows
the dissertation's benchmark methodology — clairvoyant versus causal
references, common random numbers, measured-versus-assumed labelling —
because that methodology is sound and already validated. It borrows nothing
else. If a thread starts importing F1 framing, single-car differential
dynamics or thesis chapter structure into this project, that is derailment.

---

## 2. Governing architecture

**Strict separation of concerns.** Unchanged from the draft, and the single
most important structural rule in the project:

- The **race engine** knows nothing about Gym, Streamlit or agents. It runs
  races.
- The **Gym wrapper** adapts the engine to the standard RL API. It contains
  no physics. Every lap-time, fuel, tyre and caution calculation lives in
  the engine and is called by the wrapper, never reimplemented inside it.
- The **agent** interacts only through `step`, `reset`, `action_space`,
  `observation_space`.
- The **UI** renders and passes parameters through. It contains no physics
  and no training loop.

The clause that needs stating because it was ambiguous in the draft: **there
is exactly one simulator in this project.** The Gym environment is a thin
adapter over it. If two files can both compute a lap time, the architecture
has already failed.

### Module map

```
src/endurance/
  params.py       the dials as data; scale_dials() is the lever mechanism
  calibrate.py    real lap data -> dials, one function per dial
  engine.py       the race: event queue, whole field, derived positions
  pitstop.py      02a: regulation-shaped stop cost, and pit lane status
  caution.py      02a: field compression, wave-arounds, per-series wave schedule
  strategies.py   the strategy interface and the roster
  benchmark.py    NEW in 02b: clairvoyant and causal references
  harness.py      NEW in 02c: paired-seed comparison
  gym_env.py      NEW in 03a: the adapter, and nothing else
  policy.py       NEW in 03b: a checkpoint, its card, and the one loader
  viz.py          charts, reusable by the app

scripts/
  freeze_assets.py  NEW in 03b: the only definition of the stand-in race,
                    and the three artefacts every later stage loads
  train.py          NEW in 03b: MaskablePPO, one policy per series
  evaluate.py       NEW in 03b: the roster comparison, agent included
```

**Why `scripts/` is outside the package.** `train.py` imports
Stable-Baselines3 and `freeze_assets.py` defines a stand-in for an
uncalibrated stage 00. Neither belongs in a library that `import endurance`
must satisfy, and scaffolding placed in the package starts to look
permanent. `policy.py` is inside it because 04 loads a checkpoint through
it, and a loader living in a script would be copied into the app — a second
way of turning a file into a policy, which is decision 6's failure arriving
by a quieter route.

`pitstop.py` and `caution.py` split along one line: `PitRules` and
`CautionRules` hold rulebook facts only, and every quantity that is assumed
rather than regulated lives on `ClassDials` so it is swept like any other
assumption.

### Technical stack

Python 3.11+, DuckDB, pandas, NumPy for the engine and calibration.
Gymnasium and Stable-Baselines3 (or custom PyTorch) for the agent. Streamlit
and matplotlib for the app — note **matplotlib, not Plotly**: `viz.py`
already returns bare matplotlib figures and the app was designed around
that. Changing to Plotly now means rewriting every chart for no gain.

---

## 3. Stage map

The draft blueprint and the built project both used the label "02a" for
different things. That collision is resolved here, once, in favour of the
numbering that already exists on disk.

```
[00 Data Recon] -> [01 Race Engine] -> [02a Engine Corrections]
                                            |
[03a Gym Wrapper] <- [02c Strategies] <- [02b Benchmark]
      |
[03b RL Training] -> [04 Streamlit UI] -> [05 Packaging]
```

| Stage | Focus | Primary deliverable | Boundary constraint | Status |
|---|---|---|---|---|
| 00 | Data recon | DuckDB extraction, the five dials named | Constants only; no simulation code | **done**, re-run and gated; dials real and reproducible, five named limitations |
| 01 | Race engine | Multi-class field simulator | No agent, no UI | done, re-run after 02a |
| 02a | Engine corrections | Per-car streams, `pitstop.py`, `caution.py`, 92 tests | No strategies, no benchmark | **done** |
| 02b | Per-race benchmark | Clairvoyant and causal references | No RL; benchmark is not a policy | **done** on real dials, but **degenerate**: zero foreknowledge on every seed, benchmark beaten by its own control. Not shown at 04 |
| 02c | Human strategies | Roster of five, paired comparison harness | Parameter-free; no per-race tuning | **done** |
| 03a | Gym wrapper | `gymnasium.Env` adapter | Zero physics; zero UI imports | **done** |
| 03b | RL training | Saved policy (.zip / .onnx) | Offline; checkpoints only | **done** on real dials; **both agents at the random floor**, cause diagnosed and open — see amendment 14 |
| **04** | **Streamlit UI** | **Interactive multi-page app** | **Presentation only** | **next** |
| 05 | Packaging | Hosted app and documentation | Polish and deployment | last |

**The renaming to note:** what the 02 decision record calls "02" is **02c**
here, so that the sequence reads monotonically. Nothing else about it
changes.

**What 03b's status means.** The pipeline, the gate and the two checkpoints
exist and hold. The *numbers* do not: the reward specified in section 7A is
degenerate for a timed race and the policies trained under it received no
gradient. See amendment 8. 04 may proceed — it needs a loadable checkpoint
and a decision interface, both of which it has — but no agent result may be
quoted until 03b is re-run under the amended reward.

**Superseded at the 00 re-run.** 03b has since been re-run under the amended
reward against real, gated dials. There is now an agent result and it is a
negative one: both policies sit at the random floor, gaining 0.000 and 0.005 of
races against a roster gaining 0.31 to 0.53. The cause is diagnosed — a stop
priced below its own lane-transit floor, exploited to 0.9 stops a lap — and is
open pending a `pitstop.py` fix. See amendment 14. **04 may quote the agent
result provided it quotes the diagnosis with it.**

---

## 4. Stage specifications

### 00 — Data reconnaissance · COMPLETE, RE-RUN AND GATED

The five dials named and extracted: degradation slope, caution pattern, stint
length, pit cost, traffic density. Every number traces to a query in
`calibrate.py`, and every config records which race it came from in
`source_event`.

**The re-run.** The draft recorded this stage as complete; 03b was the first
stage to read the frozen dials rather than assume them, refused them, and used
a stand-in. The re-run found two independent faults that composed: the queries
scoped by an ILIKE pattern on `event`, which carries a circuit and no edition,
and `car` was parsed as an integer, which merged entries whose numbers differ
only by a leading zero — `#7` and `#007` are both Hypercar at Le Mans. See
amendment 11 for the arithmetic and `handover_00_re_run.md` for the full
account.

**What the dials now are.** `data/processed/imsa.json` is the 2026 Rolex 24
(session 682, 60 cars, four classes); `wec.json` is the 2026 24 Hours of Le
Mans (session 1000, 62 cars, three classes). Scoped by `session_id`, which is
unique to a session and never spans an event, a year, a series or a session
type.

**Verification gate.** The stage was specified without one, which is how the
fault survived two stages. Four conditions in `src/endurance/gate00.py`, shown
in notebook 00 Part 6 and in `tests/test_calibrate.py`: the selection is one
race; the dials describe the race they came from; pooling two adjacent editions
makes conditions one and two fail; and degradation has the right sign in every
class where the sign is identifiable. Condition two was dropped and replaced
rather than restated — see amendment 12.

**Five named limitations, all live, none blocking.** The frozen dials are a
pre-fix draw and will move by 0.7% on `caution_rate` when 00 is next run in a
fresh kernel. The caution dials are measured from a single reference car, and
two cars that both finished disagree by 0.6%. Le Mans degradation for Hypercar
and LMP2 is a net within-stint trend rather than tyre wear, because tyre age
and fuel load are collinear for a class that changes tyres at every stop.
Daytona 2026 is a caution outlier — 0.353 against 0.125–0.228 across the other
five editions — so `caution_rate` must be swept rather than asserted wherever a
roster result is shown. And two assumed dials are measurably wrong; see
amendment 15.

**What this stage may be trusted for.** The dials, the gate, and the
provenance. Not the caution episode length to better than a per cent, and not
tyre degradation at Le Mans in either sign.

### 01 — Race engine · COMPLETE, RE-RUN AFTER 02a

Whole multi-class field on an event queue, position derived from cumulative
race time, no overtaking model, randomness drawn in advance. Four decisions
from the 01 thread stand: engine plus thin demo notebook; multi-car field
with derived position; `src/` package with thin notebooks; Streamlit as
eventual front end.

02a changed the engine, so 01's Part 6 validation table moved. Two things to
state in 01 rather than regenerate quietly:

- **The numbers moved**, and by how much is in the 02a table below.
- **Part 6 compared the wrong quantities.** It set a simulated caution
  *time* share against a real caution *lap* share. Both sides are time
  shares now. On the synthetic fixture the two differ by more than a third
  — a real lap share of 0.056 against a real time share of 0.086 — so this
  was not a rounding matter. `02a_engine_corrections.ipynb` Part 11 reports
  the old and new comparison side by side; 01 carries the corrected table.

### 02a — Engine corrections · COMPLETE

**Five changes shipped, not four.** The caution process — calibration in
seconds, and an alternating on/off draw replacing the merging one — was
decided and implemented ahead of the stage; Parts 1 to 8 of
`02a_engine_corrections.ipynb` are its verification. The four this document
listed followed:

1. **Per-car random streams.** Lap noise and pit cost now come from streams
   keyed by `(seed, car_idx)`, indexed by lap and by stop respectively, and
   stored as standard normals scaled where they are used — so sweeping
   `lap_noise_s` or `pit_time_std_s` resizes the noise without redrawing it.
2. **`pitstop.py`.** Stop cost is a function of fuel taken, tyre change and
   the rulebook's sequencing. The layer is anchored to the measured mean: a
   full tank plus tyres costs exactly `pit_time_mean_s` in both series by
   construction, so the level stays data and only the shape is regulation.
   IMSA overlaps refuelling and tyres under a four-crew cap (34.1.1); WEC
   forbids tools during refuelling (art. 12) so its stops are sequential.
   The module also carries pit lane status per section 7C.
3. **`caution.py`.** Field compression and wave-arounds, applied as
   adjustments to caution lap times. Everyone runs the safety car's lap
   rather than a multiple of their own, and a car with a gap closes a share
   of it each lap. The next gap works out as a weighted average of two
   positive gaps, so overtaking under yellow is prevented by the arithmetic
   rather than by a clamp.
4. **Traffic responds to current pace**, degradation included, on both sides
   of the comparison.

**`Compat` is the switchboard.** Every change is a flag that can be switched
off independently, and `Compat.v01()` turns all of them off at once. It
exists for the regression gate and for showing what each change did — it is
not a supported way to run the project.

**Verification gate — met, with the evidence for each condition.**

- *Legacy mode reproduces 01 exactly.* `Compat.v01()` reproduces reference
  numbers captured from the engine **before** 02a touched it, on three
  seeds, bit for bit. `tests/capture_golden.py` made the capture; the values
  live in `tests/test_compat.py`. A gate comparing the new engine only
  against itself would have passed whatever had broken.
- *Two strategies on one seed give identical per-car lap noise by lap
  index.* Checked end to end, with degradation, traffic and cautions off so
  that a green lap is exactly pace plus noise.
- *Compression changes no position except through lap times.* Checked by
  rebuilding every car's finishing time from its own lap times and pit costs
  and requiring `classification()` to agree.
- *The caution-timeline independence test still passes.*
- *`pytest tests/` is green:* **92 tests**, from a 42-test baseline, none of
  the original 42 removed. New files: `test_compat.py`, `test_pitstop.py`,
  `test_caution_pace.py`, `test_traffic.py`.

**What the corrections did.** IMSA-shaped 24 hours, 18 cars, 20 seeds:

| engine | winner laps | sd across seeds | headline class spread |
|---|---|---|---|
| `Compat.v01()` | 812.9 | 9.9 | 10.8 |
| + per-car streams | 801.5 | 13.9 | 11.5 |
| + pit layer | 802.5 | 14.3 | 11.4 |
| + compression and wave-arounds | 803.9 | 13.6 | 1.0 |
| + current-pace traffic | 804.4 | 14.2 | 1.1 |

Compression does nearly all the work. The wider seed spread from the streams
is the paired-comparison defect being paid off rather than a regression:
under 01 a change of strategy silently reshuffled every other car's noise,
and the stability that bought was not information.

**Independent verification of the caution process.** Part 8.2 runs the
engine's `CautionTimeline.draw` and the notebook's from-scratch
reimplementation over the same 3,000 seeds at three operating points. They
agree on realised share to four decimal places — `share_gap` is 0.0000
throughout — and on episode count and pooled coefficient of variation. Two
implementations written from one description by different routes agreeing is
a stronger statement than the test suite alone.

**The assumed dials went from five to ten.** In order:
`caution_pace_multiplier`, `pit_caution_discount`, `traffic_window_frac`,
`traffic_penalty_s`, `tyre_life_laps`, `pit_transit_frac`, `pit_tyre_frac`,
`caution_pits_open_delay_laps`, `caution_queue_gap_s`, `caution_close_frac`.
The five new ones are the shape of a pit stop (two), how long the field
takes to form up behind the safety car before the pits are declared open
(one — the class staging after it is regulation, not assumption), and how
the queue closes up (two). `pit_caution_discount` survives even though
compression now does part of its job: retiring it is a calibration decision
and 02a was not a calibration stage.

**Findings 02a produced that no decision document contained.**

1. *Compression is doing nearly all the work, and its magnitude is
   unvalidated.* Taking the headline class from a ten-lap spread to about
   one is the right direction — real endurance classifications sit there and
   01's did not — but whether it closes the field *too* hard needs the real
   event rather than an argument. This is the first thing 02b should settle,
   because a benchmark is only as good as the engine beneath it.
2. *The traffic correction is necessary and nearly inert.* It fixes a real
   defect — stop timing could not affect traffic at all — but a field
   running one strategy pits in lockstep, every car carries the same tyre
   age, degradation cancels out of the comparison and the correction changes
   nothing whatsoever. On a staggered field it is worth tenths against class
   gaps worth seconds. This bears directly on decision 2's frozen background
   field: a single-strategy background will see none of it. By decision 14
   that is a finding, not a bug, and it belongs in the write-up.
3. *A wave-around is bookkeeping, not physics.* It is a timing-system credit
   — the car is given a lap it did not drive. Implemented as a lap time so
   position stays derived, flagged `wave_by` in the lap record, and it is
   the least physical thing in the engine.
4. *`tests/sqlite_shim.py` had drifted out of step with `calibrate.py`.* Its
   `read_csv_auto` pattern predated the `quote=` and `escape=` arguments, so
   `run_notebook.py` had been failing on notebook 01 before 02a began. Fixed
   — but it means 01 had not been executed end to end for some time, which
   is worth knowing when reading its pre-02a validation numbers.

### 02b — Per-race benchmark · COMPLETE

Two-stage: dynamic programming over stop laps minimising the focal car's own
race time against the frozen caution timeline, then the top-*k* time-optimal
plans re-scored through the full engine against the frozen rival field and
ranked on position. Both a **clairvoyant** version and a **causal** version;
the gap between them is the value of foreknowledge.

**Verification gate.** The DP matches brute force in the no-caution limit.
This gate is not optional — the same check caught a benchmark in the F1 work
that passed a total-time comparison while reconstructing a suboptimal plan.
A second gate is needed in both series: no plan may stop through a closed
pit lane, which the no-caution test cannot catch because it removes the very
windows in question.

**Boundary constraint.** The benchmark is a reference, not a strategy. It
must never be handed to the engine as a callable, and the agent must never
observe it.

**Correction at the 00 re-run.** 02b had been loading
`data/processed/dials_imsa.json`, a filename that has never existed, and
silently taking its stand-in branch. Every number it produced before the fix —
the benchmark plans, the causal artefacts, the foreknowledge table — was about
a three-hour, ten-car, single-class invented race. It announced this honestly
at the top of each run, which is why it went unnoticed. Corrected to
`imsa.json`.

**And on real dials it is degenerate.** Foreknowledge is zero on every seed:
the causal and clairvoyant benchmarks make identical choices, because a
three-hour window at one caution per 103 minutes usually contains nothing to
foresee. The benchmark is also beaten by its own forced-only control — `P1, 96
laps` against `P1, 97 laps`. **02b is a documented gap and 04 shows no
benchmark row.** Resolving it means either lengthening the reduced race, which
the plan enumeration may not survive, or accepting that foreknowledge is not
measurable at this caution frequency.

**What 02b inherits from 02a.** Each of these changes what the benchmark is
allowed to do, so they are constraints rather than notes.

1. **The pit lane closes in both series.** The stop-lap search must exclude
   closed windows for IMSA *and* WEC, staged by class in IMSA, with the
   Short FCY case producing cautions that never open at all. Use
   `pitstop.lane_status` rather than reimplementing the rule.
2. **Compression widens the gap between time-optimal and position-optimal
   plans.** Decision 5 already warns that *k* must be generous; the size of
   the compression effect in the table above is why. Check *k*'s sensitivity
   rather than choosing a number.
3. **The causal benchmark is still exactly computable.** Caution durations
   remain memoryless, the timeline is still drawn before the race, and it is
   still independent of strategy. All three are guarded by tests.
4. **Pairing is now real.** With per-car streams the 200-seed bank gives
   genuinely common random numbers — which is what decision 10's paired
   position delta assumed all along and did not have.
5. **Exclude `wave_by` laps** from anything that reads caution lap times as
   pace. They are credits, not laps anybody drove.
6. **Settle the compression magnitude first**, per finding 1 above.

**What 02b delivered.** `benchmark.py`, `causal.py` and `assets.py`, both
gates met, 110 tests. Full detail is in the 02b handover and is not repeated
here; three things from it are load-bearing for later stages:

- **An engine correction landed.** `_take_wave` runs from `_next_lap`,
  before `_set_lap` refreshes the car's lap window, so the car at the line
  still carried the lap it had just *finished*, saturated its progress
  fraction at the clip and read a full lap further round than it was —
  becoming the apparent class leader and declaring everyone alongside it
  lapped. Ninety-nine wave-arounds in one six-hour race on a field that was
  never lapped; single figures after the fix. 02a's compression claim
  survives it unchanged. **Any stage touching wave-arounds must confirm the
  patch is applied before trusting a wave count** — a pre-patch `engine.py`
  produces 50 to 80 wave-arounds per six-hour race on a class finishing with
  zero lap spread, which is the cheapest way to check. 02c had to reconstruct
the correction because the patch file was not to hand; the reconstruction
reproduces the reported effect and passes 146 of 147 tests, the exception
being the compression test 02b already records as having passed on the
defect. If two versions of the patch exist, diff them before quoting any
number that depends on a wave count.
- **Model error can exceed candidate spread.** 02b's top twenty plans spanned
  six seconds of predicted time against thirty seconds of model error. When
  that ratio inverts, a ranking is noise and deeper *k* does not help —
  breadth does. Check the ratio before trusting any ranking.
- **Levels that are field-dependent get measured, not modelled.** Wave
  credits and compression were worth ~800 s and ~185 s over six hours
  against a 5 s green-lap residual, and neither is computable for one car.
  The anchor measures them off a reference run. It is a property of the
  benchmark and does not cross into the roster — see the 02 decision record.

### 02c — Human strategies and comparison · COMPLETE

Five parameter-free strategies: fuel-window baseline, caution gambler,
track-position defender, splash-and-dash planner, lap-down defender. Focal
car at P5 of the headline class; the rest of the field runs a frozen
background strategy saved to disk. Paired position deltas over 200 seeds,
reported as distributions. Sweeps of every assumed parameter, one at a time,
plus a `pit_caution_discount` × caution-rate grid.

Full detail in the 02 decision record; it is not repeated here.

**Boundary constraint.** No per-race tuning of any strategy parameter. A
tuned baseline is a second oracle wearing a disguise, and it destroys the
comparison it was meant to support. This is what keeps 02b's `anchor` out of
the roster: it is a per-seed level measured off a reference run of that
race, and handing it to a strategy is per-race tuning whatever it is called.

**Verification gate — the null is exactly the null.** This stage was
originally specified with a boundary constraint and no gate, which is a gate
nobody can fail. The gate is this: with the focal car running
`RunToFuelWindow` and the rest of the field on the frozen background, the
harness reproduces the focal car's classification row bit for bit against a
bare `run_race` call on the same seed. Against a bare call rather than
against the harness's own null arm, per 02a's convention that a regression
gate compares with something outside the code under test. Run it across
several seeds of the headline bank, not one.

It gates the apparatus rather than the roster, which is the right target for
a single condition: every paired delta this stage reports is measured *from*
this run, so a null that is not the null puts a systematic offset under
every number in 02c and everything 03b inherits from it. It also re-asserts
02a's central property at the one place 02c could break it — the focal car's
noise streams must not depend on which strategy occupies the seat — and it
catches the class of plumbing faults that produce plausible numbers rather
than errors: a mis-set focal id, a background field missing a car, a compat
flag differing between the two arms, dials scaled in one arm and not the
other, the slot rotation indexed off by one.

**Its limit, stated rather than discovered.** A `fuel_window` focal car
against a `fuel_window` background resolves to the same strategy map either
way, so on the strategy side the gate ought to pass by construction and it
says nothing about the four strategies that are not the null. The boundary
constraint is therefore guarded by review and by the sweeps, not by the
gate. The candidate not taken, if that proves too little, was a
dial-dependence test on each strategy's callable — sweep a dial the strategy
declares and require its decision boundary to move, sweep one it does not
and require the boundary to stay put — run on a constructed state rather
than through a race, since scaling a dial inside a race moves the race too.

Separately, and not part of the gate: **run 02b's ratio check before any
ranking is reported** — the spread of the roster's outcomes against the
model-and-seed error. Where the error exceeds the spread, the ordering is
noise and breadth helps where depth does not.

02c broke this rule against itself and is the reason it is stated twice. A
sixty-seed pass put a two-place gap between the two series' track-position
results, on a statistic whose per-seed spread runs to three places; it was
written into this document and the decision record as a finding, and two
hundred seeds reversed the direction. **A roster claim does not go in a
document until the difference has been checked against the spread, and the
seed count belongs in the sentence next to the number.**

**Two things 02a changed for this stage.** The splash-and-dash planner is
now implementable — `pitstop.py` prices a partial fill differently from a
full service, which 01 could not. And the sweep list has grown: five new
assumed dials joined `ASSUMED_FIELDS`, and decision 11's one-at-a-time
sweeps should cover them or say why not.

**Four things settled during 02c**, all recorded in full in the 02 decision
record:

1. **Decision 9 item 5 is amended, twice.** The lap-down defender had no
   wave-around lever, because the credit is engine-internal and
   unconditional on the frozen eligible set, so it became a standing
   condition preserving eligibility while lapped and under caution. Writing
   it then showed the condition has no threshold at all — under caution
   every car runs the same lap, so every voluntary stop costs eligibility
   and none is free. Clause two is `under_caution and laps_down >= 1`.
2. **The two series are scored in separate tables**, never a pooled row, and
   the seed banks are drawn with a per-series `draw_seed`.
3. **The anchor stays in `benchmark.py`.** The caution gambler's threshold
   is a count rather than a price and derives from the dials alone.
4. **Decision 9 item 3 is amended.** "Whoever is directly ahead on the
   road" names a car a stop cannot drop you behind. The rival is the car
   that is behind now and would be ahead once the stop is served, found by
   projecting rivals to a common lap count rather than by differencing
   crossing times.

**This stage adds nothing to the engine.** `RaceState.wave_eligible` was
proposed and is withdrawn: with no threshold to test, the roster never asks
the predicate. 02c is `strategies.py`, `harness.py` and `viz.py` only, and
the 02a boundary stands unbroken. The convention that replaces it is in
section 5.

### 03a — Gym wrapper · COMPLETE

A `gymnasium.Env` adapting the engine. The focal car is the agent; every
other car runs its background strategy through the ordinary strategy
interface. The agent's decisions enter the engine through the *same*
`(CarState, RaceState) -> PitDecision` interface the human strategies use,
and the engine cannot tell the two apart.

**What shipped.** `gym_env.py` at 257 lines, of which 122 are code; two
additions to `engine.py`; one function lifted out of `strategies.py`; one
chart in `viz.py`; 31 tests, taking the suite to 154.

**Observation space — nine rows, as built.** Every one derivable from
`RaceState` and `CarState`, and nothing computed in the wrapper:

| Feature | Range | Normalisation |
|---|---|---|
| Race progress | 0.0…1.0 | `t / duration_s` |
| Fuel remaining | 0.0…1.0 | direct |
| Tyre age | 0…`tyre_life_laps` | `tyre_age / tyre_life_laps` |
| Gap to car ahead | 0…`pit_time_mean_s` | clipped, `/ pit_time_mean_s` |
| Gap to car behind | 0…`pit_time_mean_s` | clipped, `/ pit_time_mean_s` |
| Flag state | 0 or 1 | binary |
| Stint age | 0…40 laps | `stint_laps / 40` |
| Laps down to class leader | 0…3 | /3 |
| Pit lane open | 0 or 1 | binary |

The drafted table had eight rows and then discussed both "a ninth" and "a
tenth" without the numbering adding up. Settled: `pit_lane_open` is the
ninth, on 02c's evidence that a gambler which cannot see the lane is not
gambling, and the same argument applies to an agent. `wave_eligible` is out
— under current compression the focal car is almost never a lap down, so
the predicate describes a situation the agent never meets. **That is
conditional on 02a's finding 1**: if the re-run finds compression closing
the field too hard, the row comes back, and the method goes on `RaceState`
rather than being reimplemented.

**The gap rows' normalisation changed and the draft's is superseded** — see
amendment 6 in the decision record. `/120` never met its clip and put the
median observation at 0.03; `pit_time_mean_s` puts it at 0.07 with the
ninety-ninth percentile at 0.93, and 1.0 then means a stop's worth of gap,
which is the unit the decision is taken in. **The other two chosen scales,
`stint_laps / 40` and `laps_down / 3`, are unmeasured and the same argument
may apply to them.**

**Two gap rows walk into section 5's second invariant**, which is why
`RaceState.gap_ahead_s` and `gap_behind_s` exist: at the line every slower
class rival is a lap behind, so differencing `race_time_s` measures
crossings of different laps. Rivals are projected to a common lap count.
`strategies._would_be_passed` was deliberately not rewired onto them — it
answers a different question and a refactor would have moved 02c's
published numbers.

**Action space — five discrete actions.** Stay out; pit for a full tank and
tyres; full tank, keep tyres; fill to the flag with tyres; fill to the flag,
keep tyres. This spans exactly what the roster can express and no more.
There is deliberately **no refuel level**: a level is a number, and a number
the agent chooses is a tuning surface decision 9 forbade the humans. The
fill arithmetic is `strategies.fuel_to_the_flag`, shared with the
splash-and-dash planner so there is one implementation of it.

**The agent gets no privileged path, and the mask is what makes that
tolerable.** `_must_pit` is untouched: when the rules force a stop the
engine still forces it. The wrapper removes the forced action from the
agent's choices rather than letting it be chosen and discarded, so the
policy is never trained on a decision it does not own. The mask is a
training convenience only, which is why `PolicyStrategy` — the path
`harness.compare_roster` scores through — carries none and relies on the
override exactly as the humans do.

Removing the override instead was considered and rejected. `_next_lap`
never reads `car.fuel`, so fuel exists in this model *only* through
`_must_pit`; without it an empty tank costs nothing and a car laps at full
pace for six hours. At the calibrated degradation slope that still loses;
halve `deg_slope_s_per_lap` — a `scale_dials` lever the app exposes — and it
wins the class outright.

**Engine additions, both non-physics.** `RaceEngine.run` is now a drain of
`run_stream`, a generator that suspends at the focal car's decision and
resumes on `send`. **There is still exactly one race loop**; if `run` kept
its own copy the project would have two simulators that resembled each
other. A worker thread with the focal strategy blocking on a queue pair was
the alternative and works at 1.3% overhead, so speed did not decide it: a
suspended generator is a single deterministic object that `close()` disposes
of, and a traceback out of a policy arrives at the caller rather than across
a thread boundary. The second addition is the two gap methods above.

**Verification gate — met, and one condition dropped.** `grep -r streamlit`
returns nothing and the import graph is asserted in a test rather than
reviewed; `gym_env.py` is 257 lines against a 250-line target, 122 of them
code, the remainder comment; a headless script runs 1,000 masked random
steps without error. The fourth condition — a never-pit policy reproducing
the fuel-window baseline — **is dropped rather than restated**, per
amendment 5. What replaces it: a policy returning `RunToFuelWindow`'s
decision reproduces that baseline bit for bit through the wrapper's
plumbing, on twelve seeds in both series.

**Findings 03a produced.**

1. *The stated gate did not hold, and neither did the proposed
   replacement.* 02c suggested restating it on laps and stop count, on three
   seeds where those agreed; over twelve they differ on laps in four and
   stops in two.
2. *The gap normalisation put the observations in the wrong part of the
   range.* Amendment 6. Notable less for the fix than for the fact that a
   normalisation is a modelling choice that had never been measured.
3. *Fuel exists only through the forced-stop rule.* Which means the reward's
   pit penalty is entirely inside the elapsed time, with no separate term to
   tune — see section 7A.
4. *The working tree is running the unpatched engine.* Wave-arounds on seeds
   3–6 come out at 56, 71, 50, 83 — the unpatched band. Nothing in the
   wrapper depends on a wave count, but 02c's headline table was produced on
   a different engine from the one on disk.

### 03b — RL training

DQN or PPO on the 03a environment. Evaluated on the **same pre-drawn seed
bank** the human strategies were scored on, with the held-out bank reserved
for anything the design was selected on. Scored against the **causal**
benchmark — measuring against the clairvoyant one alone would penalise a
policy for failing to predict the future.

**Deliverables.** `train.py`, an exported checkpoint, and an evaluation
script producing the same paired-delta distributions 02c produces, so agent
and human rows sit in one table.

**Settled at 03a, so no longer due here.** Pace modes are out permanently —
see section 7B and amendment 7. 03b does not reopen them.

**How the agent is scored, and the only way it may be.** Insert the policy
into `ROSTER` as a sixth member, wrapped in `gym_env.PolicyStrategy`, and
call `harness.compare_roster` on the same banks, the same frozen field and
the same pace rank as the five humans. There is no agent-specific
evaluation code and 03b must not add any: a second evaluation path that can
differ from the roster's is the failure decision 6 exists to prevent, and
it is the kind that produces plausible numbers rather than errors.

**Training draws from the headline bank; the held-out fifty are refused by
`EnduranceEnv` unless explicitly asked for.** That refusal is in the
environment rather than left to the training script on purpose.

**The mask has to be threaded through.** The env publishes `action_mask` in
`info` on every step. A policy that ignores it will spend capacity on
decisions the engine overrides, and will look worse than it is. This bears
on the algorithm choice below more than the dashboard does.

**Budget, measured at 03a.** One six-hour race is about 0.15 s and an
episode is roughly 190 steps, so the wrapper is not the bottleneck in any
training loop. A full evaluation pass over the roster is under a minute.

**Note on algorithm choice.** The UI promises visible action values or
action probabilities. DQN gives Q(s,a) directly; PPO gives P(a|s). Either
satisfies it, but pick with the dashboard in mind rather than retrofitting.

**Settled at 03b: MaskablePPO, from `sb3-contrib`.** The mask decides it.
SB3's DQN has no mask hook, so threading it means a custom policy applying
the mask at the acting argmax *and* in the target computation, and missing
the second is silent. MaskablePPO applies it inside the action distribution
during rollout and in the loss. The cost is real and is inherited by 04:
there is no Q(s,a) to show, only probabilities, and a panel wanting a
magnitude rather than a ranking is a retraining job.

**The verification gate, supplied at 03b.** This stage was specified with a
boundary constraint and no gate, which 02c had already named as a gate
nobody can fail. Two conditions, each with a falsifier:

1. **The agent's path reproduces the same decision taken directly.** A
   `PolicyStrategy` returning a constant action must produce, bit for bit,
   the classification a bare callable returning that same decision produces
   — every action, both series — with `observe` and `to_decision` in the
   path on one side and neither on the other. Asserted alongside it: that
   `observe` was called at all, and that what it returned varied.
2. **The checkpoint is the policy that was measured.** The `.zip` reloads to
   itself, and the `.onnx` export agrees with it on every observation a real
   race visits. 04 loads the export and 05 hosts it, so a divergence puts
   the app on numbers nobody produced.

Both passed on both series at 03b, including ONNX agreement.

**Two policies, not one.** Two series, two banks, two tables, two
checkpoints. Nothing is pooled and no policy crosses series; `PolicyCard`
refuses it.

**The card.** Every checkpoint carries a JSON sidecar recording the dials
fingerprint, the bank fingerprint and the training seed, and `load_policy`
refuses a policy whose dials do not match the race it is about to be scored
on. This is `assets.py`'s tripwire applied to the artefact most likely to
outlive a config change. **The reward must be recorded on the card from a
name in `gym_env.py` rather than a string typed into `train.py`** — at 03b
it was the latter, and a reward change would have left every provenance file
carrying a stale claim.

**The selection rule.** Hyperparameters and anything else chosen by looking
at a number move on the **sweep fifty** and nowhere else. The headline two
hundred are reported with the caveat that the agent trained on them and the
humans did not train at all. The held-out fifty are the generalisation row,
reported beside the headline rather than instead of it. `EnduranceEnv`
refuses held-out seeds in training; **evaluation never touches the env, so
that discipline is procedural rather than enforced.**

**What the agent found, and why it counts as a deliverable.** Given a
non-degenerate reward the IMSA policy converged on 164 stops in a 189-lap
race at 12.8 s a stop, which is correct behaviour against the model: lane
transit is `pit_time_mean_s * pit_transit_frac` and `stop_cost` charges only
for fuel actually taken, so a car that is nearly full tops up for a
thirtieth of a tank. **An adversarial policy search reached a corner of the
action space five parameter-free strategies had no reason to visit, and
found an unmeasured assumption sitting in it.** That is the stage producing
a result about the simulator rather than about racing, and it is worth more
than the agent's row in the comparison table.

**A hole worth knowing about.** SB3 seeds its environments by calling
`reset(seed=...)` with integers of its own choosing, and `_pick_seed`
accepts any integer that is not held out. Unwrapped, training draws from the
whole seed space while every artefact still claims the bank. `train.py`
wraps the env to map any supplied seed onto a headline race.

### 04 — Streamlit UI

Sidebar sliders driving `scale_dials`. Lap-by-lap stepping that runs fast
and **pauses automatically on strategic triggers** — caution called, fuel
critical, pit window opening. Manual override, so a human can take a
decision and see it compared against what the policy would have done.
Explainability panel showing action values, fuel and tyre state, and gaps.

Charts come from `viz.py` unchanged. If the app needs a chart, it is added
to `viz.py` and the notebook can use it too.

"Pit window opening" is now a real event rather than a figure of speech:
`lane_status` says when the lane opens and to which class, so the pause
trigger reads it rather than approximating it.

**Four things 04 must state on the page, settled at the 00 re-run.** The dials
come from one running of one race, and the caution share moves by a factor of
2.8 between adjacent Daytonas, so the starting point of every slider is a
sample of one. Hypercar and LMP2 degradation at Le Mans is a net within-stint
trend and not tyre wear. The agent sits at the random floor, and the honest
framing is the diagnosed one — it found a stop priced below its own floor and
exploited it — which is provisional until `stop_cost` is fixed. And there is no
benchmark row, because 02b is degenerate on real dials.

### 05 — Packaging

Hosted deployment, architectural documentation, and a portfolio narrative
covering problem formulation, data integration and what the comparison
actually showed — including the negative results.

---

## 5. Invariants

These outlive every stage. Breaking one silently is the failure mode this
document exists to prevent.

**All randomness is drawn before the race starts.** The caution timeline
especially. This is what lets two strategies be compared on exactly the same
race. **The draft blueprint's per-lap caution sampling
(`np_random.random() < p_caution_per_lap` inside `step`) is rejected** — it
would destroy the property the entire comparison rests on. Caution episodes
are drawn up front, exponential and non-overlapping.

**Noise is a function of the seed, not the strategy.** Per-car streams,
consumed by lap and stop index, drawn as standard normals and scaled where
they are used. Do not consolidate them back into one RNG.

**A strategy is just a callable** taking `(CarState, RaceState)` and
returning a `PitDecision`. The agent implements the same interface. The
engine must not be able to tell them apart, and the agent gets no privileged
path in.

**There is exactly one race loop.** Since 03a it lives in
`RaceEngine.run_stream`, and `RaceEngine.run` is a drain of it. Anything
needing another way to step a race inverts further rather than copying the
loop. A second loop is the same failure as a second simulator, arriving by
a quieter route.

**The engine's forced-stop rule applies to the agent exactly as to a
human.** A wrapper may *mask* an action the rules have already decided; it
may not remove the override. Fuel exists in this model only through
`_must_pit` — `_next_lap` never reads it — so an agent exempted from the
rule laps on an empty tank at full pace.

**Position is derived, never invented.** A car is ahead because it completed
more laps or got there sooner. Compression and wave-arounds are implemented
through lap times for this reason, and never by reaching in and reordering
cars. If one test has to be written for work in this area, write it for
this: rebuild every car's race time from its own laps and require the
classification to agree.

**At the line, a car cannot read its own position by the usual means.** Two
consequences of the event queue, and each has already caused a defect.

*Progress.* The asking car has had `laps_done` incremented but not
`_set_lap`, so its lap window still describes the lap it has just finished
and `track_fraction` saturates at the clip. Read its own progress as
`laps_done` exactly, and use `track_fraction` for other cars only. Getting
this wrong produced ninety-nine wave-arounds in one race on a field that was
never lapped.

*Gaps.* At the moment it crosses onto lap *N*, every slower car in its class
is still on lap *N - 1*, so there is usually no same-lap car behind it, and
differencing two `race_time_s` values compares crossings of different laps.
Project rivals to a common lap count — one lap down arrives at
`lap_start_t + lap_expected_s` — and compare arrivals. Getting this wrong
fired the track-position defender on a negative gap about a quarter of the
time.

Both looked correct in review, which is why they are here rather than in a
comment.

**A lap time that is not pace must be labelled.** The wave-around credit is
the only one so far, flagged `wave_by`. Anything reading caution lap times
as pace excludes it.

**Fuel is in normalised tank units.** A full tank is 1.0; burn per lap is
set so a tank lasts the observed stint. Do not "improve" this into invented
litre figures.

**The level is measured; the shape is regulated.** `pitstop.py` solves its
refuel rate backwards from `pit_time_mean_s` so a full service costs the
measured mean by construction. A rulebook may change what a *partial* stop
costs; it may not quietly restate how long a stop takes, because the timing
data already answered that.

**Every claim is labelled measured or assumed.** New assumptions go in
`ASSUMED_FIELDS` so they appear in the notebook's measured-versus-assumed
table and get swept.

**A stop never costs less than the lane.** Added at the 00 re-run, and
currently violated: `stop_cost` must not return less than
`pit_time_mean_s * pit_transit_frac` at any flag, because a car cannot traverse
the pit lane for less than the time the pit lane takes. The agent found 12.85 s
at Daytona against a 22.45 s transit and optimised into it. Until this holds,
no agent result describes strategy rather than arithmetic. See amendment 14.

---

## 6. Where the draft and the build disagreed

Recorded so no future thread re-litigates them by accident.

| Draft blueprint said | Resolution |
|---|---|
| Cautions sampled per lap inside `step` | **Rejected.** Drawn in advance; guarded by a test |
| 02a has zero race-control or wave-around logic | **Reversed.** Compression and wave-arounds are in — they are how a caution becomes positionally cheap, which the assumed discount dial cannot represent |
| Two heuristic baselines | **Superseded.** Five parameter-free strategies plus a two-stage benchmark |
| Single-car MDP env as the simulator | **Reframed.** The whole-field engine is the simulator; the env is an adapter with a focal car |
| Env file under 250 lines | **Kept**, applied to the wrapper. The engine is larger and that is fine |
| 01 complete against 1.6M real laps | **Corrected.** Validation used a synthetic fixture; re-run on real data |
| Plotly for charts | **Changed to matplotlib**, matching `viz.py` as built |
| The WEC pit lane stays open under caution | **Corrected in 02a.** It closes; see section 7C. The asymmetry is staged against unstaged reopening |
| 01's suite is 33 tests | **Corrected.** It was 42 before 02a and is 92 after. The 33 predated `test_cautions.py` |
| Decision 17: redraw starts until non-overlapping | **Superseded.** The engine implements an alternating renewal process with a stationary start. `02a_caution_verification.ipynb` Part 6 shows the stated reason for superseding decision 17 does not reproduce, and that the real failure is worse. **The decision record has been amended to match**; Part 6 was the replacement text |
| 02a is four changes | **Corrected.** Five shipped — the caution process was decided and built ahead of the stage |
| Decision 9 item 3: the rival is "whoever is directly ahead on the road" | **Amended at 02c.** A stop cannot drop a car behind something it is already behind. The rival is the car that is behind now and ahead after the stop; the two readings that describe a decision coincide on it |
| Decision 9 item 5: "takes the wave-around when one is available" | **Superseded at 02c.** No such action exists — the credit is engine-internal and unconditional on the frozen eligible set. Replaced by a standing condition preserving eligibility while lapped and under caution, with the scope limits in section 7C |
| `RaceState.wave_eligible`, added at 02c | **Withdrawn at 02c before it was written.** Clause two of the amended item 5 has no threshold, so the roster never asks the predicate and the stage adds nothing to the engine. The at-the-line hazard it was to contain is an invariant in section 5 instead |
| 03a's gate: "a policy that always returns no pit reproduces the fuel-window baseline exactly" | **Dropped at 03a**, not restated. It cannot hold — the baseline stops a lap and a half early, the forced rule a lap late — and 02c's proposed restatement on laps and stops fails on four and two seeds of twelve. Replaced by an echo-the-baseline gate through the wrapper's plumbing |
| 03a's gap rows normalised as "0…120 s, clipped, /120" | **Superseded at 03a.** The clip never binds and the median observation lands at 0.03. Rescaled on `pit_time_mean_s`, where 1.0 is a stop's worth of gap and the scale moves with the dials |
| The observation table's "ninth row" and "tenth row" | **Settled at 03a.** Nine rows: `pit_lane_open` in, `wave_eligible` out, the latter conditional on compression's unvalidated magnitude |
| Pace modes, "decide once and for all at 03a" | **Settled at 03a: out, permanently.** No engine lever, no `PitDecision` field, and they reopen the stint-length dimension decision 3 closed |
| Section 7A's reward carries "a DNF term" | **Dropped at 03a.** The engine models no retirement; `finished` is the chequered flag. Inventing one in the wrapper would be physics |
| `RaceEngine.run` as the race loop | **Inverted at 03a.** `run_stream` is the loop and `run` drains it, so a suspended race and a completed one come off one implementation |
| `pit_transit_frac = 0.25` as a background assumption | **Falsified at 03b.** 02c's `splash_and_dash` result is substantially an artefact of it: gained falls monotonically 0.40 → 0.10 in IMSA and 0.58 → 0.44 in WEC as the dial runs 0.25 → 0.65, while every other roster row wanders non-monotonically. The strategy built around a cheap short fill is the one whose result depends on how a short fill is priced. See amendment 10 |
| Section 7A's reward, "dense negative time plus pit penalty" | **Amended at 03b.** Degenerate in a timed race: the elapsed times sum to the race duration, so the return is constant and carries no gradient. Replaced by one lap credited per lap. Evidence and numbers in amendment 8 |
| 03b specified with a boundary constraint and no verification gate | **Supplied at 03b.** Two conditions with falsifiers — the wrapper's plumbing reproduces a decision taken directly, and the exported checkpoint is the policy that was measured. 02c had already named an unstated gate as a gate nobody can fail |
| 03a's replacement gate, "the agent's plumbing sits in the middle" | **Corrected at 03b.** The echo it compares is a bare callable touching neither `observe` nor `to_decision` nor `PolicyStrategy`; what it asserts is that `run_focal` does not care whether a callable is a class instance or a function. 03a's amendment stands and is not reopened — 03b's gate one closes the gap rather than restating it |
| `data/processed/{series}.json` as the dials every stage reads | **Refused at 03b.** They pool editions: 216 hours of Daytona in one config, DPi and GTLM and GTDPRO in one field, four-hour green stints, pit-time standard deviations five times their means, negative degradation at Le Mans. The stand-in is used and labelled until 00 is re-run |
| `event` ILIKE as the way a race is scoped | **Superseded at the 00 re-run.** `event` carries a circuit and no edition, so the pattern selected every running in the file. Scope is `session_id`, and the pattern path is removed rather than deprecated — a reachable pooling path is how this survived two stages |
| `car` as a numeric car identifier | **Falsified at the 00 re-run.** Parsed as an integer it merges entries whose numbers differ only by a leading zero, and the pair is sometimes in the same class — `#7` and `#007` are both Hypercar. Read as text. A composite `(car, class)` key would not have been enough |
| `MAX(stint_number)` as a stop count, and `stint_number` as a fuel stint | **Superseded at the 00 re-run.** The counter steps on driver changes, two to three fuel stints apart. Stints and stop counts come from the pit records |
| `pit_time_mean_s` as an arithmetic mean | **Superseded at the 00 re-run.** Within one scoped race the column carries hour-long repairs beside eighty-second stops. The dial is a median; the spread is the sd of the sample trimmed at 3x it. The field name is unchanged |
| "anything that is not `GF` is a caution" | **Corrected at the 00 re-run.** Caution is the named set `{FCY, SF, RF}`; `FF` and nulls leave both sides of the share |
| 00's dials as reproducible | **Falsified at the 00 re-run.** `calibrate_cautions` picked its reference car with an unbroken tie, so the caution dials were an arbitrary draw and the fingerprints moved between runs of the same notebook. Fixed by a deterministic tie-break; the residual single-car sensitivity of 0.6% is a named limitation |
| 00's condition two, "the dials reproduce the race" | **Dropped and replaced at the 00 re-run**, per 03a's precedent. It compared the engine and roster against the file, not the calibration, and missed in opposite directions in the two series for two separate reasons, neither of them a dial. Replaced by checks of the dials against the file plus one one-sided engine bound. See amendment 12 |
| 02b's dials, loaded from `dials_imsa.json` | **Corrected at the 00 re-run.** That filename never existed, so 02b silently ran on a three-hour stand-in throughout. On real dials it is degenerate: zero foreknowledge on every seed |
| `caution_pace_multiplier = 1.6` as a background assumption | **Contradicted at the 00 re-run.** Observed 1.87 in IMSA and 1.79 in WEC. Still assumed and still swept; the measurement is reported beside it. See amendment 15 |
| Decision 10's budget, read as one pooled table | **Clarified at 02c.** Two banks and two tables, one per series, never a pooled row: a rulebook difference makes the wave lever live in IMSA and dead in WEC, so an averaged row reports the mean of a measurement and a non-measurement |

---

## 7. The four points the draft left unresolved — now settled

**A. The reward credits laps; the score is position.** Training uses a dense
per-lap reward, because class position is sparse and nearly unlearnable as a
reward signal. **The DNF term this paragraph originally named is dropped at
03a**: the engine models no retirement, and a failure model invented in the
wrapper would be physics. **The pit penalty needs no separate term either** —
a stop makes that step long, consumes race time and leaves fewer laps to
credit, so it is already priced and adding a constant would double-count it.
Evaluation uses class finishing position, as everything else in the project
does. This split is deliberate and must be stated wherever agent results
appear: the agent optimises a proxy for the thing it is judged on. The gap
between the two is a finding in its own right, and it belongs in the
write-up rather than buried.

**Amended at 03b — the original "dense negative time" is degenerate here.**
This paragraph previously specified negative elapsed time, divided by the
class green lap. Summed over an episode those elapsed times are *the length
of the race*, and these are timed races, so the return was the same number
whatever the policy did. Measured over 2,664 training episodes: return
−219.85 ± 0.42 while laps completed ran 167 to 206, correlation −0.11, and
500,000 steps moved the mean by 0.016% — twelve times smaller than the
episode-to-episode noise. The IMSA policy converged on pitting during 86% of
its laps, throwing away twenty laps and four class positions, and **scored
the same return as the null**. Negative time is right for a fixed *distance*,
where finishing sooner is winning; at a fixed *duration* the time is given
and the term cancels. The replacement credits one lap per completed lap.
Any affine combination of the two is the same policy gradient, because the
old term is a constant — so this is the whole change, not a tuning choice.

**B. Pace modes are out of 02, revisited at 03a.** The draft's push /
normal / fuel-save actions need a lever the engine does not have, and
`PitDecision` carries no pace field. Adding them would reopen the
stint-length dimension that fixing fuel as the binding constraint closed
off. They stay out through 02c so the human roster and the agent face the
same action space.

**Settled at 03a: out, permanently.** They need an engine lever that does
not exist and a `PitDecision` field that does not exist; they reopen the
stint-length dimension decision 3 closed by fixing fuel as the binding
constraint; and they would have to go into the human roster too, which
reopens decision 9's parameter-freeness. The question is closed rather than
deferred again. What the agent got instead is the fill-to-the-flag action,
which reaches a lever the roster already had rather than inventing one it
did not.

**C. The pit lane closes under caution in both series; what differs is how
it reopens.** This is a rulebook fact, not an assumption, so it does not go
in `ASSUMED_FIELDS`. IMSA deems the pits closed at the announcement of Full
Course Yellow (46.1) and reopens by class — GTP and LMP2 on the first lap
after the pits are declared open, GTD PRO and GTD on the next, anyone
thereafter (46.3.1) — except under a Short FCY, declared for any caution
within thirty minutes of the start or the first within fifteen minutes of a
restart, where the lane stays shut until the green and no Final Wave-By is
run (46.3.3). WEC closes the pit entry when FCY is announced while leaving
the exit open (14.5.2), and under the Safety Car closes the entry for the
first three laps, two if the Safety Car follows an FCY (14.6.5), after which
entry is unrestricted for everyone at once. Emergency service exists in both
and is not modelled: a strategy calling for one is asking for a penalty.
Wave-around eligibility is identical in the two rulebooks — any car whose
class leader is behind it in the order circulating behind the safety car
(IMSA 46.2.2 and 46.4.1, WEC 14.6.4) — with IMSA running it up to twice and
WEC once.

This remains the largest strategic difference between the series and most of
the reason to simulate both; the difference is staged reopening against
unstaged, and a caution that offers a window against one that never does.
Consequences:

- **02a** — `pitstop.py` gains a lane-status rule per series, as above.
  **Delivered.** `PitRules.for_series` carries it; a series with no rulebook
  entry leaves the lane open rather than asserting a regulation nobody wrote.
- **02b** — the benchmark's search over stop laps must exclude closed
  windows in both series, with a gate the no-caution brute-force test
  cannot provide.
- **02c** — the caution gambler carries real risk in both series: waiting
  for a caution can mean arriving at a shut lane and losing more than it
  hoped to save.

**A consequence of the two timings, established at 02c.** The first
wave-around is announced when the pits are declared open, at
`start + open_delay_laps × caution_lap_s`; the lane opens to the earliest
class one further stagger lap after that, and the minimum stagger is 1.0 lap
in IMSA and 3.0 in WEC. **The Pass-Around's eligible set is therefore frozen
a full caution lap before any car may take a voluntary stop, in both
series.** No strategy can forfeit it by a decision. Only IMSA's Final
Wave-By (46.4.1) is forfeitable at all, and since WEC runs one wave, the
lap-down defender's second clause is inert across the whole WEC half of the
comparison. That is a rulebook consequence to report rather than an
asymmetry to engineer away, and it is one of the reasons the two series are
scored in separate tables.

**Measured at 02c, the clause is inert in IMSA too, for a different
reason.** It fired in neither series, over sixty seeds a side and again
over the full two hundred, where it comes out symmetric either side of zero
— 3% gained against 3% lost in IMSA, 7.5% against 8.5% in WEC. The WEC half
was predicted; the IMSA half is not the rulebook at all — compression takes
the headline class from a ten-lap spread to about one, so the focal car is
almost never a lap down and the situation the clause responds to has largely
been engineered out of the engine. Read this with 02a's finding 1, that
compression's magnitude is unvalidated and may close the field too hard,
rather than as a fact about the strategy.

**One reading recorded as a reading.** Article 46.3.2 disapplies the
standard procedure in the first and last thirty minutes without saying what
replaces it. `pitstop.py` treats that as the Short FCY outcome — the lane
stays shut. If that reading is wrong it is one constant, and the docstring
says so.

**D. Degradation stays linear.** `calibrate.py` fits a linear slope against
tyre age and the engine applies it linearly. The draft's quadratic constants
are not adopted, because the calibration cannot fit a form it was never
given. **Check for curvature when 00 is re-run against the real
`laps.csv`** — residuals against tyre age will show it plainly. If real
curvature appears, that is a measured finding and the model changes to
match; until then, do not add a parameter the data has not asked for.

---

## 8. Conventions

British English, plain words, except where a technical term is the right
one — RL, MDP, degradation, stint, full-course yellow. Comments explain
*why*, not *what*. Notebooks are generated from `build_nb.py`, so edits
belong there; an edit made directly in Jupyter will be lost. New assumptions
go in `ASSUMED_FIELDS`. Tests accompany the change that needs them, not a
later cleanup pass.

Two conventions 02a added, both worth keeping:

**A regression gate compares against the past, not against itself.** The
reference numbers for `Compat.v01()` were captured from the engine before it
was changed. A gate built from the new code would have passed whatever
had broken.

**A normalisation is a modelling choice and gets measured.** 03a added this
one. `/120` on the gap rows looked reasonable and put the median observation
at 0.03 of the row's range. A scale nobody has measured is a magic number
with a docstring, and the other two in the observation table are still in
that condition.

**A switch for a change that has not landed refuses to be set.** `Compat`
carries a map of flags to the work they are waiting on, and setting one
early raises rather than silently doing nothing. The map is empty now; the
next stage that adds a switch should refill it.
