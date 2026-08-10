# Handover — end of the 01 thread

Written so a fresh thread can pick this up cold. Read this, then
`notebooks/01_race_engine.ipynb`, then start on 02.

---

## What the project is

An interactive endurance-strategy sandbox. You twist the real levers —
cautions, stint length, fuel, traffic — and watch a WEC or IMSA race
unfold, with simple human-style strategies and a reinforcement learning
agent racing on the same terms so you can see what each gets right and
wrong.

The point is **not** a champion agent. It is fluency someone can click
through in two minutes and come away actually understanding how these
races are decided.

## Where we are

| Stage | Status |
|---|---|
| 00 — data recon (WEC vs IMSA lap data) | done |
| **01 — the race engine** | **done, this thread** |
| 02 — human-style strategies | next |
| 03 — the RL agent | after 02 |
| 04 — the Streamlit app | after 03 |

## What exists now

```
notebooks/01_race_engine.ipynb   thin: calibrate, run, watch, twist, compare, validate
build_nb.py                      regenerates the notebook (edit here, not the .ipynb)
src/endurance/
  params.py                      the five dials as data; scale_dials() is the lever mechanism
  calibrate.py                   real lap data -> dials, one function per dial
  engine.py                      the race: event queue, whole field, derived positions
  strategies.py                  the strategy interface + three starter baselines
  viz.py                         race charts, reusable by the app
tests/
  test_engine.py                 24 tests on engine invariants
  test_calibrate.py               9 tests: does calibration recover planted truth
  make_fixture.py                synthetic laps.csv with known values baked in
  sqlite_shim.py                 DuckDB stand-in, so tests run without DuckDB
  run_tests_nopytest.py          runs the suite where pytest isn't available
  run_notebook.py                runs the notebook from notebooks/, catching path bugs
```

`pytest tests/` should report 33 passing.

## The four decisions taken in this thread

Chosen deliberately in a decision quiz at the start, so they should not be
silently reversed later.

1. **01 is the race engine, plus a demo notebook** — not a toy simulator,
   not a calibration exercise. Everything later plugs into this core.
2. **Multi-car field, position derived from cumulative race time** — the
   whole field is simulated; a car is ahead because it completed more laps
   or got there sooner. No overtaking model: cars pass by being faster.
3. **`src/` package, notebooks stay thin** — so nothing drifts between a
   notebook, the agent and the app.
4. **Streamlit is the eventual front end** — `viz.py` returns plain
   matplotlib figures and `scale_dials` takes multipliers, both shaped for
   sliders.

## Three properties worth protecting

**All randomness is drawn before the race starts.** The caution timeline
especially. This is what lets two strategies be compared on *exactly the
same race*, so a difference between them is strategy and not luck. It is
what makes 02's comparison and 03's evaluation mean anything.
`test_caution_timeline_is_independent_of_strategy` guards it. Do not
replace it with lap-by-lap sampling.

**A strategy is just a callable** taking `(CarState, RaceState)` and
returning a `PitDecision`. The RL agent should implement this same
interface, so the engine cannot tell it apart from a human strategy. Do not
give the agent a privileged path into the engine.

**Fuel is in normalised tank units, not litres.** The lap data has no fuel
column, so a full tank is 1.0 and burn per lap is set so a tank lasts
exactly the observed stint length. This is honest and anchored. Do not
"improve" it into invented litre figures.

## Measured vs assumed

Everything in `ClassDials` is measured from timing data except these, which
are listed in `params.ASSUMED_FIELDS` and printed in the notebook:

- `pit_caution_discount` — how much cheaper a stop is under caution
- `caution_pace_multiplier` — how slow a caution lap is
- `traffic_window_frac`, `traffic_penalty_s` — what counts as traffic and what it costs
- `tyre_life_laps` — set to twice the fuel stint, so fuel binds first

`pit_caution_discount` does real work in any claim about caution strategy.
The same limitation appears in the F1 thesis: the true advantage of
stopping under caution is largely positional, and a lap-time model cannot
represent position. **Sweep it in 02 rather than quoting a single number.**

## What 01 does not do, on purpose

- No overtaking model — needs parameters no timing sheet contains.
- No weather, no reliability, no driver skill.
- Cautions arrive at random, when in reality they cluster — at night, in
  traffic, after restarts. A hazard rate varying over the race would be a
  genuine improvement and is a good 02 or 03 side quest.
- Dials come from one race per series, so they describe Daytona and Le Mans
  rather than IMSA and WEC in general.

## How well it matches reality

Against the synthetic fixture, simulated winner laps landed within 1.4%
(IMSA) and 2.7% (WEC) of the source race, with stop counts and caution
share in the right region. **These figures came from synthetic data, not
the real `laps.csv`** — DuckDB could not be installed in the environment
where this was built, so the numbers demonstrate the machinery works, not
that the calibration is right. Re-run the notebook on the real files and
check Part 6 before trusting any of it.

## Start here in 02

Build the human-style strategies properly. The three in `strategies.py`
(`RunToFuelWindow`, `OpportunistUnderCaution`, `FixedLapStint`) are
placeholders to exercise the engine, not a serious set.

What a real 02 needs:

1. **Strategies a strategist would recognise** — a full-course-yellow
   gambler, a track-position defender, a tyre-saver running long, a
   double-stint plan, a splash-and-dash at the end.
2. **A fair comparison harness** — many seeds, paired on common random
   numbers, reporting distributions rather than single races. Same-seed
   pairing is already available and already tested.
3. **A sweep of the assumed parameters**, `pit_caution_discount` first, so
   every claim can be labelled as either invariant to it or dependent on it.
4. **A per-race benchmark**, which is the harder and more interesting part:
   the best stop schedule available *for the race that actually happened*.
   Worth splitting into a clairvoyant version (knows when the cautions
   fell) and a causal one (sees only what a strategist could see). The gap
   between them is the value of foreknowledge, and it is the right zero
   point for judging the agent in 03 — measuring against the clairvoyant
   version alone would penalise a policy for failing to predict the future.

## Conventions

British English, plain words, except where a technical term is the right
one — RL, MDP, degradation, stint, full-course yellow, and so on. Comments
explain *why*, not *what*. New assumptions go in `ASSUMED_FIELDS` so they
show up in the notebook's measured-vs-assumed table.

Notebooks are generated from `build_nb.py`, so edits belong there — an edit
made directly in Jupyter will be lost next time it is regenerated.
