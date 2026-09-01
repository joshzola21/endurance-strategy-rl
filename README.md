# Endurance race strategy

A simulator of WEC and IMSA endurance racing, built from real lap timing, with
five human pit strategies, a control that never calls a stop, and an agent
trained by reinforcement learning — all driving the same car through the same
interface. Move a dial, watch a 24-hour race lap by lap, make a pit call
yourself, and see what the agent would have done.

Two races are modelled: the **Daytona 24 2026** (60 cars, four classes) and the
**Le Mans 24 2026** (62 cars, three classes). Each is calibrated from one
running of that event.

**[Open the app](https://endurance-strategy-rl.streamlit.app)** · or run it locally, below.

---

## What this project found

I set out to train an agent that beats human pit strategy. It does not, and
why not is the part worth reading.

**There is no agent result, and that is a finding rather than a gap.** Every
agent figure this project produced came from a single training run. Training
five instead, changing nothing but the seed, showed that in WEC the seed alone
moves the headline statistic by 0.45 where two hundred paired races can only
resolve 0.03 — one run gained a place in 45% of races and three others gained
one in none. In IMSA all five came out identical, which looked like stability
and was not: they had all converged on never asking to pit.

**An agent that has learned nothing is now detectable.** `never_pit` is a sixth
member of the roster that always stays out and lets the rules supply every
stop. It is the true null for a learner — an agent that cannot beat *taking no
decisions* has learned nothing — and on the fifty held-out races the trained
IMSA agent and `never_pit` score identically to three decimal places. Nothing
in this project could previously have seen that.

**Four measured ways the simulator misleads a learner.** Each was measured on
the engine directly rather than inferred from an agent's score, so all four
stand whatever any training run does:

| | what it is |
|---|---|
| A stop priced below its own lane | the cheapest caution stop the engine could price cost 13.47 s against a 22.45 s pit lane transit |
| Caution compression severs laps from the score | a car that stops forty extra times at Daytona spends 1,072 s more in the pits and loses **zero** laps; the refund tracks the caution rate, 70% at Daytona and 8% at a near-green race |
| The agent stops looking before the stop has paid off | a forced caution stop is worth **+0.89 places** twenty laps later at Daytona and **−0.67** at the flag; the agent was told the truth about the next twenty laps and the opposite about the race |
| The reward was defined on something the agent could not see | class position was rewarded while the observation's closest proxy correlated 0.091 with it |

They share one root. Caution compression makes stopping locally cheap, and a
proxy, a cost or a horizon can each fail to see that differently.

**One human strategy does clearly work.** At Daytona, a strategy that gambles
on stopping under caution gains a place in **58% of 200 races**, with a median
gain of one place. On the fifty held-out races it gains in 60%, so the share
carries over; the median on that smaller set has an interval that includes
zero, so the median does not. At Le Mans the same strategy is worth much less,
because the rulebooks differ: IMSA releases classes from a closed pit lane in
staggered order and WEC releases everyone at once, so far fewer caution stops
are reachable.

**And one dial cannot be measured at all.** `pit_transit_frac` — the share of a
stop that is driving down the lane rather than being serviced — is assumed at
0.25. Three attempts to measure it from the timing failed, for one reason:
separating what a stop costs to *enter* from what it costs to *fill* requires
observing stops that took different amounts of fuel, and endurance cars fill to
the brim. Between 62% and 90% of stops in these two races are followed by a
near-full tank. The dial stays assumed and gets swept, which is what
`scripts/sweep_pit_transit.py` is for.

---

## Run it

Python 3.11 or later.

```bash
git clone <this repository>
cd endurance-strategy-rl
pip install -r requirements.txt      # the app: about 200 MB
streamlit run app/Home.py
```

The frozen dials, seed banks, trained agents and evaluation tables are all in
the repository, so the app runs immediately with nothing to build.

For training, calibration and the notebooks:

```bash
pip install -r requirements-dev.txt  # adds PyTorch, DuckDB, Jupyter
pytest tests/                        # 266 tests
```

The suite also runs without pytest, for an environment that has not got it:
`python tests/run_tests_nopytest.py`. It skips what it cannot handle and names
it, rather than reporting a clean total.

### Reproducing the numbers

```bash
python scripts/freeze_assets.py        # the race, the seed banks, the field
python scripts/train.py                # both series, about 15 minutes each
python scripts/evaluate.py             # 200 paired races per strategy per series
python scripts/check_artefacts.py      # is everything present and about one race
```

`scripts/seed_sweep.py` runs the five-seed study and
`scripts/collect_seed_spread.py` reports it. Together they take about two
hours.

**The raw timing is not in this repository.** `laps.csv` is 582 MB and 1.66
million rows, and it is the one input that cannot be regenerated from what
ships here. The calibrated dials in `data/processed/` are its output and run to
a few kilobytes, which is why they are committed. Anything under `scripts/`
that reads the raw file — the calibration, the lap profile, the pit estimator —
needs it, and says so when it is missing.

---

## How it is laid out

| Path | What it is |
|---|---|
| `app/` | the Streamlit app: three pages, and a controller that steps one race |
| `src/endurance/params.py` | the five dials, as data, plus the lever mechanism |
| `src/endurance/calibrate.py` | real lap timing to dials |
| `src/endurance/engine.py` | the race |
| `src/endurance/pitstop.py`, `caution.py` | what the two rulebooks change |
| `src/endurance/strategies.py` | the six-member roster and the background field |
| `src/endurance/harness.py` | the paired comparison and the dial sweeps |
| `src/endurance/gym_env.py` | the Gymnasium adapter, and nothing else |
| `notebooks/` | six stages, each with its own verification gate |
| `scripts/` | freezing, training, evaluating, checking |
| `tests/` | the invariants everything else leans on |
| `docs/ARCHITECTURE.md` | how the pieces fit, and why the boundaries are where they are |

---

## What is measured and what is assumed

Every number comes from one of two places. Either it was fitted to real lap
timing, or nobody can fit it to lap timing and somebody picked it. The second
kind are listed in `params.ASSUMED_FIELDS`, marked *assumed* throughout the
app, and exposed as sliders — because the honest thing to do with a number
somebody picked is move it and see whether the conclusion survives.

**Four things to know before reading any number here**, carried on every screen
of the app and in full on its methods page:

1. **Every dial starts from one running of one race.** The caution share alone
   moves by a factor of 2.8 between two adjacent Daytonas, so the width of each
   slider tells you more about the uncertainty than its starting value does.
2. **"Degradation" at Le Mans is a within-stint trend, not tyre wear.** The
   fitted slope comes out negative for Hypercar and LMP2, which cannot be
   tyres. It is fuel burning off and a track rubbering in, and the fit cannot
   separate them.
3. **There is no agent result.** See above.
4. **There is no benchmark row.** A two-stage reference was built — one arm
   knowing the caution timeline in advance, one not — and on real dials it is
   degenerate: the arm that can see the future extracts nothing from it on any
   seed and is beaten by its own control. Rather than show a number that does
   not mean what it appears to, it is shown nowhere.

---

## What holds it together

Every artefact — each seed bank, background field, policy card and saved table
— records a hash of the dials it was built against, and anything that does not
match is refused rather than quietly used. A second hash covers the rulebook
logic, which the first cannot see, because a change to how a stop is priced
moves every race in the project while leaving every parameter untouched.

Each stage carries a verification gate with a falsifier: a case the gate itself
must fail on. The paired comparison's null must move nothing at all, on every
seed. The app's race must reproduce the harness's race bit for bit. The
benchmark's dynamic programme must agree with brute force about the *plan* and
not merely about the total.

`python scripts/check_artefacts.py` runs the whole inventory and reports
whether the tree can be shipped. `python scripts/check_figures.py --strict`
does the same for the writing: every number quoted in these documents traces to
a file on disk, and a number that appears in neither the manifest nor its list
of exclusions fails the check.

---

## What this does not claim

The agent did not beat, match or lose to the roster; there is no agent result
to compare. The four traps are not ranked against each other — each is
measured, but which dominates cannot be settled without training runs that
resolve, and at 500,000 steps they do not. The dials describe one running of
one event per series and are not an average of anything.

And the whole thing is a model. It reproduces lap times, pit costs, caution
behaviour and the two rulebooks. It does not reproduce weather, damage,
penalties or a driver having a bad night.
