# Architecture

How the pieces fit, and why the boundaries are where they are.

The short version: **there is one race loop, one strategy interface, one place
each quantity is defined, and every artefact carries a hash of what it was
built against.** Everything below is a consequence of those four, and most of
the rules exist because the alternative was tried somewhere and failed
quietly rather than loudly.

---

## Which way the dependencies point

```
        params.py          the five dials, as data
            |              plus scale_dials / set_dials, the lever mechanism
            v
   pitstop.py  caution.py  what the two rulebooks change about a race
            |              (RULES_VERSION lives here)
            v
        engine.py          the race. the only race loop in the project
            |
            +--> strategies.py      six roster members + the background field
            |         |
            |         v
            +--> harness.py         the paired comparison, the sweeps
            |         |
            |         +--> benchmark.py, causal.py   the per-race reference
            |
            +--> gym_env.py         the Gymnasium adapter, and nothing else
                      |
                      v
                 policy.py          load, export, and the policy card
                      |
                      v
                   app/             presentation only
```

Nothing points back up. `assets.py` sits beside the chain and is imported by
most of it; `viz.py` and `calibrate.py` hang off the side. The property worth
stating: **`import endurance` pulls in neither Streamlit nor the app**, and a
test asserts it in a subprocess, because `sys.modules` inside a test session
that has already imported both would answer a different question.

---

## The seven boundaries

### 1. One race loop

`RaceEngine` is the only thing that advances a race. `app/controller.py` drives
`RaceEngine.run_stream`; the whole of its stepping is eleven lines that choose
which decision to `send`.

*Why.* A user interface that steps the race slightly differently from the thing
every published number came from shows a complete, plausible race that nobody
else can reproduce. It is the failure that produces numbers rather than errors.

*What enforces it.* `tests/test_app.py` walks every module under `app/` with an
abstract syntax tree and refuses any use of `RaceEngine`, `run_race` or
`run_stream` outside the controller, plus any reach into the engine's private
working. A second test feeds the scanner a file that does all three and
requires it to catch each one — a scanner that finds nothing passes an app that
does everything.

### 2. Artefacts, not notebooks

The race configuration, the seed banks and the background field are written to
`data/processed/` once, and every later stage loads what was written.

*Why.* Training, evaluation and the notebooks must be about the *same* races,
or an agent's row is not comparable with the human rows beside it. Three files
that each rebuild the configuration in memory will agree right up until one of
them is edited.

*What enforces it.* Every artefact records the dials fingerprint it was built
against, and `freeze_assets.load_assets` refuses a mismatch rather than
reporting it. `scripts/check_artefacts.py` checks the whole inventory at once
and additionally refuses when two files in the tree answer to one name.

### 3. One strategy interface

A strategy is a callable taking `(CarState, RaceState)` and returning a
`PitDecision`. The five human strategies, the `never_pit` control and the
trained policy are all that, and all six arrive at the comparison through the
same `ROSTER` mapping.

*Why.* An agent scored by agent-specific code can differ from the roster's
scoring in ways that produce plausible numbers. `scripts/evaluate.py` contains
no evaluation logic of its own: it wraps the policy, inserts it into the roster
under the name `agent`, and calls the same function the humans' rows came from.

*Consequence worth knowing.* The action mask is a training convenience.
`PolicyStrategy` carries none, so at scoring time an agent asking to stay out
on an empty tank is overridden by the engine and scored on that — exactly as a
human strategy would be. The app draws the mask beside the policy's ranking
rather than applying it to it, and says so.

### 4. The app is presentation only

Nothing under `app/` computes a lap time, a fuel burn, a caution or a gap.
Every quantity on the screen is read off `CarState`, `RaceState` and
`LaneStatus` as the engine produced them.

*A detail that matters.* The controller does not hold a live generator as its
state. A suspended generator cannot be serialised, does not survive a Streamlit
reload and cannot be rewound. What it holds is a **decision log** — a map of
lap number to the action taken on that lap — plus the seed, the seat and the
dials. That tuple reproduces the race exactly, because every random number is
drawn from the seed before the race starts. The live generator sits beside it
as a cache for stepping forwards; any seek backwards replays from lap zero,
which takes about a second.

*What enforces it.* A race driven through the controller must reproduce
`harness.run_focal` bit for bit, and a race that was seeked through must equal
the same race driven straight.

### 5. The adapter is an adapter

`gym_env.py` converts between the engine's objects and the arrays a learner
wants: a ten-row observation, five actions, a mask, and a reward. It prices
nothing.

*Why.* A wrapper that computed its own lap cost would be a second simulator,
and the agent would be trained on one race and scored on another.

### 6. Charts live in one place

Every figure comes from `viz.py`, unchanged, whether it is drawn in a notebook
or on a page. A chart built inside the app is a chart the notebooks cannot use.

### 7. No second definition of anything

The rule the other six are instances of. The pit window's opening is a dial
comparison in one function, imported by the app rather than restated. Class
position is derived in one place and read everywhere else. The parallel
comparison shares its row-building code with the serial one, so the delta signs
exist once and both paths reach them.

---

## Determinism, and what it buys

**Every random number is drawn from the seed before the race starts.** The
caution timeline is drawn in advance, so it cannot depend on what any car does.
Per-car noise is held in indexed streams keyed by *(seed, stream, car, index)*,
so a car's third pit stop gets its third noise draw whenever it takes it.

That gives the property the whole comparison rests on: **noise is a function of
the seed and never of the strategy.** Two strategies pitting at completely
different moments see the same lap-time noise on the same lap of the same car.

It was not always true. Under a single shared generator the draws are
entangled, so changing a caution model silently reassigns every car's pace and
a comparison measures both changes at once. The old behaviour is still
reachable through a compatibility flag, and a test asserts that it still has
the defect — the fix is only legible if the thing it fixed can be shown.

**The paired comparison** follows from it. Each strategy is scored on the same
race run twice: once with the strategy in the focal car, once with the
fuel-window plan. The difference is that strategy's effect on that race, with
the cautions, the traffic and the rest of the field held identical. Deltas are
signed so that positive is better in every column — which means `d_class_pos`
is the null minus the treatment while `d_laps` is the other way round, since a
gained place is a *lower* position number. A test checks the signs against the
raw columns rather than trusting them.

The null's own row must be identically zero on every delta column, on every
seed. That is the comparison's gate, and it is checkable in the published table
without running anything.

---

## The two fingerprints

| | covers | moves when |
|---|---|---|
| `dials_fingerprint(config)` | every number in the race configuration | a dial is recalibrated or swept |
| `rules_fingerprint()` | `pitstop.RULES_VERSION`, `caution.RULES_VERSION` | the rulebook logic is rewritten |

They answer different questions and are kept apart so a mismatch says which one
moved.

The second exists because the first cannot see logic. A change to how a stop is
priced alters every race in the project while leaving every parameter — and
therefore every fingerprint, every bank, every card and every saved table —
matching perfectly. The tripwire that had caught four artefact collisions would
have reported nothing.

**What neither covers: `engine.py`.** Putting it in would mean bumping a
version on every edit to the largest and most frequently touched module in the
project, and a version nobody bumps honestly is worse than an absent one. This
is a stated limit rather than an oversight.

Rules fingerprints are checked **only where present**. Every artefact frozen
before the check existed carries none, and `None` means "predates the check"
rather than "passes" — reported as a gap on the methods page.

---

## The seed banks

Per series: **200 headline** races for every published claim, **50 sweep**
races, and **50 held-out** races.

The sweep fifty are the *first fifty of the headline two hundred*, not a
separate draw. A sweep asks how a claim moves as a dial moves, and the cleanest
form of that question compares against the same races the claim was made on; a
disjoint bank would add sampling noise to a comparison that does not need any.

The held-out fifty are genuinely disjoint, for the opposite reason: they exist
to answer whether a design chosen on the headline races generalises, and a
held-out set that overlaps the selection set answers nothing. The training
environment refuses held-out seeds outright. Evaluation does not touch the
environment, so on that bank the discipline is procedural: do not select on it.

The seeds are stored as explicit lists rather than as a rule for generating
them. A rule is shorter and invites exactly one failure — somebody changes the
generator, the lists move, and two stages are scored on different races.

---

## Gates

Each stage carries a verification gate, and each gate carries a falsifier: a
case the gate itself must fail on. A gate that cannot fail is not a gate.

| Stage | The gate | Its falsifier |
|---|---|---|
| 00 calibration | the dials regenerate exactly from the raw timing | pooling two editions must break the recovery |
| 01 engine | invariants: fuel, tyre life, the flag, position as a ranking | different seeds must give different races |
| 02a corrections | with every flag off, the engine is bit-for-bit the pre-02a engine | the old shared generator must still show its defect |
| 02b benchmark | the dynamic programme agrees with brute force about the **plan** | a plan one lap off the optimum must rank strictly worse |
| 02c comparison | the null's deltas are identically zero on every seed | a different strategy in the null seat must be caught |
| 03a wrapper | the wrapper prices nothing and drains one stream | — |
| 03b policy | the ONNX export agrees with the checkpoint over an evaluation pass | — |
| 04 app | the app's race reproduces the harness's race bit for bit | two different seats must produce different races |
| 05 packaging | every artefact present, current, and about one race | an ambiguous or orphaned artefact must be refused |

---

## Assumed and measured

`params.ASSUMED_FIELDS` lists every quantity that lap timing cannot identify —
the caution pace multiplier, how the field bunches behind a safety car, tyre
life, how a stop divides between transit, tyres and fuel, and the traffic
window. Each is exposed as a slider and swept rather than trusted.

The rule when a measured counterpart appears and disagrees: **the dial stays at
its assumed value and the disagreement is shown.** Changing a dial changes
every number measured against it, which is a recalibration rather than an edit.

One of them turns out to be unmeasurable in principle from this data.
Separating what a stop costs to enter from what it costs to fill requires
observing stops that took different amounts of fuel, and endurance cars fill to
the brim — between 62% and 90% of stops in these two races are followed by a
near-full tank. `scripts/estimate_pit_transit.py` records the three attempts and why each
fails.

---

## Structural limits

**The pit model cannot express a large fixed overhead.** A full service is
anchored at the measured mean and split into shares, so raising the transit
share lowers the refuel share and a full stop always costs the same. Only the
ratio moves. That is the right shape for asking what a *partial* stop saves and
the wrong shape for a real Daytona pit lane, where entry, the speed limit and
exit are most of a stop.

**Cautions are a property of the race, not of the field.** They are drawn in
advance from a rate and a mean duration, so nothing a car does can cause or
extend one. That is what makes the comparison paired, and it means the
simulator cannot represent a caution *caused* by an incident.

**The background field runs one strategy.** Every car that is not the focal car
runs the fuel-window plan, so the field pits in lockstep. Traffic modelling
only bites when stints are out of phase, so a single-strategy background hides
most of it. The field is held as a per-car map rather than a single name
precisely so that a mixed field is a change of values rather than a change of
format.

**500,000 training steps is denominated in laps, not races.** Daytona costs 742
steps an episode and Le Mans 385, so the same budget buys 674 races against
1,299. The asymmetry was never tested, because the five-seed sweep made it
moot.
