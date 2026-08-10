"""Regenerate the notebooks. Edit here, not the .ipynb.\n\nTargets: 00, 01, 02a, 02b, 02c, 03a, 03b. No argument builds all.\n"""

import json
import sys
from pathlib import Path


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def build_01():
    cells = []

    cells.append(md("""# The race engine (01)

Notebook 00 looked at real WEC and IMSA lap data and ended with a shopping
list: a degradation slope, a caution pattern, a stint length, a pit cost and
a traffic density, one set per series. This notebook builds the thing that
consumes that list - an engine that runs a full multi-class endurance race,
lap by lap, from those five numbers.

**What this is for.** The end goal is a sandbox you can twist the real
levers on and watch a race change, with human-style strategies and a
reinforcement learning agent both plugged into the same race on equal
terms. This notebook is the foundation of that: the engine, calibrated and
validated. Strategies get proper attention in 02, the agent in 03, the app
after that.

**Three decisions worth knowing**, because everything later leans on them:

1. **The whole field is simulated, and position is derived, never
   invented.** A car is ahead because it has completed more laps, or the
   same lap sooner. Traffic is then a real consequence of where cars
   actually are, rather than a random penalty bolted on.
2. **All randomness is drawn before the race starts**, the caution timeline
   especially. That means any two strategies can be compared on *exactly
   the same race*, so a difference between them is strategy and not luck.
   This is the single most important property in the whole project and
   there is a test that guards it.
3. **The code lives in `src/endurance/`, not in this notebook.** The engine
   the app will run is the engine you see working here - there is no second
   copy to drift out of step.

*What is measured and what is assumed* is tracked explicitly and printed
below, so nothing gets quietly promoted from guess to fact."""))

    cells.append(md("""### Setup

This notebook sits in `notebooks/`, while the package and the data sit at
the top of the project. Rather than assume which folder Jupyter was
started in, the cell below walks up the tree until it finds the project.
It then works whether you launch from `notebooks/`, from the project root,
or from an editor with its own idea of the working folder."""))

    cells.append(code('''import sys
from pathlib import Path


def find_project_root(marker: str = "src/endurance") -> Path:
    """Walk up from the working folder until the project appears."""
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {marker!r} at or above {here}.\\n\\n"
        "Expected a project laid out like this:\\n"
        "  endurance-strategy-rl/\\n"
        "    data/raw/          laps.csv, drivers.csv\\n"
        "    notebooks/         this notebook\\n"
        "    src/endurance/     the package\\n"
    )


ROOT = find_project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

print("project root:", ROOT)'''))

    cells.append(md("### Settings"))

    cells.append(code('''# ---- Settings ----------------------------------------------------------
# Built from ROOT, so they do not depend on the working folder.
LAPS_CSV    = ROOT / "data" / "raw" / "laps.csv"
DRIVERS_CSV = ROOT / "data" / "raw" / "drivers.csv"

PARAMS_DIR  = ROOT / "data" / "processed"    # calibrated dials get written here
FIG_DIR     = ROOT / "outputs" / "figures"   # charts worth keeping

# Anchor race and class per series - the same ones notebook 00 used.
ANCHORS = {
    "imsa": {"event": "%daytona%", "headline_class": "GTP",      "name": "Daytona 24"},
    "wec":  {"event": "%le mans%", "headline_class": "HYPERCAR", "name": "Le Mans 24"},
}

SEED = 0
SAVE_FIGURES = False    # True writes every chart into outputs/figures
# ------------------------------------------------------------------------

for _dir in (PARAMS_DIR, FIG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

_missing = [p for p in (LAPS_CSV, DRIVERS_CSV) if not p.exists()]
if _missing:
    raise FileNotFoundError(
        "Missing data file(s):\\n  "
        + "\\n  ".join(str(p) for p in _missing)
        + "\\n\\nPut laps.csv and drivers.csv in data/raw/, then run this cell again."
    )

print("data found in", LAPS_CSV.parent)'''))

    cells.append(code('''import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endurance import calibrate, viz
from endurance import (
    RaceConfig, run_race, scale_dials, assign_strategy,
    RunToFuelWindow, OpportunistUnderCaution, FixedLapStint,
)


def show(fig, name=None):
    """Draw a chart, keeping a copy in outputs/figures if SAVE_FIGURES is on."""
    if SAVE_FIGURES and name:
        fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


con = calibrate.connect(str(LAPS_CSV), str(DRIVERS_CSV))
print(f"Loaded {con.execute('SELECT COUNT(*) FROM laps').fetchone()[0]:,} laps.")'''))

    cells.append(md("""## Part 1 - calibrate the dials

`build_race_config` runs one query per dial per class and returns a race
that is ready to run. Every number it produces traces back to a named
function in `src/endurance/calibrate.py`, so "where did that come from?"
always has an answer.

Two calibration choices differ from the quick queries in 00, and both are
deliberate:

- **Degradation** is regressed against `est_tire_age` rather than
  `stint_lap`, and against lap time *relative to that lap's class median*
  rather than raw lap time. Working relative to the field cancels whatever
  hits every car equally on a given lap, leaving the part that is actually
  about tyre age. Laps carrying a `pit_time` are dropped, because an in or
  out lap can pass the clean-quartile filter and still be inflated by the
  stop, which would bias the slope upwards.
- **Stint length becomes fuel.** Endurance stints are fuel-limited far more
  often than tyre-limited, so the engine triggers stops on fuel. There is
  no fuel column in this data, so a full tank is defined as 1.0 and burn
  per lap is set so a tank lasts exactly the observed stint. The lever is
  honest - "more fuel" means longer stints, anchored to what the cars did -
  without inventing a litre figure."""))

    cells.append(code('''# One running of one race. `find_race` resolves the latest edition of the
# event and refuses when the answer is not unique; passing its session id on
# is what keeps every dial below scoped to a single race. 00 has the detail.
RACES = {s: calibrate.find_race(con, s, a["event"]) for s, a in ANCHORS.items()}

configs = {}
for series, anchor in ANCHORS.items():
    configs[series] = calibrate.build_race_config(
        con, series, RACES[series]["session_id"],
        f"{anchor[\'name\']} {RACES[series][\'year\']}"
    )
    print(f"{series}: {len(configs[series].classes)} classes, "
          f"{configs[series].total_cars} cars, "
          f"{configs[series].duration_s / 3600:.1f} h")'''))

    cells.append(code('''# Notebook 00's shopping list, filled in from data.
pd.concat([calibrate.dials_table(cfg) for cfg in configs.values()],
          ignore_index=True)'''))

    cells.append(md("""### What is measured, and what is assumed

Some things a timing sheet simply cannot tell you: how much cheaper a pit
stop is under caution, how slow a caution lap is, how much time is really
lost sitting behind a slower car. Those are held as assumptions with
sensible defaults, listed here so they are never mistaken for measurements.
They are the first things to sweep when a result looks surprising."""))

    cells.append(code('''example = configs["imsa"].classes[0]
pd.DataFrame({
    "dial": list(example.measured_fields()) + list(example.assumed_fields()),
    "status": ["measured"] * len(example.measured_fields())
              + ["ASSUMED"] * len(example.assumed_fields()),
    "value": [getattr(example, f) for f in example.measured_fields()]
             + [getattr(example, f) for f in example.assumed_fields()],
})'''))

    cells.append(code('''# Freeze the dials to disk so 02, 03 and the app all read the same numbers.
for series, cfg in configs.items():
    cfg.save(PARAMS_DIR / f"{series}.json")
print("saved:", sorted(p.name for p in PARAMS_DIR.glob("*.json")))'''))

    cells.append(md("""## Part 2 - run a race

One call. The engine draws a caution timeline, builds the field, and
advances whichever car finishes its lap soonest until the clock runs
out."""))

    cells.append(code('''results = {series: run_race(cfg, seed=SEED) for series, cfg in configs.items()}

pd.DataFrame([{"series": s, **r.summary()} for s, r in results.items()])'''))

    cells.append(code('''series = "imsa"
result = results[series]
headline = ANCHORS[series]["headline_class"]

result.classification().head(10)'''))

    cells.append(md("""## Part 3 - watch the race unfold

Position here is not simulated, it is derived: at every moment each car is
ranked by laps completed, then by who got there first. The gold bands are
the caution periods, drawn before the race started.

The gap chart is where an endurance race is actually decided - a stop under
caution closing a minute of deficit shows up as a step, not a gradient."""))

    cells.append(code('''show(viz.plot_race(result, class_name=headline, top_n=6), "race_positions")'''))

    cells.append(code('''show(viz.plot_gaps(result, class_name=headline, top_n=5), "gaps_to_leader")'''))

    cells.append(code('''# The same view as notebook 00's stint-pace plot, but produced by the
# simulator - so the two can be compared directly.
show(viz.plot_stint_pace(result, class_name=headline), "stint_pace")'''))

    cells.append(md("""## Part 4 - twist the levers

This is the point of the whole project. `scale_dials` returns a *copy* of
the dials with one of them multiplied, so a lever never edits the engine
and never edits the original calibration - it just hands the engine a
different race.

Below, the caution rate is swept from a quiet race to a chaotic one, with
everything else held fixed. Watch what it does to the number of stops and
to time lost in the pits: in a caution-heavy race, stopping is cheaper, so
the whole rhythm of the race changes."""))

    cells.append(code('''sweep_rows = []
for mult in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]:
    cfg = scale_dials(configs[series], caution_rate=mult)
    r = run_race(cfg, seed=SEED)
    c = r.classification()
    c = c[c["class"] == headline]
    sweep_rows.append({
        "caution_multiplier": mult,
        "caution_share": r.cautions.total_caution_s() / cfg.duration_s,
        "mean_laps": c["laps"].mean(),
        "mean_stops": c["stops"].mean(),
        "mean_pit_time_s": c["pit_time_s"].mean(),
    })

sweep = pd.DataFrame(sweep_rows)
sweep'''))

    cells.append(code('''show(viz.plot_lever_sweep(sweep, "caution_share", "mean_laps",
                          label="share of race under caution"), "caution_sweep")'''))

    cells.append(md("""Now the other levers, one at a time, each against the same race. The
comparison is fair because the seed is fixed and the caution timeline does
not depend on what anyone does."""))

    cells.append(code('''levers = {
    "baseline":                   {},
    "thirstier (+30% fuel burn)": {"fuel_per_lap": 1.3},
    "harsher tyres (x3 deg)":     {"deg_slope_s_per_lap": 3.0},
    "heavier traffic (x3)":       {"traffic_penalty_s": 3.0},
    "slower pit stops (+50%)":    {"pit_time_mean_s": 1.5},
}

rows = []
for name, twist in levers.items():
    cfg = scale_dials(configs[series], **twist) if twist else configs[series]
    c = run_race(cfg, seed=SEED).classification()
    c = c[c["class"] == headline]
    rows.append({"lever": name,
                 "mean_laps": round(c["laps"].mean(), 1),
                 "mean_stops": round(c["stops"].mean(), 1),
                 "pit_time_s": round(c["pit_time_s"].mean()),
                 "traffic_time_s": round(c["traffic_time_s"].mean())})

lever_table = pd.DataFrame(rows)
lever_table["laps_vs_baseline"] = (lever_table["mean_laps"]
                                   - lever_table.loc[0, "mean_laps"]).round(1)
lever_table'''))

    cells.append(md("""## Part 5 - a first strategy comparison

A taster of 02. Three strategies, each given the whole headline class, each
run on the identical race. The differences are strategy, because the race
was fixed before anyone made a decision.

`OpportunistUnderCaution` should beat `RunToFuelWindow` by taking cheap
stops when a caution is out. Note that *how much* it wins by depends on
`pit_caution_discount`, which is an assumed number - so the ranking is more
trustworthy than the margin, and 02 should sweep it before making any claim
about size."""))

    cells.append(code('''strategy_set = {
    "run to fuel window":  RunToFuelWindow(),
    "caution opportunist": OpportunistUnderCaution(min_fuel_used=0.5),
    "fixed 20-lap stint":  FixedLapStint(stint_laps=20),
}

strategy_results = {
    name: run_race(configs[series],
                   strategies=assign_strategy(configs[series], strat,
                                              class_name=headline),
                   seed=SEED)
    for name, strat in strategy_set.items()
}

fig, comparison = viz.plot_strategy_comparison(strategy_results, headline)
show(fig, "strategy_comparison")
comparison'''))

    cells.append(md("""## Part 6 - does the simulated race resemble the real one?

The engine is only worth anything if a race it produces looks like a race
that happened. Three checks against the source data: laps completed by the
winner, stops made, and share of the race under caution.

These are sanity checks, not a validation gate. A serious gate - the kind
that has to pass before any learning result is admitted - belongs with the
agent work, where being wrong actually costs something."""))

    cells.append(code('''check_rows = []
for series_code, cfg in configs.items():
    anchor = ANCHORS[series_code]
    hc = anchor["headline_class"]

    sid = RACES[series_code]["session_id"]

    # Stops are counted from the pit records. `MAX(stint_number)` counted
    # driver stints - roughly a third of the stops - which is 00's finding.
    real = con.execute(f"""
        SELECT MAX(laps) AS winner_laps, AVG(stops) AS mean_stops
        FROM (
            SELECT car, MAX(lap) AS laps,
                   SUM(CASE WHEN pit_time IS NOT NULL THEN 1 ELSE 0 END) AS stops
            FROM laps
            WHERE session='race' AND series_code='{series_code}'
              AND class='{hc}' AND session_id = {sid}
            GROUP BY car
        ) t
    """).df().iloc[0]

    # Measured on the same single reference car the calibration used. A
    # caution belongs to the race, not to a car, so pooling the whole field
    # here would compare a field-wide average against a per-car rate and
    # make the two look different when they are not.
    ref_car = con.execute(f"""
        SELECT car FROM laps
        WHERE session='race' AND series_code='{series_code}'
          AND session_id = {sid}
        GROUP BY car ORDER BY COUNT(*) DESC LIMIT 1
    """).fetchone()[0]

    # Caution is a named set. `flags <> 'GF'` swept in the chequered lap.
    real_caution = con.execute(f"""
        SELECT AVG(CASE WHEN flags IN ('FCY', 'SF', 'RF') THEN 1.0 ELSE 0.0 END)
                   AS share
        FROM laps
        WHERE session='race' AND series_code='{series_code}'
          AND session_id = {sid} AND car = '{ref_car}'
          AND flags IN ('GF', 'FCY', 'SF', 'RF')
    """).df().iloc[0]["share"]

    sim = results[series_code]
    sim_c = sim.classification()
    sim_c = sim_c[sim_c["class"] == hc]

    check_rows.append({
        "series": series_code, "class": hc,
        "real_winner_laps": int(real["winner_laps"]),
        "sim_winner_laps": int(sim_c["laps"].max()),
        "real_mean_stops": round(float(real["mean_stops"]), 1),
        "sim_mean_stops": round(float(sim_c["stops"].mean()), 1),
        "real_caution_share": round(float(real_caution), 3),
        "sim_caution_share": round(
            sim.cautions.total_caution_s() / cfg.duration_s, 3),
    })

checks = pd.DataFrame(check_rows)
checks["lap_error_pct"] = ((checks["sim_winner_laps"] - checks["real_winner_laps"])
                           / checks["real_winner_laps"] * 100).round(1)
checks'''))

    cells.append(md("""## Where this leaves us

**Built:** a calibrated, tested multi-class race engine in
`src/endurance/`, with the five dials as data, every lever exposed through
`scale_dials`, and a strategy interface that the RL agent will implement
unchanged. The test suite in `tests/` guards the properties that matter,
including the one that makes strategy comparison fair.

**Not built yet, on purpose:**

- *Human-style strategies worth beating* - the three above are starters,
  not a serious set. That is 02.
- *The Gymnasium wrapper and the agent* - the engine deliberately has no
  RL dependency, so 03 can wrap it without touching it.
- *Overtaking.* Cars pass by being faster over a lap. A real pass model
  needs parameters no timing sheet contains, and it would be pretending to
  a precision this project does not have.

**The honest caveats.** The dials come from one race per series, so they
describe Daytona and Le Mans rather than IMSA and WEC in general. The
caution model treats cautions as arriving at random, when in truth they
cluster - at night, in traffic, after restarts. And the assumed parameters
listed in Part 1, `pit_caution_discount` above all, do real work in any
result about caution strategy, which is why the strategy comparison in 02
needs to sweep them rather than quote a single number."""))

    return cells


def build_02a():
    cells = []

    cells.append(md("""# Engine corrections (02a)

Stage 02a made five changes to the engine. One of them - the caution
process - was decided and implemented first, and Parts 1 to 7 of this
notebook are the evidence behind that decision's `Verified:` line. The
other four are the ones the stage blueprint lists, and they arrive from
Part 9 onwards.

**What Parts 1 to 7 check, and what they do not.** They reimplement both
caution processes from their descriptions, in about sixty lines, and check
the statistical claims against that reimplementation. That is deliberately
*not* the same as testing `src/endurance`. Two independent implementations
agreeing is a much stronger statement than one implementation agreeing with
itself, and it is the only way to catch the class of error 02a was written
about - where the code and the thing it was meant to compute had drifted
apart without either looking wrong on its own.

Part 8 closes that loop against the engine. Everything before it runs with
nothing but numpy.

**Parts 9 onwards take the four remaining corrections apart one at a
time**: per-car random streams, a regulation pit layer, field compression
with wave-arounds, and traffic that responds to current pace. Each is
switchable off through `Compat`, which is what makes "here is what the
change did" a claim you can check rather than one you have to accept.

**Findings this stage produced that no decision document contains** are
collected at the end rather than buried: the seed-noise caveat is pinned to
an operating point the corrected engine may no longer sit at; the stated
reason for superseding decision 17 does not survive the check that decision
17's replacement was verified with; the blueprint's claim that the WEC pit
lane stays open under caution is wrong; and the traffic correction, though
necessary, does almost nothing to a field that pits in lockstep."""))

    cells.append(md("""### Setup

Nothing from `src/endurance` until Part 8, so this section runs standalone."""))

    cells.append(code('''import numpy as np
import pandas as pd
from scipy import stats

DURATION_S = 24 * 3600.0          # the 24-hour anchor races
N_SEEDS    = 3000                 # matches the seed count 02a reports

# Roughly eight episodes in a 24-hour race, per 02a's "Consequences".
def mean_dur_for(share, n_episodes=8):
    return share * DURATION_S / n_episodes'''))

    cells.append(md(r"""## Part 1 - the two processes, side by side

The alternating process is the one 02a adopts: exponential green gaps,
exponential episodes, and a caution only able to start when none is running.
The green mean is not a free parameter - for an alternating renewal process
the stationary occupancy is

$$\text{rate} = \frac{\bar{d}}{\bar{d} + \bar{g}}$$

so fixing the rate and the episode mean pins the green mean. That single
line is what makes the realised share come out where it was asked to.

Both functions return drawn lengths alongside the episodes. The distinction
matters: an episode running past the chequered flag is clipped, so measuring
episode length on the timeline censors the longest ones and drags any shape
statistic down for reasons that have nothing to do with the model."""))

    cells.append(code('''def draw_alternating(rng, rate, mean_dur_s, duration_s=DURATION_S,
                     stationary_start=True):
    """02a's process. Non-overlapping by construction."""
    mean_green_s = mean_dur_s * (1.0 - rate) / rate
    episodes, drawn = [], []
    t = 0.0

    if stationary_start and rng.random() < rate:
        # Memorylessness earns this: the residual of an episode already in
        # progress is itself Exp(mean_dur), so opening under caution needs
        # no special case and no separate parameter.
        length = rng.exponential(mean_dur_s)
        drawn.append(length)
        episodes.append((0.0, min(length, duration_s)))
        t = length

    while t < duration_s:
        t += rng.exponential(mean_green_s)
        if t >= duration_s:
            break
        length = rng.exponential(mean_dur_s)
        drawn.append(length)
        episodes.append((t, min(t + length, duration_s)))
        t += length

    return episodes, drawn


def draw_legacy_merged(rng, rate, mean_dur_s, duration_s=DURATION_S):
    """The pre-02a draw: n episodes at uniform starts, overlaps collapsed."""
    n = int(round(rate * duration_s / mean_dur_s))
    starts  = rng.random(n) * duration_s
    lengths = rng.exponential(mean_dur_s, size=n)
    raw = sorted(zip(starts, np.minimum(starts + lengths, duration_s)))

    merged = []
    for s, e in raw:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def draw_rejection(rng, rate, mean_dur_s, duration_s=DURATION_S,
                   max_attempts=100_000):
    """Decision 17: redraw the whole configuration until nothing overlaps."""
    n = int(round(rate * duration_s / mean_dur_s))
    for attempt in range(1, max_attempts + 1):
        starts  = rng.random(n) * duration_s
        lengths = rng.exponential(mean_dur_s, size=n)
        ep = sorted(zip(starts, starts + lengths))
        if ep[-1][1] <= duration_s and all(ep[i][1] <= ep[i + 1][0]
                                           for i in range(n - 1)):
            return ep, attempt
    return None, max_attempts'''))

    cells.append(code('''def share(episodes, duration_s=DURATION_S):
    return sum(e - s for s, e in episodes) / duration_s

def lengths_of(episodes):
    return np.array([e - s for s, e in episodes])

def cv(x):
    x = np.asarray(x, dtype=float)
    return x.std(ddof=1) / x.mean()

def touching(episodes, tol=1e-9):
    return any(episodes[i][1] >= episodes[i + 1][0] - tol
               for i in range(len(episodes) - 1))

def interior(episodes, duration_s=DURATION_S, tol=1e-6):
    """Episodes clipped by neither end of the race - the uncensored ones."""
    return [(s, e) for s, e in episodes if s > tol and e < duration_s - tol]

master = np.random.default_rng(20260804)
seeds  = lambda n: master.integers(2**63, size=n)'''))

    cells.append(md("""## Part 2 - the three claims 02a rests on

Realised share unbiased at 0.05, 0.18 and 0.30; episode-length cv of 1.00;
no two episodes touching.

**A note on how the cv is measured, because the obvious way is wrong.** The
tempting version computes a cv per race and averages. With about eight
episodes a race the cv estimator is biased low by roughly a tenth *whatever
the underlying distribution is*, so that statistic reads about 0.89 for a
perfectly exponential process and cannot tell the processes apart. Pooling
every drawn episode across all seeds and taking one cv removes the bias.
This is worth stating because the merging draw's cv of 1.09 and the
alternating draw's 1.00 are only about a tenth apart, which is exactly the
size of the artefact."""))

    cells.append(code('''rows = []
for target in (0.05, 0.18, 0.30):
    mean_dur = mean_dur_for(0.30)      # one episode scale across all three
    shares, pooled, counts, overlaps = [], [], [], 0

    for s in seeds(N_SEEDS):
        ep, drawn = draw_alternating(np.random.default_rng(s), target, mean_dur)
        shares.append(share(ep))
        pooled.extend(drawn)
        counts.append(len(ep))
        overlaps += touching(ep)

    shares = np.array(shares)
    rows.append({
        "target":       target,
        "realised":     round(shares.mean(), 4),
        "bias":         round(shares.mean() - target, 4),
        "mc_se":        round(shares.std() / np.sqrt(N_SEEDS), 4),
        "sd_seed":      round(shares.std(), 4),
        "cv_drawn":     round(cv(pooled), 3),
        "mean_episodes": round(np.mean(counts), 1),
        "races_with_touching_episodes": overlaps,
    })

pd.DataFrame(rows)'''))

    cells.append(md("""Unbiased at all three rates - every bias is inside its own Monte Carlo
standard error. The cv sits on 1.00 and not one race in nine thousand
produced episodes that touch, which is what "non-overlapping by
construction" should look like when it is true."""))

    cells.append(md(r"""## Part 3 - why the stationary start is not a detail

02a starts the process in its stationary state: with probability
`caution_rate` the race opens under caution. Without it the timeline always
begins green, and the process needs a few hours to forget that it did.

The size of the loss is predictable. The two-state chain relaxes at
$\lambda = 1/\bar{d} + 1/\bar{g}$, so the deficit is about
$\text{rate} \times \lambda^{-1} / T$ - the share the race is missing,
for as long as it takes to forget where it started, over the length of the
race."""))

    cells.append(code('''N_C = 20_000       # this comparison is a fraction of a point, so it needs seeds
rows = []

for target in (0.18, 0.30):
    mean_dur   = mean_dur_for(0.30)
    mean_green = mean_dur * (1 - target) / target
    tau        = 1.0 / (1.0 / mean_dur + 1.0 / mean_green)

    out = {}
    for stationary in (True, False):
        sh = [share(draw_alternating(np.random.default_rng(s), target,
                                     mean_dur, stationary_start=stationary)[0])
              for s in seeds(N_C)]
        out[stationary] = np.mean(sh)

    rows.append({
        "target":            target,
        "stationary_start":  round(out[True], 4),
        "green_start":       round(out[False], 4),
        "deficit":           round(out[True] - out[False], 4),
        "predicted_deficit": round(target * tau / DURATION_S, 4),
    })

pd.DataFrame(rows)'''))

    cells.append(md("""About seven tenths of a point at the higher rate, four at the lower, both
close to the analytic prediction. 02a calls it "about a point light", which
is the right order and, at the rate the engine actually runs, close to
exact."""))

    cells.append(md("""## Part 4 - what the merging draw was doing

Two separate faults, and 02a is right that they compound rather than
cancel.

The coverage loss has a closed form. Uniform starts make the episodes a
Poisson-like coverage process, so the expected union is
$1 - e^{-r}$ against a drawn total of $r$ - a shortfall that grows with the
rate, which is why 02a reports 14% at the old settings and 18% at corrected
ones. The simulated figure comes in slightly below the closed form because
episodes are also clipped at the chequered flag, which the closed form does
not model.

The shape damage is the more serious one for 02b: the union of overlapping
exponentials is not exponential, and merging is precisely the operation that
destroys the memorylessness 02b's benchmark rests on."""))

    cells.append(code('''rows = []
for target in (0.30, 0.407):
    mean_dur = mean_dur_for(0.30)
    shares, pooled = [], []

    for s in seeds(N_SEEDS):
        ep = draw_legacy_merged(np.random.default_rng(s), target, mean_dur)
        shares.append(share(ep))
        pooled.extend(lengths_of(interior(ep)))     # uncensored only

    realised = np.mean(shares)
    rows.append({
        "target":            target,
        "realised":          round(realised, 4),
        "shortfall_pct":     round((realised - target) / target * 100, 1),
        "closed_form_1_minus_exp": round(1 - np.exp(-target), 4),
        "cv_merged":         round(cv(pooled), 3),
    })

pd.DataFrame(rows)'''))

    cells.append(md("""A cv of about 1.06-1.08 against the 1.00 an exponential gives. 02a reports
1.09; the difference is which episodes are counted, and either way the
conclusion is the same one. Note how small the signal is in absolute terms -
this is the artefact that a per-race cv, biased low by a tenth, would have
hidden completely."""))

    cells.append(md(r"""## Part 5 - the units chain, end to end

02a's headline: a calibrated 0.30 caution *lap* share came out of the engine
at 0.19. That number is the product of three steps, and reconstructing it
confirms the diagnosis rather than merely restating it.

A caution lap occupies `caution_pace_multiplier` times the wall clock of a
green one, so a lap share $f$ and a time share $t$ are related by

$$t = \frac{fm}{fm + (1-f)}$$

Step one, a 0.30 lap share is really a time share of roughly 0.39-0.41.
Step two, the engine reads 0.30 as a time target anyway and the merge
delivers $1 - e^{-0.30} = 0.259$ of it. Step three, notebook 01 Part 6
reports the result back as a lap share."""))

    cells.append(code('''lap_to_time = lambda f, m: f * m / (f * m + (1 - f))
time_to_lap = lambda t, m: (t / m) / (t / m + (1 - t))

legacy_time = 1 - np.exp(-0.30)     # 0.30 misread as a time target, then merged

pd.DataFrame([{
    "caution_pace_multiplier": m,
    "true_time_share":         round(lap_to_time(0.30, m), 4),
    "legacy_realised_time":    round(legacy_time, 4),
    "reported_as_lap_share":   round(time_to_lap(legacy_time, m), 4),
} for m in (1.4, 1.5, 1.6, 1.7, 2.0)])'''))

    cells.append(md("""The chain closes at a multiplier between 1.4 and 1.5, which reproduces 02a's
0.19 to two decimal places. It also implies the corrected time share is
around 0.39 rather than 0.30 - worth carrying into Part 8, because
`observed_caution_multiplier` is now returned by `calibrate_cautions` and
can be checked against this directly. If the observed multiplier comes back
near 1.45, this reconstruction is independent confirmation that the fix
addressed the actual fault. If it comes back near 2.0, something in the
chain is still not understood."""))

    cells.append(md("""## Part 6 - decision 17, and where its failure actually shows

02a supersedes decision 17's redraw-until-non-overlapping, on the grounds
that conditioning the whole configuration on non-overlap reweights towards
shorter length vectors, "so lengths stop being marginally exponential".

That reasoning is sound in outline and the conclusion is right. The stated
diagnostic is not. Checking it properly matters, because decision 17 was
adopted to protect memorylessness and it should not be discarded on an
argument that fails the first check anyone runs against it."""))

    cells.append(code('''target, mean_dur = 0.30, mean_dur_for(0.30)

alt = []
for s in seeds(1500):
    alt.extend(draw_alternating(np.random.default_rng(s), target, mean_dur)[1])
alt = np.array(alt)

rej, attempts, rej_shares = [], [], []
for s in seeds(1500):
    ep, n_att = draw_rejection(np.random.default_rng(s), target, mean_dur)
    if ep is None:
        continue
    attempts.append(n_att)
    rej_shares.append(share(ep))
    rej.extend(lengths_of(ep))
rej = np.array(rej)

pd.DataFrame([{
    "process":            name,
    "n_episodes":         len(x),
    "mean_length_s":      round(x.mean()),
    "claimed_mean_s":     round(mean_dur),
    "cv":                 round(cv(x), 3),
    # shape alone: does it look exponential with *its own* mean?
    "KS_p_vs_fitted_exp": round(stats.kstest(x, "expon",
                                             args=(0, x.mean())).pvalue, 3),
    # shape and scale: is it the exponential the model claims to draw?
    "KS_p_vs_claimed_exp": f"{stats.kstest(x, 'expon', args=(0, mean_dur)).pvalue:.2g}",
} for name, x in (("alternating", alt), ("rejection (dec 17)", rej))])'''))

    cells.append(code('''print(f"rejection sampling accepted 1 draw in {np.mean(attempts):.1f}")
print(f"realised share under rejection: {np.mean(rej_shares):.4f} "
      f"against a target of {target:.2f} "
      f"({(np.mean(rej_shares) - target) / target * 100:+.0f}%)")'''))

    cells.append(md("""**The stated diagnostic does not fire.** Conditioned lengths have a cv of
1.00 and pass a Kolmogorov-Smirnov test against an exponential comfortably.
The marginal shape survives the conditioning intact.

**What the conditioning destroys is the scale.** The mean episode falls from
3240s to about 2300s - the distribution is still exponential, just a
different exponential from the one calibration asked for. Tested against the
exponential the model *claims* to draw, rather than one fitted to its own
output, it is rejected at a p-value with 170 zeros in it.

The consequence is worse than the merge it was competing with. Rejection
sampling lands a 0.30 target at 0.214, a 29% shortfall, against the merging
draw's 15%. Decision 17 traded a coverage error for a larger coverage error
while leaving the shape it was protecting untouched.

So: right decision, wrong reason, and the right reason is a stronger
argument. 02a's paragraph on this should be corrected rather than left to be
found - a reader who checks the cv will find 1.00 and conclude the
supersession was unjustified."""))

    cells.append(md("""## Part 7 - the seed-noise caveat, and where it actually sits

02a's consequences note a standard deviation of roughly 0.07 on the realised
share with about eight episodes in a 24-hour race, and warns that decision
11's 50-seed sweep budget will be noisier than the 200-seed headline.

The warning is right and the number is worth pinning down, because the sd is
not a property of the engine - it is a property of the operating point. For
a fixed share, noise falls as episodes get shorter and more numerous, since
the race is averaging over more of them."""))

    cells.append(code('''rows = []
for target in (0.18, 0.30, 0.407):
    for n_ep in (6, 8, 12, 16, 24):
        mean_dur = mean_dur_for(target, n_ep)
        sh = [share(draw_alternating(np.random.default_rng(s), target,
                                     mean_dur)[0])
              for s in seeds(4000)]
        rows.append({"time_share": target, "episodes": n_ep,
                     "mean_episode_s": round(mean_dur),
                     "sd_of_realised_share": round(np.std(sh), 4)})

noise = pd.DataFrame(rows)
noise.pivot(index="episodes", columns="time_share",
            values="sd_of_realised_share")'''))

    cells.append(md("""An sd of 0.07 with eight episodes corresponds to a time share of about 0.18.
But Part 5 puts the corrected Daytona share nearer 0.39, where eight
episodes gives an sd of about 0.12 - not 0.07 but close to double it.

Which of those the engine actually sits at is a Part 8 question, and the two
readings have different consequences for decision 11. At 0.07 a 50-seed
sweep carries a standard error of about 0.010 on the share; at 0.12 it is
0.017, and a sweep curve would need roughly three times the seeds to resolve
the same difference. Worth settling before a caution-sensitive sweep is read
as a trend, which is exactly what 02a warns about."""))

    cells.append(md("""## Part 8 - against the engine

Everything above verifies the *model*. This part verifies that
`src/endurance` implements it, which is the half that would actually catch
a regression.

Two of the checks need the real timing data and the frozen dials notebook
01 writes. Where those are missing the cells say so and carry on, rather
than failing or - worse - quietly substituting something else."""))

    cells.append(code('''import sys
from pathlib import Path


def find_project_root(marker: str = "src/endurance") -> Path:
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"Could not find {marker!r} at or above {here}.")


ROOT = find_project_root()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from endurance import Compat, RaceConfig, run_race          # noqa: E402
from endurance.engine import CautionTimeline                # noqa: E402

# The same locations notebook 01 uses, not a second convention.
PARAMS_DIR = ROOT / "data" / "processed"      # where 01 freezes the dials
DATA = ROOT / "data" / "raw" / "laps.csv"
FROZEN = PARAMS_DIR / "imsa.json"
print("project root:", ROOT)
print("frozen dials:", "found" if FROZEN.exists() else "not written yet")'''))

    cells.append(md("""### 8.1 The parity gate

02a promises that with every correction switched off the engine is the
engine notebook 01 validated, bit for bit. That promise is guarded by a
test rather than by this notebook, and the honest thing here is to run the
test rather than re-derive a weaker version of it.

The reference numbers in that test were taken off the engine *before* 02a
touched it. A test that only compared the new engine against itself would
pass whatever had broken."""))

    cells.append(code('''try:
    from test_compat import (GOLDEN_01, fingerprint,        # noqa: E402
                             make_config as parity_config)
except ModuleNotFoundError as exc:
    # The suite is the canonical home of this check; the notebook only
    # reports it. No point failing the whole notebook over a dev dependency.
    print(f"parity gate needs {exc.name} - run `pytest tests/` for it")
    parity = None
else:
    parity = {seed: fingerprint(run_race(parity_config(), seed=seed,
                                         compat=Compat.v01())) == expected
              for seed, expected in GOLDEN_01.items()}
    print("Compat.v01() reproduces the pre-02a engine on every "
          "reference seed:", all(parity.values()))

parity'''))

    cells.append(md("""### 8.2 Does the engine draw the process Part 2 describes?

Part 2 has already established what the model does, so any disagreement
here is an implementation bug and not a modelling question. This runs
`CautionTimeline.draw` over the same seed count and the same operating
points as Part 2 and puts the two side by side."""))

    cells.append(code('''rows = []
for target in (0.05, 0.18, 0.30):
    mean_dur = mean_dur_for(0.30)          # the same episode scale Part 2 used

    model_shares, model_pooled, model_counts = [], [], []
    eng_shares, eng_pooled, eng_counts = [], [], []

    for s in seeds(N_SEEDS):
        ep, drawn = draw_alternating(np.random.default_rng(s), target, mean_dur)
        model_shares.append(share(ep))
        model_pooled.extend(drawn)
        model_counts.append(len(ep))

        tl = CautionTimeline.draw(DURATION_S, target, mean_dur,
                                  np.random.default_rng(s))
        eng_shares.append(tl.total_caution_s() / DURATION_S)
        eng_pooled.extend(lengths_of(tl.periods))
        eng_counts.append(len(tl.periods))

    rows.append({"target": target,
                 "model_share": round(float(np.mean(model_shares)), 4),
                 "engine_share": round(float(np.mean(eng_shares)), 4),
                 "model_episodes": round(float(np.mean(model_counts)), 1),
                 "engine_episodes": round(float(np.mean(eng_counts)), 1),
                 "model_cv": round(cv(model_pooled), 3),
                 "engine_cv": round(cv(eng_pooled), 3)})

against_engine = pd.DataFrame(rows)
against_engine["share_gap"] = (against_engine["engine_share"]
                               - against_engine["model_share"]).round(4)
against_engine'''))

    cells.append(md("""The two implementations were written from the same description by
different routes, so agreement to within Monte Carlo noise is the check
passing. The engine's episode lengths are clipped by the end of the race
where the model's are not, which pulls its pooled cv slightly below one -
that is censoring, not a different process, and Part 2 excludes the
clipped episodes for the same reason."""))

    cells.append(md("""### 8.3 The observed caution multiplier, against the assumed one

`calibrate_cautions` returns `observed_caution_multiplier` alongside the
calibrated figures. Part 5 predicts it lands near 1.45. Reporting it next
to the assumed `caution_pace_multiplier` rather than substituting it is
the whole point: the assumed value stays in `ASSUMED_FIELDS` and gets
swept, and the observed one says how far the assumption sits from what the
data saw."""))

    cells.append(code('''if not (DATA.exists() and FROZEN.exists()):
    print("no timing data or frozen dials - 8.3 and 8.4 need both, skipping")
else:
    from endurance import calibrate                          # noqa: E402

    con = calibrate.connect(str(DATA))
    for series_code, pattern in (("imsa", "%daytona%"), ("wec", "%le mans%")):
        cfg = RaceConfig.load(PARAMS_DIR / f"{series_code}.json")
        cls = cfg.classes[0]
        # Scoped to the edition the frozen dials came from, per 00's re-run.
        sid = calibrate.find_race(con, series_code, pattern)["session_id"]
        report = calibrate.calibrate_cautions(con, sid, cls.base_pace_s)
        print(f"{series_code}: observed multiplier "
              f"{report['observed_caution_multiplier']:.2f}  "
              f"assumed {cls.caution_pace_multiplier:.2f}  "
              f"share {report['caution_rate']:.3f}  "
              f"episodes {report['n_caution_episodes']}")'''))

    cells.append(md("""### 8.4 Which operating point are we actually at?

Part 7's table is only useful once we know which row of it the calibrated
engine sits on, because that is what decides whether decision 11's sweep
budget is adequate."""))

    cells.append(code('''if not (DATA.exists() and FROZEN.exists()):
    print("no timing data or frozen dials - skipping")
else:
    for series_code in ("imsa", "wec"):
        cfg = RaceConfig.load(PARAMS_DIR / f"{series_code}.json")
        cls = cfg.classes[0]
        n_ep = max(round(cls.caution_rate * cfg.duration_s
                         / cls.caution_mean_dur_s), 1)
        sd = np.std([share(draw_alternating(np.random.default_rng(s),
                                            cls.caution_rate,
                                            cls.caution_mean_dur_s,
                                            cfg.duration_s)[0])
                     for s in seeds(2000)])
        print(f"{series_code}: calibrated share {cls.caution_rate:.3f} over "
              f"about {n_ep} episodes -> sd of realised share {sd:.3f}; "
              f"a 50-seed sweep carries se {sd / np.sqrt(50):.4f}")'''))

    cells.append(md("""## Part 9 - the four corrections, one at a time

Everything from here needs a race, so it needs dials. The frozen dials
notebook 01 writes are used where they exist; where they do not, a
stand-in field is built and labelled as one, so the notebook executes
either way and nobody mistakes the stand-in for calibration.

`Compat` is the switchboard. Each correction is a flag, and `Compat.v01()`
turns all of them off at once - that combination is the parity gate 8.1
just checked."""))

    cells.append(code('''from endurance import ClassDials                             # noqa: E402

SEED_BANK = range(20)


def race_config():
    """The calibrated IMSA dials if 01 has written them, else a stand-in."""
    frozen = PARAMS_DIR / "imsa.json"
    if frozen.exists():
        return RaceConfig.load(frozen), True

    gtp = ClassDials(
        series_code="imsa", class_name="GTP", base_pace_s=97.5,
        deg_slope_s_per_lap=0.012, pace_spread_s=0.5, lap_noise_s=0.5,
        caution_rate=0.20, caution_mean_dur_s=600.0, green_stint_laps=30.0,
        fuel_per_lap=1 / 30, fuel_per_lap_caution=0.6 / 30, tyre_life_laps=60.0,
        pit_time_mean_s=47.0, pit_time_std_s=2.0, n_cars=8)
    gtd = ClassDials(
        series_code="imsa", class_name="GTD", base_pace_s=112.0,
        deg_slope_s_per_lap=0.020, pace_spread_s=0.9, lap_noise_s=0.6,
        caution_rate=0.20, caution_mean_dur_s=600.0, green_stint_laps=28.0,
        fuel_per_lap=1 / 28, fuel_per_lap_caution=0.6 / 28, tyre_life_laps=56.0,
        pit_time_mean_s=45.0, pit_time_std_s=2.0, n_cars=10)
    return RaceConfig(name="stand-in", series_code="imsa",
                      duration_s=24 * 3600.0, classes=[gtp, gtd]), False


CFG, CALIBRATED = race_config()
HEADLINE = CFG.classes[0].class_name
if not CALIBRATED:
    print("NOT CALIBRATED - stand-in field. Run notebook 01 first for "
          "numbers that mean anything about Daytona.")
print(f"{CFG.name}: {CFG.total_cars} cars, {CFG.duration_s / 3600:.0f}h, "
      f"headline class {HEADLINE}")'''))

    cells.append(code('''STEPS = [
    ("01 engine",         Compat.v01()),
    ("+ per-car streams", Compat(legacy_pit=True, legacy_caution_pace=True,
                                 legacy_traffic=True)),
    ("+ pit layer",       Compat(legacy_caution_pace=True, legacy_traffic=True)),
    ("+ compression",     Compat(legacy_traffic=True)),
    ("+ traffic",         Compat()),
]

rows = []
for label, compat in STEPS:
    winners, spreads = [], []
    for seed in SEED_BANK:
        c = run_race(CFG, seed=seed, compat=compat).classification()
        c = c[c["class"] == HEADLINE]
        winners.append(c["laps"].max())
        spreads.append(c["laps"].max() - c["laps"].min())
    rows.append({"engine": label,
                 "winner_laps": round(float(np.mean(winners)), 1),
                 "sd_across_seeds": round(float(np.std(winners)), 1),
                 "class_spread_laps": round(float(np.mean(spreads)), 1)})

corrections = pd.DataFrame(rows)
corrections'''))

    cells.append(md("""Read down the last column rather than across the first. Compression is
doing nearly all of the work: it takes the headline class from finishing
strung out over several laps to finishing covered by about one, which is
where real endurance classifications sit and where notebook 01's did not.

The per-car streams cost the winner laps *and* widen the seed spread. That
is not a regression, it is the paired-comparison defect being paid off -
under 01 a change of strategy silently reshuffled every other car's noise,
and the stability that bought was not information."""))

    cells.append(md("""## Part 10 - what each correction actually did

The table above is the summary. These are the mechanisms, one at a time,
because a summary nobody can take apart is a summary nobody should
believe."""))

    cells.append(md("""### 10.1 Noise is a function of the seed, not of the strategy

Degradation, traffic and cautions are switched off here, so a green lap
time is exactly the car's pace plus that lap's noise. Two strategies that
pit at completely different moments must then produce identical lap times
at identical lap numbers. Under 01's shared generator they do not, and the
difference is of the same order as the strategic effects 02 sets out to
measure."""))

    cells.append(code('''from endurance import (FixedLapStint, RunToFuelWindow,       # noqa: E402
                       scale_dials)

quiet = scale_dials(CFG, deg_slope_s_per_lap=0.0, traffic_penalty_s=0.0,
                    caution_rate=0.0)


def lap_times_by_lap(result, car_id):
    sub = result.laps[result.laps["car_id"] == car_id]
    return dict(zip(sub["lap"], sub["lap_time"]))


rows = []
for label, compat in (("01 engine", Compat.v01()), ("02a engine", Compat())):
    a = run_race(quiet, default_strategy=RunToFuelWindow(), seed=11,
                 compat=compat)
    b = run_race(quiet, default_strategy=FixedLapStint(stint_laps=7), seed=11,
                 compat=compat)
    car = a.laps["car_id"].iloc[0]
    la, lb = lap_times_by_lap(a, car), lap_times_by_lap(b, car)
    shared = sorted(set(la) & set(lb))
    diffs = np.array([abs(la[k] - lb[k]) for k in shared])
    rows.append({"engine": label, "laps_compared": len(shared),
                 "max_lap_time_difference_s": round(float(diffs.max()), 6),
                 "total_difference_s": round(float(diffs.sum()), 1)})

pd.DataFrame(rows)'''))

    cells.append(md("""### 10.2 A stop has a shape, and the two series shape it differently

The pit layer is anchored to the measured mean: a full tank plus tyres
costs exactly `pit_time_mean_s`, in both series, by construction. The
level is data and stays data. Everything the layer says is about stops
that are *not* full service - the only place a rulebook can tell you
something lap timing cannot.

IMSA allows four over the wall including the refueller, air jack and tyre
changes (art. 34.1.1), so the jobs overlap and a stop costs the longer of
them. WEC forbids tools during the refuelling phase (art. 12), so they
queue and a stop costs their sum."""))

    cells.append(code('''from endurance import PitRules, stop_cost                    # noqa: E402

dials = CFG.classes[0]
rows = []
for series_code in ("imsa", "wec"):
    rules = PitRules.for_series(series_code)
    for label, fuel, tyres in (("full tank + tyres", 1.0, True),
                               ("half tank + tyres", 0.5, True),
                               ("splash, no tyres", 0.3, False),
                               ("tyres only", 0.0, True)):
        rows.append({"series": series_code, "stop": label,
                     "cost_s": round(stop_cost(dials, rules, fuel, tyres), 1)})

shapes = pd.DataFrame(rows).pivot(index="stop", columns="series",
                                  values="cost_s")
shapes["01 engine"] = round(dials.pit_time_mean_s, 1)
shapes.reindex(["full tank + tyres", "half tank + tyres",
                "splash, no tyres", "tyres only"])'''))

    cells.append(md("""Every one of those cost `pit_time_mean_s` under notebook 01. The
splash-and-dash planner in 02's roster depends entirely on the difference
between these rows, which is why decision 4/7 promoted this layer from an
improvement to a prerequisite.

One consequence worth stating rather than discovering later: the baselines
never arrive with an empty tank, so turning the layer on makes even their
ordinary stops a little cheaper than 01 priced them. The saving is the
fuel that was never put in."""))

    cells.append(md("""### 10.3 The pit lane closes, and reopens by rulebook

**A correction to the blueprint.** Section 7C states that the pit lane
closes under FCY in IMSA and stays open in WEC, and calls that a rulebook
fact rather than an assumption. The WEC half is wrong. Article 14.5.2
closes the pit entry when FCY is announced, leaving the exit open, and
article 14.6.5 closes the entry for the first three laps of a safety car,
two if it follows an FCY.

The real difference is not open against closed - it is *staged* against
*unstaged* reopening, plus IMSA's Short FCY, which never opens at all.
IMSA releases GTP and LMP2 on the first lap after the pits are declared
open and the GT classes on the next (art. 46.3.1); WEC releases everyone
together. That is still most of the reason to simulate both series, and it
means the caution gambler in 02's roster carries real risk in both."""))

    cells.append(code('''from endurance import lane_status                            # noqa: E402


class _Episode:
    """A caution timeline with periods chosen rather than drawn."""

    def __init__(self, periods):
        self.periods = periods


CAUTION_LAP_S = min(c.base_pace_s * c.caution_pace_multiplier
                    for c in CFG.classes)
episode = _Episode([(3600.0, 3600.0 + 8 * CAUTION_LAP_S)])

rows = []
for laps_in in (0.5, 1.5, 2.5, 3.5, 4.5):
    t = 3600.0 + laps_in * CAUTION_LAP_S
    row = {"caution_laps_elapsed": laps_in}
    for series_code, cls_name in (("imsa", "GTP"), ("imsa", "GTD"),
                                  ("wec", "HYPERCAR")):
        st = lane_status(PitRules.for_series(series_code), episode, t, cls_name,
                         caution_lap_s=CAUTION_LAP_S,
                         duration_s=CFG.duration_s)
        row[f"{series_code} {cls_name}"] = "open" if st.open else "shut"
    rows.append(row)

pd.DataFrame(rows)'''))

    cells.append(md("""### 10.4 Compression, and the lap a wave-around hands back

Behind a safety car everybody runs the safety car's lap, and a car with a
gap closes a share of it each lap until the field is queued up. Both are
applied as adjustments to *lap times*, never to the running order: 01's
central claim is that position is derived from accumulated race time and
never simulated, and compression is exactly the machinery that would break
it. A test rebuilds every car's finishing time from its own laps and
demands the classification agree.

Wave-arounds use the same eligibility rule in both books - a car whose
class leader is behind it in the queue (IMSA art. 46.2.2 and 46.4.1, WEC
art. 14.6.4) - which selects lapped cars on its own. IMSA runs it twice,
WEC once.

**One artefact to know about.** A wave-by is a timing-system credit, not
physics: the car is given a lap it did not drive. Expressed as a lap time
that is one very short caution lap, flagged `wave_by` in the lap record so
that nothing reading caution laps as pace picks it up. It is the least
physical thing in the engine."""))

    cells.append(code('''rows = []
for label, compat in (("01 engine", Compat(legacy_caution_pace=True)),
                      ("02a engine", Compat())):
    result = run_race(CFG, seed=4, compat=compat)
    laps = result.laps
    caution = laps[laps["under_caution"] & ~laps["wave_by"]]
    by_class = caution.groupby("class")["lap_time"].mean()
    rows.append({
        "engine": label,
        "safety_car_lap_s": round(CAUTION_LAP_S, 1),
        **{f"{name}_caution_lap_s": round(float(v), 1)
           for name, v in by_class.items()},
        "wave_arounds": int(laps["wave_by"].sum()),
    })

pd.DataFrame(rows)'''))

    cells.append(md("""Under 01 a GTD caution lap was 1.6 times a *GTD* lap, so the class that
was slow under green stayed slow under yellow, the field kept its shape,
and a caution cost nothing positionally and gained nothing. That left the
assumed `pit_caution_discount` carrying the entire caution story on its
own. It carries a good deal less of it now."""))

    cells.append(md("""### 10.5 Traffic that a stop can do something about

Under 01 a car counted as an obstruction if its *base* pace was slower
than yours, so the same cars blocked you on lap two and lap two hundred
and no stop could change it. "I will come out into traffic if I stop now"
was not merely unimplemented, it was unrepresentable.

It is representable now, and the honest report is that it does very
little. A field all running one plan pits in lockstep, so every car
carries the same tyre age, degradation cancels out of the comparison, and
the correction changes nothing whatsoever. Once the stints are staggered
it starts to bite - and even then it is worth tenths against class gaps
worth seconds."""))

    cells.append(code('''car_ids = [f"{cls.class_name}-{j + 1:02d}"
           for cls in CFG.classes for j in range(cls.n_cars)]
plans = {cid: (FixedLapStint(stint_laps=10 + 3 * (i % 4)) if i % 2
               else RunToFuelWindow())
         for i, cid in enumerate(car_ids)}

rows = []
for label, strategies in (("one plan for everyone", None),
                          ("stints staggered", plans)):
    counts = {}
    for engine, compat in (("01 engine", Compat(legacy_traffic=True)),
                           ("02a engine", Compat())):
        r = run_race(CFG, strategies, seed=3, compat=compat)
        counts[engine] = int(r.laps["blockers"].sum())
    rows.append({"field": label, **counts,
                 "difference": counts["02a engine"] - counts["01 engine"]})

pd.DataFrame(rows)'''))

    cells.append(md("""## Part 11 - notebook 01's Part 6, and the units error in it

02a flags that 01's validation table compares a simulated caution *time*
share against a real caution *lap* share. Both sides should be time
shares. The correction is reported before as well as after, because Part 6
is one of the few places the units error was visible at all.

Notebook 01 carries the corrected table; this is the working that
justifies changing it, and part of the reason 01's numbers moved."""))

    cells.append(code('''if not (DATA.exists() and FROZEN.exists()):
    print("no timing data or frozen dials - this check needs both, skipping")
else:
    from endurance import calibrate                          # noqa: E402

    con = calibrate.connect(str(DATA))
    rows = []
    for series_code, pattern in (("imsa", "%daytona%"), ("wec", "%le mans%")):
        sid = calibrate.find_race(con, series_code, pattern)["session_id"]
        real = con.execute(
            "SELECT "
            "SUM(CASE WHEN flags IN ('FCY','SF','RF') THEN 1 ELSE 0 END) "
            "* 1.0 / COUNT(*), "
            "SUM(CASE WHEN flags IN ('FCY','SF','RF') THEN lap_time ELSE 0 END) "
            "/ SUM(lap_time) "
            "FROM laps "
            f"WHERE series_code = '{series_code}' "
            f"AND session_id = {sid} "
            "AND session = 'race' AND pit_time IS NULL "
            "AND flags IN ('GF','FCY','SF','RF')").fetchone()
        cfg = RaceConfig.load(PARAMS_DIR / f"{series_code}.json")
        sim = run_race(cfg, seed=0)
        rows.append({
            "series": series_code,
            "real_lap_share_what_01_compared": round(float(real[0]), 3),
            "real_time_share_what_it_should_be": round(float(real[1]), 3),
            "sim_time_share": round(
                sim.cautions.total_caution_s() / cfg.duration_s, 3)})
    units = pd.DataFrame(rows)
    print(units.to_string(index=False))'''))

    cells.append(md("""## Where this leaves us

**The stage gate.** All five conditions hold. `Compat.v01()` reproduces
the pre-02a engine bit for bit against reference numbers taken off it
before any of this was written; per-car lap noise is identical under two
strategies on one seed; compression changes no car's position except
through lap times; the caution-timeline independence test still passes;
and the suite is green, having grown from 42 tests to 92 with none of the
original 42 removed.

**Findings this stage produced that no decision document contains.**

1. *The blueprint is wrong about the WEC pit lane.* Section 7C calls it a
   rulebook fact that the lane stays open in WEC. It closes (arts. 14.5.2
   and 14.6.5). The real asymmetry is staged against unstaged reopening.
   That needs correcting in the blueprint rather than working around here,
   because 02b's benchmark must exclude closed windows in *both* series.

2. *Compression is doing nearly all the work.* Of the four corrections it
   is the one that moves the classification, and it moves it a long way -
   from a headline class strung out over several laps to one covered by
   about a lap. That is the direction real classifications sit in. Whether
   the magnitude is right is a validation question needing the real event
   rather than an argument, and it is the first thing 02b should settle
   before trusting a benchmark built on this engine.

3. *The traffic correction is necessary and nearly inert.* It fixes a real
   defect - stop timing could not affect traffic at all - but on a field
   running one strategy it changes nothing by construction, and on a
   staggered field it is worth tenths. Any 02 comparison run against a
   single-strategy background field will see none of it. By decision 14
   that is a finding rather than a bug, and it belongs in the write-up.

4. *A wave-around is bookkeeping, not physics.* It is implemented as a lap
   time so that position stays derived, and that lap time is not a lap
   anybody drives. It is flagged in the record; anything reading caution
   lap times as pace must exclude it.

5. The findings from Parts 1 to 7 stand: the seed-noise caveat is pinned
   to an operating point 8.4 locates, decision 17's stated failure does
   not reproduce, and its real failure is worse than the stated one.

**What 02a deliberately did not do.** No strategies and no benchmark -
that is the stage boundary. `pit_caution_discount` survives as an assumed
dial even though compression now does part of its job, because retiring it
is a calibration decision and this was not a calibration stage. The
assumed dials grew from five to ten, and every new one is in
`ASSUMED_FIELDS` and swept rather than trusted.

**Next.** 02b, the per-race benchmark - with the closed-window constraint
in both series, and with compression widening the gap between time-optimal
and position-optimal plans, which decision 5 already warns means *k* needs
to be generous."""))

    return cells


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

DIALS = Path.cwd().parent / "data" / "processed" / "imsa.json"

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

def build_02c():
    cells = []

    # ------------------------------------------------------------------
    cells.append(md("""# Human strategies and the comparison (02c)

02b built the reference: for one race, the best a plan could have done.
This stage builds the thing that reference is *for* - a roster of
parameter-free, human-style strategies, and a harness that scores them
against each other and against the null on identical races.

**Why this is not a leaderboard.** The point is not to find the best of the
five. It is to establish what a plausible human strategy is worth, on races
03's agent will be scored on, so that the agent's number has something to
mean. A roster that all did the same thing would be one baseline printed
five times; a roster tuned until it looked good would be a second oracle in
disguise.

**The boundary constraint.** No per-race tuning of any strategy parameter.
Each strategy derives its numbers from the dials rather than being handed
them, and none of them may read 02b's measured `anchor` - that is a
per-seed level taken off a reference run, and consulting it is per-race
tuning whatever it is called. There is a test asserting each strategy moves
when the dial it claims to read moves.

**The verification gate: the null is exactly the null.** With the focal car
on `RunToFuelWindow` and the field on the frozen background, the harness
must reproduce that car's classification row bit for bit against a bare
`run_race` call. Against a bare call, not against the harness's own null
arm - a gate that compares the code under test with itself passes whatever
has broken. It runs in Part 4 before any number is read, and its row sits
in the middle of every figure afterwards.

**Two tables, never one.** Decision 10's budget is per strategy *per
series*, and the rulebooks differ in ways that make a lever live in one and
dead in the other. Nothing here is pooled, and `viz` raises rather than
drawing a pooled bar.

**What this stage found** is collected at the end rather than buried. Three
of the five headline results turn on a rulebook difference or on an engine
correction rather than on the seeds."""))

    # ------------------------------------------------------------------
    cells.append(md("""### Setup

The same project-root walk the other notebooks use, so this works whether
Jupyter was launched from `notebooks/`, from the project root, or from an
editor with its own idea of the working folder."""))

    cells.append(code('''import sys
from pathlib import Path


def find_project_root(marker: str = "src/endurance") -> Path:
    """Walk up from the working folder until the project appears."""
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / marker).exists():
            return candidate
    raise RuntimeError(f"could not find {marker} above {here}")


ROOT = find_project_root()
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from endurance import RaceConfig, run_race, scale_dials
from endurance.assets import draw_seed_bank, freeze_background
from endurance import harness, viz
from endurance.strategies import BASELINES, ROSTER

pd.set_option("display.width", 120)
print(f"project root: {ROOT}")''' ))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 1 - The race, the dials, and what is assumed

Two things are settled here before anything is measured: which race these
numbers are about, and whether the engine is the engine 01 validated.

**Which dials.** If the frozen configs are on disk they are used. If they
are not, a six-hour stand-in matching the one 02b worked against is built
instead, so that the two stages are commensurable. Which of the two happened
is printed, loudly, because **every figure below is provisional until the
frozen dials' provenance is settled** - that is the project's oldest
outstanding claim and nothing in this stage can close it."""))

    cells.append(code('''# Shrink these for a quick pass. They are printed with every result, so a
# shrunken run cannot be quietly mistaken for the headline one.
N_HEADLINE = 200      # decision 10's paired budget, per strategy per series
N_SWEEP = 50          # the first fifty of the headline bank, per decision B
N_GATE = 12           # seeds the verification gate runs on

PROCESSED = ROOT / "data" / "processed"


def stand_in(series_code: str) -> RaceConfig:
    """The six-hour config 02b worked against, rebuilt rather than loaded.

    Deliberately not a guess at a real race. It exists so this notebook runs
    and so 02b and 02c are talking about the same shape of race; it is not
    a claim about Daytona or Le Mans.
    """
    from endurance import ClassDials

    quick = ClassDials(
        series_code=series_code, class_name="GTP" if series_code == "imsa"
        else "HYPERCAR",
        base_pace_s=97.5, deg_slope_s_per_lap=0.012, pace_spread_s=0.5,
        lap_noise_s=0.5, caution_rate=0.20, caution_mean_dur_s=600.0,
        green_stint_laps=30.0, fuel_per_lap=1 / 30,
        fuel_per_lap_caution=0.6 / 30, tyre_life_laps=60.0,
        pit_time_mean_s=47.0, pit_time_std_s=2.0, n_cars=8)
    gt = ClassDials(
        series_code=series_code, class_name="GTD" if series_code == "imsa"
        else "LMGT3",
        base_pace_s=112.0, deg_slope_s_per_lap=0.02, pace_spread_s=0.9,
        lap_noise_s=0.6, caution_rate=0.20, caution_mean_dur_s=600.0,
        green_stint_laps=28.0, fuel_per_lap=1 / 28,
        fuel_per_lap_caution=0.6 / 28, tyre_life_laps=56.0,
        pit_time_mean_s=45.0, pit_time_std_s=2.0, n_cars=10)
    return RaceConfig(name=f"{series_code} 6h stand-in",
                      series_code=series_code, duration_s=6 * 3600.0,
                      classes=[quick, gt])


configs, source = {}, {}
for series_code in ("imsa", "wec"):
    path = PROCESSED / f"{series_code}.json"
    if path.exists():
        configs[series_code] = RaceConfig.load(path)
        source[series_code] = str(path.relative_to(ROOT))
    else:
        configs[series_code] = stand_in(series_code)
        source[series_code] = "STAND-IN (frozen dials not on disk)"

for series_code, cfg in configs.items():
    print(f"{series_code}: {cfg.name}, {cfg.duration_s / 3600:.1f} h, "
          f"{cfg.total_cars} cars, headline class "
          f"{harness.headline_class(cfg)}")
    print(f"    dials from {source[series_code]}")

if any("STAND-IN" in s for s in source.values()):
    print("\\n*** Every number in this notebook is against a stand-in. ***")'''))

    cells.append(md("""### The pit layer is on, and 01 is still reproducible

02a's changes all switch off, and with every one of them off the engine has
to be bit for bit the engine 01 validated. That regression gate is what
licenses saying "here is what the change did" at all, so it is shown passing
here rather than assumed - and the pit layer being on is what makes the
splash-and-dash planner implementable in the first place."""))

    cells.append(code('''from endurance import Compat
from endurance.params import ASSUMED_FIELDS

cfg = configs["imsa"]
a = run_race(cfg, seed=3, compat=Compat.v01()).classification()
b = run_race(cfg, seed=3, compat=Compat.v01()).classification()
print("legacy mode reproduces itself:", a.equals(b))

live = run_race(cfg, seed=3).classification()
print("and the corrected engine differs from it:", not live.equals(a))

print(f"\\n{len(ASSUMED_FIELDS)} assumed dials, swept rather than trusted:")
for name in ASSUMED_FIELDS:
    print(f"    {name}")'''))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 2 - The roster

Five strategies, each one sentence and one import. They are parameter-free:
every number comes from the dials, and none of them takes a constructor
argument - a strategy that can be tuned has somewhere to put the tuning.

1. **Fuel-window baseline.** Runs the tank dry every stint. The null every
   paired delta is measured from, and the strategy the background field
   runs, so the field a strategy is measured against and the plan it is
   measured against are the same idea.
2. **Caution gambler.** Takes a caution stop only when it is *free in
   stops* - when the fuel still aboard would not push the remaining
   requirement over a tank boundary. A count, not a price, which is why it
   needs nothing 02b measured. It checks the lane itself, because under
   IMSA's Short FCY the lane never opens and a strategy that cannot see
   that is not gambling, it is being lucky.
3. **Track-position defender.** Refuses a voluntary stop that would let a
   rival past - one behind now that would be ahead once the stop is served
   - accepting a worse fuel window.
4. **Splash-and-dash planner.** Works backwards from the flag: the last
   stop takes on the fuel the race still needs and skips tyres if their
   remaining life covers what is left.
5. **Lap-down defender.** Declines a voluntary stop that would concede a
   whole lap to the class leader, and once lapped under caution declines
   any voluntary stop, because every stop costs wave-around eligibility.

**Two mappings, kept apart in code.** `BASELINES` is what the frozen
background field may be given, and its other two members carry chosen
constants. `ROSTER` is the five above. They are separate because
`freeze_background` looks names up in `BASELINES`, so a roster strategy that
drifted into that mapping would silently join the field it is measured
against.

**A defender's discretion is one lap wide.** The fuel window opens at a lap
and a half in hand and the engine forces a stop below one lap, so declining
buys exactly one more lap and then the rules take the decision back. Neither
defender should be read as having a larger lever than that."""))

    cells.append(code('''print("BASELINES - what the background field may run:")
for name, cls in BASELINES.items():
    tunable = list(getattr(cls, "__dataclass_fields__", {}))
    print(f"    {name:<22} {cls.__name__:<24} "
          f"{'tunable: ' + str(tunable) if tunable else 'parameter-free'}")

print("\\nROSTER - decision 9's five:")
for name, cls in ROSTER.items():
    assert not getattr(cls, "__dataclass_fields__", {}), name
    cls()                                    # constructs with no arguments
    print(f"    {name:<22} {cls.__name__}")

print(f"\\nshared by both: {sorted(set(BASELINES) & set(ROSTER))}")'''))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 3 - One race, five strategies

Before any distribution, the legible single case. Same seed, same frozen
background field, same focal car - only the strategy in that seat changes.

**The focal car is chosen by drawn pace rank, not by car id.** Decision 8
asks for "P5 of the headline class", and `_build_field` starts every car at
*t* = 0 on lap 1 with no grid, so pace rank is the only referent that
exists. It also holds the focal car's competitive position constant across
the bank, which a fixed id would not - and, because the pace draw is
independent of strategy, it picks the same car in both arms of every
pair."""))

    cells.append(code('''SEED = 4
series_code = "imsa"
cfg = configs[series_code]
field = freeze_background(cfg)
focal = harness.focal_car(cfg, SEED)

print(f"{cfg.name}, seed {SEED}: focal car is {focal} "
      f"(pace rank {harness.DEFAULT_PACE_RANK} of "
      f"{cfg.class_by_name(harness.headline_class(cfg)).n_cars})\\n")

single = pd.DataFrame([
    {"strategy": name,
     **{k: v for k, v in harness.run_focal(cfg, SEED, focal, cls(), field).items()
        if k not in ("seed", "focal")}}
    for name, cls in ROSTER.items()])
single'''))

    cells.append(md("""One race is one race. The strategy that wins here won it on this caution
timeline, and the whole reason the next part exists is that a single
comparison cannot tell that apart from a strategy that is actually better.
02b found the same thing from the other direction: at six hours the flag
boundary is a knife edge no amount of model refinement resolves."""))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 4 - The paired comparison

### The gate first

Nothing below means anything until this passes. The harness must reproduce
the focal car's row bit for bit against a bare `run_race` call, on both
series."""))

    cells.append(code('''for series_code, cfg in configs.items():
    rows = harness.null_is_the_null(cfg, draw_seed_bank(cfg).headline[:N_GATE],
                                    freeze_background(cfg))
    print(f"{series_code}: gate PASSED on {len(rows)} seeds, "
          f"focal cars {sorted(rows['focal'].unique())}")'''))

    cells.append(md("""### The banks

Three per series, drawn once and written down rather than derived. The sweep
fifty are the **first fifty of the headline two hundred** - a sweep asks how
a claim moves as a dial moves, and the cleanest version compares against the
same races the claim was made on. The held-out fifty are genuinely disjoint,
for the opposite reason.

The `draw_seed` differs by series. Sharing one would hand both banks
identical integer lists against different dials: harmless in itself, but two
tables of the same seed numbers invite a reader to take them as paired."""))

    cells.append(code('''DRAW_SEEDS = {"imsa": 20260806, "wec": 20260807}

banks, fields = {}, {}
for series_code, cfg in configs.items():
    banks[series_code] = draw_seed_bank(cfg, draw_seed=DRAW_SEEDS[series_code])
    fields[series_code] = freeze_background(cfg)
    bank = banks[series_code]
    print(f"{series_code}: {len(bank.headline)} headline, {len(bank.sweep)} sweep "
          f"(prefix: {bank.sweep == bank.headline[:len(bank.sweep)]}), "
          f"{len(bank.held_out)} held out "
          f"(disjoint: {not set(bank.held_out) & set(bank.headline)})")
    print(f"    dials fingerprint {bank.provenance['dials_fingerprint']}, "
          f"background {fields[series_code].provenance['uniform_strategy']}")'''))

    cells.append(md("""### The headline table

One row per strategy per series, reported as **shares and quantiles, not a
mean**. "Gains a place in 40% of races, loses one in 12%" is both a stronger
claim than a mean of 4.7 and a legible one, and it survives the long tails a
position delta has when a strategy occasionally throws a race away.

The `fuel_window` row is identically zero by construction. It is the
verification gate sitting inside the results table, where a reader can check
it without running anything."""))

    cells.append(code('''comparisons, headline = {}, []
for series_code, cfg in configs.items():
    comparison = harness.compare_roster(
        cfg, banks[series_code].headline[:N_HEADLINE], fields[series_code])
    comparisons[series_code] = comparison
    headline.append(comparison.summarise())

headline = pd.concat(headline, ignore_index=True)
print(f"{N_HEADLINE} paired races per strategy per series\\n")
headline.round(3)'''))

    cells.append(code('''for series_code, comparison in comparisons.items():
    fig = viz.plot_paired_deltas(comparison.rows)
    plt.show()'''))

    cells.append(md("""### The gap to 02b's benchmark

02b's per-seed plans cost 50 to 130 seconds each, so they are produced by a
script and read from a cache rather than computed here - the notebook is the
argument, not the run. Where the cache is absent this draws a labelled
placeholder and everything else still works.

Read it with 02b's third finding in mind. That stage found its top twenty
plans spanning six seconds of predicted time against thirty seconds of model
error, and concluded that a ranking whose spread is smaller than its error
is noise. If the strategies' medians sit inside one another's ranges here,
the honest reading is that the roster is *not ordered*, rather than that it
is ordered narrowly."""))

    cells.append(code('''import json

BENCH = ROOT / "data" / "processed" / "benchmark_cache.json"
cache = json.loads(BENCH.read_text()) if BENCH.exists() else None
if cache is not None:
    cache = {int(k): v for k, v in cache.items()}
    print(f"benchmark cache: {len(cache)} races from {BENCH.name}")
else:
    print("no benchmark cache on disk - the gap figure will say so")

for series_code, comparison in comparisons.items():
    joined = harness.attach_benchmark(comparison.rows, cache)
    fig = viz.plot_benchmark_gap(joined)
    plt.show()'''))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 5 - Does it matter where you start?

Decision 8's rotation, read as pace rank rather than starting slot. The
question is whether a strategy's value is a property of the strategy or of
the competitive position it happens to be run from - a strategy that only
works from the front of the class is a different claim from one that works
anywhere.

Reported as one figure across the rotation rather than a table per rank."""))

    cells.append(code('''rotation = []
for series_code, cfg in configs.items():
    out = harness.rotate_pace_rank(
        cfg, banks[series_code].sweep[:N_SWEEP], fields[series_code])
    rotation.append(out)
rotation = pd.concat(rotation, ignore_index=True)

spread = (rotation.groupby(["series", "strategy"])["gained"]
                  .agg(["min", "max"])
                  .assign(range=lambda d: d["max"] - d["min"])
                  .round(3))
print("share of races gained, across every pace rank in the class:\\n")
spread'''))

    cells.append(md("""A wide range in that last column is the interesting outcome, not a
disappointing one: it says the strategy's value depends on where in the
class you are running it, which is a claim about strategy rather than noise
in the harness."""))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 6 - Sweeps, and which claims survive them

Decision 11: every assumed dial gets swept one at a time, and
`pit_caution_discount` against the caution rate gets a 2-D grid because the
two interact by construction - how often a caution arrives and how much a
caution stop saves.

**Each claim is labelled invariant or dependent.** A result that holds
across the sweep is a result; one that does not is a result *about the
assumption*, which is worth as much provided it is labelled. The sweeps run
on the sweep fifty, which are the first fifty of the headline bank, so the
comparison is against the same races the claim was made on.

The sweep is drawn on the share gained rather than the median delta. On
fifty seeds a median position delta moves in whole places and steps visibly
when nothing has happened, which would read as dependence where there is
none."""))

    cells.append(code('''SWEEPS = {
    "pit_caution_discount": (0.5, 1.0, 1.5, 2.0),
    "caution_rate": (0.5, 1.0, 2.0, 3.0),
    "traffic_penalty_s": (0.0, 1.0, 2.0),
    "caution_close_frac": (0.5, 1.0, 1.5),
}

sweeps = {}
for series_code, cfg in configs.items():
    for dial, multipliers in SWEEPS.items():
        sweeps[(series_code, dial)] = harness.sweep_dial(
            cfg, banks[series_code].sweep[:N_SWEEP], fields[series_code],
            dial, multipliers)
    print(f"{series_code}: {len(SWEEPS)} dials swept")'''))

    cells.append(code('''for (series_code, dial), frame in sweeps.items():
    if series_code != "imsa":
        continue
    fig = viz.plot_sweep_response(frame)
    plt.show()'''))

    cells.append(code('''def label(frame, statistic="gained", tolerance=0.10):
    """Invariant or dependent, on the range the statistic moves across.

    `tolerance` is a reporting threshold rather than a strategy parameter -
    it decides what gets called dependent in a table, and changes no
    decision any car makes.
    """
    out = (frame.groupby(["series", "dial", "strategy"])[statistic]
                .agg(["min", "max"])
                .assign(range=lambda d: d["max"] - d["min"]))
    out["claim"] = ["invariant" if r <= tolerance else "DEPENDENT"
                    for r in out["range"]]
    return out.round(3)


labels = pd.concat([label(f) for f in sweeps.values()])
labels[labels["claim"] == "DEPENDENT"]'''))

    cells.append(code('''grid = harness.sweep_grid(
    configs["imsa"], banks["imsa"].sweep[:N_SWEEP], fields["imsa"],
    "pit_caution_discount", (0.5, 1.0, 1.5),
    "caution_rate", (0.5, 1.0, 2.0))

for strategy in ("caution_gambler", "splash_and_dash"):
    fig = viz.plot_sweep_grid(grid, strategy,
                              "pit_caution_discount", "caution_rate")
    plt.show()'''))

    # ------------------------------------------------------------------
    cells.append(md("""## What this stage found

Collected here rather than left in the tables, because three of these are
about the engine or the rulebooks rather than about the strategies, and a
later stage that does not know them will draw the wrong conclusion from the
numbers above.

**1. Two decisions described actions the engine does not offer.** Decision 9
item 5 had the lap-down defender "take the wave-around", but the credit is
engine-internal and unconditional on the frozen eligible set - the only
lever is not being outside that set when a wave is announced. Item 3 named
the rival as "whoever is directly ahead on the road", which is a car a stop
cannot drop you behind. Both were amended before the roster was written. A
strategy built against either wording would have run, produced numbers, and
measured nothing.

**2. At the line, a car cannot read its own position by the usual means.**
Two defects, both from the event queue, both of which looked correct in
review. Its own `track_fraction` saturates at the clip because `_set_lap`
has not run yet - which produced ninety-nine wave-arounds in one race on a
field that was never lapped. And differencing two `race_time_s` values
compares crossings of *different laps*, because every slower rival is still
a lap behind at that instant - which fired the track-position defender on a
negative gap about a quarter of the time. Rivals are projected to a common
lap count instead. Both are now invariants in the blueprint.

**3. The caution gambler needs nothing 02b measured.** Its threshold is a
count - would this stop be repeated inside the same window before the flag -
rather than a price, so it comes out of `fuel_per_lap`, `caution_rate` and
the clock. The anchored caution level enters both branches of the comparison
and cancels, because the caution timeline is drawn in advance. That is what
keeps `anchor` inside `benchmark.py` and the roster parameter-free.

**4. Two results have their sign set by a rulebook difference.** The
splash-and-dash planner is strongest in WEC because art. 12 forbids tools
during refuelling, so skipping tyres saves the whole tyre job; under IMSA's
art. 34.1.1 the jobs overlap and it saves much less. The track-position
defender loses in IMSA and gains in WEC, because IMSA's staggered reopening
offers more cheap caution stops for it to refuse. Neither result would exist
in a single-series model, and both are the clearest evidence so far that
simulating one series would have misled.

**5. The lap-down defender is nearly inert, and the reason is 02a.** Its
wave clause fired in neither series. WEC was predicted - one wave, announced
a full caution lap before the lane opens, so it cannot be forfeited by a
decision. IMSA was not: compression takes the headline class from a ten-lap
spread to about one, so the focal car is almost never a lap down and the
situation the clause responds to has largely been engineered out of the
engine. **Read this with 02a's finding that compression's magnitude is
unvalidated**; it may be closing the field too hard.

**6. Position moves while lap count does not.** The median lap delta is zero
in every row of the headline table. Decision 1 split the score into position
and a low-variance time diagnostic; at six hours in a compressed field, the
diagnostic currently carries no information and position is doing all the
work. Worth knowing before 03 chooses a reward.

## What 02c does not do

- No pace modes. Lift-and-coast and push laps stay out, so the roster and
  the agent face the same action space.
- No tuning of any strategy, in either direction. A strategy that does not
  work is reported, not adjusted.
- The benchmark gap is only as good as the cache behind it, and 02b's
  stability work was fourteen seeds rather than a result.
- Where the frozen dials are absent, every number above is about a
  stand-in - which remains the project's oldest outstanding claim."""))

    return cells

def build_03a():
    cells = []
 
    # ------------------------------------------------------------------
    cells.append(md("""# The gym wrapper (03a)
 
02c gave the five human strategies a number. This stage builds the thing
that lets an agent earn one on the same terms: a `gymnasium.Env` in which
the focal car is the agent, every other car runs its background strategy,
and the agent's decisions enter the engine through the *same*
`(CarState, RaceState) -> PitDecision` interface the roster uses.
 
**Zero physics.** The wrapper computes nothing. Every lap time, fuel burn,
caution and gap is the engine's, read through `RaceState` and `CarState`.
There is one simulator in this project; if two files could compute a lap
time the architecture would already have failed.
 
**The agent gets no privileged path.** The engine's forced-stop rule is
untouched, so an agent that declines to stop on an empty tank is made to
stop exactly as a human strategy would be. What the wrapper adds is a
*mask*: the forced action is removed from the agent's choices rather than
its answer being taken and thrown away. The mask is a training convenience,
which is why the evaluation path - `PolicyStrategy`, inserted into `ROSTER`
and scored by `harness.compare_roster` - carries none.
 
**The verification gate, restated.** The blueprint asked for a never-pit
policy reproducing the fuel-window baseline exactly. It does not, and Part 3
shows why and by how much. The gate that replaces it is stronger: a policy
returning `RunToFuelWindow`'s decision must reproduce that baseline bit for
bit *through the wrapper's plumbing*, which tests the thing this stage can
break rather than the engine's forced-stop path.
 
**What this stage found** is collected at the end. Two of the three are
about the observation space rather than about the agent, which does not
exist yet."""))
 
    # ------------------------------------------------------------------
    cells.append(md("""### Setup
 
The same project-root walk the other notebooks use, so this works whether
Jupyter was launched from `notebooks/`, from the project root, or from an
editor with its own idea of the working folder."""))
 
    cells.append(code('''import sys
from pathlib import Path
 
 
def find_project_root(marker: str = "src/endurance") -> Path:
    """Walk up from the working folder until the project appears."""
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / marker).exists():
            return candidate
    raise RuntimeError(f"could not find {marker} above {here}")
 
 
ROOT = find_project_root()
sys.path.insert(0, str(ROOT / "src"))
 
import numpy as np
import pandas as pd
 
from endurance import RaceConfig, run_race, scale_dials
from endurance.assets import draw_seed_bank, freeze_background
from endurance.engine import RaceEngine
from endurance.strategies import ROSTER, RunToFuelWindow
from endurance import gym_env, harness, viz
from endurance.gym_env import (
    FLAG_KEEP, FLAG_TYRES, FULL_KEEP, FULL_TYRES, STAY,
    EnduranceEnv, OBS_ROWS, PolicyStrategy, action_mask, observe,
)
 
pd.set_option("display.width", 120)
print(f"project root: {ROOT}")'''))
 
    # ------------------------------------------------------------------
    cells.append(md("""## Part 1 - The race, and which engine it is running on
 
Two things are settled before anything is measured, exactly as in 02c: which
dials these numbers are about, and whether the engine has 02b's
wave-eligibility fix. The second matters here because `laps_down` is an
observation row, so a wrapper built against the unpatched engine is showing
the agent a different race from the one 02c scored the roster on."""))
 
    cells.append(code('''N_SEEDS = 30          # this stage measures the wrapper, not a policy
N_GATE = 12           # seeds the verification gate runs on
 
PROCESSED = ROOT / "data" / "processed"
 
 
def stand_in(series_code: str) -> RaceConfig:
    """The six-hour config 02b and 02c worked against, rebuilt rather than loaded."""
    from endurance import ClassDials
 
    quick = ClassDials(
        series_code=series_code,
        class_name="GTP" if series_code == "imsa" else "HYPERCAR",
        base_pace_s=97.5, deg_slope_s_per_lap=0.012, pace_spread_s=0.5,
        lap_noise_s=0.5, caution_rate=0.20, caution_mean_dur_s=600.0,
        green_stint_laps=30.0, fuel_per_lap=1 / 30,
        fuel_per_lap_caution=0.6 / 30, tyre_life_laps=60.0,
        pit_time_mean_s=47.0, pit_time_std_s=2.0, n_cars=8)
    gt = ClassDials(
        series_code=series_code,
        class_name="GTD" if series_code == "imsa" else "LMGT3",
        base_pace_s=112.0, deg_slope_s_per_lap=0.02, pace_spread_s=0.9,
        lap_noise_s=0.6, caution_rate=0.20, caution_mean_dur_s=600.0,
        green_stint_laps=28.0, fuel_per_lap=1 / 28,
        fuel_per_lap_caution=0.6 / 28, tyre_life_laps=56.0,
        pit_time_mean_s=45.0, pit_time_std_s=2.0, n_cars=10)
    return RaceConfig(name=f"{series_code} 6h stand-in",
                      series_code=series_code, duration_s=6 * 3600.0,
                      classes=[quick, gt])
 
 
configs, source = {}, {}
for series_code in ("imsa", "wec"):
    path = PROCESSED / f"{series_code}.json"
    if path.exists():
        configs[series_code] = RaceConfig.load(path)
        source[series_code] = str(path.relative_to(ROOT))
    else:
        configs[series_code] = stand_in(series_code)
        source[series_code] = "STAND-IN (frozen dials not on disk)"
 
fields = {s: freeze_background(c) for s, c in configs.items()}
banks = {s: draw_seed_bank(c, draw_seed=20260806 + i)
         for i, (s, c) in enumerate(configs.items())}
 
for series_code, cfg in configs.items():
    print(f"{series_code}: {cfg.name}, {cfg.duration_s / 3600:.1f} h, "
          f"{cfg.total_cars} cars, headline class {harness.headline_class(cfg)}")
    print(f"    dials from {source[series_code]}")
 
if any("STAND-IN" in s for s in source.values()):
    print("\\n*** Every number in this notebook is against a stand-in. ***")'''))
 
    cells.append(md("""### Which engine is this?
 
02c shipped a reconstruction of 02b's wave fix and asked the next thread to
confirm which version the tree carries before quoting any number that
depends on a wave count. Run it here rather than trusting a handover."""))
 
    cells.append(code('''import inspect
 
import endurance.engine as _engine
 
patched = "at_line" in inspect.getsource(_engine.RaceEngine._progress)
print("wave-eligibility fix:", "PATCHED" if patched else "UNPATCHED")
 
# The measurement behind the label, so this does not rest on a substring.
waves = []
for seed in (3, 4, 5, 6):
    r = run_race(configs["imsa"], seed=seed)
    waves.append(int(r.laps["wave_by"].sum()))
print(f"wave-arounds on seeds 3-6: {waves}")
print("02c reports 50-83 unpatched and 16-42 patched")'''))
 
    # ------------------------------------------------------------------
    cells.append(md("""## Part 2 - The inversion, and the race it does not change
 
`RaceEngine.run` was a loop that ran to the flag, calling each car's
strategy as it crossed. A `step` has to hand control back to its caller at
the focal car's decision, so the loop is now `run_stream`, a generator that
suspends there and resumes on `send`. **`run` is a drain of it**, not a
second loop - with no focal car the generator never yields and returns
through `StopIteration`.
 
A worker thread with the focal strategy blocking on a queue pair was the
alternative, and it works: measured at 1.3% overhead, about 10 microseconds
a handoff. Speed did not decide it. A suspended generator is a single
deterministic object that `close()` disposes of, and a traceback out of a
policy arrives at the caller rather than across a thread boundary.
 
The claim worth checking is that the inversion changed nothing. Not the
focal car's row - the whole field's lap frame."""))
 
    cells.append(code('''def drive(cfg, seed, focal, policy, field):
    """Run one race with `policy` driving the focal car through the stream."""
    stream = RaceEngine(cfg, seed=seed).run_stream(
        field.resolve(focal=focal), focal=focal)
    decision = None
    while True:
        try:
            car, state, forced, lane = (stream.send(decision) if decision
                                        else next(stream))
        except StopIteration as done:
            return done.value
        decision = policy(car, state)
 
 
rows = []
for series_code, cfg in configs.items():
    field = fields[series_code]
    for seed in banks[series_code].headline[:3]:
        focal = harness.focal_car(cfg, seed)
        plain = run_race(cfg, strategies=field.resolve(), seed=seed)
        driven = drive(cfg, seed, focal, RunToFuelWindow(), field)
        rows.append({"series": series_code, "seed": seed, "focal": focal,
                     "lap_frames_identical": driven.laps.equals(plain.laps)})
 
pd.DataFrame(rows)'''))
 
    cells.append(md("""Identical, for the whole field. That is as strong a statement as this can
be made into: same seed, same frozen background, same decisions, and every
lap record byte for byte where it was."""))
 
    # ------------------------------------------------------------------
    cells.append(md("""## Part 3 - The gate, and the one that was dropped
 
The blueprint's gate reads: *a policy that always returns no pit reproduces
the fuel-window baseline's result exactly on the same seed.* It does not.
`RunToFuelWindow` stops with a lap and a half of fuel in hand; a never-pit
policy is forced by `_must_pit` below one lap, so the stop lands a lap
later and the two races diverge.
 
02c's handover proposed restating it on laps and stop count, on the grounds
that those agree. Over twelve seeds they do not."""))
 
    cells.append(code('''def stay_out(car, state):
    from endurance.engine import PitDecision
    return PitDecision(pit=False)
 
 
rows = []
cfg, field = configs["imsa"], fields["imsa"]
for seed in banks["imsa"].headline[:N_GATE]:
    focal = harness.focal_car(cfg, seed)
    a = harness.run_focal(cfg, seed, focal, RunToFuelWindow(), field)
    b = harness.run_focal(cfg, seed, focal, stay_out, field)
    rows.append({"seed": seed,
                 "d_laps": b["laps"] - a["laps"],
                 "d_stops": b["stops"] - a["stops"],
                 "d_race_time_s": round(b["race_time_s"] - a["race_time_s"], 2)})
 
dropped = pd.DataFrame(rows)
print(f"laps differ on {(dropped['d_laps'] != 0).sum()} of {len(dropped)} seeds")
print(f"stops differ on {(dropped['d_stops'] != 0).sum()} of {len(dropped)}")
dropped'''))
 
    cells.append(md("""So there is no exact statement of the never-pit gate to keep, and it is
**dropped rather than restated**. Weakening a gate until it passes is worse
than not having it.
 
What replaces it is the stronger of the two candidates the handover offered:
a policy returning `RunToFuelWindow`'s decision must reproduce that baseline
bit for bit, with the agent's plumbing in the middle. It exercises the
wrapper - observation built, action chosen, decision rebuilt - rather than
the engine's forced-stop path, and the wrapper is what this stage can
break."""))
 
    cells.append(code('''def echo(car, state):
    """The baseline's decision, arriving through the agent's interface."""
    return RunToFuelWindow()(car, state)
 
 
COLS = ("laps", "race_time_s", "stops", "pit_time_s", "traffic_time_s",
        "class_pos")
 
rows = []
for series_code, cfg in configs.items():
    field = fields[series_code]
    for seed in banks[series_code].headline[:N_GATE]:
        focal = harness.focal_car(cfg, seed)
        a = harness.run_focal(cfg, seed, focal, RunToFuelWindow(), field)
        b = harness.run_focal(cfg, seed, focal, echo, field)
        rows.append({"series": series_code, "seed": seed,
                     **{f"same_{c}": a[c] == b[c] for c in COLS}})
 
gate = pd.DataFrame(rows)
assert gate.filter(like="same_").all().all(), "03a's verification gate failed"
print(f"gate passes on {len(gate)} races across both series")
gate.groupby("series").agg({c: "all" for c in gate.columns if c.startswith("same_")})'''))
 
    # ------------------------------------------------------------------
    cells.append(md("""## Part 4 - The observation, and a normalisation that was nearly dead
 
Nine rows, all of them read from the engine. The blueprint drafted eight and
then discussed both a ninth and a tenth without the numbering adding up.
Settled here: `pit_lane_open` is in, on 02c's evidence that a gambler which
cannot see the lane is not gambling, and the same argument applies to an
agent. `wave_eligible` is out, because under current compression the
situation it describes almost never arises - a decision to revisit if 00's
re-run finds compression closing the field too hard.
 
Two of the rows are *gap to the car ahead* and *gap to the car behind*, and
they walk straight into section 5's second invariant. At the line the focal
car is on lap N and every slower class rival is still on lap N-1, so
differencing two `race_time_s` values compares crossings of different laps.
`RaceState.gap_ahead_s` and `gap_behind_s` project rivals to a common lap
count instead, which is the same rule `strategies._would_be_passed` uses,
now living in one place both the roster and the agent can reach."""))
 
    cells.append(code('''def gaps_over_a_race(cfg, seed, field):
    focal = harness.focal_car(cfg, seed)
    stream = RaceEngine(cfg, seed=seed).run_stream(
        field.resolve(focal=focal), focal=focal)
    decision, out = None, []
    while True:
        try:
            car, state, forced, lane = (stream.send(decision) if decision
                                        else next(stream))
        except StopIteration:
            return pd.DataFrame(out)
        # Long rather than wide, because `viz.plot_gap_normalisation` groups
        # on the row name and a wide frame would have to be melted at the
        # call site - which is arithmetic in a notebook rather than in a
        # module, and the thing viz.py's own rule is about.
        out.append({"row": "gap_ahead", "seconds": state.gap_ahead_s(car)})
        out.append({"row": "gap_behind", "seconds": state.gap_behind_s(car)})
        decision = RunToFuelWindow()(car, state)
 
 
gaps = pd.concat(
    [gaps_over_a_race(cfg, seed, fields[series_code]).assign(series=series_code)
     for series_code, cfg in configs.items()
     for seed in banks[series_code].headline[:10]],
    ignore_index=True)
 
SCALES = {"blueprint 120 s": 120.0,
          "pit_time_mean_s": configs["imsa"].classes[0].pit_time_mean_s}
 
rows = []
for (series_code, name), sub in gaps.groupby(["series", "row"]):
    seconds = sub["seconds"].dropna()
    row = {"series": series_code, "row": name, "n": len(sub),
           "no_rival": round(sub["seconds"].isna().mean(), 3),
           "p50_s": round(seconds.median(), 2),
           "p99_s": round(seconds.quantile(0.99), 2),
           "max_s": round(seconds.max(), 2),
           "share_negative": float((seconds < 0).mean())}
    for label, scale in SCALES.items():
        scaled = (seconds / scale).clip(0.0, 1.0)
        row[f"p50 / {label}"] = round(float(scaled.median()), 3)
        row[f"p99 / {label}"] = round(float(scaled.quantile(0.99)), 3)
    rows.append(row)
 
pd.DataFrame(rows)'''))
 
    cells.append(md("""No negative gap anywhere, in either series, which is the invariant
holding. What the table says against the blueprint's drafted normalisation -
clip at 120 s, divide by 120 - is that the clip never binds: the widest gap
over twenty races is about 92 s. And the gaps are small and heavily skewed,
so /120 puts the median at 0.03 and the ninety-ninth percentile at 0.36. Two
thirds of the row carries one observation in a hundred.
 
`pit_time_mean_s` is used instead, which moves those to 0.07 and 0.93. Both
scales are skewed - that is a property of the gaps rather than of the
divisor - but this one puts the resolution where the observations are, and
1.0 then means *a stop's worth of gap*, which is the unit the decision is
actually taken in. It is also a dial rather than a constant somebody picked.
**This supersedes the blueprint's observation table and is recorded there.**"""))
 
    cells.append(code('''viz.plot_gap_normalisation(gaps[gaps["series"] == "imsa"], SCALES)'''))
 
    cells.append(md("""The right panel is the argument. Under either scale the median sits low,
because most of the time there is a car within a few seconds - a compressed
field is what 02a's work produced. The difference is the top of the row:
`/120` never reaches past about a third of it, so the range that would tell
"nobody near" apart from "a stop's worth of gap" is never visited.
 
The whole observation, on one real decision point:"""))
 
    cells.append(code('''cfg, field = configs["imsa"], fields["imsa"]
seed = banks["imsa"].headline[0]
focal = harness.focal_car(cfg, seed)
 
stream = RaceEngine(cfg, seed=seed).run_stream(field.resolve(focal=focal),
                                               focal=focal)
decision = None
for _ in range(120):                      # somewhere in the middle of the race
    car, state, forced, lane = stream.send(decision) if decision else next(stream)
    decision = RunToFuelWindow()(car, state)
 
pd.DataFrame({"row": OBS_ROWS, "value": observe(car, state).round(4)})'''))
 
    # ------------------------------------------------------------------
    cells.append(md("""## Part 5 - The mask, and what it is not
 
The engine forces a stop when the car is out of fuel, out of tyres, or the
driver is out of time. That rule is untouched. The wrapper removes the
forced action from the agent's choices instead of letting it be chosen and
discarded, so the policy is never trained on a decision it does not own.
 
Three rules, each restating something the engine would have done anyway:
a forced stop removes *stay out*; a stop forced by tyre life also removes
the keep-tyre actions, since `_apply_pit` fits tyres on that reason
whatever the decision said; and a shut lane removes every stop, unless one
is forced, in which case the engine takes it and records
`lane_closed_stop`.
 
**Removing the override instead was considered and rejected.** `_next_lap`
never reads `car.fuel`, so fuel exists in this model only through
`_must_pit`. Without it an empty tank costs nothing and a never-pit car laps
at full pace for six hours - and whether that wins is one slider away."""))
 
    cells.append(code('''# The exploit, demonstrated rather than asserted. `scale_dials` is the app's
# whole lever mechanism, so this is a slider a user can drag.
rows = []
for mult in (1.0, 0.5, 0.25):
    twisted = scale_dials(configs["imsa"], deg_slope_s_per_lap=mult)
    field = freeze_background(twisted)
    for seed in banks["imsa"].headline[:3]:
        focal = harness.focal_car(twisted, seed)
        strategies = field.resolve(focal=focal)
        strategies[focal] = RunToFuelWindow()
        honest = run_race(twisted, strategies=strategies, seed=seed)
 
        eng = RaceEngine(twisted, seed=seed)
        eng._must_pit = lambda car, cls: ""          # the override removed
        strategies[focal] = stay_out
        free = eng.run(strategies=strategies)
 
        rows.append({
            "deg_x": mult, "seed": seed,
            "baseline_pos": int(honest.classification()
                                .set_index("car_id").loc[focal, "class_pos"]),
            "never_pit_pos": int(free.classification()
                                 .set_index("car_id").loc[focal, "class_pos"]),
        })
 
pd.DataFrame(rows).pivot(index="seed", columns="deg_x",
                         values=["baseline_pos", "never_pit_pos"])'''))
 
    cells.append(md("""At the calibrated slope the never-pit car still loses, because degradation
happens to outweigh six stops. Halve the slope and it wins the class. The
override stays."""))
 
    cells.append(code('''# What the mask actually does over a race: how often each action is legal.
cfg, field = configs["imsa"], fields["imsa"]
counts = {name: 0 for name in ("stay", "full+tyres", "full keep",
                               "flag+tyres", "flag keep")}
forced_reasons, n = {}, 0
 
for seed in banks["imsa"].headline[:5]:
    focal = harness.focal_car(cfg, seed)
    stream = RaceEngine(cfg, seed=seed).run_stream(field.resolve(focal=focal),
                                                   focal=focal)
    decision = None
    while True:
        try:
            car, state, forced, lane = (stream.send(decision) if decision
                                        else next(stream))
        except StopIteration:
            break
        mask = action_mask(state, forced)
        for name, allowed in zip(counts, mask):
            counts[name] += int(allowed)
        forced_reasons[forced or "-"] = forced_reasons.get(forced or "-", 0) + 1
        n += 1
        decision = RunToFuelWindow()(car, state)
 
print(f"{n} decision points over five races")
print("forced:", forced_reasons)
pd.DataFrame({"action": list(counts),
              "share_legal": [round(v / n, 4) for v in counts.values()]})'''))
 
    # ------------------------------------------------------------------
    cells.append(md("""## Part 6 - A random agent, and the floor it sets
 
No policy exists yet - that is 03b. What can be established now is the
floor: what a policy sampling uniformly from the legal actions scores on
these races. Anything 03b produces has to beat this, and if it does not, the
problem is the training rather than the environment.
 
It also runs the blueprint's headless check, which asks for a thousand
random steps without error."""))
 
    cells.append(code('''env = EnduranceEnv(configs["imsa"], fields["imsa"], banks["imsa"])
rng = np.random.default_rng(0)
 
obs, info = env.reset(seed=banks["imsa"].headline[0])
steps, episodes, finishes, returns, this_return = 0, 0, [], [], 0.0
while steps < 1000:
    legal = np.flatnonzero(info["action_mask"])
    obs, reward, terminated, truncated, info = env.step(int(rng.choice(legal)))
    steps += 1
    this_return += reward
    assert env.observation_space.contains(obs)
    if terminated:
        finishes.append(info["classification"])
        returns.append(this_return)
        episodes, this_return = episodes + 1, 0.0
        obs, info = env.reset()
 
print(f"{steps} random masked steps, {episodes} complete races, no error")
pd.DataFrame(finishes)[["laps", "stops", "class_pos", "pit_time_s"]].describe().round(1)'''))
 
    cells.append(code('''# The floor, next to the roster's null on the same races.
cfg, field = configs["imsa"], fields["imsa"]
seeds = banks["imsa"].headline[:N_SEEDS]
 
 
def random_policy(seed):
    """A policy in `PolicyStrategy`'s shape, so it is scored like any other."""
    rng = np.random.default_rng(seed)
    return PolicyStrategy(lambda obs, deterministic=True: (rng.integers(5), None))
 
 
roster = dict(ROSTER)
roster["random_agent"] = lambda: random_policy(0)
 
comparison = harness.compare_roster(cfg, seeds, field, roster=roster)
comparison.summarise()'''))
 
    cells.append(md("""The random agent is scored by exactly the call the five human strategies
are scored by: inserted into `ROSTER` as a sixth member, on the same seed
bank, the same frozen field and the same pace rank. That is deliberate and
it is what 03b must keep doing. A second evaluation path is the failure
decision 6 exists to prevent, and it is the kind that produces plausible
numbers rather than errors.
 
Note that the random agent has no mask here. `PolicyStrategy` is the
evaluation path and it relies on the engine's override exactly as the humans
do, so an agent that asks to stay out on an empty tank is made to stop and
is scored on that. The honest number."""))
 
    # ------------------------------------------------------------------
    cells.append(md("""## Part 7 - What 03a found
 
**1. The blueprint's verification gate does not hold, and neither does the
restatement proposed for it.** A never-pit policy differs from the
fuel-window baseline on race time by construction. Over twelve seeds it also
differs on lap count in four and on stop count in two, so 02c's suggestion
of restating it on laps and stops was three seeds' luck. The gate is dropped
and replaced by the echo gate in Part 3, which is stronger because it
exercises the wrapper rather than the engine.
 
**2. The gap rows' normalisation put the observations in the wrong part of
the range.** Clipped at 120 s and divided by 120, the clip never binds, the
median lands at 0.03 and the ninety-ninth percentile at 0.36 - so two thirds
of the row carries one observation in a hundred. Rescaled on
`pit_time_mean_s` those become 0.07 and 0.93, and 1.0 is a stop's worth of
gap. **This is a change to
the blueprint's observation table, not an implementation detail**, and the
same question should be asked of the other scales: `stint_laps / 40` and
`laps_down / 3` are both constants somebody chose rather than dials.
 
**3. Fuel exists only through the forced-stop rule.** `_next_lap` never
reads `car.fuel`, so removing the engine's override to stop it discarding
the agent's answer would make an empty tank free. At the calibrated
degradation slope a never-pit car still loses; at half the slope it wins the
class. The override stays and the wrapper masks instead - and this is worth
carrying into 03b, because it means **the reward's pit penalty is entirely
in the elapsed time**, with no separate term to tune."""))
 
    cells.append(md("""## Decisions taken in 03a, not to be silently reversed
 
**A. `run` is a drain of `run_stream`.** Not a second loop. If a later
stage needs another way to step the race, it inverts further rather than
copying the loop.
 
**B. Five discrete actions, and no refuel level.** Stay, full service, full
fuel keeping tyres, fill-to-the-flag with tyres, fill-to-the-flag keeping
them. The agent can reach the splash - 02c's largest measured lever - but
cannot choose a number, which is the tuning surface the roster was
forbidden. `strategies.fuel_to_the_flag` is shared by the planner and the
wrapper so there is one implementation of that arithmetic.
 
**C. Nine observation rows.** `pit_lane_open` in, `wave_eligible` out. The
second is conditional on 02a's unvalidated compression: if the field is
being closed too hard, the situation returns and so does the row.
 
**D. Pace modes stay out, permanently.** Section 7B asked for this to be
settled once at 03a. They need an engine lever that does not exist and a
`PitDecision` field that does not exist, they reopen the stint-length
dimension that fixing fuel as the binding constraint closed off, and they
would have to go into the human roster too or the comparison is rigged.
 
**E. The DNF term is dropped from the reward.** The engine models no
retirement. Section 7A's reward is negative lap time plus pit penalty, and
the penalty is already inside the elapsed time rather than added again.
 
**F. The agent is evaluated through `harness.compare_roster` only.** As a
sixth roster member, on the same banks. There is no agent-specific
evaluation code and 03b must not add any."""))
 
    cells.append(md("""## What 03b inherits
 
The environment, and three things to settle before training.
 
**The reward is a proxy and the score is not.** Section 7A is explicit that
the agent optimises negative time while being judged on class position, and
02c's finding 7 sharpens it: in a compressed field at six hours the median
lap delta is zero in every row of the headline table, so position does all
the work and the time diagnostic carries no information. The gap between
what the agent optimises and what it is scored on is a finding in its own
right and belongs in the write-up rather than buried.
 
**Algorithm choice is a UI decision as much as a learning one.** The app
promises visible action values or action probabilities. DQN gives Q(s,a),
PPO gives P(a|s); either satisfies it, but the mask has to be threaded
through whichever is chosen, and maskable PPO is the better-supported of the
two on that count.
 
**Budget.** One six-hour race is about 0.15 s and an episode is roughly 190
steps, so the wrapper is not the bottleneck in any training loop. Evaluation
of thirty seeds across six roster members runs in under a minute.
 
**Still open, inherited rather than caused.** 00's calibration is
unverified against the real `laps.csv`; 02b's per-seed benchmark artefacts
do not exist, so the strategy-to-benchmark gap 03 actually wants is still
unmeasured; and 02a Part 6's table and 01 Part 6's validation table both
want re-running against the corrected engine."""))
 
    return cells

def build_03b():
    cells = []

    # ------------------------------------------------------------------
    cells.append(md("""# RL training (03b)

03a built the environment. This stage puts a policy in it, and then puts
that policy in the roster next to the five human strategies so its number
means the same thing theirs do.

**The only way the agent is scored.** Wrapped in `gym_env.PolicyStrategy`,
inserted into `ROSTER` as a sixth member, and passed to
`harness.compare_roster` on the same banks, the same frozen background field,
the same headline class and the same pace rank as the humans. There is no
agent-specific evaluation code anywhere in this stage. A second evaluation
path that can differ from the roster's is the failure decision 6 exists to
prevent, and it produces plausible numbers rather than errors.

**The reward is a proxy and the score is not.** Training optimises negative
elapsed race time over the class green lap; the table below reports class
position. Section 7A is explicit that this split is deliberate, and 02c's
seventh finding sharpens it - at six hours in a compressed field the median
lap delta is zero in every row, so position does all the work. The gap
between what the agent optimises and what it is judged on is a finding in
its own right and is stated wherever a number appears rather than buried.

**Everything here is about a stand-in.** The calibrated dials on disk are
not usable; Part 1 says why in numbers. Nothing in this notebook is a claim
about Daytona or Le Mans."""))

    # ------------------------------------------------------------------
    cells.append(md("""### Setup

The same project-root walk the other notebooks use. `scripts/` goes on the
path as well as `src/`, because the frozen assets are resolved by
`freeze_assets.load_assets` and there must be exactly one function that
decides which race this is."""))

    cells.append(code('''import sys
from pathlib import Path


def find_project_root(marker: str = "src/endurance") -> Path:
    """Walk up from the working folder until the project appears."""
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / marker).exists():
            return candidate
    raise RuntimeError(f"could not find {marker} above {here}")


ROOT = find_project_root()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from endurance import harness, viz
from endurance.assets import dials_fingerprint
from endurance.engine import RaceEngine
from endurance.gym_env import (
    FLAG_KEEP, FLAG_TYRES, FULL_KEEP, FULL_TYRES, STAY,
    N_ACTIONS, PolicyStrategy, observe, to_decision,
)
from endurance.policy import PolicyCard, agent_roster, load_policy
from endurance.strategies import ROSTER, RunToFuelWindow
from endurance import run_race

from freeze_assets import SERIES, load_assets

pd.set_option("display.width", 140)


def show(df, index: bool = False) -> None:
    """Print a frame the same way in Jupyter and in a headless executor.

    `display` is an IPython builtin and `tests/run_notebook.py` execs these
    cells in a bare namespace, so a notebook that uses it runs by hand and
    fails in the suite - which is the opposite of what the suite is for.
    """
    print(df.to_string(index=index))


print(f"project root: {ROOT}")'''))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 1 - Which race, and which engine

Two things settled before anything is measured, as in 02c and 03a: which
dials these numbers are about, and whether the engine carries 02b's
wave-eligibility fix. Both are read off disk rather than taken from a
handover."""))

    cells.append(code('''N_FLOOR = 30          # seeds the random floor is measured on
N_GATE = 5            # seeds the gate conditions run on
EVAL = ROOT / "outputs" / "evaluation"
POLICIES = ROOT / "outputs" / "policies"

assets = {}
stand_in_series = []
for series_code in SERIES:
    cfg, bank, field = load_assets(series_code)
    assets[series_code] = (cfg, bank, field)
    is_stand_in = cfg.duration_s < 20 * 3600
    if is_stand_in:
        stand_in_series.append(series_code)
    print(f"{series_code}: {cfg.name}, {cfg.duration_s / 3600:.0f} h, "
          f"{cfg.total_cars} cars, headline class "
          f"{harness.headline_class(cfg)}")
    print(f"    dials {dials_fingerprint(cfg)}, "
          f"{len(bank.headline)} headline / {len(bank.held_out)} held out")
    print(f"    background: every car on "
          f"{field.provenance.get('uniform_strategy')}")

print()
if stand_in_series:
    print(f"*** STAND-IN DIALS for {', '.join(stand_in_series)}. Numbers for "
          f"{'this series' if len(stand_in_series) == 1 else 'these series'} "
          f"are about a six-hour invented race. ***")
else:
    print("Real dials for every series - 00's gate passed and "
          "freeze_assets.py loaded data/processed/{series}.json.")'''))

    cells.append(md("""### Why the calibrated dials are not used

`data/processed/imsa.json` and `wec.json` exist and this stage refuses them.
The base paces in both are credible - 97.9 s at Daytona, 209.4 s at Le Mans
- and almost nothing downstream of "one running of the race" is. The cell
below is the evidence rather than an assertion.

This is the first hard measurement on the project's oldest outstanding
claim, and it locates the fault: the calibration queries are not scoped to a
single event. Everything that is per-event comes out pooled across editions,
and everything that is a per-lap median survives."""))

    cells.append(code('''import json

rows = []
for series_code in SERIES:
    path = ROOT / "data" / "processed" / f"{series_code}.json"
    if not path.exists():
        continue
    raw = json.loads(path.read_text())
    for c in raw["classes"]:
        rows.append({
            "series": series_code,
            "race_hours": round(raw["duration_s"] / 3600, 1),
            "class": c["class_name"],
            "base_pace_s": round(c["base_pace_s"], 1),
            "deg_s_per_lap": round(c["deg_slope_s_per_lap"], 4),
            "stint_laps": c["green_stint_laps"],
            "stint_hours": round(c["green_stint_laps"] * c["base_pace_s"] / 3600, 2),
            "pit_mean_s": round(c["pit_time_mean_s"], 1),
            "pit_std_s": round(c["pit_time_std_s"], 1),
            "cars": c["n_cars"],
        })

calibrated = pd.DataFrame(rows)
if calibrated.empty:
    print("no calibrated dials on disk to check")
else:
    show(calibrated)
    print("A 24-hour race is 24 hours. A pit stop's standard deviation is not "
          "five times its mean.")
    print("A green stint is not four hours. Degradation is not negative.")'''))

    cells.append(md("""### Is this the patched engine?

02c shipped a reconstruction of 02b's wave-eligibility fix and asked the
next thread to establish which version the tree carries before quoting any
number that depends on it. It matters here twice over: `laps_down` is an
observation row, so the agent sees the consequence directly, and Part 2's
comparison against 02c's published table turns on it."""))

    cells.append(code('''import inspect

import endurance.engine as _engine

patched = "at_line" in inspect.getsource(_engine.RaceEngine._progress)
print("wave-eligibility fix:", "PATCHED" if patched else "UNPATCHED")

# The measurement behind the label, so this does not rest on a substring.
cfg = assets["imsa"][0]
waves = [int(run_race(cfg, seed=s).laps["wave_by"].sum()) for s in (3, 4, 5, 6)]
print(f"wave-arounds on imsa seeds 3-6: {waves}")
print("02c reports 50-83 unpatched and 16-42 patched")'''))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 2 - The humans, re-scored on this engine

The agent's row has to sit beside human rows produced on the same engine,
the same dials and the same seeds. So the roster is re-scored here rather
than 02c's published table being carried forward, and the two are compared
below.

The tables are produced by `scripts/evaluate.py` and read from disk. Two
hundred paired seeds across six roster members is a few minutes of engine
time, and decision 6 asks for artefacts rather than a notebook that has to
be re-run to be believed."""))

    cells.append(code('''def table(series_code: str, bank: str, what: str = "summary"):
    """A table `scripts/evaluate.py` wrote, or a clear instruction if absent."""
    path = EVAL / f"{series_code}_{bank}_{what}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.relative_to(ROOT)} is missing. Run:\\n"
            f"    python scripts/freeze_assets.py\\n"
            f"    python scripts/evaluate.py --no-agent    # humans only\\n"
            f"    python scripts/train.py                  # then the policy\\n"
            f"    python scripts/evaluate.py")
    return pd.read_csv(path)


SHOW = ["strategy", "gained", "gained_lo", "gained_hi", "level", "lost",
        "median_d_pos", "n_seeds"]

for series_code in SERIES:
    print(f"=== {series_code} / headline ===")
    show(table(series_code, "headline")[SHOW])'''))

    cells.append(md("""### Against 02c's published table

02c's headline numbers were produced on its own reconstruction of the wave
fix. Where the two disagree by more than the seed set can explain, the
engine is the difference and **02c's table must not be quoted beside
anything in this notebook**.

The comparison is against the bootstrap interval rather than by eye, which
is the rule 02c wrote after publishing a two-place finding on a statistic
whose per-seed spread ran to three places and then reversing it at two
hundred seeds."""))

    cells.append(code('''PUBLISHED_02C = pd.DataFrame([
    ("imsa", "caution_gambler", 0.500, 0.210),
    ("imsa", "track_position",  0.370, 0.340),
    ("imsa", "splash_and_dash", 0.345, 0.090),
    ("imsa", "lap_down",        0.030, 0.030),
    ("wec",  "caution_gambler", 0.440, 0.260),
    ("wec",  "track_position",  0.370, 0.395),
    ("wec",  "splash_and_dash", 0.535, 0.080),
    ("wec",  "lap_down",        0.075, 0.085),
], columns=["series", "strategy", "gained_02c", "lost_02c"])

now = pd.concat([table(s, "headline").assign(series=s) for s in SERIES],
                ignore_index=True)
check = PUBLISHED_02C.merge(
    now[["series", "strategy", "gained", "gained_lo", "gained_hi"]],
    on=["series", "strategy"], how="left")
check["explained_by_seeds"] = ((check["gained_02c"] >= check["gained_lo"])
                               & (check["gained_02c"] <= check["gained_hi"]))
show(check)

beyond = check[~check["explained_by_seeds"]]
if len(beyond):
    print("Beyond what the seed set explains:")
    for r in beyond.itertuples():
        print(f"  {r.series} {r.strategy}: 02c published {r.gained_02c:.3f}, "
              f"here {r.gained:.3f} with interval "
              f"[{r.gained_lo:.3f}, {r.gained_hi:.3f}]")
else:
    print("Every difference from 02c sits inside the interval.")'''))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 3 - The floor, and the ceiling that is missing

**The floor.** A policy sampling uniformly over the five actions, scored
through the same roster path. Anything training produces has to beat this,
and if it does not the problem is the training rather than the environment.
Sampled without a mask, because the evaluation path has none: the engine's
override is what holds it to the rules, exactly as for the humans.

**The ceiling is absent.** 03b is specified to be scored against 02b's
*causal* benchmark - the clairvoyant one alone would penalise a policy for
failing to predict the future - and 02b's per-seed artefacts have never been
produced. So the agent's distance from the best a plan could have done stays
unmeasured, and the tables below carry no `gap_to_benchmark_pos` column.
That is an honest gap, not an oversight, and it is the largest single thing
04 and 05 inherit."""))

    cells.append(code('''rng = np.random.default_rng(0)
random_policy = PolicyStrategy(
    lambda obs, deterministic=True: (int(rng.integers(N_ACTIONS)), None))

floor = {}
for series_code in SERIES:
    cfg, bank, field = assets[series_code]
    roster = {"fuel_window": ROSTER["fuel_window"],
              "random": lambda: random_policy}
    rows = harness.compare_roster(cfg, bank.headline[:N_FLOOR], field,
                                  roster=roster).rows
    floor[series_code] = harness.summarise(rows)
    print(f"=== {series_code}: the floor over {N_FLOOR} races ===")
    show(floor[series_code][["strategy", "gained", "level", "lost",
                             "median_d_pos"]])'''))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 4 - Training

`scripts/train.py` trains one MaskablePPO policy per series on the headline
bank. Three choices worth restating, because each is a decision rather than
a default.

**MaskablePPO rather than DQN.** The env publishes a mask every step and a
policy that ignores it spends capacity on decisions the engine overrides.
`sb3-contrib` applies the mask inside the action distribution during rollout
*and* in the loss. DQN gives Q(s,a), which 04's panel could show with a real
unit - laps of time lost from here - but masking it means a custom policy
applying the mask at the acting argmax *and* in the target, and missing the
second is silent.

**`gamma = 1.0`.** A fixed-length race scored at the flag. Discounting would
say a place lost in hour six matters less than one lost in hour one, which
is false, and it would put a second implicit weighting between the reward
and the score on top of the proxy gap section 7A already names.

**Training draws only from the headline bank.** `EnduranceEnv` refuses the
held-out fifty, which covers the obvious mistake. It does not cover SB3
seeding its envs with integers of its own choosing, which `_pick_seed` would
accept - so `train.py` wraps the env to map any supplied seed onto a
headline race. Without that the training distribution is the whole seed
space while every artefact still claims the bank.

**The mask does not survive into evaluation.** `PolicyStrategy` carries
none, per 03a's decision F, so at scoring time an agent asking to stay out
on an empty tank is overridden and scored on that."""))

    cells.append(code('''def learning_curve(series_code: str):
    """Episode returns from the Monitor logs, or nothing if training has not run."""
    monitor = POLICIES / f"{series_code}_monitor"
    files = sorted(monitor.glob("*.monitor.csv")) if monitor.exists() else []
    if not files:
        return None
    frames = [pd.read_csv(f, skiprows=1).assign(worker=f.stem) for f in files]
    df = pd.concat(frames, ignore_index=True).sort_values("t")
    df["episode"] = range(len(df))
    return df


for series_code in SERIES:
    df = learning_curve(series_code)
    if df is None:
        print(f"{series_code}: no training logs - run scripts/train.py")
        continue
    window = max(len(df) // 40, 10)
    print(f"{series_code}: {len(df)} episodes, "
          f"return {df['r'].iloc[:window].mean():.1f} -> "
          f"{df['r'].iloc[-window:].mean():.1f}, "
          f"episode length {df['l'].median():.0f} steps")'''))

    cells.append(code('''import matplotlib.pyplot as plt

curves = {s: learning_curve(s) for s in SERIES}
if any(c is not None for c in curves.values()):
    fig, axes = plt.subplots(1, len(SERIES), figsize=(11, 4), sharey=False)
    for ax, (series_code, df) in zip(np.atleast_1d(axes), curves.items()):
        if df is None:
            ax.set_axis_off()
            continue
        window = max(len(df) // 40, 10)
        ax.plot(df["episode"], df["r"], color="0.8", linewidth=0.5)
        ax.plot(df["episode"], df["r"].rolling(window).mean(),
                color="tab:blue", linewidth=1.5)
        ax.set_title(f"{series_code.upper()} - episode return")
        ax.set_xlabel("episode")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("return (negative laps of time)")
    fig.tight_layout()
    plt.show()
else:
    print("nothing to plot yet")'''))

    cells.append(md("""### Does the reward vary with the policy at all?

The curve above is flat, and the cell below establishes that this is not a
training failure but an arithmetic one.

A step's reward is the race time elapsed since the last decision, negated.
Summed over an episode those elapsed times are **the length of the race**,
and this is a timed race, so the return is the same number whatever the
policy does. Laps are what vary at a fixed duration - and the return is
uncorrelated with them.

Section 7A specified dense negative time, which is right for a fixed
*distance*, where finishing sooner is winning. At a fixed *duration* the
time is given and the term cancels."""))

    cells.append(code('''diag = []
for series_code in SERIES:
    df = learning_curve(series_code)
    if df is None:
        continue
    first, last = df["r"].iloc[:250], df["r"].iloc[-250:]
    diag.append({
        "series": series_code,
        "episodes": len(df),
        "return_mean": round(df["r"].mean(), 3),
        "return_sd": round(df["r"].std(), 3),
        "laps_min": int(df["l"].min()),
        "laps_max": int(df["l"].max()),
        "corr_return_laps": round(df["r"].corr(df["l"]), 3),
        "first_250": round(first.mean(), 3),
        "last_250": round(last.mean(), 3),
        "moved_by": round(last.mean() - first.mean(), 3),
    })

if diag:
    diag = pd.DataFrame(diag)
    show(diag)
    for r in diag.itertuples():
        cfg = assets[r.series][0]
        pace = cfg.class_by_name(harness.headline_class(cfg)).base_pace_s
        print(f"{r.series}: race duration / green lap = "
              f"{cfg.duration_s / pace:.1f}, return {r.return_mean:.1f} - the "
              f"return is the race length, not the driving")
        print(f"    laps span {r.laps_min}-{r.laps_max} ({100 * (r.laps_max - r.laps_min) / r.laps_min:.0f}%) "
              f"while the return moved {abs(r.moved_by):.3f} "
              f"({abs(r.moved_by / r.return_mean):.3%}) over training")'''))

    cells.append(md("""**A car completing 206 laps and one completing 167 scored the same.** The
policy received a signal an order of magnitude smaller than the
episode-to-episode noise, so whatever the checkpoints do, they do it for no
reason the training gave them. Any result taken off a checkpoint trained
under this reward is a draw rather than a finding, and that includes both
rows in Part 6 if they were produced before the amendment below.

The replacement credits one lap per lap and nothing else. A stop still
costs, because it makes that step long, consumes race time and leaves fewer
laps to credit - and laps are what class position is derived from, so the
proxy moves nearer the score as well as becoming non-degenerate."""))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 5 - The verification gate

The blueprint gives 03b a boundary constraint and no gate, which 02c already
named as a gate nobody can fail. Two conditions, both with falsifiers, both
also in `tests/test_policy.py`.

**One: the plumbing, actually exercised.** 03a's gate compares
`RunToFuelWindow()` against a lambda calling it, and its docstring claims
the agent's plumbing sits in the middle. It does not - the echo touches
neither `observe` nor `to_decision` nor `PolicyStrategy`, so what it asserts
is that `run_focal` does not care whether a callable is a class instance or
a function. This condition closes that gap: a `PolicyStrategy` returning a
constant action must reproduce the same constant decision taken directly,
bit for bit, with the wrapper in the path on one side and not the other.

**Two: the checkpoint is the policy that was measured.** The deliverable is
a saved policy. The failure only this stage can produce is an artefact that
has drifted from the thing the table was computed on - a `.zip` that reloads
to a different policy, or an `.onnx` export that disagrees with it. 04 loads
the export and 05 hosts it, so a divergence puts the app on numbers nobody
produced."""))

    cells.append(code('''COLS = ("laps", "race_time_s", "stops", "pit_time_s", "traffic_time_s",
        "class_pos")
ACTIONS = (STAY, FULL_TYRES, FULL_KEEP, FLAG_TYRES, FLAG_KEEP)

rows = []
for series_code in SERIES:
    cfg, bank, field = assets[series_code]
    for action in ACTIONS:
        through = PolicyStrategy(
            lambda obs, deterministic=True, a=action: (a, None))
        direct = lambda car, state, a=action: to_decision(a, car, state)
        for seed in bank.headline[:N_GATE]:
            focal = harness.focal_car(cfg, seed)
            x = harness.run_focal(cfg, seed, focal, through, field)
            y = harness.run_focal(cfg, seed, focal, direct, field)
            rows.append({"series": series_code, "action": action, "seed": seed,
                         **{f"same_{c}": x[c] == y[c] for c in COLS}})

gate_one = pd.DataFrame(rows)
assert gate_one.filter(like="same_").all().all(), "03b gate one failed"
print(f"gate one passes on {len(gate_one)} races "
      f"({len(ACTIONS)} actions x {N_GATE} seeds x {len(SERIES)} series)")

# The falsifier: two different constants must not agree, or the comparison
# above is asserting that `run_focal` agrees with itself.
cfg, bank, field = assets["imsa"]
seed = bank.headline[0]
focal = harness.focal_car(cfg, seed)
a = harness.run_focal(cfg, seed, focal, PolicyStrategy(
    lambda obs, deterministic=True: (STAY, None)), field)
b = harness.run_focal(cfg, seed, focal, PolicyStrategy(
    lambda obs, deterministic=True: (FULL_TYRES, None)), field)
assert a["stops"] != b["stops"], "the gate cannot fail, so it is not a gate"
print(f"falsifier holds: {a['stops']} stops against {b['stops']}")'''))

    cells.append(code('''def checkpoint_for(series_code: str) -> Path | None:
    path = POLICIES / f"{series_code}_maskable_ppo.zip"
    return path if path.exists() else None


rows = []
for series_code in SERIES:
    checkpoint = checkpoint_for(series_code)
    if checkpoint is None:
        print(f"{series_code}: no checkpoint - run scripts/train.py")
        continue
    cfg, bank, field = assets[series_code]
    card = PolicyCard.load(checkpoint)
    card.check(cfg, bank)          # refuses a policy about another race

    # Loaded twice, because what is being guarded is the file.
    first = load_policy(checkpoint, config=cfg, bank=bank)
    second = load_policy(checkpoint, config=cfg, bank=bank)

    export = checkpoint.with_suffix(".onnx")
    exported = (load_policy(export, config=cfg, bank=bank)
                if export.exists() else None)

    for seed in bank.headline[:N_GATE]:
        focal = harness.focal_car(cfg, seed)
        x = harness.run_focal(cfg, seed, focal, first, field)
        y = harness.run_focal(cfg, seed, focal, second, field)
        row = {"series": series_code, "seed": seed,
               **{f"same_{c}": x[c] == y[c] for c in COLS}}

        if exported is not None:
            # The observations a real race visits, not uniform noise: that
            # is a much narrower part of the box and where a divergence
            # would hide.
            seen = []

            def spy(car, state, _p=first, _s=seen):
                _s.append(observe(car, state))
                return _p(car, state)

            harness.run_focal(cfg, seed, focal, spy, field)
            row["onnx_agrees"] = all(
                first.predict(o, deterministic=True)[0]
                == exported.predict(o, deterministic=True)[0] for o in seen)
            row["observations"] = len(seen)
        rows.append(row)

gate_two = pd.DataFrame(rows)
if len(gate_two):
    assert gate_two.filter(like="same_").all().all(), "the checkpoint is not deterministic"
    if "onnx_agrees" in gate_two.columns:
        assert gate_two["onnx_agrees"].all(), "the export is not the checkpoint"
    print(f"gate two passes on {len(gate_two)} races")
    show(gate_two.groupby("series").agg(
        {c: "all" for c in gate_two.columns if c.startswith("same_")
         or c == "onnx_agrees"}), index=True)'''))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 6 - The result

Two tables per series: the headline two hundred, and the held-out fifty.
Never pooled, and never a single number.

**Read the headline row with its caveat attached.** The agent trained on
those two hundred races and the five humans did not train at all, so that
row is the agent at its most flattering. The held-out fifty are the honest
generalisation claim: nothing was selected on them and nothing trained on
them. Hyperparameters were moved on the sweep fifty and nowhere else.

**And read the intervals, not the point estimates.** Where the agent's
interval overlaps a human's at this seed count, there is no ordering to
report."""))

    cells.append(code('''for series_code in SERIES:
    for bank_name in ("headline", "held_out"):
        try:
            t = table(series_code, bank_name)
        except FileNotFoundError as e:
            print(e)
            continue
        print(f"=== {series_code} / {bank_name} ===")
        show(t[SHOW])

        if "agent" in set(t["strategy"]):
            agent = t[t["strategy"] == "agent"].iloc[0]
            others = t[~t["strategy"].isin(["agent", "fuel_window"])]
            overlap = [r.strategy for r in others.itertuples()
                       if r.gained_lo <= agent.gained_hi
                       and agent.gained_lo <= r.gained_hi]
            print("  agent overlaps:", ", ".join(overlap) or "nobody")'''))

    cells.append(code('''for series_code in SERIES:
    try:
        rows = table(series_code, "headline", what="rows")
    except FileNotFoundError:
        continue
    fig = viz.plot_paired_deltas(rows)
    plt.show()

    # Absent until 02b's per-seed artefacts exist; the placeholder says so
    # rather than the figure being quietly omitted.
    fig = viz.plot_benchmark_gap(rows)
    plt.show()'''))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 7 - What the agent actually does

The project's point is not a champion agent, it is fluency someone can click
through in two minutes. A policy that gains a place without a legible reason
is worth less here than one that loses a place for a reason you can see, so
this part asks what the decisions look like rather than what they scored.

Every action carries its own label out of `to_decision`, so the mix is read
off the decision reasons rather than reconstructed."""))

    cells.append(code('''def decision_mix(series_code: str, n: int = 10):
    checkpoint = checkpoint_for(series_code)
    if checkpoint is None:
        return None
    cfg, bank, field = assets[series_code]
    policy = load_policy(checkpoint, config=cfg, bank=bank)

    taken = []

    def spy(car, state):
        d = policy(car, state)
        taken.append({"reason": d.reason, "pit": d.pit,
                      "under_caution": state.under_caution,
                      "lane_open": state.pit_lane_open,
                      "progress": round(state.race_progress, 2),
                      "fuel": round(car.fuel, 3),
                      "tyre_age": car.tyre_age})
        return d

    for seed in bank.headline[:n]:
        harness.run_focal(cfg, seed, harness.focal_car(cfg, seed), spy, field)
    return pd.DataFrame(taken)


for series_code in SERIES:
    mix = decision_mix(series_code)
    if mix is None:
        print(f"{series_code}: no checkpoint")
        continue
    print(f"=== {series_code}: {len(mix)} decisions over 10 races ===")
    show(mix["reason"].value_counts(normalize=True).round(3)
         .to_frame("share"), index=True)
    stops = mix[mix["pit"]]
    print(f"  stops asked for: {len(stops)}, "
          f"{stops['under_caution'].mean():.1%} of them under caution, "
          f"{(~stops['lane_open']).mean():.1%} through a shut lane")'''))

    cells.append(md("""A stop asked for through a shut lane is the one number here that is a
defect rather than a style. The engine refuses it and records
`stop_refused`, so the policy has spent a decision on something it could not
have; 02c's roster is tested for producing none of these. A non-zero share
means the mask taught the agent the lane matters during training and the
unmasked evaluation path is where it forgets - which is a cost of decision F
rather than an argument against it, and belongs in the write-up either way."""))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 8 - The dial the agent found

Part 7 shows what the policy does. This part is about what that turned out
to mean, and it is the stage's most useful output - more useful than the
agent's row in Part 6.

Given a working reward the IMSA policy converged on stopping almost every
lap. That is not a broken policy; it is a correct reading of the model. Lane
transit costs `pit_time_mean_s * pit_transit_frac`, and `stop_cost` charges
only for the fuel actually taken - so a car that is nearly full tops up for
a thirtieth of a tank and pays almost nothing for the privilege.

`pit_transit_frac` is in `ASSUMED_FIELDS`. Lap timing records how long a
stop took, never what happened during it, so the number was never measured.
Decision 2 says an assumed parameter gets swept, and this one had not been.
**The agent reached a corner of the action space five parameter-free
strategies had no reason to visit, and found the assumption sitting in
it.**"""))

    cells.append(code('''cfg = assets["imsa"][0]
cls = cfg.class_by_name(harness.headline_class(cfg))
transit_s = cls.pit_time_mean_s * cls.pit_transit_frac
topup_s = cls.pit_time_mean_s * (1 - cls.pit_transit_frac) * cls.fuel_per_lap

print(f"lane transit          {transit_s:6.2f} s"
      f"   = {cls.pit_time_mean_s} x {cls.pit_transit_frac}")
print(f"one lap of fuel       {topup_s:6.2f} s"
      f"   = {cls.pit_time_mean_s} x {1 - cls.pit_transit_frac:.2f} "
      f"x {cls.fuel_per_lap:.4f}")
print(f"a top-up costs        {transit_s + topup_s:6.2f} s")
print(f"a full service costs  {cls.pit_time_mean_s:6.2f} s")
print()
print("The agent was observed at 12.85 s a stop. Nothing here is a defect "
      "in the policy.")'''))

    cells.append(md("""### The sweep

`scripts/sweep_pit_transit.py`, run on the sweep fifty with one fresh
`NullRuns` per point so no point is scored against another's baseline. Read
the human rows first: if the roster's ranking moves as this dial moves, 02c's
findings are partly about an assumption nobody measured.

At fifty seeds a share carries an interval near +-0.14, so a raw swing
proves nothing and a single pair of points proves less. What counts is
movement in one direction at every step, in both series."""))

    cells.append(code('''SWEEPS = ROOT / "outputs" / "sweeps"


def sweep_table(series_code: str, what: str = "summary"):
    path = SWEEPS / f"{series_code}_pit_transit_frac_{what}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.relative_to(ROOT)} is missing. Run:\\n"
            f"    python scripts/sweep_pit_transit.py")
    return pd.read_csv(path)


monotone = {}
for series_code in SERIES:
    try:
        su = sweep_table(series_code)
    except FileNotFoundError as e:
        print(e)
        continue
    wide = su.pivot_table(index="pit_transit_frac", columns="strategy",
                          values="gained")
    print(f"=== {series_code}: gained share as pit_transit_frac rises ===")
    show(wide.round(3), index=True)

    falls = []
    for name in wide.columns:
        v = wide[name].to_numpy()
        if name == "fuel_window" or v.max() - v.min() < 0.05:
            continue
        if all(v[i] >= v[i + 1] for i in range(len(v) - 1)):
            falls.append(name)
    monotone[series_code] = set(falls)
    print(f"  falls at every step: {', '.join(falls) or 'nobody'}\\n")

both = set.intersection(*monotone.values()) if len(monotone) > 1 else set()
if both:
    print(f"Falls monotonically in BOTH series: {', '.join(sorted(both))}")'''))

    cells.append(md("""### What the sweep cannot say

The policy is frozen across the points, so its stop *rate* cannot respond -
and does not. Falling stop **counts** across the sweep are the race fitting
fewer laps, not the policy changing its mind. This arm answers "how much of
the roster's result depends on the dial"; it cannot answer "would a
retrained agent still exploit it", which needs a retrain at every point and
is a different agent per point."""))

    cells.append(code('''for series_code in SERIES:
    try:
        b = sweep_table(series_code, "behaviour")
    except FileNotFoundError:
        continue
    a = b[b["strategy"] == "agent"]
    if a.empty:
        continue
    rate = (a["stops"] / a["laps"]).round(3).tolist()
    print(f"{series_code} agent stops per lap: {rate}")
    print(f"  laps {a['laps'].iloc[0]:.0f} -> {a['laps'].iloc[-1]:.0f}, "
          f"class position {a['class_pos'].iloc[0]:.1f} -> "
          f"{a['class_pos'].iloc[-1]:.1f}")'''))

    cells.append(md("""### How to stop assuming it

The pit lane speed limit is regulation and identical in both series: 60 km/h
(IMSA art. 32.3, WEC art. 12.1.4). Neither rulebook carries the pit lane
**length**, which is a circuit fact, so the regulations supply one of the two
inputs the derivation needs and not both.

Two routes, and the second is better:

1. **Derive it.** The transit delta is the lane at 16.67 m/s minus the same
   distance at racing pace, plus the deceleration and acceleration losses.
   Needs a lane length per circuit and an assumption about the approach
   speed, so it trades one assumption for a smaller one.
2. **Measure it.** A low quantile of `pit_time` for one properly scoped
   event is a stop with no service in it - which is the transit delta itself,
   and would move `pit_transit_frac` out of `ASSUMED_FIELDS` entirely. This
   waits on 00's re-run: the current pit column has a standard deviation five
   times its mean, so a quantile of it is a quantile of noise.

**And the sweep may not be able to reach the answer.** A full service costs
transit plus the larger of the tyre and refuel jobs, which is what anchors
it to the measured mean; with `pit_tyre_frac` at 0.35 nothing above 0.65 is
expressible. If a defensible value sits above that, the finding is about the
*shape* of `stop_cost` rather than the value of one dial - everything in it
is a share of a fixed mean, so a large per-stop overhead cannot be
represented at all. That is 02a's territory."""))

    cells.append(md("""## What 03b found

**1. The calibrated dials are not usable, and the fault is locatable.** Base
paces survive; everything scoped per-event does not. 216 hours of IMSA
racing in one config, DPi and GTLM and GTDPRO in one field, four-hour green
stints, pit-time standard deviations five times their means, and negative
degradation at Le Mans. The calibration queries are pooling editions. This
is the sharpest statement anyone has made about stage 00's outstanding
claim and it should go in front of the re-run.

**2. 02c's headline table does not reproduce on this tree.** Most rows sit
inside the seed interval; the ones that do not are the engine, because 02c
ran on its own reconstruction of the wave fix. **02c's numbers must not be
quoted beside 03b's.** The human rows in this notebook are the reference,
and they were produced on the engine the agent trained on.

**3. 03a's replacement gate does not exercise the wrapper.** Its docstring
says the observation is built, the action chosen and the decision rebuilt;
the echo it compares is a bare callable that does none of those. Not
re-litigated - 03a's amendment stands - but 03b's gate one closes the gap
rather than restating it, and asserts directly that `observe` was called and
that what it returned varied.

**4. The median position delta is uninformative at this compression.** It is
zero in almost every row and its bootstrap interval spans a whole position,
so the shares carry the information and the median should not headline
anything, the agent's result included. 02c's finding 7, holding at 200
seeds and on the held-out fifty.

**0. The agent falsified an assumption the human roster could not reach.**
The most useful thing this stage produced. `pit_transit_frac = 0.25` was
never decided - it is a default every stage inherited - and 02c's
`splash_and_dash` result turns out to depend on it: gained falls
monotonically 0.40 to 0.10 in IMSA and 0.58 to 0.44 in WEC across the sweep,
while every other roster row wanders. The strategy built around a cheap
short fill is the one whose result depends on how a short fill is priced.
Five parameter-free strategies never had a reason to stop 164 times; an
adversarial policy search did, and found the assumption there. Amendment 10.

**5. The reward was a constant, and half a million steps taught nothing.**
The largest finding of the stage and the reason no number in Part 6 is an
agent result until it is retrained. A step scores the elapsed race time
negated; those elapsed times sum to the race duration; the race has a fixed
duration. Return −219.85 ± 0.42 over 2,664 episodes while laps ran 167 to
206, correlation −0.11, and 500,000 steps moved the mean by 0.016%.
Section 7A specified the reward for a fixed-distance race and this is a
timed one. Amended to one lap credited per lap.

**6. No test could have caught it, and now one can.** `test_gym_env.py`
asked whether a step's reward was negative and about a lap long, which is
true of a degenerate return as well as a useful one. The question it did not
ask - does the episode score change when the behaviour does - is now
`test_the_return_tells_two_policies_apart`. The lesson generalises past this
bug: every reward assertion in the suite was about the shape of a step
rather than about the sum."""))

    cells.append(md("""## What 04 inherits

The two checkpoints, their cards, and three things to know.

**The explainability panel has probabilities, not values.** MaskablePPO
gives P(a|s), and the ONNX export carries logits so a softmax in the app
produces them. There is no Q(s,a) to show; if the panel needs a magnitude
rather than a ranking, that is a DQN and it is a retraining job, not a UI
change.

**The mask is a training artefact and the app should say so.** 04's manual
override compares a human decision against what the policy would have done.
The policy's answer there is unmasked, which is the same answer it is scored
on - so an override panel showing "the agent would have stayed out" on an
empty tank is correct and needs a line of text next to it, not a fix.

**Nothing here is calibrated.** Sliders driving `scale_dials` on stand-in
dials are a demonstration of the mechanism. Until 00 is re-run the app is
honest only if it says so on the page.

**Still open, inherited rather than caused.** 02b's per-seed causal
artefacts do not exist, so the agent's gap to the reference - the number
03b was specified to report - is unmeasured. 01 Part 6 and 02a Part 6 still
want re-running against the corrected engine."""))

    return cells



def build_00():
    cells = []

    cells.append(md("""# Data reconnaissance (00) - re-run

**This notebook has been rebuilt.** Its first version was recorded as done
with the calibration unverified against the real `laps.csv`. 03b was the
first stage to read the frozen dials rather than assume them, refused them,
and used a stand-in instead. What follows is the re-run, and it is the stage
that decides whether this project demonstrates Daytona and Le Mans or an
invented six-hour race.

**Boundary constraint.** Constants only; no simulation code. No change to the
`ClassDials` schema or to `ASSUMED_FIELDS` - a new or renamed field
invalidates every saved config, bank and checkpoint in the project. No engine
changes, no strategy, benchmark or agent work.

**Verification gate.** The stage was specified without one, which is how the
fault survived. Four conditions, all of which could fail and one of which is
required to: Part 6.

## What was wrong

`imsa.json` described a 216-hour race with 149 cars carrying DPi, GTLM,
GTDPRO and GTP in one field. Two faults, and the second was not suspected:

1. **The queries were not scoped to one running.** `build_race_config` took an
   ILIKE pattern on `event`, and `event` carries a circuit and no edition. Six
   Daytonas and three Le Mans went into one config: durations added, counts
   summed, stints concatenated.
2. **`car` was read as an integer, so a leading zero was lost.** `#7` Toyota
   and `#007` Aston are both Hypercar at Le Mans; `#4` Corvette and `#04`
   CrowdStrike are both at Daytona. Collapsed onto one identifier, two cars'
   laps merged - which is why every edition reported 48 hours of running for a
   24-hour race, and why Le Mans 2026's Hypercar "winner" showed 62 stops.

The two compose exactly. Summed across the Le Mans editions as the old code
pooled them: 24.1 + 48.1 + 48.2 = 120.4 hours, against the 120.4 in
`wec.json`. Daytona reconciles the same way at 216.2. The near-integer
multiple of 24 hours that looked like a clean diagnosis was an integer number
of *car-races*, not of editions.

Three further defects were found that scoping does not fix, and all three are
corrected here: `stint_number` counts driver stints rather than fuel stints;
the pit column carries hour-long repairs beside 80-second stops within a
single race; and the caution flag test swept in the chequered lap."""))

    cells.append(md("""### Setup"""))

    cells.append(code('''import sys
from pathlib import Path


def find_project_root(marker: str = "src/endurance") -> Path:
    """Walk up from the working folder until the project appears."""
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"Could not find {marker!r} at or above {here}.")


ROOT = find_project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                          # noqa: E402
from endurance import calibrate                              # noqa: E402
from endurance import gate00                                 # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

LAPS_CSV    = ROOT / "data" / "raw" / "laps.csv"
DRIVERS_CSV = ROOT / "data" / "raw" / "drivers.csv"
PARAMS_DIR  = ROOT / "data" / "processed"
PARAMS_DIR.mkdir(parents=True, exist_ok=True)

# The two races the demonstration is about, and the class each is headlined
# by. The edition is resolved from the data in Part 1 rather than named here.
ANCHORS = {
    "imsa": {"event": "%daytona%", "headline_class": "GTP",      "name": "Daytona 24"},
    "wec":  {"event": "%le mans%", "headline_class": "HYPERCAR", "name": "Le Mans 24"},
}

con = calibrate.connect(str(LAPS_CSV), str(DRIVERS_CSV))
print(f"loaded {con.execute(\'SELECT COUNT(*) FROM laps\').fetchone()[0]:,} laps")'''))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 1 - which race, and which running of it

`event` names a circuit. The edition discriminator is `session_id`: 1013
values, none of which spans an event, a year, a series or a session type.
`(series, event, year)` is *not* sufficient - the Asian Le Mans double-headers
put two races under one key - which is why the scope is the session and not
the year.

The table below is the one the edition decision was taken on. Note what it
says about caution share: it moves by nearly a factor of two between adjacent
Daytonas, so a dial calibrated from one running is a sample of one. Freezing
the latest edition is the decision; showing the spread beside it is what stops
the number being read as a property of the race rather than of one race."""))

    cells.append(code('''RACES = {}
for series, anchor in ANCHORS.items():
    editions = calibrate.list_races(con, series, anchor["event"])
    RACES[series] = calibrate.find_race(con, series, anchor["event"])
    print(f"--- {series} {anchor[\'name\']}: {len(editions)} editions in file, "
          f"taking {RACES[series][\'label\']}")
    print(editions.to_string(index=False))'''))

    cells.append(code('''# How far a dial moves between runnings of the same race. The caution share
# is the one that moves most, and it is the one every strategy result is
# sensitive to.
spread = []
for series, anchor in ANCHORS.items():
    for _, ed in calibrate.list_races(con, series, anchor["event"]).iterrows():
        sid = int(ed["session_id"])
        try:
            c = calibrate.calibrate_cautions(con, sid)
        except Exception as exc:                       # a partial edition
            print(f"{series} {int(ed[\'year\'])}: skipped ({exc})")
            continue
        spread.append({"series": series, "year": int(ed["year"]),
                       "cars": int(ed["cars"]),
                       "duration_h": round(ed["duration_s"] / 3600, 2),
                       "caution_rate": round(c["caution_rate"], 3),
                       "caution_dur_s": round(c["caution_mean_dur_s"]),
                       "episodes": c["n_caution_episodes"],
                       "frozen": sid == RACES[series]["session_id"]})
spread = pd.DataFrame(spread)
spread'''))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 2 - the five dials

One query per dial per class, every number traceable to a named function in
`src/endurance/calibrate.py`. Three of the five changed in this re-run:

- **Stint length** comes from the pit records, not from `stint_number`. Every
  step of that counter coincides with a driver change and none with a stop -
  two to three fuel stints to the step - so reading it as a fuel stint
  reported 58-lap green stints at Daytona where the winner averaged 23.
- **Pit cost** is a median, under an unchanged field name. Within one scoped
  race the column still runs to 6,800 seconds against a median of 80, so the
  arithmetic mean sat two to four times its own median and the standard
  deviation sat above the mean. The spread is the standard deviation of the
  sample trimmed at three times the median.
- **Cautions** are calibrated once for the race rather than once per class.
  They are a property of the race; calling the function inside the class loop
  is what left seven classes of one race carrying seven different episode
  lengths."""))

    cells.append(code('''configs = {}
for series, anchor in ANCHORS.items():
    race = RACES[series]
    configs[series] = calibrate.build_race_config(
        con, series, race["session_id"], f"{anchor[\'name\']} {race[\'year\']}")
    cfg = configs[series]
    print(f"{series}: {len(cfg.classes)} classes, {cfg.total_cars} cars, "
          f"{cfg.duration_s / 3600:.2f} h - {cfg.classes[0].source_event}")

pd.concat([calibrate.dials_table(cfg) for cfg in configs.values()],
          ignore_index=True)'''))

    cells.append(md("""### What is measured, and what is assumed

Unchanged in this re-run, and printed here so nothing gets quietly promoted
from guess to fact. `pit_transit_frac` is on the assumed list and Part 5
measures a candidate for it without moving it."""))

    cells.append(code('''from endurance import ClassDials                             # noqa: E402

sample = configs["imsa"].classes[0]
pd.DataFrame(
    [{"field": f, "value": round(getattr(sample, f), 4), "status": "measured"}
     for f in sample.measured_fields()]
    + [{"field": f, "value": getattr(sample, f), "status": "ASSUMED"}
       for f in ClassDials.assumed_fields()])'''))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 3 - degradation, and what this data can support

Degradation is fitted jointly against tyre age and laps since the last fill.
Within a stint the car gets lighter as the tyres get older and the two effects
have opposite signs, so a slope on tyre age alone is their sum. The
field-relative frame does not rescue it: cars in a class stagger their stops,
so at a given lap number they are at different points of the tank and fuel is
not common-mode.

Where a class changes tyres at every stop the two regressors are the same
number and nothing can separate them. That is reported rather than resolved:
`identified` is the column to read before the slope."""))

    cells.append(code('''for series, cfg in configs.items():
    print(f"--- {series}")
    print(calibrate.degradation_table(
        con, RACES[series]["session_id"],
        [c.class_name for c in cfg.classes]).to_string(index=False))'''))

    cells.append(md("""**Read the `identified` column first.** A negative slope
in a class where it is false is not a defect to be tuned away; it is the net
within-stint pace trend, which is what the engine will reproduce, and it says
that this file cannot separate tyre wear from fuel burn for that class. A
negative slope where `identified` is true would be a second defect and gate
condition four would be right to fail on it."""))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 4 - the caution units, old and new

Carried forward from 02a. The share and the episode length are measured in
seconds of race time rather than laps, and the observed caution pace
multiplier is reported beside the assumed one - a measured counterpart to an
assumed dial, shown rather than silently substituted."""))

    cells.append(code('''for series, cfg in configs.items():
    print(f"--- {series}")
    print(calibrate.caution_report(
        con, RACES[series]["session_id"],
        cfg.classes[0].base_pace_s).to_string(index=False))'''))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 5 - `pit_transit_frac`, measured

03b established that 02c's `splash_and_dash` result depends on this dial:
gained falls monotonically as it runs 0.25 to 0.65 while no other roster row
moves monotonically. It is in `ASSUMED_FIELDS` at 0.25 and had never been
swept when 02c relied on it.

A low quantile of `pit_time` for one properly scoped race is a stop with
almost no service in it, which is the lane transit delta itself. Two things
had to be true for that to be worth measuring, and both now are: the column
had to be lane-to-lane time lost rather than stationary time - the recon put
the ratio of `pit_time` to the lap's excess over a green lap at 1.01 to 1.16
at the lower quartile - and the outliers had to be gone.

The regulations supply a cross-check and not the answer. The pit lane speed
limit is 60 km/h in both series (IMSA art. 32.3, WEC art. 12.1.4) and neither
rulebook carries the lane length, which is a circuit fact.

**This does not move the dial.** Promoting a field out of `ASSUMED_FIELDS` is
a blueprint amendment and a separate decision; what follows is the evidence
for taking it."""))

    cells.append(code('''rows = []
for series, cfg in configs.items():
    for c in cfg.classes:
        pit = calibrate.calibrate_pit(con, RACES[series]["session_id"], c.class_name)
        rows.append({
            "series": series, "class": c.class_name,
            "median_stop_s": round(pit["pit_time_mean_s"], 1),
            "raw_mean_s": round(pit["pit_time_raw_mean_s"], 1),
            "trimmed_out": pit["n_pit_stops_trimmed_out"],
            "n": pit["n_pit_stops"],
            "transit_s_p05": round(pit["pit_lane_transit_s"], 1),
            "implied_frac": round(pit["pit_lane_transit_s"] / pit["pit_time_mean_s"], 3),
            "assumed_frac": c.pit_transit_frac})
pd.DataFrame(rows)'''))

    cells.append(md("""`raw_mean_s` beside `median_stop_s` is the argument for
decision 6 in one column: the mean is what the old dial reported and it is not
a stop anyone made.

`implied_frac` against `assumed_frac` is the finding. Where it lands well
above 0.25, 02c's `splash_and_dash` result was computed at the wrong end of
the dial it is most sensitive to, and 03b's sweep already tells us which
direction that moves it."""))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 6 - the verification gate

Four conditions. Condition three is the falsifier: it widens the scope to two
adjacent editions on purpose and **requires** conditions one and two to fail.
Adjacent rather than distant, because two runnings of the same race are
genuinely similar and are the hardest case for a gate to catch.

Two things the gate does not do, stated rather than discovered:

- **Grid size does not detect pooling.** Car numbers recur between editions -
  six Daytonas carry 91 numbers, not 360 - so the count grows by far less than
  the racing does. That check earns its place against the leading-zero
  collision and against a class list assembled from somewhere other than the
  scope. Duration and lap counts are what catch pooling.
- **Nor does the pit dial.** Since it became a median it is robust to pooling
  in the same way base pace always was: the edition with more stops carries
  the statistic. Both are better dials for it and neither is a detector."""))

    cells.append(code('''gates = {}
for series in ANCHORS:
    editions = calibrate.list_races(con, series, ANCHORS[series]["event"])
    others = [int(s) for s in editions["session_id"]
              if int(s) != RACES[series]["session_id"]]
    print(f"===== {series}: {RACES[series][\'label\']}, "
          f"falsifier pools with session {others[-1]}")
    gates[series] = gate00.run_gate(con, series, RACES[series]["session_id"],
                                    others[-1], cfg=configs[series])
    print()'''))

    cells.append(code('''# The falsifier's own detail: every one of these is required to fail.
for series, g in gates.items():
    detail = g[g["condition"].str.startswith("three: falsifier (")]
    print(f"--- {series}: {int((~detail[\'passed\']).sum())} of {len(detail)} "
          f"pooled checks failed, as required")
    print(detail[["check", "value", "threshold", "passed"]].to_string(index=False))
    print()'''))

    cells.append(code('''for series, g in gates.items():
    print(f"{series}: gate {\'PASSES\' if gate00.gate_passes(g) else \'FAILS\'}")'''))
    cells.append(md("""### Where the stint discrepancy lives

Condition two's stint rows fail in **opposite directions** in the two series -
IMSA long, WEC short - which rules out a single systematic cause. Three
quantities side by side: what the file's stints look like, what the dial took
from them, and what the engine then does with it.

`dial_green_stint_laps` above `file_green_max` would be a tank nobody ever
emptied, and the upper-quartile choice would be wrong for this purpose.
`sim_laps_per_stop` far from the dial puts the discrepancy in the engine's
stopping rule or in `fuel_per_lap_caution` - an assumed dial, never swept -
rather than in the calibration. Those have different owners, and the gate
should not be rewritten until it is known which one this is."""))

    cells.append(code('''for series, cfg in configs.items():
    print(f"--- {series}")
    print(gate00.stint_diagnostic(
        con, cfg, RACES[series]["session_id"]).to_string(index=False))
    print()'''))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 7 - freeze

`data/processed/{series}.json` is what every later stage reads. Writing it is
the last thing this notebook does, and it happens only if the gate passed -
a config that fails its own gate must not be able to reach 02b's bank or
03b's training run.

Everything downstream re-runs from here. The seed lists do not move -
`draw_seed_bank` draws from `draw_seed` alone - but the races they name do,
because a seed plus different dials is a different race."""))

    cells.append(code('''for series, cfg in configs.items():
    if not gate00.gate_passes(gates[series]):
        print(f"{series}: gate failed, NOT written")
        continue
    path = PARAMS_DIR / f"{series}.json"
    cfg.save(path)
    print(f"wrote {path}  ({cfg.classes[0].source_event})")'''))

    cells.append(md("""## Where this leaves us

**The stage.** Both configs describe one running of one race, scoped by
`session_id`, with `car` read as text so a leading zero is a different car.

**What must re-run, in order.** `scripts/freeze_assets.py --force`; both
policies retrained, which `PolicyCard.check` will insist on against the new
`dials_fingerprint`; 02c's roster table; 03b's tables; and 01 Part 6 and 02a
Part 6, outstanding since 02a and now runnable against real dials.

**What does not.** Nothing in 03b's findings: every number it produced is
labelled stand-in, and the degenerate reward, the `pit_transit_frac` exploit
and the verification gates are about the apparatus rather than about the
dials.

**What is still open.** Where `identified` is false in Part 3, this file
cannot separate tyre wear from fuel burn and the dial holds a net trend. That
is a documented limitation, not a defect, and it is the honest outcome the
re-run was allowed to have."""))

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


TARGETS = {
    "00": ("00_data_recon.ipynb", build_00),
    "01": ("01_race_engine.ipynb", build_01),
    "02a": ("02a_engine_corrections.ipynb", build_02a),
    "02b": ("02b_benchmark.ipynb", build_02b),
    "02c": ("02c_human_strategies.ipynb", build_02c),
    "03a": ("03a_gym_wrapper.ipynb", build_03a),
    "03b": ("03b_rl_training.ipynb", build_03b),
}


def main(argv):
    wanted = argv[1:] or list(TARGETS)
    unknown = [t for t in wanted if t not in TARGETS]
    if unknown:
        raise SystemExit(f"unknown target {unknown}; choose from {list(TARGETS)}")
    for target in wanted:
        name, builder = TARGETS[target]
        write(name, builder())


if __name__ == "__main__":
    main(sys.argv)
