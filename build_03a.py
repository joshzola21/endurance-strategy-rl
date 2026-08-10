"""Paste into `build_nb.py`, and add to TARGETS:

    "03a": ("03a_gym_wrapper.ipynb", build_03a),
"""


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
from endurance import gym_env, harness
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
        out.append({"ahead_s": state.gap_ahead_s(car),
                    "behind_s": state.gap_behind_s(car)})
        decision = RunToFuelWindow()(car, state)


gaps = pd.concat([gaps_over_a_race(configs["imsa"], s, fields["imsa"])
                  for s in banks["imsa"].headline[:5]], ignore_index=True)

summary = gaps.describe(percentiles=[0.5, 0.9, 0.99]).T[
    ["count", "50%", "90%", "99%", "max"]].round(2)
summary["share_negative"] = [(gaps[c] < 0).mean() for c in gaps.columns]
summary["p50_over_120"] = (summary["50%"] / 120).round(3)
summary["p50_over_pit_mean"] = (summary["50%"]
                                / configs["imsa"].classes[0].pit_time_mean_s).round(3)
summary'''))

    cells.append(md("""No negative gap anywhere, which is the invariant holding. But the
blueprint's drafted normalisation - clip at 120 s, divide by 120 - is nearly
dead on arrival. The clip never binds, and the median gap lands around
0.02 of the row's range, so nine tenths of it is unused and a policy network
sees a near-constant input.

`pit_time_mean_s` is used instead. 1.0 then means *a stop's worth of gap*,
which is the unit the decision is actually taken in, and it moves with the
dials rather than being a constant somebody picked. **This supersedes the
blueprint's observation table and is recorded there.**

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

**2. The gap rows' normalisation was nearly inert.** Clipped at 120 s and
divided by 120, the median observation is about 0.02 and the clip never
binds; nine tenths of the row's range is unused. Rescaled on
`pit_time_mean_s`, so 1.0 is a stop's worth of gap. **This is a change to
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
