# Handover — the stage 00 re-run, closed

**The dials are real.** `data/processed/imsa.json` and `wec.json` describe one
running of one race each: the 2026 Rolex 24 at Daytona (session 682, 60 cars,
four classes) and the 2026 24 Hours of Le Mans (session 1000, 62 cars, three
classes). Both passed a four-condition gate before they were allowed to write.
Every artefact downstream — seed banks, background fields, both policies, both
evaluations, both sweeps — carries a matching `dials_fingerprint`, and the
guard in `load_assets` now refuses to let that silently drift again.

04 may proceed. What it may and may not claim is in section 10.

Read `PROJECT_BLUEPRINT.md` first, then `handover_02_decision_record.md`, then
this. Amendments 11 to 16 in the decision record are this thread's.

---

## 1. What was wrong

The incoming handover's hypothesis — the queries pool editions — was right and
incomplete. There were **two independent faults and they composed**.

**One: the scope.** `build_race_config` selected with an ILIKE pattern on
`event`, which carries a circuit name and no edition. Six Daytonas and three Le
Mans went into one config: durations added, counts summed, per-stint sequences
concatenated.

**Two: `car` was parsed as an integer.** Both series field entries whose
numbers differ only by a leading zero, and the pair is sometimes **in the same
class** — `#7` Toyota and `#007` Aston are both Hypercar at Le Mans; `#21` and
`#021` are both GTD at Daytona. Collapsed onto one identifier, two cars' laps
merged. Every Daytona edition reported 48.0 hours of running for a 24-hour
race; Le Mans 2022, which had no LMGT3 and therefore no `#007`, reported 24.1.

**The arithmetic closes exactly.** Summed across the Le Mans editions as the
old code pooled them: 24.09 + 48.12 + 48.17 = **120.38 h**, against the 120.36
in the old `wec.json`. Daytona is four collided editions plus one clean:
4 × 48.04 + 24.02 = **216.2 h**, against 216.2.

A composite `(car, class)` key would not have fixed it. Three of the nine
collisions in the four races examined are inside one class.

### The claim that did not survive checking

The incoming handover read `duration_s / 86400 = 9.009` as nine runnings of the
Daytona 24 and called it "the whole diagnosis in one number". **The file holds
six seasons.** There are at most six Rolex 24s in it, and a car's summed lap
time telescopes to elapsed time, so no edition can contribute more than ~24
hours. Nine editions cannot be in a six-season file.

The near-integer multiple was real and meant something — an integer number of
*car-races*, where a collided number contributes two per edition — but it was
not the edition count, and reading it as one would have sent the fix at the
scope alone and left the collision in place.

**Recorded because the failure mode is a number that looks like a diagnosis.**

---

## 2. Three defects that scoping did not fix

Found by probing one correctly scoped race. All three are corrected.

**`stint_number` counts driver stints, not fuel stints.** Every step of that
counter coincides with a driver change and none with a stop — 1.9 to 3.1 stops
per step across the four races probed. Reading it as a fuel stint gave a q75 of
58 green laps for GTP at Daytona 2025, where the winner did 781 laps in 34
stops, or 23 a stint. **Two and a half times too long on one correctly scoped
race.** Stints now come from the pit records.

**The pit column keeps its outliers inside one race.** Daytona 2025 GTD: mean
131.9 s, sd 313.5, max 3,626. Le Mans 2026 Hypercar: mean 104.8, sd 322.2, max
6,856. Medians are 77–92 s everywhere. An arithmetic mean over that column
describes no stop anyone made.

**Degradation at Le Mans is not identifiable.** `max_tire_age` is 10 for
Hypercar and 9 for LMP2 against fuel stints of 11–12 laps, so tyres change at
every stop and tyre age is the same number as laps-since-fill. LMGT3, which
triple-stints tyres, comes out positive at +0.032. The proposed fix — a
per-car-per-stint intercept — does not work, and the reason is instructive:
demeaning removes a level, not a trend, and fuel burn-off is a trend.

---

## 3. The largest find: the calibration was nondeterministic

Three different values for the same quantity, from **one notebook run**:

| where | WEC caution rate | mean episode | episodes |
|---|---|---|---|
| Part 1, per-edition spread | 0.082 | 1,181 s | 6 |
| Part 2, `dials_table` | 0.080 | 1,718 s | 4 |
| the file it then wrote | 0.0883 | 1,271 s | — |

`calibrate_cautions` chose its reference car with
`GROUP BY car ORDER BY COUNT(*) DESC LIMIT 1`. At Le Mans many cars finish on
the same lap count, `COUNT(*)` ties, and DuckDB's parallel scan broke the tie
differently each run. A different reference car means a different flag
sequence, a different caution share and a different episode segmentation.

This is why WEC passed through three fingerprints — `262f31d8`, `8a147c1a`,
`5af03450` — while nobody could say what had changed, and why IMSA later
alternated between `76bac232` and `7792d2ed`. **Every result this project
produced before the fix was computed against dials that were an arbitrary draw
from a distribution nobody knew existed.** Not wrong; unrepeatable.

**Fixed** by a deterministic tie-break, `ORDER BY COUNT(*) DESC, car ASC`.
Verified across five separate processes: identical reference car, identical
rate to ten decimal places, identical episode count.

---

## 4. Decisions taken

1. **`session_id` is the scope key, and the pattern path is gone.** 1013 ids,
   none spanning an event, year, series or session type. `(series, event,
   year)` is not sufficient — the Asian Le Mans double-headers put two races
   under one key. Removed rather than deprecated: a reachable pooling path is
   how this survived two stages. `find_race` resolves the latest edition and
   raises when the answer is not unique.
2. **`car` is read as VARCHAR**, in `connect` and in `sqlite_shim` both. If the
   two disagree, the tests pass against a car identity the production path does
   not have, which is this fault's exact shape.
3. **`source_event` carries the edition** — `"Daytona 2026 (imsa session
   682)"`. A value change on an existing field, so no schema moves.
4. **`duration_s` is measured elapsed session time**, summed per session so a
   widened scope reports the racing it contains and the falsifier fails on
   duration as it should.
5. **Stints come from pit records**, with each car's last stint dropped as
   truncated by the flag.
6. **`pit_time_mean_s` holds a median**, field name unchanged; `pit_time_std_s`
   is the sd of the sample trimmed at 3× that median.
7. **Caution is the named set `{FCY, SF, RF}`.** `FF` and nulls leave both
   sides. Cautions are calibrated **once per race**, not once per class.
8. **Degradation is fitted jointly** on tyre age and laps-since-fill, with
   `deg_identified` reported beside the slope.
9. **The reference car is chosen deterministically.** Added after the fact; see
   section 3.

**Editions frozen: the latest of each.** Le Mans had no real alternative —
only 2022, 2025 and 2026 are in the file and 2022 is partial at 32 cars.
Daytona 2026 is structurally identical to 2025 at four classes and 60 cars.

---

## 5. The verification gate

00 was specified with a boundary constraint and no gate — the pattern 02c and
03b both had to fix. Four conditions, in `src/endurance/gate00.py`, shown in
notebook 00 Part 6 and in `tests/test_calibrate.py`.

1. **The race is one race.** Duration within 2% of 24 h; the sum of per-class
   car counts equals the number of distinct numbers, which holds for a single
   grid and breaks under pooling; every class competed; no car's laps exceed
   what the clock allowed.
2. **The dials describe the race.** The stint dial lies between the median and
   the longest stint the cars completed; the pit dial lies inside the
   interquartile range of observed green stops; `pit_time_std_s` is no greater
   than its own mean; the simulated winner beats the real winner by between 0
   and 10%.
3. **The falsifier.** Pool the frozen edition with the adjacent one and require
   one and two to fail. Adjacent rather than distant, because two runnings of
   the same race are the hardest case to catch.
4. **Degradation sign, where the sign means anything.** Tested only where
   `deg_identified` is true; unidentified classes are exempt and **named**, in
   the gate table and in Part 3.

### Condition two was dropped and replaced, not weakened

The incoming specification asked for simulated stint length and mean pit time
within 10% of observed. Neither can be met, and the diagnostic located why —
twice, in opposite directions, neither of them a dial:

- At Le Mans the simulated stint is 1.4 laps short of the tank in all three
  classes. That is `RunToFuelWindow` stopping a lap and a half early, which
  03a recorded.
- At Daytona the file's cars average 26.4 laps between stops against a 30-lap
  tank, *despite* a caution share that should stretch stints by 11%. Real cars
  take fuel under yellow with half a tank; the default field never does. The
  effect scales with caution rate, which is why the two series miss in opposite
  directions.

Both are the engine and the roster consuming the dials, and the stage boundary
is explicit that a dial the engine misuses is 02a's problem. Per 03a's
precedent, a gate that cannot be met is dropped and replaced. See amendment 12.

### Two things the gate does not detect, stated rather than discovered

**Grid size does not detect pooling.** Car numbers recur between editions — six
Daytonas carry 91 numbers, not 360 — so the count grows far less than the
racing does. That check earns its place against the leading-zero collision and
against a class list assembled from outside the scope.

**Nor does the pit dial.** Making it a median moved it into the same robust set
as base pace: pooling two WEC fixture editions moved it from 58.11 to 59.39
against a planted 58.0. Only quantities that count, sum or sequence detect
pooling.

---

## 6. Known limitations

Five, all live, none blocking.

**The frozen dials are a pre-fix draw.** Notebook 00 was last run in a Jupyter
kernel started before the determinism fix landed, so `imsa.json` holds
`caution_rate = 0.3528007170810149` where the fixed code returns
`0.3552535212`. The difference is 0.7% on one dial. Everything on disk is
internally consistent and the fingerprints chain, so nothing is invalid — but
re-running 00 with a fresh kernel will move both fingerprints. **Do that at the
start of the next full pass, not on its own.**

**The caution dials are measured from one car.** Two cars that both completed
Daytona 2026 disagree by 0.6% on caution share, because one was in the pits
when a yellow started. The tie-break makes the choice stable; it does not make
it a fact. The durable fix is to measure the share from the field — caution
lap-time over total lap-time across all cars, with episode structure from the
per-lap majority flag. Deterministic by construction, uses 36,756 lap records
instead of about 660, and removes the question entirely. Deferred because it
moves the dials again for 0.6% on a quantity whose edition-to-edition spread is
180%.

**Le Mans degradation is a net trend, not tyre wear**, for Hypercar and LMP2.
See section 2. This is a statement about what the data can support.

**Daytona 2026 is a caution outlier.**

```
 year  caution_rate  caution_dur_s  episodes
 2021         0.125            899        12
 2022         0.228           1155        17
 2023         0.160            988        14
 2024         0.154            886        15
 2025         0.179           1101        14
 2026         0.353           3388         9   <- frozen
```

Red-flag laps excluded: zero, so the 56-minute mean episode is real — nine long
full-course yellows rather than fourteen short ones. The frozen race has double
the caution share of any other edition, and `caution_rate` is the dial every
roster strategy is most sensitive to. This is not a reason to pick a different
edition; cherry-picking a median race would be worse. It is a reason to sweep
`caution_rate` across the observed 0.125–0.353 range wherever a roster result
is shown.

**Two assumed dials are measurably wrong.** Caution pace multiplier: assumed
1.6, observed **1.87** (IMSA) and **1.79** (WEC). `pit_transit_frac`: assumed
0.25, implied **0.51–0.66** in IMSA. The WEC figures of 0.87–0.92 are not
transit — LMP2's p05 of 81.8 s against a median of 88.6 is a tight distribution
with no service-free stops in it — so the measurement is an upper bound and the
honest conclusion is that 0.25 is too low, not that 0.51 is right. Note also
that `pit_transit_frac + pit_tyre_frac` must not exceed 1, and GTP's 0.66 +
0.35 already breaks it. See amendment 15.

---

## 7. State on disk

| artefact | IMSA | WEC |
|---|---|---|
| `data/processed/{series}.json` | `76bac232842e2d4f` | `5af034500e722532` |
| seed bank, background field | matching | matching |
| policy card | matching | matching |
| headline and held-out evaluation | matching | matching |
| `pit_transit_frac` sweep | matching | matching |

Notebooks 00, 01, 02a, 02b, 02c and 03b executed against these. Test suite: 212
passing.

**Two housekeeping items.** `data/processed/seed_bank_imsa.json` and
`background_field_imsa.json` are strays written by 02b under a third
fingerprint; nothing reads them and they should be deleted. And 03b's notebook
code still differs from `build_nb.py` by the stand-in guard, so its results are
not reproducible from source until that block is folded into `build_03b` — no
re-run required, the notebook already has the code.

**There is still no version control.** Every provenance question asked during
this thread — did the patch land, is this notebook current, when did the dials
change — is `git log` in a repository. It is five minutes and it should happen
before 04.

---

## 8. The agent, and the one open decision

```
                  gained  level   lost  median_d_pos
imsa  agent        0.000  0.050  0.950          -6.0
      caution_gambler 0.500                      0.5
      lap_down        0.395                      0.0
      splash_and_dash 0.390                      0.0
      track_position  0.375                      0.0
wec   agent        0.005  0.005  0.990         -12.0
      splash_and_dash 0.525                      1.0
      caution_gambler 0.365                      0.0
      track_position  0.310                      0.0
      lap_down        0.095                      0.0
```

**Both agents are at the random floor** and are beaten by every human strategy
in both series. This supersedes an earlier claim that WEC's agent gained
62–70%; that figure came from a policy trained against a superseded config and
did not survive refreezing.

**The leading explanation is a stop-cost floor violation, and it is open.** At
Daytona the lane transit alone is `89.81 × 0.25 = 22.45 s`, a top-up costs
24.70 s, a full service 89.81 s. **The agent's realised cost was 12.85 s a
stop**, and it stops on 85.8% of decisions — about 0.9 stops a lap. A stop
cannot cost less than driving the length of the pit lane. The likely route is
`pit_caution_discount` at 0.4 applied to the whole stop including transit, with
35.8% of the agent's stops taken under caution.

**This is reward hacking with a diagnosed mechanism, not a demonstration that
RL fails at endurance strategy.** The policy is behaving correctly given a cost
function that is wrong. The distinction matters for what 04 claims.

The fix belongs in `pitstop.py` — `stop_cost` should never return less than
`pit_time_mean_s × pit_transit_frac` at any flag — which is 02a's territory and
requires retraining both policies. **Deferred deliberately**, on runtime
grounds. See section 9.

---

## 9. The runtime constraint, and what to do about it

A full pass — 00, refreeze, train both, evaluate, sweep, then 02b, 02c, 03b —
takes about five hours, of which 02c alone is 2 h 52 m. That has been the
binding constraint on this project all week and it deserves recording as one.

**It is almost certainly unnecessary.** One race is 0.72 s and the machine has
8 cores. 02c's work is roughly 200 seeds × 5 strategies × 2 series plus the
dial sweeps — order a thousand races, or about 12 minutes of compute, run
serially on one core. The `fuel_window` null is also re-simulated for every
strategy and every dial setting when it need only be computed once per seed.

**Parallelising over seeds and caching the null is the highest-value work
available in this project right now.** It is an afternoon, it needs no
recalibration, and it converts every future pass from an evening into an hour.
Everything deferred above becomes cheap the moment it lands.

Recommended order for the next pass, once 02c is fast:

1. Parallelise 02c and cache the null. Verify against the current table — same
   seeds, same dials, same numbers.
2. Fix the `stop_cost` floor in `pitstop.py`, with a test.
3. Fresh kernel, re-run 00, take the post-fix dials.
4. Refreeze, retrain, re-evaluate, re-sweep, re-run the notebooks.

Steps 2 and 3 together answer the only real question left about the agent.

---

## 10. What 04 inherits

**Calibrated dials for two real races**, each self-describing through
`source_event`, each having passed a gate that can fail and that fails on the
defect it was written for. `scale_dials` now drives numbers that mean
something.

**A working roster comparison**: five parameter-free strategies with paired
deltas on 200 seeds per series, plus an invariance analysis showing which
results survive a dial being wrong.

**Four things 04 must put on the page rather than in a comment.**

The dials come from **one running**, and the caution share moves by a factor of
2.8 between adjacent Daytonas. A reader twisting that slider should know the
starting point is a sample of one.

**Hypercar and LMP2 degradation at Le Mans is a net within-stint trend**, not
tyre wear, because this data cannot separate the two for a class that changes
tyres at every stop.

**The agent is at the random floor**, and the honest framing is the diagnosed
one: it found a stop priced below its own floor and exploited it. That is a
more interesting result than a policy that quietly wins, because a reader can
check it — and it is provisional until the floor is fixed.

**No benchmark row.** 02b now runs on real dials but reports zero foreknowledge
on every seed, and its benchmark is beaten by its own forced-only control. 04
shows the roster and the agent; 02b is a documented gap.

### Still open, inherited rather than caused

02c's `gained_02c` literals in `build_nb.py` are stale, and 03b cell 12 reports
two of them as unexplained discrepancies. They should be updated from the
current 02c before that cell is read as a finding.

---

## 11. Corrections to prior documents, including my own

Recorded because this project's characteristic failure has been a document
asserting something nobody checked.

- **"Nine editions"** was arithmetically impossible; see section 1.
- **The first nondeterminism diagnosis** blamed tie-ordering in `ORDER BY lap`
  and was retracted after a five-call test came back stable. The retraction was
  wrong: the test ran in one process and happened to land on IMSA. The real
  mechanism was the reference-car tie-break, found four days later.
- **"IMSA is stable"** was luck, not determinism. It alternated too.
- **A stale uploaded file** was read as the current state of the tree and
  reported as a missing deliverable. The verification report was right and was
  overridden by an artefact from two days earlier.
- **The claim that WEC's agent gains 62–70%** came from a superseded config.
- **02b had been reading `dials_imsa.json`**, a filename that never existed,
  and silently falling back to a stand-in. It said so honestly at the top of
  every run, which is why nobody noticed.
