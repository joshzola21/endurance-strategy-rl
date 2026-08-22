# Amendments 28 to 31 — for `handover_02_decision_record.md`

05's amendments. Numbering continues from 27; the post-04 pass closed at 27
with amendment 14 retired.

Amendment 25 does not reappear here. `never_pit` was already decided; 05 only
built it, which is recorded under 28's note rather than as a decision of its
own.

---

## 28 — `set_dials` beside `scale_dials`

**The decision.** `params.py` gains `set_dials(config, **values)`, which writes
a value onto a dial rather than multiplying what is there. Both go through one
private `_adjusted`, so there remains a single place a twisted copy of the
dials is made. `harness.sweep_dial` takes `multipliers` or `values` and refuses
both or neither; `sweep_grid` takes `how="scale"` or `how="set"`, defaulting to
the behaviour every published sweep used.

**Why.** Amendment 23 separated `pit_transit_caution_discount` from
`pit_caution_discount` and said both were sweepable. They were not. The new
dial defaults to 0.0 and every lever in this project is a multiplier, so no
setting of any slider could move it and no sweep could reach it. The amendment
had been recorded and could not be exercised.

**The refusal that came with it.** Setting writes the *same* number onto every
class, while scaling moves each class's own value. That is fine for a dial the
classes share and wrong for one they do not — `base_pace_s` differs by fifteen
seconds between GTP and GTD, and flattening it would be a different race
presented as a lever. `set_dials` refuses when the classes disagree and names
the dial, rather than reporting it afterwards.

**Consequence.** `viz.plot_sweep_response` now picks its x axis from the frame.
A sweep by value carries no multiplier — `sweep_dial` leaves the column empty
rather than back-computing one from a zero default, which would invent a
number — and the "calibrated" marker at 1.0 is meaningless on a value axis.

**Built alongside:** `never_pit` as `ROSTER`'s sixth member, per amendment 25.
It is exempted by name from `test_strategies.py`'s no-forced-stops condition,
and the exemption is the point of it: it runs itself dry every stint by
construction, and a control that had to be a competent strategy would not be a
control. A second test asserts the other half — every stop it takes carries one
of `_must_pit`'s reasons and never a reason of its own.

---

## 29 — `rules_fingerprint`, beside `dials_fingerprint` and not inside it

**The decision.** `pitstop.py` and `caution.py` carry a `RULES_VERSION`.
`assets.rules_fingerprint()` hashes them; `assets.rules_mismatch(recorded)`
compares and returns `None` when nothing was recorded. New banks and fields
stamp it. It is **checked only where present**.

**Why not inside `dials_fingerprint`.** That is the stronger design and it was
rejected on cost. `dials_fingerprint` hashes `config.to_dict()`, so adding
anything to that payload changes the hash — and every seed bank, background
field, policy card and saved evaluation table in the project would stop
matching, and the app would refuse to start. Amendment 21's exposure is real
and it is not worth invalidating the whole tree immediately before packaging.
Beside rather than inside closes the gap going forward and breaks nothing.

**What `None` means.** "Frozen before the check existed", which is the state of
every artefact the project had when this landed. Not a pass. It is reported as
a gap by `scripts/check_artefacts.py` and stated on the methods page. A strict
check would have refused the entire project on its first run.

**What it does not cover: `engine.py`.** Including it would mean bumping a
version on every edit to the largest and most frequently touched module in the
project, and a version nobody bumps honestly is worse than an absent one. The
rules layers are covered, the engine is not, and this is written down rather
than left to be discovered.

---

## 30 — 05's verification gate, supplied

**The decision.** 05 is specified with a boundary constraint, a deliverable
list and no gate — the sixth stage in a row, after 03a, 03b, 00, 04 and the
post-04 pass. Two conditions, each with a falsifier, per 02c's standing rule
that a stage specified without a gate is a gate nobody can fail.

**Condition one — the committed tree is sufficient and about one race.**
`scripts/check_artefacts.py`. Every artefact the app and the scripts open is
present; every one names the same race; policy cards match their dials, their
bank and their files' hashes; the saved tables carry the sixth roster member;
no artefact is orphaned from every config in the tree.

*Falsifier:* remove an artefact, or introduce a second file under one name, and
the check must fail. It did, three times, on its first three runs — a renamed
banked config, a duplicated policy card, and a card whose `dials_source` was a
literal from a superseded `train.py`.

**Condition two — every number in the write-up traces to a file on disk.**
Outstanding: it needs the write-up to exist. Figures in the documents are
tagged with the artefact they come from and checked against it; changing one
digit must fail the check.

**What the gate caught that a reader would not have.** The `.onnx` and the
`.zip` have different consequences when missing — one stops the app, the other
stops reproduction — and a tree can host perfectly while being unable to
reproduce a single published number. It reports them separately for that
reason.

---

## 31 — `pit_transit_frac` has no measured counterpart, and cannot have one

**The decision.** The note claiming a low quantile of real pit times implies
0.51–0.66 is **withdrawn**. `app/loading.MEASURED_COUNTERPART` now records that
the quantity is not measurable from lap timing. The dial stays at its assumed
0.25 and is swept, which is what amendment 15's rule and decision 2 required
all along — now with a reason rather than a shrug.

**Why, in three parts, each checkable.** `scripts/estimate_pit_transit.py`
records all three.

1. **The quantile is biased, and the bias is measurable.** Against a planted
   truth of 0.407 the fifth percentile returns 0.548 — out by +0.14, and in the
   direction that matters: the cheapest stops in any real sample still have
   service in them, so pricing the lane off them overstates it.
2. **It is not stable across samples.** Across the seven classes of these two
   races the same statistic gives 0.50, 0.50, 0.52, 0.73, 0.88, 0.90 and 0.93.
   A measurement that ranges from a half to almost all of a stop depending on
   which class is picked is not measuring a property of a pit lane.
3. **The regression that would settle it has no leverage.** Separating what a
   stop costs to *enter* from what it costs to *fill* requires observing stops
   that took different amounts of fuel. Between 62% and 90% of stops in these
   two races are followed by a near-full tank; the median stop is a brim-full
   tank in all seven classes. Endurance racing does not produce the variation
   the question needs.

**Two results withdrawn on the way, and the second is the instructive one.**
The estimator first reported a transit of 0.95 of a stop for IMSA GTD. That was
an artefact: with a proxy carrying no signal the fit is a flat line, and a flat
line hands its whole budget to the intercept, which then *is* the mean stop
cost. It now refuses on the slope's t statistic rather than on the signs of the
coefficients.

It then reported floors of 0.84 to 0.98 — "at least this much of a stop is
fixed" — from the top of the slope's interval times a tankful. Also withdrawn.
That bound assumed a short stint means a small fill, and it does not: a stint
ends early for a small fill, a caution, or the flag, and nothing in the lap
data says which. The bound machinery is removed rather than adjusted.

**The general rule this yields.** *A refusal that offers a number anyway is a
refusal nobody heeds.* Where the script cannot identify a quantity it reports
the interval over the range the data actually spans, and nothing else.

**What this does to the write-up.** It adds a fifth thing the project can say
honestly, and it is about the data rather than the reward: an assumed dial
whose measured counterpart does not exist and cannot be made to exist from lap
timing. It also sharpens the sample-of-one statement — it is not only that
these dials come from one running of one race, but that one of them could not
be measured from a thousand runnings either.

**One thing it does not settle.** Whether `stop_cost`'s *shape* can express a
real pit lane. A full service is anchored at the measured mean and split into
shares, so raising the transit share lowers the refuel share and a full stop
always costs the same; only the ratio moves.
`scripts/sweep_pit_transit.py`'s docstring has warned since 03b that this may
be the finding rather than any dial's value. It remains open, and it is now
open for a stated reason rather than as a worry.

---

## Files

Changed: `src/endurance/{params,pitstop,caution,assets,harness,strategies,viz,
__init__}.py`, `app/{loading,controller,panels,statements,Home}.py`,
`app/pages/{1_Race,2_Comparison,3_Methods}.py`,
`scripts/{freeze_assets,train,evaluate,sweep_pit_transit}.py`,
`tests/{test_app,test_gym_env,test_strategies,run_tests_nopytest}.py`.

New: `scripts/{check_artefacts,estimate_pit_transit}.py`,
`tests/test_05_changes.py`, `pyproject.toml`, `requirements-dev.txt`,
`.gitignore`, `.streamlit/config.toml`, `README.md`, `docs/ARCHITECTURE.md`.

Renamed: `data/processed/{code}_standin.json` → `{code}_banked.json`. The old
name is still read so an unmigrated tree loads; nothing writes it.

Deleted: `data/processed/{seed_bank_imsa,background_field_imsa}.json` — a
three-hour IMSA race whose config was not in the tree, amendment 19's live
instance, which the app had been avoiding by the order of a tuple rather than
by a check.

Three stale tests retired: `test_gym_env.py` still asserted the lap reward and
a nine-row observation, both superseded at amendment 24. The replacements say
what they supersede, because that file has now outlived two rewards and the
second superseded test was left in place for a whole pass while quietly
failing.
