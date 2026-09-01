# Handover — 05, the packaging pass: it shipped, and four things it found on the way

**The app is live at `endurance-strategy-rl.streamlit.app`.** The tree passes
its own inventory check, 266 tests pass, and both halves of 05's verification
gate are green. That is the stage's deliverable and it is done.

**Read the four findings before quoting anything about pit stops.** This pass
produced two figures for `pit_transit_frac` and withdrew both. What survives is
better and is stated in section 5: that dial cannot be measured from lap
timing, and the reason is a property of endurance racing rather than of the
method.

Order of reading: `PROJECT_BLUEPRINT.md`, `handover_02_decision_record.md`
(amendments 28-31 are this pass's), `handover_00_re_run.md`,
`handover_04_app.md`, the post-04 pass handover, then this. Amendments 28 to 31
are listed in section 9 and written out in `docs/amendments_28_to_31.md`.

---

## 1. What this pass did

05 was specified as *hosted app and documentation* with the boundary constraint
*polish and deployment*, and it turned out to be four different jobs:

1. **Build what earlier amendments had recorded and not built.** `never_pit`
   was decided at amendment 25 and did not exist. The dial amendment 23 added
   could not be swept by any mechanism in the project.
2. **Close the carried-over defects.** All four: the false `dials_source`
   literal, `RULES_VERSION` for `caution.py`, `build_nb.py`'s stale
   `gained_02c` literals, and the gates printing nothing.
3. **Package and deploy.** `pyproject.toml`, a deployment requirements set
   distinct from the development one, and a committed artefact set.
4. **Write the thing.** `README.md`, `docs/ARCHITECTURE.md`,
   `docs/WRITE_UP.md`.

**Dials unchanged:** IMSA `54de878a6d26afbf`, WEC `5cb7cf63e4446105`. Banks
unchanged: `d9b110e4014ad7f4`, `fb5c4439f77c420d`. **New:** a rules fingerprint,
`f3588be9ac572351`, now stamped on both banks and both fields.

---

## 2. The roster on real dials, with the control in it

`scripts/evaluate.py` re-run on both series and both banks, with `never_pit` as
a sixth roster member. The tables in `outputs/evaluation/` are these.

**IMSA, headline, 200 seeds.** `caution_gambler` gains a place in **0.580** of
races against a **0.270** loss share, median position delta **+1.0 [1.0, 2.0]**.
It holds on the held-out fifty at **0.60** — but there the median's interval is
**[0.0, 2.0]** and includes zero. **The strictly-positive-median claim is a
headline-bank claim.** The held-out row supports the share and not the median,
and the write-up says it that way.

**WEC, headline.** `splash_and_dash` gains in **0.450** and loses in **0.010**,
which is a different shape of result from the gambler's and worth reporting as
such. `lap_down` gains in **0.050**: the wave-around clause is live in IMSA and
nearly dead in WEC, exactly as decision 10's per-series budget anticipated.

**And the finding amendment 25 was added for.** On IMSA's held-out fifty the
trained policy and `never_pit` are **identical to three decimal places on every
column** — 0.32 gained, 0.14 level, 0.54 lost, median -1.0, same interval. The
policy has learned to take no decisions, and the roster now shows it as an
ordinary row rather than requiring somebody to suspect it and go looking.

`scripts/evidence.py` computes that gap and reports it, so it does not depend on
a reader noticing two adjacent rows.

---

## 3. Four things this pass found, none of which it went looking for

**A three-hour IMSA race was still in `data/processed/`.** `seed_bank_imsa.json`
and `background_field_imsa.json`, fingerprint `420186160d860b42`,
`duration_s: 10800`, with no config in the tree matching. Amendment 19's live
instance, still there. `app/loading.py` accepted both naming conventions and was
avoiding it **only because `{code}_seeds.json` sorted first in a tuple** - and
only for IMSA, since no WEC pair was ever written. Deleted, and the older
convention is no longer read at all.

**Two generations of policy card were being crossed.** `check_artefacts.py`
reported both series' policies as trained on the wrong race with hashes that
did not match. They were not: a second `{code}_maskable_ppo.card.json` existed
elsewhere in the tree, `rglob` took whichever sorted first, and a card from one
generation was being compared against weights from another. **Every script in
`scripts/` resolves artefacts this way.** The checker now refuses an ambiguous
name outright and requires a policy's three files to share a directory.

**The IMSA checkpoint had gone missing without anything noticing.** The `.onnx`
and the card were in `outputs/policies/`; the `.zip` was not. The app was
unaffected - it reads the export - and `scripts/evaluate.py` could not run.
**A tree can host perfectly while being unable to reproduce a single published
number**, and the checker reports the two absences separately for that reason.
Recovered by retraining at seed 0, which reproduced the lost weights exactly:
the `.zip` hash moved and the `.onnx` hash did not, because an SB3 archive
embeds timestamps and an ONNX export has nowhere to put one.

**`test_app.py` could not be collected.** It is the only test file that imports
`app`, and the only one that never put anything on `sys.path`. It had been
working by accident, in trees where the working directory happened to be the
root. Its two subprocess gates had the same problem one layer down.

---

## 4. Three stale tests, and one that was stale in the other direction

`test_gym_env.py` still asserted a nine-row observation and the **lap reward**,
both superseded at amendment 24. The post-04 pass wrote
`tests/test_position_reward.py` **beside** them rather than **over** them, so
three tests had been failing quietly for a whole pass while a fourth asserted
the opposite.

That file's own docstrings record superseding an *earlier* reward test the same
way. It has now outlived two rewards. The replacements say what they supersede
and why, on the theory that the next reward change will look for exactly that.

`test_strategies.py` failed in the useful direction: it names the roster-only
members explicitly, so `never_pit` could not arrive quietly. Its
no-forced-stops condition now exempts `never_pit` **by name**, and the
exemption is the point - the control runs itself dry every stint by
construction, and a control that had to be a competent strategy would not be a
control.

---

## 5. `pit_transit_frac` cannot be measured, and that is the result

The dial deciding what share of a stop is driving down the lane rather than
being serviced. Assumed at 0.25. `app/loading.MEASURED_COUNTERPART` claimed a
low quantile of real pit times implies 0.51-0.66.

**That claim is withdrawn.** Three lines of evidence, all in
`scripts/estimate_pit_transit.py`:

1. **The quantile is biased and the bias is measurable.** Against a planted
   truth of 0.407 it returns 0.548 - out by +0.14, in the direction that
   matters: the cheapest stops in a real sample still have service in them.
2. **It is not stable.** Across the seven classes of these two races the same
   statistic gives 0.50, 0.50, 0.52, 0.73, 0.88, 0.90, 0.93.
3. **The regression that would settle it has no leverage.** Between **62% and
   90%** of stops in these two races are followed by a near-full tank, and the
   median stop is a brim-full tank in all seven classes. Separating what a stop
   costs to enter from what it costs to fill needs stops that took different
   amounts of fuel, and this sport barely produces any.

**Two figures were produced and withdrawn on the way, and the second is the
instructive one.**

The first reported a transit of 0.95 of a stop for IMSA GTD, "identified". An
artefact: a proxy carrying no signal gives a flat line, and a flat line hands
its whole budget to the intercept, which is then the mean stop cost. The
refusals checked the *signs* of coefficients and not whether the slope could be
told from zero.

The second turned the failed fit into a floor - "at least 0.84 of a stop is
fixed" - from the top of the slope's interval times a tankful. Also withdrawn.
It assumed a short stint means a small fill; a stint also ends early for a
caution or for the flag, and nothing in the data says which. **The bound
machinery was removed rather than adjusted**, because it had now produced two
numbers that had to be taken back.

**The rule that came out of it, and it generalises:** *a refusal that offers a
number anyway is a refusal nobody heeds.*

**What this does not settle.** Whether `stop_cost`'s *shape* can express a real
pit lane. A full service is anchored at the measured mean and split into
shares, so only the ratio moves. `scripts/sweep_pit_transit.py` has warned since
03b that this may be the finding rather than any dial's value. Still open, now
for a stated reason.

---

## 6. What 05 built

**`scripts/check_artefacts.py`** - gate condition one. Every artefact present,
naming one race, cards matching their dials, bank and file hashes, tables
carrying the sixth roster member, nothing orphaned, no ambiguous names. It
caught three real faults on its first three runs.

**`scripts/evidence.py`** - the carried-over "gates print nothing". Runs the
gates a committed tree can run, prints the quantity each turns on, writes 159
figures to `outputs/evidence.json`. It reports *not run* separately from
*broken*, because a gate that was not run is not a gate that passed. 00's gate
is the one it cannot run.

**`scripts/check_figures.py` and `docs/figures.json`** - gate condition two.
Every figure the documents quote, tied to the artefact it comes from and to the
exact text that must appear. Two failure modes: the artefact moved, or somebody
edited a number in the writing. `--self-test` plants both and requires each to
be caught. It found four wording mismatches across the three documents before
it was even committed.

**`scripts/estimate_pit_transit.py`** - section 5.

Plus `pyproject.toml`, a deployment requirements set that excludes torch,
Stable-Baselines3, DuckDB and Jupyter, `README.md`, `docs/ARCHITECTURE.md` and
`docs/WRITE_UP.md`.

---

## 7. Known limitations

**The write-up's voice is not the author's.** It was written in the register of
the project's existing documents. The structure, the figures and the framing
hold; the wording is due a pass. The property worth preserving through that
rewrite is the shape where a claim is followed immediately by what would
falsify it.

**`docs/figures.json` covers 21 figures.** `check_figures.py` reports 15
numbers in `docs/WRITE_UP.md` and 7 in `README.md` outside the manifest. Adding
one is three lines. A count that climbs while the manifest does not is the
manifest going stale.

**The rules fingerprint does not cover `engine.py`**, which includes the
stop-cost floor clamp. Covering it would mean bumping a version on every edit to
the largest module in the project. Stated, not solved.

**`outputs/sweeps/` predates `never_pit`.** Four tables with five strategies.
Not committed; re-run `sweep_pit_transit.py` if the write-up ever cites them.

**The two series' policies were trained at different seeds until this pass.**
WEC's was seed 4, left in place by `scripts/seed_sweep.py`, which copies each
seed's policy into `outputs/policies/` to score it. Both are now seed 0, and
`check_artefacts.py` reports the seeds as a gap if they ever diverge again.

**02b's per-seed causal artefacts still do not exist**, so every evaluation
prints `benchmark: NOT JOINED`. Unchanged since 02b and correctly reported
rather than passed over.

---

## 8. What 06 inherits, and what it must not say

**It may say:** the app is deployed and the tree reproduces itself; the roster
result in section 2, with the held-out median's interval stated; the four
measured traps from the post-04 pass; that `pit_transit_frac` has no measured
counterpart and why; and that an agent indistinguishable from taking no
decisions is now detectable by an ordinary roster row.

**It must not say** that the agent beat, matched or lost to the roster. There
is no agent result. It must not quote 0.590 or 0.445. It must not rank the four
traps. And it must not quote **0.95, 0.84, 0.92, 0.93 or 0.98** as shares of a
pit stop - those are this pass's withdrawn figures and they are the newest way
to be wrong about this project.

**The next pass is presentation**, not engineering: a colourway
(`#8968CD` with white, silver and black), typography, and a rewrite of every
block of writing in the author's own voice. Two notes for it.

`.streamlit/config.toml` already carries a `[theme]` block and will take
`primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, `textColor` and
`font`, which reaches every widget. It does **not** reach `viz.py`, which has
its own palette and is shared with the notebooks - so a chart restyle is one
file and applies in both places.

And **`check_figures.py` will fail during a rewrite**, by design: it matches
exact strings. Rewording a sentence containing a quoted figure means updating
that entry's `quoted` field. Run it after each pass; it names the sentences a
number has moved out of.

---

## 9. Amendments and files

Amendments 28 to 31, written out in `docs/amendments_28_to_31.md`:

- **28** - `set_dials` beside `scale_dials`, because a multiplier cannot move a
  dial that sits at zero and amendment 23's dial does. `never_pit` built under
  amendment 25.
- **29** - `rules_fingerprint`, beside `dials_fingerprint` rather than inside
  it. Inside is the stronger design and was rejected on cost: it would
  invalidate every artefact in the tree.
- **30** - 05's verification gate, supplied. Two conditions, two falsifiers.
- **31** - `pit_transit_frac` has no measured counterpart and cannot have one.

Changed: `src/endurance/{params,pitstop,caution,assets,harness,strategies,viz,
__init__}.py`, `app/{loading,controller,panels,statements,Home}.py`,
`app/pages/{1_Race,2_Comparison,3_Methods}.py`,
`scripts/{freeze_assets,train,evaluate,sweep_pit_transit}.py`,
`tests/{test_app,test_gym_env,test_strategies,run_tests_nopytest}.py`,
`build_nb.py`.

New: `scripts/{check_artefacts,evidence,check_figures,estimate_pit_transit}.py`,
`tests/test_05_changes.py`, `docs/{ARCHITECTURE,WRITE_UP,figures,
amendments_28_to_31}`, `pyproject.toml`, `requirements-dev.txt`, `.gitignore`,
`.streamlit/config.toml`, `README.md`.

Renamed: `data/processed/{code}_standin.json` to `{code}_banked.json`. The old
name is still read; nothing writes it.

Deleted: `data/processed/{seed_bank_imsa,background_field_imsa}.json`, and
`build_nb.py`'s `PUBLISHED_02C` block - removed rather than updated, because
amendment 23 superseded every number in it and a comparison against those
literals could only report a disagreement already known.

**Still open:** `RULES_VERSION` does not cover `engine.py`; 02b's per-seed
artefacts; `outputs/sweeps/` predates the sixth roster member; the manifest
covers 21 of the figures in the documents; and the write-up needs its author's
voice.
