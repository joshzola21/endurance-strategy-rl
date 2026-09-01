# Amendments 32 and 33 — for `handover_02_decision_record.md`

06's amendments. Numbering continues from 31; the packaging pass closed at 31.

---

## 32 — 06 exists, and its verification gate

**The decision.** The blueprint's stage map in section 3 ends at 05, marked
*last*. It gains a sixth row:

| Stage | Focus | Primary deliverable | Boundary constraint |
|---|---|---|---|
| 06 | Presentation | Colourway, typography, and every block of writing in the author's voice | Presentation only: no number moves |

**Why this is recorded rather than assumed.** 06 existed only in the closing
paragraph of the packaging handover. A stage that appears in a handover and not
in the blueprint is a stage with no boundary constraint, and the boundary
constraint is the whole of what makes this one safe: a rewrite is the single
easiest way to change a result by accident, because the sentence and the number
move together and only one of them was checked.

**The boundary constraint, stated plainly.** No dial, no artefact, no
computation. The one place the constraint was tested in practice: three of the
figures being rewritten around were *wrong* in the sense of overstating what
the evaluation supports — see 33's note on the held-out row — and correcting
those is inside the constraint, because the artefact was always right and only
the sentence was not.

**The gate, two conditions, each with a falsifier.** The seventh stage in this
project to arrive without one.

**Condition one — nothing but presentation moved.** Snapshot
`outputs/evidence.json` before the pass. Afterwards: all 159 figures identical,
both dials fingerprints and the rules fingerprint unmoved, `check_artefacts.py`
green, the suite green.

*Falsifier:* change one dial by one digit and the comparison must fail.

**Status: passing.** Run on the committed tree after the pass. All 159 figures
in `outputs/evidence.json` identical, both dials fingerprints and the rules
fingerprint unmoved.

**Condition two — every figure quoted still traces to a file on disk.**
`scripts/check_figures.py --strict` green across all six documents, and
`--self-test` green on both of its halves.

*Falsifiers, both of them:* a planted digit must fail the check, **and** a
re-wrapped line must not. The second is the half that makes the gate usable
during the job it exists for. A check that fires when a paragraph is reflowed
gets switched off in the first hour of a rewrite, and a gate that gets switched
off is worth less than no gate, because the document still carries the claim
that it was checked.

**Status: passing**, against the real `outputs/evidence.json` rather than a
stand-in. Fourteen keyed entries checked against the artefact and fourteen
agree. Also checked, and clean in all four documents: `0.590` and `0.445`,
which the packaging handover names as figures 06 must not quote.

**What the gate caught that a reader would not have.** Two things.

**The write-up quoted a stop count that the artefact does not hold.** The five
WEC training seeds have median stop counts of 38, 65, **72.5**, 174 and 194,
and every document in this project has said 72 since the figure was first
written down. It survived the packaging pass, this stage's own rewrite, and
several readings by people looking for exactly this kind of fault. It was found
in the ninety seconds after `wec.train_seed_stops` was promoted from a
source-backed entry to a keyed one, which is the entire argument for keying an
entry rather than citing one. Corrected in `docs/WRITE_UP.md`.

**Its own falsifier was broken.** `--self-test` reported NOT CAUGHT on the digit it had just planted:
`docs/WRITE_UP.md` quotes `58% of 200 races` twice and the second is wrapped
over a line, so removing the literal string removed one of two and the survivor
passed. The check was sound and the falsifier was planting a fault that was not
the fault it meant to plant. It now plants on the flattened text and removes
every occurrence.

---

## 33 — the colourway, and four things it decided

**The decision.** `#8968CD` with white, silver and black, as the packaging
handover specified, resolved into a palette in `.streamlit/config.toml` and
`src/endurance/viz.py`.

**`#8968CD` cannot carry text, and that shaped everything else.** It is 4.26:1
against white and 4.93:1 against black; WCAG AA wants 4.5:1 for body text. So
the accent draws fills, borders, focus rings and chart marks, and a darkened
counterpart, `#7A57BC` at 5.4:1, is what any purple *word* uses. Black became
`#2A2730` at 13.6:1 and white became `#F7F6FA`. The silver split in two: a
`#6E6A78` that passes for text at 4.9:1 and a `#8A8694` for marks, because the
obvious single mid-grey comes out at 3.3:1 and this app's captions are where
the caveats live.

**Green and red were reconsidered and kept.** The paired-delta chart carries
the strongest result the project has, and roughly one man in twelve cannot
separate that pair. A purple-and-orange replacement was built and rejected: the
convention is one every reader of a motorsport chart already has, and the
meaning is also carried by the bar order and the axis label. Recorded as a
decision, in the file, so the next person to notice the problem finds the
reasoning rather than re-making the change.

**Two palettes, and they cannot be merged.** The Streamlit theme reaches every
widget and reaches `viz.py` not at all, because `st.pyplot` renders a figure
that was drawn before Streamlit saw it. The two files carry the same hexes and
have to be edited together, which is now stated in both and in
`docs/ARCHITECTURE.md` boundary 6. `viz.py` applies its style per figure
through a decorator rather than writing to `plt.rcParams` at import, so
importing it in a notebook no longer restyles every other figure in that
notebook. Asserted in the smoke run: `rcParams` is identical before and after.

**A bug fixed on the way.** `plot_sweep_grid` wrote every cell label in white,
including over the pale end of the colour ramp, where white on near-white is
not there at all — worse than an unlabelled grid, because the label exists and
the reader assumes they have read it. It now computes the contrast of ink and
of paper against each cell and takes the better. The first version used a
threshold picked by eye at 0.35 and was wrong through the middle of the ramp,
which is exactly where cells are hardest to read and where a guess feels
safest.

### What 06 removed from the app, and why

Three pieces of furniture, all of them provenance rendered for the wrong
reader:

- **"Where the app is reading from"** on the landing page. It printed resolved
  absolute paths, which on the hosted app are the deployment's internals.
  `loading.where()` still exists; the question is now answered from a terminal.
- **"What this table was measured on"** on the comparison page. A raw
  `st.json` dump of the provenance block.
- **"Settled at ..."** under each of the four statements. A citation to an
  internal decision record is furniture to a visitor, who cannot follow it.
  `Statement.source` is kept and no longer rendered, because the point of the
  field is that a caveat can be chased to where it was settled, and that is a
  property of the project rather than of the page.

The methods page keeps its provenance and renders it as a table rather than a
JSON blob. It is the page a reader has opted into the machinery to reach.

### One claim corrected rather than reworded

The write-up said the caution gambler "holds up on fifty races the strategy was
never selected on". That is stronger than the held-out row supports: the share
holds at 0.60 and the median's interval includes zero. Both documents now state
the share and say the median does not carry over. This is the correction the
packaging handover's section 2 asked for and it had reached the README only.

Two withdrawn figures are also gone. The write-up quoted `95% of a stop` and a
bound of `84% to 98%` while explaining that both had been withdrawn. Amendment
31 named those as the newest way to be wrong about this project, and a
withdrawn number that keeps being printed is the one a reader carries away. The
paragraph still says both results were withdrawn and why each was wrong; it no
longer restates either. Their two manifest entries went with them.

---

## The manifest, which is now a gate rather than a report

`docs/figures.json` went from 21 figures across three documents to 43 across
six. `check_figures.py` gained three things:

- **`--strict`**, which fails on a number in neither the manifest nor
  `not_figures`. Coverage was counted and not enforced until this pass, and a
  count nobody fails is a count nobody reads.
- **whitespace flattening** on both sides, so a reflow is not an edit.
- **`.py` documents**, read through the syntax tree: implicit concatenation
  joined so a quotation may span the source lines it is wrapped over, and
  docstrings excluded so what is checked is what a visitor sees.

`app/statements.py`, `app/loading.py` and `app/pages/3_Methods.py` are now
documents. The four statements are the most-read text in the project and were
guarded by nothing.

**Fourteen of the 46 entries are keyed** into `outputs/evidence.json`; the
other 32 carry a `source`. Everything `evidence.py` actually records is now
keyed, including three entries promoted at the end of this pass, and one of
those promotions immediately caught a wrong figure that had been in the
documents for two stages.

**The remaining 32 are source-backed because nothing computes them.** The
rulebook articles, the file size, the row count, the trap measurements from the
post-04 pass, the pit-transit estimates. A source-only entry checks that the
text says what it said yesterday and not that the artefact still agrees, so the
honest way to shrink that number is to teach `evidence.py` to emit those
figures — not to invent keys for it.

---

## Files

Changed: `src/endurance/viz.py`, `app/{Home,statements,panels,loading,
controller}.py`, `app/pages/{1_Race,2_Comparison,3_Methods}.py`,
`scripts/check_figures.py`, `docs/{figures.json,WRITE_UP.md,ARCHITECTURE.md}`,
`README.md`.

New: `.streamlit/config.toml`, `docs/amendments_32_to_33.md`.

Removed from the app: the landing page's reading-from panel, the comparison
page's provenance expander, and the source caption under each statement. No
module was deleted.

---

## Where this ended

The decision record's status table says **05 — the write-up: next**. It should
now read:

| Stage | Status |
|---|---|
| 05 — packaging and the write-up | done — `handovers/handover_06_packaging.md`, amendments 28-31 |
| 06 — presentation | done and gated, both conditions — amendments 32-33 |

**There is no 06 handover, and that is deliberate.** Every handover in this
project was written for a thread that had not happened yet. This was the last
one, so the audience for a handover does not exist, and writing one would
produce a document whose only reader is the person who commissioned it. What a
future reader actually needs is already in the tree and in this file: the
blueprint says what the project is, `ARCHITECTURE.md` says how it fits
together, `WRITE_UP.md` says what it found, and the decision record plus its
amendments say why each of those is the shape it is.

**What is still open, in full, and none of it is 06's:**

- `RULES_VERSION` does not cover `engine.py`. Amendment 29 states why and does
  not solve it.
- 02b's per-seed causal artefacts do not exist, so every evaluation prints
  `benchmark: NOT JOINED`.
- `outputs/sweeps/` predates `never_pit` — four tables with five strategies.
  Re-run `sweep_pit_transit.py` before citing them.
- 32 of the 46 manifest entries are source-backed rather than
  artefact-backed. Closing that gap means `evidence.py` emitting the trap
  measurements and the pit-transit estimates, not editing the manifest.
- Whether `stop_cost`'s *shape* can express a real pit lane is unsettled, now
  for a stated reason. Amendment 31.

**The thing worth saying last.** This project set out to train an agent that
beats human pit strategy and did not. What it has instead is an apparatus that
could tell it had not — a control that scores what taking no decisions is
worth, a seed sweep that shows a single training run measures nothing, four
traps measured on the engine rather than inferred from a score, and a dial
proved unmeasurable rather than guessed at. Every one of those is a result
about the method rather than about racing, and every one of them arrived
because a check was built to fail on something specific rather than to pass.
