"""Is the tree that ships actually sufficient, and is everything in it current?

    python scripts/check_artefacts.py
    python scripts/check_artefacts.py --quiet     # verdict lines only

05's verification gate, second condition. The stage was specified with a
boundary constraint and a deliverable list and no gate - the sixth in a row,
after 03a, 03b, 00, 04 and the post-04 pass - and this is half of what was
supplied instead. The other half is that every number in the write-up traces
to a file, which needs the write-up to exist first.

What it asks
------------
**Is every artefact the app and the scripts open actually there?** A missing
file is not always an error at the moment it goes missing: the app reads the
`.onnx` and `scripts/evaluate.py` reads the `.zip`, so a tree can host
perfectly while being unable to reproduce a single number in the write-up.
That is not a hypothetical - it is the state this script was written in.

**Does everything agree about which race it is about?** Every artefact carries
a `dials_fingerprint` and each was checked where it is used. Nothing checked
them all against each other, and an artefact whose fingerprint matches nothing
in the tree is the worst case rather than the loudest: loaded by name it
produces a complete, plausible race no other stage can reproduce. Amendment 19
records one live instance of exactly that.

**Do the rules agree too?** Amendment 29's `rules_fingerprint`, reported
rather than enforced. Artefacts frozen before it existed carry none, which is
a gap and not a failure, and saying which is the point.

What it cannot ask
------------------
Whether the numbers in the tables are right. It checks that the files exist,
that they name the same race, and that nothing is orphaned. A table computed
correctly against these dials and a table computed wrongly against them look
identical from here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_project_root(marker: str = "src/endurance") -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(
        f"could not find {marker!r} at or above {here.parent}.\n\n"
        "This script belongs in `scripts/`, beside `src/` and `notebooks/`.")


ROOT = find_project_root()
sys.path.insert(0, str(ROOT / "src"))

from endurance.assets import (  # noqa: E402
    BackgroundField,
    SeedBank,
    dials_fingerprint,
    rules_fingerprint,
    rules_mismatch,
)
from endurance.params import RaceConfig  # noqa: E402
from endurance.policy import PolicyCard, bank_fingerprint  # noqa: E402

SERIES = ("imsa", "wec")
BANKS = ("headline", "held_out")

# Filled in as the cards are read. Which training seed each series ships is
# not a property of either series alone, so it cannot be checked inside
# `check_series` and is reported at the end.
TRAIN_SEEDS: dict[str, int] = {}

# Where each kind of thing lives, as a pattern rather than a path, because the
# app resolves its folders the same way and a checker that hard-coded them
# would pass on a tree the app cannot read.
PATTERNS = {
    "config": "{code}.json",
    "seed bank": "{code}_seeds.json",
    "background field": "{code}_field.json",
    "onnx export": "{code}_maskable_ppo.onnx",
    "checkpoint": "{code}_maskable_ppo.zip",
    "policy card": "{code}_maskable_ppo.card.json",
}


class Report:
    """Rows and a verdict. Nothing here writes."""

    def __init__(self, quiet: bool = False):
        self.rows: list[tuple[str, str, str]] = []
        self.quiet = quiet

    def ok(self, what: str, detail: str = "") -> None:
        self._add("ok", what, detail)

    def gap(self, what: str, detail: str = "") -> None:
        """Something absent that is allowed to be absent, and is worth saying."""
        self._add("gap", what, detail)

    def bad(self, what: str, detail: str = "") -> None:
        self._add("BAD", what, detail)

    def _add(self, verdict: str, what: str, detail: str) -> None:
        self.rows.append((verdict, what, detail))
        if not self.quiet or verdict != "ok":
            line = f"  {verdict:4} {what}"
            print(f"{line}{'  - ' + detail if detail else ''}", flush=True)

    @property
    def failed(self) -> int:
        return sum(1 for v, _, _ in self.rows if v == "BAD")


def find_all(name: str) -> list[Path]:
    """Every file in the tree answering to a name, in path order."""
    return [hit for hit in sorted(ROOT.rglob(name))
            if ".ipynb_checkpoints" not in hit.parts]


def find(name: str) -> Path | None:
    """First match, which is what every other script in `scripts/` takes.

    Kept because this checker has to resolve names the way the rest of the
    project resolves them - reporting on a tidier answer of its own would be
    reporting on a tree nobody runs.
    """
    hits = find_all(name)
    return hits[0] if hits else None


def resolve(name: str, what: str, report: "Report") -> Path | None:
    """The one file that answers to a name, or a refusal if several do.

    **Ambiguity is a failure, not a tie to be broken.** Every script here
    resolves artefacts by `rglob` and takes the first hit, so two files under
    one name means those scripts disagree about which artefact they are
    holding depending only on where each happens to sit in the tree. That is
    amendment 19's failure in a second place, and unlike the orphan case
    nothing was looking for it.
    """
    hits = find_all(name)
    if not hits:
        return None
    if len(hits) > 1:
        report.bad(f"{what}: {len(hits)} files answer to {name!r}",
                   "every script here takes whichever sorts first, so they "
                   "are not all reading the same artefact: "
                   + ", ".join(rel(h) for h in hits))
        return hits[0]
    return hits[0]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ----------------------------------------------------------------------
# One series
# ----------------------------------------------------------------------
def check_series(code: str, report: Report) -> str | None:
    """Everything for one series. Returns its dials fingerprint, or `None`."""
    print(f"\n=== {code.upper()} ===")

    found = {name: resolve(pattern.format(code=code), code, report)
             for name, pattern in PATTERNS.items()}

    config_path = found["config"]
    if config_path is None:
        report.bad(f"{code}: no config", f"expected {code}.json")
        return None

    config = RaceConfig.load(config_path)
    fingerprint = dials_fingerprint(config)
    report.ok(f"{code}: config", f"{rel(config_path)} -> {fingerprint} "
                                 f"({config.name}, "
                                 f"{config.duration_s / 3600:.1f} h, "
                                 f"{config.total_cars} cars)")

    # The banked config is what `freeze_assets.load_assets` reads, and the
    # check that it still matches 00's output is the one that says a
    # recalibration has not happened underneath the banks.
    #
    # Both names, new first. `{code}_banked.json` replaced
    # `{code}_standin.json` at 05; a tree that has migrated has only the
    # first, one that has not has only the second, and one that has *both* is
    # holding two answers to the same question and is refused.
    banked = None
    for name in (f"{code}_banked.json", f"{code}_standin.json"):
        banked = banked or resolve(name, code, report)
    if find_all(f"{code}_banked.json") and find_all(f"{code}_standin.json"):
        report.bad(f"{code}: both banked-config names are present",
                   f"{code}_standin.json was renamed to {code}_banked.json; "
                   f"delete the old one rather than leaving two")

    if banked is None:
        report.bad(f"{code}: no banked config",
                   f"expected {code}_banked.json; scripts/freeze_assets.py "
                   f"writes it and every script downstream reads it")
    elif dials_fingerprint(RaceConfig.load(banked)) != fingerprint:
        report.bad(f"{code}: banked config is a different race",
                   f"{rel(banked)} does not match {rel(config_path)}; run "
                   f"scripts/freeze_assets.py {code} --force and retrain")
    else:
        report.ok(f"{code}: banked config matches", rel(banked))

    bank = _check_artefact(code, "seed bank", found["seed bank"], SeedBank,
                           fingerprint, report)
    _check_artefact(code, "background field", found["background field"],
                    BackgroundField, fingerprint, report)

    _check_policy(code, found, config, bank, fingerprint, report)
    _check_evaluation(code, fingerprint, report)
    return fingerprint


def _check_artefact(code: str, what: str, path: Path | None, cls,
                    fingerprint: str, report: Report):
    """A bank or a field: present, about this race, and rules-stamped or not."""
    if path is None:
        report.bad(f"{code}: no {what}")
        return None

    artefact = cls.load(path)
    recorded = artefact.provenance.get("dials_fingerprint")
    if recorded != fingerprint:
        report.bad(f"{code}: {what} is about another race",
                   f"{rel(path)} was built against {recorded!r}, config is "
                   f"{fingerprint!r}")
        return artefact

    report.ok(f"{code}: {what}", rel(path))

    stamped = artefact.provenance.get("rules_fingerprint")
    if stamped is None:
        report.gap(f"{code}: {what} carries no rules fingerprint",
                   "frozen before amendment 29; run scripts/refreeze_assets.py "
                   "--write to stamp it")
    else:
        message = rules_mismatch(stamped, what=f"the {code} {what}")
        if message:
            report.bad(f"{code}: {what} was built against other rules", message)
        else:
            report.ok(f"{code}: {what} rules fingerprint", stamped)
    return artefact


def _check_policy(code: str, found: dict, config, bank, fingerprint: str,
                  report: Report) -> None:
    """The three files a policy is, and what each of them is for.

    Split deliberately. The app loads the `.onnx` and `scripts/evaluate.py`
    loads the `.zip`, so their absences have completely different consequences
    and reporting them as one line would hide which of the two happened.
    """
    card_path = found["policy card"]
    if card_path is None:
        report.bad(f"{code}: no policy card",
                   "a checkpoint without one cannot be checked against the "
                   "race it is scored on")
        return

    # **The three files have to be in one folder.** A card is written beside
    # the weights it describes and its hashes are only meaningful there, so a
    # card resolved from one directory and an export from another produces a
    # hash mismatch that reads as "these are the wrong weights" when what
    # actually happened is that two generations of policy are in the tree and
    # the resolver crossed them. That is a completely different repair, and
    # the message has to say which one it is.
    onnx, checkpoint = found["onnx export"], found["checkpoint"]
    elsewhere = [rel(p) for p in (onnx, checkpoint)
                 if p is not None and p.parent != card_path.parent]
    if elsewhere:
        report.bad(f"{code}: the policy's files are in different folders",
                   f"the card is at {rel(card_path)} and {', '.join(elsewhere)} "
                   f"is not beside it. The hashes below are comparing one "
                   f"generation's card with another's weights; fix the folders "
                   f"before reading them")

    card = PolicyCard.load(card_path.with_suffix("").with_suffix(".zip"))
    report.ok(f"{code}: policy card read from", rel(card_path))
    if card.dials_fingerprint != fingerprint:
        report.bad(f"{code}: policy trained on another race",
                   f"card says {card.dials_fingerprint!r}, config is "
                   f"{fingerprint!r}")
    elif bank is not None and card.bank_fingerprint != bank_fingerprint(bank):
        report.bad(f"{code}: policy trained on another bank",
                   f"card says {card.bank_fingerprint!r}; the held-out set may "
                   f"no longer be held out")
    else:
        report.ok(f"{code}: policy card",
                  f"{card.algorithm}, train seed {card.train_seed}, "
                  f"{card.total_timesteps:,} steps")
    TRAIN_SEEDS[code] = card.train_seed

    source = str(card.notes.get("dials_source", ""))
    if "STAND-IN" in source:
        report.bad(f"{code}: the card's dials_source is false",
                   f"{source!r} - these dials are calibrated from "
                   f"{config.classes[0].source_event!r}. This card was written "
                   f"by a `train.py` that held the string as a literal. Either "
                   f"retrain the series against the fixed script, which is "
                   f"minutes and gives a card true in every field, or restamp "
                   f"the note from the config - but do not edit it by hand, "
                   f"because a card nobody can regenerate is worse than a "
                   f"card that is wrong in a way this catches")

    if onnx is None:
        report.bad(f"{code}: no .onnx export",
                   "the app loads this; without it the race page cannot put "
                   "the policy in the seat")
    else:
        try:
            card.check_file(onnx)
            report.ok(f"{code}: .onnx export", rel(onnx))
        except ValueError as e:
            report.bad(f"{code}: .onnx does not match its card", str(e))

    if checkpoint is None:
        report.bad(f"{code}: no .zip checkpoint",
                   "scripts/evaluate.py loads this, so this series cannot be "
                   "re-scored; the app is unaffected because it reads the "
                   ".onnx")
    else:
        try:
            card.check_file(checkpoint)
            report.ok(f"{code}: .zip checkpoint", rel(checkpoint))
        except ValueError as e:
            report.bad(f"{code}: .zip does not match its card", str(e))


def _check_evaluation(code: str, fingerprint: str, report: Report) -> None:
    """The saved tables, which are what the comparison page reads."""
    for bank_name in BANKS:
        rows = find(f"{code}_{bank_name}_rows.csv")
        summary = find(f"{code}_{bank_name}_summary.csv")
        provenance = find(f"{code}_{bank_name}_provenance.json")

        if rows is None or summary is None:
            report.bad(f"{code}: no {bank_name} table",
                       "the comparison page falls back to offering a handful "
                       "of races")
            continue

        if provenance is None:
            report.gap(f"{code}: {bank_name} table has no provenance",
                       "the page counts distinct seeds instead, which gives "
                       "the right number and nothing to check it against")
        else:
            meta = json.loads(provenance.read_text())
            recorded = meta.get("dials_fingerprint")
            if recorded != fingerprint:
                report.bad(f"{code}: {bank_name} table is about another race",
                           f"measured against {recorded!r}")
                continue
            source = str(meta.get("dials_source", ""))
            if "STAND-IN" in source:
                report.bad(f"{code}: {bank_name} provenance dials_source is "
                           f"false", repr(source))

        # The sixth roster member, which is the reason this check exists at 05
        # rather than being a tidy-up. A table written before amendment 25 has
        # five strategies and cannot answer the question `never_pit` was added
        # to ask.
        header = summary.read_text().splitlines()
        names = {line.split(",")[1] for line in header[1:] if "," in line}
        if "never_pit" not in names:
            report.bad(f"{code}: {bank_name} table predates never_pit",
                       f"{len(names)} strategies and no control; re-run "
                       f"scripts/evaluate.py {code} --bank {bank_name}")
        else:
            report.ok(f"{code}: {bank_name} table", f"{len(names)} strategies")


# ----------------------------------------------------------------------
# The orphan scan
# ----------------------------------------------------------------------
def check_orphans(fingerprints: set[str], report: Report) -> None:
    """Artefacts in the tree that no config in the tree can explain.

    The failure amendment 19 recorded: a bank and a field for a race whose
    config is not here, sitting under a name the loaders also accept. Nothing
    refuses them, because every check in the project asks "does this match the
    config I am holding" and never "is there a config this matches at all".
    """
    print("\n=== orphans ===")
    before = report.failed          # so this section reports its own verdict
    seen = 0
    for pattern in ("*_seeds.json", "seed_bank_*.json",
                    "*_field.json", "background_field_*.json"):
        for path in sorted(ROOT.rglob(pattern)):
            if ".ipynb_checkpoints" in path.parts:
                continue
            try:
                provenance = json.loads(path.read_text()).get("provenance", {})
            except (json.JSONDecodeError, OSError):
                continue
            recorded = provenance.get("dials_fingerprint")
            if recorded is None:
                continue
            seen += 1
            if recorded not in fingerprints:
                report.bad(f"orphan: {rel(path)}",
                           f"built against dials {recorded!r} ({provenance.get('race')!r}, "
                           f"{provenance.get('duration_s', 0) / 3600:.1f} h), and no "
                           f"config in this tree has that fingerprint. Loaded by "
                           f"name it produces a race nothing else can reproduce - "
                           f"delete it or move it outside the tree")
    if seen == 0:
        report.gap("no banked artefacts found to scan for orphans")
    elif report.failed == before:
        report.ok(f"every one of {seen} banked artefacts matches a config "
                  f"in this tree")


# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quiet", action="store_true",
                        help="print gaps and failures only")
    args = parser.parse_args()

    print(f"tree: {ROOT}")
    print(f"rule logic: {rules_fingerprint()}")

    report = Report(quiet=args.quiet)
    fingerprints = {f for f in (check_series(c, report) for c in SERIES) if f}

    # A gap rather than a failure: two series shipping policies from different
    # training seeds is not wrong, and amendment 26 says a single run is not a
    # reportable result either way. It is worth knowing because it is easy to
    # arrive at by accident - `scripts/seed_sweep.py` puts each seed's policy
    # in place to score it - and because anything written about the two series
    # side by side would otherwise imply they are the same experiment.
    if len(set(TRAIN_SEEDS.values())) > 1:
        report.gap("the two series ship policies from different training seeds",
                   ", ".join(f"{c} at seed {s}"
                             for c, s in sorted(TRAIN_SEEDS.items()))
                   + " - deliberate or left behind by a sweep, but not "
                     "something to write about as one result")

    check_orphans(fingerprints, report)

    gaps = sum(1 for v, _, _ in report.rows if v == "gap")
    print("\n=== verdict ===")
    if report.failed:
        print(f"  {report.failed} failing, {gaps} gap(s). This tree cannot be "
              f"shipped as it stands.")
        return 1
    if gaps:
        print(f"  no failures, {gaps} gap(s). Shippable; the gaps are things "
              f"the write-up should say rather than things to fix.")
        return 0
    print("  every artefact present, current, and about the same race.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
