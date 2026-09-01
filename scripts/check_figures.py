"""Does every number in the write-up trace to a file on disk?

    python scripts/check_figures.py
    python scripts/check_figures.py --self-test    # plant a wrong digit

05's verification gate, second condition. The first - is the tree sufficient
and about one race - is `scripts/check_artefacts.py`. This one is about the
documents.

The claim being checked is the one the write-up closes on: *every figure quoted
traces to something on disk.* That is easy to write and, until something checks
it, impossible to rely on. Documents drift from the artefacts under them
silently and in one direction: a number gets updated in a table and not in the
sentence that quotes it, and the sentence is what a reader believes.

How it works
------------
`docs/figures.json` is a manifest. Each entry names a figure, the value it must
have, the exact text that must appear, and which documents must contain it.
Values are checked against `outputs/evidence.json`, which
`scripts/evidence.py` writes by running the gates.

**Two failures, and they mean different things.**

*The artefact no longer yields the value.* The project has moved and the
documents are stale. Re-run `scripts/evidence.py`, read what changed, and edit
the sentence.

*The text is not in the document.* Somebody edited a number in the writing
without touching what produced it. This is the failure the file exists for and
the only one nothing else in the project would catch.

What 06 changed
---------------
**Coverage is enforceable.** `--strict` fails when a document holds a number
the manifest does not account for. It was counted and not enforced until the
write-up rewrite made the difference matter: a rewrite is exactly when a
number gets introduced, moved or quietly dropped, and a count nobody fails is
a count nobody reads. The default is unchanged, so a run without the flag
still reports rather than refuses.

**Whitespace no longer counts.** Document and quotation are both collapsed to
single spaces before matching. Re-wrapping a paragraph is not editing a
figure, and a check that cannot tell those apart teaches its reader to ignore
it - which costs more than the failure it was catching.

**A `.py` file can be a document.** `app/statements.py` carries the four
things the app must say, which is the most-read text in the project and was
guarded by nothing. A Python document is read through its syntax tree rather
than as raw text: implicit string concatenation is joined, so a quotation may
span the source lines it happens to be wrapped over, and docstrings are
excluded, so what is checked is what a visitor sees rather than what the
module says about itself.

What it does not check
----------------------
It cannot check that a number is being quoted *about the right thing*. A
correct value in a sentence that misdescribes it passes here and is caught
only by somebody reading.

The coverage count matches by substring, so a number contained in a longer
quoted one counts as covered - `0.7` is covered by a manifest holding `0.73`.
That is the direction of error worth having, because the alternative reports
figures that are in fact accounted for, and a check that cries wolf is the
one thing this file cannot afford to be.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


def find_project_root(marker: str = "src/endurance") -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(
        f"could not find {marker!r} at or above {here.parent}")


ROOT = find_project_root()
MANIFEST = ROOT / "docs" / "figures.json"
EVIDENCE = ROOT / "outputs" / "evidence.json"

# How close a stored figure has to be to the manifest's value. Not exact:
# `evidence.py` rounds to four places on the way in, and a share that moves in
# the fifth is not a change to any claim.
TOLERANCE = 5e-4


def flatten(text: str) -> str:
    """One space everywhere, so a line break cannot fail a check.

    Applied to both sides. A figure that has been edited still fails; a
    paragraph that has been re-wrapped no longer does.
    """
    return " ".join(text.split())


def document_text(path: Path) -> str:
    """What a reader sees, which for a `.py` file is not what `read_text` gives.

    Markdown is its own text. Python is read through its syntax tree, for two
    reasons. Implicit concatenation means a sentence in the source is a
    column of quoted fragments, and a quotation that spans two of them would
    never match however the manifest was written. And docstrings are excluded
    deliberately: they explain the module to whoever maintains it and are not
    on screen, so a figure that appears only in one is not a figure the app
    has published.
    """
    raw = path.read_text()
    if path.suffix != ".py":
        return raw

    tree = ast.parse(raw)
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)) or not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            docstrings.add(id(first.value))

    return "\n\n".join(
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in docstrings)


def load_json(path: Path, what: str) -> dict | None:
    if not path.exists():
        print(f"no {what} at {path.relative_to(ROOT)}")
        return None
    return json.loads(path.read_text())


def matches(expected, actual) -> bool:
    if isinstance(expected, str) or isinstance(actual, str):
        return str(expected) == str(actual)
    if isinstance(expected, list) or isinstance(actual, list):
        return list(expected) == list(actual)
    return abs(float(expected) - float(actual)) <= TOLERANCE


def check(manifest: dict, evidence: dict, texts: dict[str, str]) -> list[str]:
    """Every entry, both ways. Returns the problems."""
    problems: list[str] = []
    figures = evidence.get("figures", {})

    for entry in manifest["figures"]:
        key, value = entry.get("key"), entry.get("value")
        quoted, docs = entry.get("quoted"), entry.get("docs") or []

        # --- the value against the artefact ---
        if key is not None:
            if key not in figures:
                problems.append(
                    f"{key}: not in outputs/evidence.json. Either "
                    f"scripts/evidence.py no longer produces it, or it never "
                    f"did and this entry was written by hand")
            elif not matches(value, figures[key]):
                problems.append(
                    f"{key}: the manifest says {value!r} and the artefact says "
                    f"{figures[key]!r}. The project has moved; re-read the "
                    f"sentences that quote it")
        elif entry.get("source") is None:
            problems.append(
                f"{quoted!r}: has no key and no source, so nothing says where "
                f"this number came from")

        # --- the text against the documents ---
        if quoted is None:
            continue
        for doc in docs:
            if doc not in texts:
                problems.append(f"{doc}: named in the manifest and not on disk")
            elif flatten(quoted) not in flatten(texts[doc]):
                problems.append(
                    f"{doc}: does not contain {quoted!r}. A number has been "
                    f"edited in the writing without the manifest following it")
    return problems


def unchecked_numbers(texts: dict[str, str], manifest: dict) -> dict[str, list]:
    """Which numbers in each document the manifest does not account for.

    Returns the numbers rather than a count, because a count tells you the
    manifest has gone stale and not which line to go and read.

    `not_figures` is the escape hatch, and it takes a reason. A language
    version is a number and is not a claim, and the choice is between saying
    so once in the manifest or loosening the pattern until it stops noticing.
    Loosening it would also stop it noticing the next real one.
    """
    quoted = " ".join(e["quoted"] for e in manifest["figures"] if e.get("quoted"))
    excused = " ".join(str(n.get("text", "")) for n in manifest.get("not_figures", []))
    out = {}
    for doc, text in texts.items():
        # Ignore anything inside a fenced block: those are commands and paths.
        body = re.sub(r"```.*?```", "", text, flags=re.S)
        numbers = set(re.findall(r"\b\d+\.\d+\b|\b\d{2,}%", body))
        out[doc] = sorted(n for n in numbers
                          if n not in quoted and n not in excused)
    return out


def self_test(manifest: dict, evidence: dict, texts: dict[str, str]) -> int:
    """Plant a wrong digit and require the check to fail on it.

    A gate that cannot fail is not a gate - 02c's rule, applied to this one.
    Nothing is written: the planted document exists only in memory.
    """
    print("=== self-test ===")
    entry = next((e for e in manifest["figures"]
                  if e.get("quoted") and e.get("docs")), None)
    if entry is None:
        print("  no entry with both a quotation and a document; cannot test")
        return 1

    doc = entry["docs"][0]
    planted = dict(texts)
    # Flattened, and every occurrence. `docs/WRITE_UP.md` quotes the first
    # entry twice and the second one is wrapped over a line, so removing the
    # literal string removed one of the two and the surviving copy passed the
    # check. The self-test reported NOT CAUGHT and was right to: what it had
    # planted was not the fault it meant to plant.
    planted[doc] = flatten(texts[doc]).replace(flatten(entry["quoted"]),
                                               "a different number")
    problems = check(manifest, evidence, planted)
    caught = any(doc in p and "edited in the writing" in p for p in problems)
    print(f"  removed {entry['quoted']!r} from {doc}: "
          f"{'caught' if caught else 'NOT CAUGHT'}")

    # The other half of what 06 added: re-wrapping must *not* fail. A check
    # that fires on a reflow is a check that gets switched off during the one
    # job it was built for, which is a rewrite.
    rewrapped = dict(texts)
    rewrapped[doc] = flatten(texts[doc]).replace(
        flatten(entry["quoted"]), flatten(entry["quoted"]).replace(" ", "\n", 1))
    survived = not any(doc in p and "edited in the writing" in p
                       for p in check(manifest, evidence, rewrapped))
    print(f"  re-wrapped {entry['quoted']!r} in {doc}: "
          f"{'still passes' if survived else 'FAILED ON A LINE BREAK'}")

    bent = json.loads(json.dumps(evidence))
    keyed = next((e for e in manifest["figures"] if e.get("key")
                  and isinstance(e.get("value"), (int, float))), None)
    caught_value = False
    if keyed:
        bent["figures"][keyed["key"]] = float(keyed["value"]) + 1.0
        problems = check(manifest, bent, texts)
        caught_value = any(keyed["key"] in p for p in problems)
        print(f"  moved {keyed['key']} by 1.0: "
              f"{'caught' if caught_value else 'NOT CAUGHT'}")

    if caught and survived and (caught_value or keyed is None):
        print("  both failure modes are detected, and a reflow is not one")
        return 0
    if not survived:
        print("  a line break fails this check, so a rewrite cannot use it")
    else:
        print("  this check cannot fail, which makes it worth nothing")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true",
                        help="plant a wrong digit and require a failure")
    parser.add_argument("--strict", action="store_true",
                        help="also fail when a document holds a number the "
                             "manifest does not account for")
    args = parser.parse_args()

    manifest = load_json(MANIFEST, "manifest")
    evidence = load_json(EVIDENCE, "evidence")
    if manifest is None:
        return 2
    if evidence is None:
        print("run scripts/evidence.py first - it writes the file this "
              "checks against")
        return 2

    docs = sorted({d for e in manifest["figures"] for d in (e.get("docs") or [])})
    texts = {d: document_text(ROOT / d) for d in docs if (ROOT / d).exists()}

    if args.self_test:
        return self_test(manifest, evidence, texts)

    print(f"manifest: {len(manifest['figures'])} figures across "
          f"{len(docs)} document(s)")
    print(f"evidence: written {evidence.get('written_at')}, "
          f"rule logic {evidence.get('rules_fingerprint')}")

    problems = check(manifest, evidence, texts)

    for doc, numbers in sorted(unchecked_numbers(texts, manifest).items()):
        if not numbers:
            print(f"  {doc}: every number accounted for")
            continue
        print(f"  {doc}: {len(numbers)} number(s) outside the manifest: "
              f"{', '.join(numbers)}")
        if args.strict:
            problems.append(
                f"{doc}: {', '.join(numbers)} appear(s) in the text and in "
                f"neither `figures` nor `not_figures`. Add the entry, or say "
                f"in `not_figures` why it is not a claim")

    print("\n=== verdict ===")
    if problems:
        print(f"  {len(problems)} problem(s):")
        for p in problems:
            print(f"    {p}")
        return 1
    print("  every figure in the manifest matches its artefact and appears "
          "in the documents that quote it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
