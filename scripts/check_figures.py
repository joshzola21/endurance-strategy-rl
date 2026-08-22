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

What it does not check
----------------------
Figures absent from the manifest. That is a real gap rather than a technicality,
and it is smaller than it looks: what belongs in the manifest is every number a
reader could act on, not every number that appears. Adding one is three lines.

It also cannot check that a number is being quoted *about the right thing*. A
correct value in a sentence that misdescribes it passes here and is caught only
by somebody reading.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
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
            elif quoted not in texts[doc]:
                problems.append(
                    f"{doc}: does not contain {quoted!r}. A number has been "
                    f"edited in the writing without the manifest following it")
    return problems


def unchecked_numbers(texts: dict[str, str], manifest: dict) -> dict[str, int]:
    """Roughly how much of each document is outside the manifest.

    Reported rather than enforced. A count that climbs while the manifest does
    not is the manifest going stale, which is worth seeing before it matters.
    """
    quoted = " ".join(e["quoted"] for e in manifest["figures"] if e.get("quoted"))
    out = {}
    for doc, text in texts.items():
        # Ignore anything inside a fenced block: those are commands and paths.
        body = re.sub(r"```.*?```", "", text, flags=re.S)
        numbers = set(re.findall(r"\b\d+\.\d+\b|\b\d{2,}%", body))
        out[doc] = len({n for n in numbers if n not in quoted})
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
    planted[doc] = texts[doc].replace(entry["quoted"], "a different number")
    problems = check(manifest, evidence, planted)
    caught = any(doc in p and "edited in the writing" in p for p in problems)
    print(f"  removed {entry['quoted']!r} from {doc}: "
          f"{'caught' if caught else 'NOT CAUGHT'}")

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

    if caught and (caught_value or keyed is None):
        print("  both failure modes are detected")
        return 0
    print("  this check cannot fail, which makes it worth nothing")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true",
                        help="plant a wrong digit and require a failure")
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
    texts = {d: (ROOT / d).read_text() for d in docs if (ROOT / d).exists()}

    if args.self_test:
        return self_test(manifest, evidence, texts)

    print(f"manifest: {len(manifest['figures'])} figures across "
          f"{len(docs)} document(s)")
    print(f"evidence: written {evidence.get('written_at')}, "
          f"rule logic {evidence.get('rules_fingerprint')}")

    problems = check(manifest, evidence, texts)
    for doc, n in sorted(unchecked_numbers(texts, manifest).items()):
        print(f"  {doc}: {n} number(s) outside the manifest")

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
