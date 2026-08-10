"""Did the re-run land on disk, or only in a transcript?

Read-only. Touches nothing in the project: the notebooks are regenerated into
a temporary directory for comparison, never over the real ones.

    python verify_00_landed.py > verify_report.txt

Answers, in order: is the working tree what git thinks it is; do the frozen
dials describe one real race; do the fingerprints agree across every artefact
that carries one; is the modification order consistent with the claimed
sequence; do the notebooks on disk match what `build_nb.py` generates; and do
the tests pass.

Nothing here trusts a summary. Every line comes from a file.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path.cwd()

NOTEBOOKS = {
    "00": "00_data_recon.ipynb",
    "01": "01_race_engine.ipynb",
    "02a": "02a_engine_corrections.ipynb",
    "02b": "02b_benchmark.ipynb",
    "02c": "02c_strategies.ipynb",
    "03a": "03a_rl_setup.ipynb",
    "03b": "03b_rl_training.ipynb",
}


def head(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True,
                             text=True, timeout=900)
        return (out.stdout + out.stderr).strip()
    except Exception as exc:                                  # noqa: BLE001
        return f"<could not run {' '.join(cmd)}: {exc}>"


def age(p: Path) -> str:
    if not p.exists():
        return "MISSING"
    m = p.stat().st_mtime
    return f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(m))}"


# ----------------------------------------------------------------------
def section_git() -> None:
    head("A. Git — did files on disk actually change?")
    if not (ROOT / ".git").exists():
        print("no git repository; skipping. Everything below is still valid, "
              "but there is no record of what changed when.")
        return
    print("--- git log, last 15\n" + run(["git", "log", "--oneline", "-15"]))
    print("\n--- git status --porcelain (empty means nothing uncommitted)\n"
          + (run(["git", "status", "--porcelain"]) or "(clean)"))
    print("\n--- files changed in the working tree or last commit, by area")
    print(run(["git", "diff", "--stat", "HEAD~1", "--",
               "src", "scripts", "tests", "notebooks", "data/processed"]))


# ----------------------------------------------------------------------
def section_dials() -> None:
    head("B. The frozen dials — do they describe one real race?")
    for series in ("imsa", "wec"):
        p = ROOT / "data" / "processed" / f"{series}.json"
        print(f"\n--- {p}   (modified {age(p)})")
        if not p.exists():
            print("  MISSING — nothing downstream can be trusted")
            continue
        d = json.loads(p.read_text())
        dur = d["duration_s"]
        print(f"  name        : {d.get('name')}")
        print(f"  duration    : {dur:,.0f} s = {dur / 3600:.2f} h "
              f"= {dur / 86400:.3f} x 24 h")
        print(f"  classes     : {len(d['classes'])}, "
              f"{sum(c['n_cars'] for c in d['classes'])} cars")
        for c in d["classes"]:
            print(f"    {c['class_name']:9s} pace={c['base_pace_s']:7.2f} "
                  f"deg={c['deg_slope_s_per_lap']:+.5f} "
                  f"stint={c['green_stint_laps']:5.1f} "
                  f"pit={c['pit_time_mean_s']:6.2f} sd={c['pit_time_std_s']:6.2f} "
                  f"cars={c['n_cars']:3d} caution={c['caution_rate']:.3f} "
                  f"| {c['source_event']}")

        # The tells. Any of these means the file is not what it claims.
        bad = []
        if abs(dur - 86400) / 86400 > 0.02:
            bad.append(f"duration is {dur / 3600:.1f} h, not ~24 h")
        srcs = {c["source_event"] for c in d["classes"]}
        if len(srcs) != 1:
            bad.append(f"classes disagree about their source: {srcs}")
        src = next(iter(srcs))
        if "%" in src:
            bad.append("source_event still holds an ILIKE pattern")
        if "POOLED" in src:
            bad.append("source_event says POOLED — more than one session")
        if "session" not in src:
            bad.append("source_event names no session id")
        if len({c["caution_rate"] for c in d["classes"]}) != 1:
            bad.append("caution_rate differs between classes — calibrated "
                       "per class rather than per race")
        if any(c["pit_time_std_s"] > c["pit_time_mean_s"] for c in d["classes"]):
            bad.append("a pit sd exceeds its own mean — untrimmed")
        print("  VERDICT     : " + ("OK" if not bad else "PROBLEMS"))
        for b in bad:
            print(f"    - {b}")


# ----------------------------------------------------------------------
def section_fingerprints() -> None:
    head("C. Fingerprints — does every artefact agree on which dials it used?")
    found: dict[str, list[str]] = {}
    for p in sorted(ROOT.rglob("*")):
        if p.is_dir() or p.suffix not in (".json", ".txt", ".yaml", ".yml"):
            continue
        if ".git" in p.parts or "node_modules" in p.parts:
            continue
        try:
            text = p.read_text(errors="ignore")
        except Exception:                                     # noqa: BLE001
            continue
        if "dials_fingerprint" not in text:
            continue
        try:
            obj = json.loads(text)
        except Exception:                                     # noqa: BLE001
            continue

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == "dials_fingerprint":
                        yield str(v)
                    else:
                        yield from walk(v)
            elif isinstance(o, list):
                for v in o:
                    yield from walk(v)

        for fp in walk(obj):
            found.setdefault(fp, []).append(
                f"{p.relative_to(ROOT)}  ({age(p)})")

    if not found:
        print("no artefact on disk carries a dials_fingerprint.")
        print("If freeze_assets.py and the policy cards were re-run, at least")
        print("one should. This is the single strongest end-to-end check and")
        print("its absence is worth explaining.")
        return

    for fp, where in found.items():
        print(f"\n  fingerprint {fp}")
        for w in where:
            print(f"    {w}")
    if len(found) > 1:
        print(f"\n  VERDICT: {len(found)} DIFFERENT fingerprints on disk. "
              "Artefacts built against different dials are being used together.")
    else:
        print("\n  VERDICT: one fingerprint everywhere — consistent.")


# ----------------------------------------------------------------------
def section_order() -> None:
    head("D. Modification order — is it consistent with the claimed sequence?")
    watch = [
        "data/raw/laps.csv",
        "src/endurance/calibrate.py",
        "src/endurance/gate00.py",
        "scripts/freeze_assets.py",
        "build_nb.py",
        "data/processed/imsa.json",
        "data/processed/wec.json",
    ]
    watch += [f"notebooks/{n}" for n in NOTEBOOKS.values()]
    for pat in ("data/**/*bank*", "data/**/*field*", "**/*.pt", "**/*.ckpt",
                "**/*policy*.json", "**/*card*.json"):
        watch += [str(p.relative_to(ROOT)) for p in ROOT.glob(pat)
                  if p.is_file() and ".git" not in p.parts]

    rows = []
    for rel in dict.fromkeys(watch):
        p = ROOT / rel
        rows.append((p.stat().st_mtime if p.exists() else 0, rel, age(p)))
    for mtime, rel, when in sorted(rows):
        print(f"  {when:>16}  {rel}")
    print("\n  Expected order: laps.csv, then source, then the processed JSONs,")
    print("  then the frozen assets, then the policy checkpoints, then the")
    print("  notebooks. A checkpoint older than imsa.json was not retrained.")


# ----------------------------------------------------------------------
def section_notebooks() -> None:
    head("E. Notebooks — do they match build_nb.py, and were they really run?")
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(ROOT / "build_nb.py", tmp / "build_nb.py")
    (tmp / "notebooks").mkdir(exist_ok=True)
    gen = run([sys.executable, "build_nb.py", *NOTEBOOKS.keys()], cwd=tmp)
    print("--- regenerated into a temp dir (the real notebooks are untouched)")
    print("   " + gen.replace("\n", "\n   ")[:1500])

    for key, name in NOTEBOOKS.items():
        real, fresh = ROOT / "notebooks" / name, tmp / "notebooks" / name
        print(f"\n--- {name}")
        if not real.exists():
            print("  MISSING on disk")
            continue
        nb = json.loads(real.read_text())
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        ran = [c for c in code if c.get("execution_count")]
        with_out = [c for c in code if c.get("outputs")]
        counts = [c["execution_count"] for c in ran]
        print(f"  {len(code)} code cells, {len(ran)} executed, "
              f"{len(with_out)} carrying output   (modified {age(real)})")
        if counts and counts != sorted(counts):
            print("  execution counts are NOT monotonic — cells were run out "
                  "of order or re-run piecemeal")
        if len(ran) < len(code):
            print(f"  {len(code) - len(ran)} code cells were never executed")

        if not fresh.exists():
            print("  build_nb.py has no target for this notebook")
            continue
        fnb = json.loads(fresh.read_text())
        fcode = ["".join(c["source"]) for c in fnb["cells"]
                 if c["cell_type"] == "code"]
        rcode = ["".join(c["source"]) for c in code]
        if rcode == fcode:
            print("  code cells MATCH build_nb.py")
        else:
            print(f"  code cells DIFFER from build_nb.py "
                  f"({len(rcode)} on disk vs {len(fcode)} generated)")
            for i, (a, b) in enumerate(zip(rcode, fcode)):
                if a != b:
                    print(f"    first difference at code cell {i}:")
                    print(f"      on disk  : {a.splitlines()[0][:90]}")
                    print(f"      generated: {b.splitlines()[0][:90]}")
                    break
            print("    -> a hand edit in the .ipynb is lost on the next build")


# ----------------------------------------------------------------------
def section_source() -> None:
    head("F. Source — is the code on disk the code that was agreed?")
    checks = {
        "src/endurance/calibrate.py": [
            ("types={'car': 'VARCHAR'}", "car read as text"),
            ("def find_race", "session resolver present"),
            ("def _fuel_stints", "stints from pit records"),
            ("CAUTION_FLAGS", "named caution set"),
            ("event ILIKE", "!! ILIKE scoping still reachable"),
        ],
        "src/endurance/gate00.py": [
            ("GREEN", "GREEN referenced"),
            ("from .calibrate import", "calibrate import line"),
            ("stint dial inside observed stints", "replacement condition two"),
            ("EXEMPT", "condition four exemption"),
            ("def stint_diagnostic", "diagnostic present"),
        ],
        "scripts/freeze_assets.py": [
            ("data/processed", "reads the frozen dials"),
            ("6", "any literal 6 — check by eye for a hardcoded 6h race"),
        ],
        "tests/sqlite_shim.py": [("TEXT_COLUMNS", "car read as text in tests")],
    }
    for rel, items in checks.items():
        p = ROOT / rel
        print(f"\n--- {rel}   (modified {age(p)})")
        if not p.exists():
            print("  MISSING")
            continue
        text = p.read_text()
        for needle, label in items:
            print(f"  {'yes' if needle in text else 'NO ':>3}  {label}")

    g = ROOT / "src" / "endurance" / "gate00.py"
    if g.exists():
        text = g.read_text()
        if "GREEN" in text and "import GREEN" not in text \
                and "GREEN =" not in text:
            print("\n  !! gate00.py uses GREEN but does not import or define it")
        print("\n--- gate00.py imports")
        for line in text.splitlines():
            if line.startswith(("import ", "from ")):
                print("    " + line)


# ----------------------------------------------------------------------
def section_tests() -> None:
    head("G. Tests")
    out = run([sys.executable, "-m", "pytest", "-q", "--no-header",
               "tests"], cwd=ROOT)
    print(out[-4000:] if out else "<no output>")


def main() -> None:
    print(f"verify_00_landed — {ROOT}")
    print(f"run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"python {sys.version.split()[0]}, cwd {os.getcwd()}")
    for fn in (section_git, section_dials, section_fingerprints, section_order,
               section_notebooks, section_source, section_tests):
        try:
            fn()
        except Exception as exc:                              # noqa: BLE001
            head(f"{fn.__name__} FAILED")
            import traceback
            traceback.print_exc()
            print(exc)


if __name__ == "__main__":
    main()
