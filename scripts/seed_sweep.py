"""Train and evaluate the same configuration five times, and keep all of it.

Every agent number this project has reported is one training run at seed 0.
Four configuration changes have each fixed one series and broken the other,
and the swings between them are three to five times the width of the
evaluation interval - so they are either real effects or training-seed
variance, and nothing so far distinguishes the two. This measures the second.

    python scripts/seed_sweep.py --dry-run     # see the plan, run nothing
    python scripts/seed_sweep.py               # about two hours
    python scripts/collect_seed_spread.py      # read the answer

`train.py` and `evaluate.py` each write to one fixed filename, so a second run
overwrites the first. This moves each artefact aside as it appears, then puts
each policy back in place one at a time to be scored. The live policy is backed
up first and restored at the end, so a sweep leaves the tree as it found it.

**Resumable.** Anything already in `outputs/seed_sweep/` is skipped, so an
interrupted sweep continues rather than starting over. Delete a file to redo
just that run.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "outputs" / "seed_sweep"
BACKUP = SWEEP / "_restore"

SERIES = ("imsa", "wec")
BANKS = ("headline", "held_out")
STEM = "{series}_maskable_ppo"
SUFFIXES = (".zip", ".onnx", ".card.json")


def find_dir(pattern: str, fallbacks: tuple[str, ...]) -> Path:
    """Where a kind of artefact lives, found rather than assumed."""
    for name in fallbacks:
        folder = ROOT / name
        if folder.is_dir() and any(folder.glob(pattern)):
            return folder
    hits = [p.parent for p in ROOT.rglob(pattern)
            if not any(part.startswith(".") for part in p.parts)]
    if hits:
        return hits[0]
    return ROOT / fallbacks[0]


POLICIES = find_dir("*_maskable_ppo.onnx", ("outputs/policies", "checkpoints"))
EVALS = find_dir("*_headline_summary.csv", ("outputs/evaluation", "results"))


def run(cmd: list[str], dry: bool) -> bool:
    print("    $ " + " ".join(cmd))
    if dry:
        return True
    started = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT)
    print(f"      ({time.perf_counter() - started:.0f} s)")
    if result.returncode != 0:
        print(f"      FAILED with exit code {result.returncode}")
        return False
    return True


def trained_paths(series: str, seed: int) -> list[Path]:
    return [SWEEP / f"{series}_s{seed}{suffix}" for suffix in SUFFIXES]


def backup_live(dry: bool) -> None:
    """Keep whatever is currently in place, so the sweep is reversible."""
    if BACKUP.exists() or dry:
        return
    BACKUP.mkdir(parents=True, exist_ok=True)
    for series in SERIES:
        for suffix in SUFFIXES:
            live = POLICIES / (STEM.format(series=series) + suffix)
            if live.exists():
                shutil.copy2(live, BACKUP / live.name)
    print(f"  backed up the live policies to "
          f"{BACKUP.relative_to(ROOT)}/")


def restore_live(dry: bool) -> None:
    if dry or not BACKUP.is_dir():
        return
    for path in BACKUP.glob("*"):
        shutil.copy2(path, POLICIES / path.name)
    print(f"\nrestored the policies that were in place before the sweep")


def train(series: str, seed: int, timesteps: int | None, dry: bool) -> bool:
    """One training run, moved aside under its seed."""
    want = trained_paths(series, seed)
    if all(p.exists() for p in want):
        print(f"  {series} seed {seed}: already trained, skipping")
        return True

    cmd = [sys.executable, "scripts/train.py", series, "--seed", str(seed)]
    if timesteps:
        cmd += ["--timesteps", str(timesteps)]
    if not run(cmd, dry):
        return False
    if dry:
        return True

    for suffix, dest in zip(SUFFIXES, want):
        src = POLICIES / (STEM.format(series=series) + suffix)
        if not src.exists():
            print(f"      expected {src.name} and it is not there")
            return False
        shutil.move(str(src), dest)
    print(f"      -> {', '.join(p.name for p in want)}")
    return True


def evaluate(series: str, seed: int, dry: bool) -> bool:
    """Put one policy back, score it, tag the tables with its seed."""
    wanted = [SWEEP / f"{series}_{bank}_summary_s{seed}.csv" for bank in BANKS]
    if all(p.exists() for p in wanted):
        print(f"  {series} seed {seed}: already evaluated, skipping")
        return True

    if not dry:
        for suffix in SUFFIXES:
            src = SWEEP / f"{series}_s{seed}{suffix}"
            if not src.exists():
                print(f"      no {src.name}; train it first")
                return False
            shutil.copy2(src, POLICIES / (STEM.format(series=series) + suffix))

    cmd = [sys.executable, "scripts/evaluate.py", series,
           "--bank", *BANKS]
    if not run(cmd, dry):
        return False
    if dry:
        return True

    for bank in BANKS:
        for kind in ("summary", "rows"):
            src = EVALS / f"{series}_{bank}_{kind}.csv"
            if src.exists():
                shutil.copy2(src, SWEEP / f"{series}_{bank}_{kind}_s{seed}.csv")
        prov = EVALS / f"{series}_{bank}_provenance.json"
        if prov.exists():
            shutil.copy2(prov, SWEEP / f"{series}_{bank}_provenance_s{seed}.json")
    print(f"      -> tables tagged _s{seed}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("series", nargs="*", default=list(SERIES),
                    help="imsa, wec, or nothing for both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="training seeds (default 0 1 2 3 4)")
    ap.add_argument("--timesteps", type=int, default=None,
                    help="passed through to train.py; omit for its default")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and run nothing")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    series = [s for s in args.series if s in SERIES]
    if not series:
        print(f"unknown series {args.series}; expected any of {SERIES}")
        return 2

    print(f"policies: {POLICIES.relative_to(ROOT)}")
    print(f"tables:   {EVALS.relative_to(ROOT)}")
    print(f"sweep:    {SWEEP.relative_to(ROOT)}")
    print(f"{len(series)} series x {len(args.seeds)} seeds = "
          f"{len(series) * len(args.seeds)} runs\n")

    SWEEP.mkdir(parents=True, exist_ok=True)
    backup_live(args.dry_run)

    failed = []
    if not args.skip_train:
        print("\n--- training ---")
        for s in series:
            for seed in args.seeds:
                if not train(s, seed, args.timesteps, args.dry_run):
                    failed.append(f"train {s} s{seed}")

    if not args.skip_eval:
        print("\n--- evaluating ---")
        for s in series:
            for seed in args.seeds:
                if not evaluate(s, seed, args.dry_run):
                    failed.append(f"evaluate {s} s{seed}")

    restore_live(args.dry_run)

    print("\n=== done ===")
    if failed:
        print("  these did not complete: " + ", ".join(failed))
        print("  fix and rerun - everything finished is skipped")
        return 1
    if args.dry_run:
        print("  dry run; nothing was executed")
    else:
        print("  now: python scripts/collect_seed_spread.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
