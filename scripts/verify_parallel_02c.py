"""Step 1's gate: parallelising 02c changed the runtime and nothing else.

Three conditions, each with the property that a plausible bug would break it.

1. **Serial equals parallel, cell for cell.** Same seeds, same dials, same
   roster, compared as a frame rather than by eye. A race that picked up
   anything from a neighbour - a shared RNG, a mutated config, a reused
   strategy instance - shows up here.
2. **Both equal what is already on disk.** 03b's saved rows are the reference
   every published number came from. Serial matching parallel proves the two
   paths agree; only this proves they agree with the project.
3. **The null is run once per seed, not once per arm.** Asserted by counting
   engine constructions rather than inferred from the runtime, because a cache
   that silently stopped working would look like nothing at all.

Run:  python scripts/verify_parallel_02c.py [n_seeds] [n_workers]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                              # noqa: E402

from endurance import harness                                    # noqa: E402
from endurance.assets import BackgroundField, SeedBank           # noqa: E402
from endurance.params import RaceConfig                          # noqa: E402

SERIES = ("imsa", "wec")


def find(name: str) -> Path | None:
    for hit in ROOT.rglob(name):
        if ".ipynb_checkpoints" not in hit.parts:
            return hit
    return None


def load(code: str):
    return (RaceConfig.load(find(f"{code}.json")),
            SeedBank.load(find(f"{code}_seeds.json")),
            BackgroundField.load(find(f"{code}_field.json")))


# Amendment 22: comparing saved floats for exact equality fails for reasons
# that have nothing to do with strategy - re-running under a different NumPy
# moves a summation by ~1e-13. So the *integer* columns, which carry every
# claim this project makes, must match exactly, and the float columns are
# allowed a tolerance well below anything a strategy could produce. The first
# version of this script compared everything exactly and reported a false
# alarm on its first run, which is the amendment earning its place.
EXACT = ("class_pos", "overall_pos", "laps", "stops", "caution_laps",
         "d_class_pos", "d_laps", "d_stops", "seed", "pace_rank")
FLOAT_TOL = 1e-9


def identical(a: pd.DataFrame, b: pd.DataFrame, label: str,
              tolerant: bool = False) -> bool:
    """Every cell, after aligning on the keys rather than on row order.

    `tolerant` allows `FLOAT_TOL` on the non-integer columns, and is for
    comparisons against a table written by a different interpreter. Two frames
    built in one process must match exactly and are compared with it off.
    """
    keys = ["series", "strategy", "seed"]
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)

    shared = [c for c in a.columns if c in b.columns]
    missing = sorted(set(a.columns) ^ set(b.columns))
    if missing:
        print(f"    columns only on one side: {missing}")

    if len(a) != len(b):
        print(f"  {label}: {len(a)} rows vs {len(b)}")
        return False

    bad, noise = [], []
    for col in shared:
        left, right = a[col], b[col]
        if left.equals(right):
            continue
        if not pd.api.types.is_numeric_dtype(left):
            bad.append(col)
            continue
        delta = float((left - right).abs().max())
        if tolerant and col not in EXACT and delta < FLOAT_TOL:
            noise.append(f"{col} {delta:.2g}")
        else:
            bad.append(f"{col} (max delta {delta:.3g})")

    if bad:
        print(f"  {label}: DIFFERS on {', '.join(bad)}")
        return False
    if noise:
        print(f"  {label}: every integer column identical across {len(a)} "
              f"rows; float columns within {FLOAT_TOL:g} "
              f"({', '.join(noise)}) - summation order, per amendment 22")
    else:
        print(f"  {label}: identical across {len(shared)} columns, "
              f"{len(a)} rows")
    return True


def count_engines(fn):
    """How many races were actually run, counted at the constructor."""
    from endurance import engine as engine_module

    original = engine_module.RaceEngine.__init__
    calls = {"n": 0}

    def counted(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    engine_module.RaceEngine.__init__ = counted
    try:
        out = fn()
    finally:
        engine_module.RaceEngine.__init__ = original
    return out, calls["n"]


def main() -> int:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if n_workers <= 0:
        import os
        n_workers = max(1, (os.cpu_count() or 1) - 1)
    print(f"{n_seeds} seeds per series, {n_workers} workers\n")

    ok = True
    for code in SERIES:
        print(f"=== {code.upper()} ===")
        config, bank, field = load(code)
        seeds = bank.headline[:n_seeds]

        t0 = time.perf_counter()
        serial, engines = count_engines(
            lambda: harness.compare_roster(config, seeds, field))
        t_serial = time.perf_counter() - t0

        # One null plus four arms per seed, plus one field build per seed for
        # `focal_car`. Anything more means the null is being recomputed.
        expected = n_seeds * (len(harness.ROSTER) + 1)
        print(f"  races run: {engines} (expected {expected} = "
              f"{n_seeds} seeds x {len(harness.ROSTER)} arms + "
              f"{n_seeds} focal-car field builds)")
        if engines != expected:
            print("  the null cache is not doing its job")
            ok = False

        t0 = time.perf_counter()
        parallel = harness.compare_roster(config, seeds, field,
                                          n_workers=n_workers)
        t_parallel = time.perf_counter() - t0

        print(f"  serial {t_serial:.1f}s -> parallel {t_parallel:.1f}s "
              f"({t_serial / max(t_parallel, 1e-9):.2f}x on {n_workers} "
              f"workers)")
        print(f"  projected for 200 seeds: {t_parallel / n_seeds * 200 / 60:.1f} min")

        ok &= identical(serial.rows, parallel.rows, "serial vs parallel")

        saved = find(f"{code}_headline_rows.csv")
        if saved is None:
            print("  no saved rows found; skipped the disk comparison")
        else:
            on_disk = pd.read_csv(saved)
            on_disk = on_disk[on_disk["seed"].isin(seeds)
                              & on_disk["strategy"].isin(harness.ROSTER)]
            ok &= identical(serial.rows, on_disk, "serial vs 03b on disk",
                            tolerant=True)
        print()

    # The refusal, which is a feature rather than a limitation.
    config, bank, field = load("imsa")
    try:
        harness.compare_roster(config, bank.headline[:2], field,
                               roster={**harness.ROSTER, "agent": lambda: None},
                               n_workers=2)
    except ValueError as e:
        print(f"non-ROSTER member refused for parallel: {str(e)[:70]}…")
    else:
        print("a non-ROSTER roster was NOT refused")
        ok = False

    print("\n=== verdict ===")
    print("  parallelising 02c changed the runtime and nothing else"
          if ok else "  SOMETHING MOVED - do not adopt")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
