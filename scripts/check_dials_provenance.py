"""Did these dials come out of the lap data, or were they a stand-in?

Every evaluation provenance file and both policy cards say `"dials_source":
"STAND-IN, not calibrated"`, while the config they fingerprint carries a
`source_event` and an `n_laps_observed` - which are calibration outputs. One
of those is stale and nobody remembers which.

The artefacts can settle it without anybody remembering. `dials_fingerprint`
is a hash of every number in a config, so recalculating the dials from the
raw timing and hashing them either reproduces the frozen fingerprint or it
does not.

    python scripts/check_dials_provenance.py

A hash says *that* two things differ and never *what*, so on a mismatch this
falls back to a field-by-field comparison, which is the answer actually
wanted: a difference in `name` alone means the dials are calibrated and the
race was retitled, while a difference in `base_pace_s` means they are not
calibrated at all.

Nothing here writes. It reads the lap data, builds a config in memory and
compares.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from endurance import calibrate                                  # noqa: E402
from endurance.assets import dials_fingerprint                   # noqa: E402
from endurance.params import RaceConfig                          # noqa: E402

LAPS = ROOT / "data" / "raw" / "laps.csv"

# Which session each race was calibrated from. The fingerprint is deliberately
# *not* recorded here: it moves whenever the dials do, and a script that
# carried a hard-coded copy would start reporting a stale expectation as a
# finding - which it did, once, immediately after the post-fix re-run. What is
# checked is that the config on disk regenerates from the raw timing, which is
# a question about the data rather than about any particular freeze.
FROZEN = {
    "imsa": {"session": 682, "event": "%daytona%", "name": "Daytona 24 2026"},
    "wec": {"session": 1000, "event": "%le mans%", "name": "Le Mans 24 2026"},
}


def find_config(series_code: str) -> Path | None:
    """The frozen config, wherever it is in this tree."""
    for pattern in (f"{series_code}.json", f"{series_code}_standin.json"):
        for hit in ROOT.rglob(pattern):
            if ".ipynb_checkpoints" not in hit.parts:
                return hit
    return None


# `build_race_config(con, series_code, session_ids, name=None, classes=None,
#                    min_cars=3, legacy_cautions=False, trim_factor=3.0)`
#
# Three of those defaults are choices 00 made and did not write down anywhere
# the config can be checked against. So if the defaults do not reproduce the
# frozen dials, this searches the small grid below before concluding anything:
# "the defaults do not reproduce it" and "nothing reproduces it" are very
# different findings, and only the second means the dials are a stand-in.
GRID = [{"legacy_cautions": legacy, "trim_factor": trim}
        for legacy in (False, True)
        for trim in (3.0, 2.5, 5.0)]


def build(con, series_code: str, frozen: RaceConfig, session_ids: list[int],
          **extra) -> RaceConfig:
    """One call to `build_race_config`, scoped to the frozen race.

    `name` and `classes` are taken from the frozen config rather than left to
    default, so that a difference in the *title* or in which classes cleared
    `min_cars` cannot be mistaken for a difference in the dials. The dials are
    the question.
    """
    return calibrate.build_race_config(
        con, series_code, session_ids,
        name=frozen.name,
        classes=[c.class_name for c in frozen.classes],
        **extra)


def reproduce(con, series_code: str, frozen: RaceConfig, frozen_fp: str,
              session_ids: list[int]):
    """Try the defaults, then the grid. Returns (config, settings) or (None, None)."""
    print(f"  trying the defaults …", flush=True)
    try:
        first = build(con, series_code, frozen, session_ids)
    except Exception as e:                          # noqa: BLE001 - report it
        print(f"  could not rebuild at all: {type(e).__name__}: {e}")
        return None, None

    if dials_fingerprint(first) == frozen_fp:
        return first, {}

    print(f"  defaults give {dials_fingerprint(first)}, not {frozen_fp}. "
          f"Searching {len(GRID)} settings …", flush=True)
    for settings in GRID:
        if settings == {"legacy_cautions": False, "trim_factor": 3.0}:
            continue                                # already done above
        try:
            candidate = build(con, series_code, frozen, session_ids, **settings)
        except Exception as e:                      # noqa: BLE001
            print(f"    {settings}: {type(e).__name__}: {e}")
            continue
        mark = dials_fingerprint(candidate)
        hit = mark == frozen_fp
        print(f"    {settings} -> {mark}{'  MATCH' if hit else ''}", flush=True)
        if hit:
            return candidate, settings
    return first, None


def differences(frozen: dict, rebuilt: dict) -> list[str]:
    """Which numbers moved, and by how much. Classes matched by name."""
    out = []
    for key in sorted(set(frozen) | set(rebuilt)):
        if key == "classes":
            continue
        if frozen.get(key) != rebuilt.get(key):
            out.append(f"    {key}: frozen {frozen.get(key)!r} "
                       f"vs rebuilt {rebuilt.get(key)!r}")

    by_name = {c["class_name"]: c for c in rebuilt.get("classes", [])}
    for cls in frozen.get("classes", []):
        other = by_name.get(cls["class_name"])
        if other is None:
            out.append(f"    class {cls['class_name']}: missing from rebuild")
            continue
        for key in sorted(set(cls) | set(other)):
            a, b = cls.get(key), other.get(key)
            if a == b:
                continue
            if isinstance(a, float) and isinstance(b, float):
                out.append(f"    {cls['class_name']}.{key}: {a!r} vs {b!r} "
                           f"(delta {b - a:+.6g})")
            else:
                out.append(f"    {cls['class_name']}.{key}: {a!r} vs {b!r}")
    extra = set(by_name) - {c["class_name"] for c in frozen.get("classes", [])}
    out += [f"    class {name}: in the rebuild but not the frozen config"
            for name in sorted(extra)]
    return out


def main() -> int:
    if not LAPS.exists():
        print(f"no lap data at {LAPS}. Point LAPS at it and rerun.")
        return 2

    print(f"lap data: {LAPS}")
    print(f"build_race_config{inspect.signature(calibrate.build_race_config)}")
    con = calibrate.connect(str(LAPS))
    verdicts = {}

    for series_code, spec in FROZEN.items():
        print(f"\n=== {series_code.upper()} ===")
        path = find_config(series_code)
        if path is None:
            print(f"  no frozen config found for {series_code}; skipped")
            continue

        frozen = RaceConfig.load(path)
        frozen_fp = dials_fingerprint(frozen)
        print(f"  frozen:   {path.relative_to(ROOT)}  ->  {frozen_fp}")

        rebuilt, settings = reproduce(con, series_code, frozen, frozen_fp,
                                      [spec["session"]])
        if rebuilt is None:
            print("  send me the signature printed above and I will fix the "
                  "call.")
            verdicts[series_code] = "could not rebuild"
            continue

        rebuilt_fp = dials_fingerprint(rebuilt)
        print(f"  rebuilt:  from session {spec['session']}  ->  {rebuilt_fp}")

        if rebuilt_fp == frozen_fp:
            how = ("the default settings" if settings == {}
                   else f"settings {settings}")
            print(f"  IDENTICAL, under {how}. These dials came out of the lap "
                  f"data; the STAND-IN label is stale.")
            verdicts[series_code] = f"calibrated ({how})"
            continue

        diffs = differences(frozen.to_dict(), rebuilt.to_dict())
        print(f"  DIFFERENT, in {len(diffs)} field(s):")
        for line in diffs[:40]:
            print(line)
        if len(diffs) > 40:
            print(f"    ... and {len(diffs) - 40} more")

        # `name` and `classes` are pinned to the frozen config, so anything
        # left here is either a derived label or a genuine dial.
        dial_fields = [d for d in diffs
                       if not any(k in d for k in ("name:", "source_event",
                                                   "n_laps_observed"))]
        tiny = dial_fields and all(
            "delta " in d and abs(float(d.rsplit("delta ", 1)[1].rstrip(")")))
            < 1e-9 for d in dial_fields)

        if not dial_fields:
            print("  ...but every difference is a label, not a number. The "
                  "dials are calibrated; only the provenance strings moved.")
            verdicts[series_code] = "calibrated, labels differ"
        elif tiny:
            print("  ...and every numeric difference is below 1e-9, which is "
                  "floating-point summation order rather than a different "
                  "calibration. The dials are calibrated.")
            verdicts[series_code] = "calibrated, float noise only"
        else:
            print("  ...and the differences include real dials, at a size no "
                  "rounding explains. Either these were not built from this "
                  "session, or 00 used settings outside the grid above.")
            verdicts[series_code] = "not reproduced"

    print("\n=== verdict ===")
    for series_code, verdict in verdicts.items():
        print(f"  {series_code}: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
