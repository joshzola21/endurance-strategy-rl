"""Write the race, the seed banks and the background field to disk, once.

Decision 6 asks for artefacts rather than a notebook, and 03b is the stage
where that stops being tidiness. Training, evaluation and the notebook all
have to be about the *same* races or the agent's row is not comparable with
the five human rows beside it, and three files that each rebuild the config
in memory will agree right up until one of them is edited.

So this script holds the only definition of the stand-in race in the
project, and everything downstream loads what it wrote.

Real dials, when they exist
----------------------------
`data/processed/imsa.json` and `wec.json` are 00's frozen output: a single
scoped edition, gated on four conditions before they were allowed to write.
When a series' file is present, `freeze()` loads it and that is what gets
banked - real Daytona, real Le Mans.

Why a stand-in otherwise
-------------------------
Before 00 was re-run, `imsa.json` and `wec.json` described a 216-hour,
nine-editions-pooled race with green stints at 199 laps and degradation of
the wrong sign - the fault was in the event scoping in `calibrate.py`, not
in the dial arithmetic, and those files could not be the thing 03b trained
against. The six-hour stand-in below is what 02b, 02c and 03a worked against
instead, and it remains the fallback for a series whose real config is
missing or fails its gate. **Any number produced from the stand-in is about
a stand-in**, and the notebook and 03b say so wherever one appears.

Usage
-----
    python scripts/freeze_assets.py            # both series
    python scripts/freeze_assets.py imsa       # one
    python scripts/freeze_assets.py --force    # overwrite what is there

Refusing to overwrite by default is the point of the script. A bank silently
redrawn between training and evaluation is two different experiments wearing
one set of seed numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

def find_project_root(marker: str = "src/endurance") -> Path:
    """Walk up until the project appears, as the notebooks do.

    Not `parents[1]`. A hard-coded depth is correct only while the file sits
    where its author left it, and silently resolves to the wrong folder the
    moment somebody moves it - which puts a non-existent `src` on the path
    and reports a missing package rather than a misplaced script.
    """
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(
        f"could not find {marker!r} at or above {here.parent}.\n\n"
        "This script belongs in `scripts/`, beside `src/` and `notebooks/`:\n"
        "  endurance-strategy-rl/\n"
        "    data/processed/\n"
        "    notebooks/\n"
        "    scripts/          this file\n"
        "    src/endurance/    the package\n")


ROOT = find_project_root()
sys.path.insert(0, str(ROOT / "src"))

from endurance import ClassDials, RaceConfig                    # noqa: E402
from endurance.assets import (                                  # noqa: E402
    BackgroundField,
    SeedBank,
    dials_fingerprint,
    draw_seed_bank,
    freeze_background,
)

SERIES = ("imsa", "wec")
PROCESSED = ROOT / "data" / "processed"

# Per series rather than an index into a loop, per 02c's second settled
# point. An index makes the bank depend on the order the dict was built in,
# which is exactly the kind of dependency nobody notices until it moves.
DRAW_SEEDS = {"imsa": 20260806, "wec": 20260807}

BACKGROUND = "fuel_window"      # decision 10's null, so field and null agree


def stand_in(series_code: str) -> RaceConfig:
    """The six-hour config 02b and 02c worked against.

    Copied forward unchanged from 03a's notebook and given a single home.
    Two classes rather than a full grid because the comparison is a class
    position in the headline class, and a fuller field costs runtime without
    changing what is being measured.
    """
    quick = ClassDials(
        series_code=series_code,
        class_name="GTP" if series_code == "imsa" else "HYPERCAR",
        base_pace_s=97.5, deg_slope_s_per_lap=0.012, pace_spread_s=0.5,
        lap_noise_s=0.5, caution_rate=0.20, caution_mean_dur_s=600.0,
        green_stint_laps=30.0, fuel_per_lap=1 / 30,
        fuel_per_lap_caution=0.6 / 30, tyre_life_laps=60.0,
        pit_time_mean_s=47.0, pit_time_std_s=2.0, n_cars=8)
    gt = ClassDials(
        series_code=series_code,
        class_name="GTD" if series_code == "imsa" else "LMGT3",
        base_pace_s=112.0, deg_slope_s_per_lap=0.02, pace_spread_s=0.9,
        lap_noise_s=0.6, caution_rate=0.20, caution_mean_dur_s=600.0,
        green_stint_laps=28.0, fuel_per_lap=1 / 28,
        fuel_per_lap_caution=0.6 / 28, tyre_life_laps=56.0,
        pit_time_mean_s=45.0, pit_time_std_s=2.0, n_cars=10)
    return RaceConfig(name=f"{series_code} 6h stand-in",
                      series_code=series_code, duration_s=6 * 3600.0,
                      classes=[quick, gt])


# ----------------------------------------------------------------------
# Where things live, in one place so nothing has to guess
# ----------------------------------------------------------------------
def config_path(series_code: str, processed: Path = PROCESSED) -> Path:
    """Where the banked config is **written**.

    Deliberately not `{series}.json` - that name is 00's own output, read
    directly by `freeze()` when present rather than copied here, so there is
    one place a reader checks to see which dials a bank was drawn against.

    It used to be `{series}_standin.json`, and by 05 that name was the last
    surviving piece of the stand-in story: the file has been a byte-for-byte
    copy of the real calibrated config since 00 was re-run. A file named for
    something it stopped being is where a false `dials_source` came from and
    kept coming back, so it is now named for what it is.
    """
    return processed / f"{series_code}_banked.json"


# The old name, still read so a tree frozen before the rename works untouched.
# Nothing writes it. `mv data/processed/imsa_standin.json
# data/processed/imsa_banked.json` migrates without re-drawing anything.
LEGACY_CONFIG = "{code}_standin.json"


def banked_config_path(series_code: str, processed: Path = PROCESSED) -> Path | None:
    """Where the banked config is **read**: the new name, then the old one."""
    for name in (config_path(series_code, processed).name,
                 LEGACY_CONFIG.format(code=series_code)):
        candidate = processed / name
        if candidate.exists():
            return candidate
    return None


def real_config_path(series_code: str, processed: Path = PROCESSED) -> Path:
    """00's frozen output, if it exists and passed its gate."""
    return processed / f"{series_code}.json"


def bank_path(series_code: str, processed: Path = PROCESSED) -> Path:
    return processed / f"{series_code}_seeds.json"


def field_path(series_code: str, processed: Path = PROCESSED) -> Path:
    return processed / f"{series_code}_field.json"


def load_assets(series_code: str, processed: Path = PROCESSED):
    """The race, its banks and its field, with the fingerprint checked.

    Every consumer goes through here rather than through three `load` calls,
    so the one check that matters cannot be skipped by forgetting it: a bank
    drawn against different dials describes different races under the same
    seed numbers, and nothing downstream would notice.
    """
    banked = banked_config_path(series_code, processed)
    paths = (banked or config_path(series_code, processed),
             bank_path(series_code, processed), field_path(series_code, processed))
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing frozen assets:\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\n\nRun `python scripts/freeze_assets.py` first.")

    cfg = RaceConfig.load(paths[0])
    bank = SeedBank.load(paths[1])
    field = BackgroundField.load(paths[2])

    fingerprint = dials_fingerprint(cfg)
    real_path = real_config_path(series_code, processed)
    if real_path.exists():
        real_fingerprint = dials_fingerprint(RaceConfig.load(real_path))
        if real_fingerprint != fingerprint:
            raise ValueError(
                f"{series_code}: the banked config is {fingerprint!r} but "
                f"{real_path.relative_to(ROOT)} is now {real_fingerprint!r}. "
                f"00 has been re-run since the assets were frozen, so the "
                f"bank names different races under the same seed numbers. "
                f"Run `python scripts/freeze_assets.py {series_code} --force` "
                f"and retrain that series.")
    for name, artefact in (("bank", bank), ("field", field)):
        recorded = artefact.provenance.get("dials_fingerprint")
        if recorded != fingerprint:
            raise ValueError(
                f"{series_code} {name} was built against dials "
                f"{recorded!r}, but the config on disk is {fingerprint!r}. "
                f"These are different races sharing seed numbers. Re-run "
                f"scripts/freeze_assets.py --force and retrain.")
    return cfg, bank, field


# ----------------------------------------------------------------------
# Writing them
# ----------------------------------------------------------------------
def freeze(series_code: str, processed: Path = PROCESSED,
           force: bool = False) -> None:
    paths = (config_path(series_code, processed), bank_path(series_code, processed),
             field_path(series_code, processed))
    # Existence is asked of the *read* path, so a tree still carrying
    # `{code}_standin.json` counts as frozen and is left alone rather than
    # redrawn under the new name.
    banked = banked_config_path(series_code, processed)
    existing = [p for p in (banked, *paths[1:]) if p is not None and p.exists()]
    if existing and not force:
        print(f"{series_code}: already frozen, leaving alone "
              f"({', '.join(p.name for p in existing)})")
        return

    real_path = real_config_path(series_code, processed)
    if real_path.exists():
        cfg = RaceConfig.load(real_path)
        source = f"real dials ({real_path.relative_to(ROOT)})"
    else:
        cfg = stand_in(series_code)
        source = "STAND-IN (6h invented race; no gated config on disk)"

    bank = draw_seed_bank(cfg, draw_seed=DRAW_SEEDS[series_code])
    field = freeze_background(cfg, strategy=BACKGROUND)

    cfg.save(paths[0])
    bank.save(paths[1])
    field.save(paths[2])

    print(f"{series_code}: {source}")
    print(f"    {cfg.name}, {cfg.duration_s / 3600:.0f} h, "
          f"{cfg.total_cars} cars, dials {dials_fingerprint(cfg)}")
    print(f"    {len(bank.headline)} headline / {len(bank.sweep)} sweep / "
          f"{len(bank.held_out)} held out, draw seed {DRAW_SEEDS[series_code]}")
    print(f"    background: every car on {BACKGROUND}")
    for p in paths:
        print(f"    wrote {p.relative_to(ROOT)}")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("series", nargs="*", default=list(SERIES),
                        help="imsa, wec, or nothing for both")
    parser.add_argument("--force", action="store_true",
                        help="redraw and overwrite (invalidates any checkpoint)")
    args = parser.parse_args(argv[1:])

    unknown = [s for s in args.series if s not in SERIES]
    if unknown:
        raise SystemExit(f"unknown series {unknown}; choose from {list(SERIES)}")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for series_code in (args.series or SERIES):
        freeze(series_code, force=args.force)

    if args.force:
        print("\n*** Banks redrawn. Every existing checkpoint is now about "
              "different races and must be retrained. ***")


if __name__ == "__main__":
    main(sys.argv)
