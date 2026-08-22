"""Step 4, part one: refreeze the banks and fields against the new dials.

The dials moved twice — the determinism fix landed and `stop_cost` gained a
dial — so every artefact frozen against the old ones is now about a different
race and the tripwires will say so. This redraws them.

**The seeds do not change, and that is the point.** `draw_seed_bank` is a
function of its `draw_seed` alone, so redrawing with the recorded seed returns
the same two hundred numbers. The paired comparison therefore runs on the same
races it always did; what those races *contain* is different, which is exactly
the change being made. A refreeze that also moved the seeds would confound the
two and nothing downstream could tell them apart.

That also means `bank_fingerprint` is unchanged, because it hashes the seed
lists rather than the dials. Only the bank's provenance moves. This is asserted
rather than assumed, because a redraw that quietly produced different seeds is
the one failure here that would look like success.

    python scripts/refreeze_assets.py            # report only
    python scripts/refreeze_assets.py --write    # write them

Nothing is written without `--write`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from endurance.assets import (BackgroundField, SeedBank, dials_fingerprint,
                              draw_seed_bank, freeze_background)
from endurance.params import RaceConfig
from endurance.policy import bank_fingerprint

# The draw seeds are recorded in each bank's provenance and are the only thing
# that had to be kept. Read from disk rather than restated here, so this script
# cannot disagree with the artefact it is replacing.
SERIES = ("imsa", "wec")


def find(name: str) -> Path | None:
    for hit in ROOT.rglob(name):
        if ".ipynb_checkpoints" not in hit.parts:
            return hit
    return None


def main() -> int:
    write = "--write" in sys.argv
    print("writing" if write else "reporting only (pass --write to commit)\n")

    ok = True
    for code in SERIES:
        print(f"=== {code.upper()} ===")
        config = RaceConfig.load(find(f"{code}.json"))
        new_fp = dials_fingerprint(config)
        print(f"  dials: {new_fp}  ({config.name})")

        bank_path = find(f"{code}_seeds.json")
        field_path = find(f"{code}_field.json")
        old_bank = SeedBank.load(bank_path)
        old_field = BackgroundField.load(field_path)

        draw_seed = old_bank.provenance.get("draw_seed")
        if draw_seed is None:
            print("  the old bank records no draw_seed; cannot redraw safely")
            ok = False
            continue
        was = old_bank.provenance.get("dials_fingerprint")
        print(f"  old bank was frozen against {was}, draw_seed {draw_seed}")

        bank = draw_seed_bank(config, draw_seed=int(draw_seed),
                              headline_n=len(old_bank.headline),
                              sweep_n=len(old_bank.sweep),
                              held_out_n=len(old_bank.held_out))

        # The three assertions that make this a refreeze rather than a redraw.
        if bank.headline != old_bank.headline:
            print("  SEEDS MOVED: the headline bank is not the one every "
                  "published number was measured on")
            ok = False
        elif bank.held_out != old_bank.held_out:
            print("  SEEDS MOVED: the held-out bank changed")
            ok = False
        else:
            print(f"  seeds unchanged: {len(bank.headline)} headline, "
                  f"{len(bank.sweep)} sweep, {len(bank.held_out)} held out")

        before, after = bank_fingerprint(old_bank), bank_fingerprint(bank)
        print(f"  bank fingerprint {before} -> {after}"
              f"{'  (unchanged, as it should be)' if before == after else '  MOVED'}")
        if before != after:
            ok = False

        field = freeze_background(
            config, strategy=old_field.provenance.get("uniform_strategy",
                                                      "fuel_window"))
        if field.strategies != old_field.strategies:
            added = sorted(set(field.strategies) - set(old_field.strategies))
            gone = sorted(set(old_field.strategies) - set(field.strategies))
            print(f"  FIELD CHANGED: +{added} -{gone}")
            ok = False
        else:
            print(f"  background field unchanged: {len(field.strategies)} cars, "
                  f"all {old_field.provenance.get('uniform_strategy')}")

        # `freeze_assets.py` writes `dials_source: "STAND-IN, not calibrated"`
        # as a literal, which stopped being true when the real dials landed and
        # was still being copied into policy cards at 03b. Derived here instead.
        source = config.classes[0].source_event if config.classes else ""
        for artefact in (bank, field):
            artefact.provenance["dials_source"] = (
                f"calibrated from {source}" if source else "unknown")

        if write:
            bank.save(bank_path)
            field.save(field_path)
            print(f"  written: {bank_path.relative_to(ROOT)}, "
                  f"{field_path.relative_to(ROOT)}")
        print()

    print("=== verdict ===")
    if not ok:
        print("  something moved that should not have; nothing else should be "
              "refrozen until it is understood")
        return 1
    print("  same seeds, same field, new dials" if write else
          "  same seeds, same field, new dials — rerun with --write to commit")
    print("\n  Every policy card and saved evaluation table is now about the "
          "old dials and\n  will be refused on load. That is correct: retrain "
          "and re-evaluate next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
