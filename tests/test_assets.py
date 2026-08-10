"""The artefacts have to be boring, and boring is a testable property.

Decision 6 exists because 03 must evaluate on the races 02 scored. That only
holds if the banks are identical every time they are read, if the sweep and
held-out sets stand in the relationship they are claimed to, and if a bank
built against one set of dials cannot be silently used with another.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from endurance import ClassDials, RaceConfig, run_race  # noqa: E402
from endurance.assets import (  # noqa: E402
    BackgroundField,
    SeedBank,
    dials_fingerprint,
    draw_seed_bank,
    freeze_background,
)


def config(series="imsa") -> RaceConfig:
    fast = ClassDials(
        series_code=series, class_name="GTP", base_pace_s=100.0,
        deg_slope_s_per_lap=0.012, pace_spread_s=0.5, lap_noise_s=0.35,
        caution_rate=0.30, caution_mean_dur_s=600.0, green_stint_laps=30.0,
        fuel_per_lap=1 / 30, fuel_per_lap_caution=0.6 / 30, tyre_life_laps=60.0,
        pit_time_mean_s=45.0, pit_time_std_s=3.0, n_cars=6)
    slow = ClassDials(
        series_code=series, class_name="GTD", base_pace_s=112.0,
        deg_slope_s_per_lap=0.02, pace_spread_s=0.9, lap_noise_s=0.6,
        caution_rate=0.30, caution_mean_dur_s=600.0, green_stint_laps=28.0,
        fuel_per_lap=1 / 28, fuel_per_lap_caution=0.6 / 28, tyre_life_laps=56.0,
        pit_time_mean_s=44.0, pit_time_std_s=3.0, n_cars=4)
    return RaceConfig(name="test race", series_code=series,
                      duration_s=6 * 3600.0, classes=[fast, slow])


# ----------------------------------------------------------------------
# The banks
# ----------------------------------------------------------------------
def test_the_banks_stand_in_the_relationship_decision_10_describes():
    bank = draw_seed_bank(config())
    bank.check()
    assert len(bank.headline) == 200
    assert len(bank.held_out) == 50
    assert bank.sweep == bank.headline[:50]
    assert not set(bank.held_out) & set(bank.headline)


def test_the_same_draw_seed_gives_the_same_bank():
    """Otherwise 'the same races' is a hope rather than a fact."""
    assert draw_seed_bank(config()).headline == draw_seed_bank(config()).headline
    other = draw_seed_bank(config(), draw_seed=1)
    assert other.headline != draw_seed_bank(config()).headline


def test_a_bank_round_trips_and_is_checked_on_the_way_in(tmp_path=None):
    import tempfile

    tmp_path = Path(tmp_path or tempfile.mkdtemp())
    bank = draw_seed_bank(config())
    path = tmp_path / "seeds.json"
    bank.save(path)
    assert SeedBank.load(path).to_dict() == bank.to_dict()


def test_a_bank_that_breaks_its_own_promises_is_refused():
    """The check is on load, not only on draw, because a file can be edited."""
    bank = draw_seed_bank(config())
    broken = SeedBank(series_code=bank.series_code, headline=bank.headline,
                      sweep=bank.headline[1:51],          # not a prefix
                      held_out=bank.held_out)
    with pytest.raises(ValueError):
        broken.check()

    overlapping = SeedBank(series_code=bank.series_code, headline=bank.headline,
                           sweep=bank.sweep,
                           held_out=bank.headline[:50])   # not disjoint
    with pytest.raises(ValueError):
        overlapping.check()


def test_recalibrated_dials_are_visible_rather_than_silent():
    """The tripwire. Same seeds against different dials are different races."""
    a = config()
    b = config()
    b.classes[0].base_pace_s += 0.5
    assert dials_fingerprint(a) != dials_fingerprint(b)
    assert draw_seed_bank(a).provenance["dials_fingerprint"] == dials_fingerprint(a)


# ----------------------------------------------------------------------
# The field
# ----------------------------------------------------------------------
def test_the_frozen_field_covers_every_car_in_the_race():
    cfg = config()
    field = freeze_background(cfg)
    assert len(field.strategies) == cfg.total_cars
    for cls in cfg.classes:
        for i in range(cls.n_cars):
            assert f"{cls.class_name}-{i + 1:02d}" in field.strategies


def test_the_focal_car_is_left_out_so_its_plan_can_go_in():
    cfg = config()
    field = freeze_background(cfg)
    resolved = field.resolve(focal="GTP-05")
    assert "GTP-05" not in resolved
    assert len(resolved) == cfg.total_cars - 1


def test_the_field_actually_runs_the_race_it_describes():
    """A frozen field that the engine will not accept is not an artefact.

    Three hours rather than one: with a thirty-lap tank and a caution-heavy
    race, an hour does not oblige anybody to stop, and a test that asserted
    stops over an hour would be asserting the caution draw rather than the
    field.
    """
    cfg = config()
    cfg.duration_s = 3 * 3600.0
    field = freeze_background(cfg)
    result = run_race(cfg, strategies=field.resolve(), seed=1)
    classification = result.classification()
    assert len(classification) == cfg.total_cars
    assert (classification["laps"] > 0).all()
    assert (classification["stops"] > 0).all()


def test_an_unknown_strategy_name_is_refused_rather_than_ignored():
    with pytest.raises(KeyError):
        freeze_background(config(), strategy="not_a_strategy")
    bad = BackgroundField(strategies={"GTP-01": "not_a_strategy"})
    with pytest.raises(KeyError):
        bad.resolve()


def test_the_field_round_trips(tmp_path=None):
    import tempfile

    tmp_path = Path(tmp_path or tempfile.mkdtemp())
    field = freeze_background(config())
    path = tmp_path / "field.json"
    field.save(path)
    assert BackgroundField.load(path).to_dict() == field.to_dict()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
