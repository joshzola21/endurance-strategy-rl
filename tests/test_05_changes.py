"""The three package changes 05 made, each tested for what it can break.

One file rather than three because they were decided together and a reader
asking "what did 05 change about the package" should find it in one place.

**Written to run under both runners.** `run_tests_nopytest.py` stubs `pytest`
with a shim that supplies `tmp_path` and nothing else: a fixture argument
arrives as a missing positional, `monkeypatch` does not exist, and
`raises` takes no `match=`. So the helpers below are plain functions called
inside the tests, the way `test_assets.py` and `test_harness.py` do it, and
`refusal()` stands in for `raises(..., match=...)` where the wording is part
of what is being tested. Nothing here reads an artefact from disk, so unlike
`test_stop_cost_floor.py` it runs in a clean clone.

Every test asks the question its change could fail rather than the question it
obviously passes. That `never_pit` returns `pit=False` is not worth asserting;
that it still stops, and that it did not leak into the background field, is.
That `set_dials` writes a number is not worth asserting; that it refuses to
flatten classes which disagree, is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from endurance import harness, run_race  # noqa: E402
from endurance.assets import (  # noqa: E402
    BackgroundField,
    dials_fingerprint,
    freeze_background,
    rules_fingerprint,
    rules_mismatch,
)
from endurance.params import (  # noqa: E402
    ClassDials,
    RaceConfig,
    scale_dials,
    set_dials,
)
from endurance.strategies import (  # noqa: E402
    BASELINES,
    ROSTER,
    NeverPit,
    assign_strategy,
)

SEEDS = [11, 12, 13]

# The columns the paired comparison is phrased in. The null's are identically
# zero by construction, which is 02c's gate seen from inside the table.
DELTAS = ("d_class_pos", "d_laps", "d_race_time_s", "d_stops", "d_pit_time_s")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def config() -> RaceConfig:
    """Two classes and three hours.

    Two classes because one property under test is what happens when the
    classes disagree about a dial. Three hours rather than one because a
    thirty-lap tank does not oblige anybody to stop inside an hour, and a test
    asserting stops over an hour would be asserting the caution draw.
    """
    fast = ClassDials(
        series_code="imsa", class_name="GTP", base_pace_s=97.5,
        deg_slope_s_per_lap=0.012, pace_spread_s=0.5, lap_noise_s=0.5,
        caution_rate=0.20, caution_mean_dur_s=600.0, green_stint_laps=30.0,
        fuel_per_lap=1 / 30, fuel_per_lap_caution=0.6 / 30, tyre_life_laps=60.0,
        pit_time_mean_s=47.0, pit_time_std_s=2.0, n_cars=6)
    slow = ClassDials(
        series_code="imsa", class_name="GTD", base_pace_s=112.0,
        deg_slope_s_per_lap=0.02, pace_spread_s=0.9, lap_noise_s=0.6,
        caution_rate=0.20, caution_mean_dur_s=600.0, green_stint_laps=28.0,
        fuel_per_lap=1 / 28, fuel_per_lap_caution=0.6 / 28, tyre_life_laps=56.0,
        pit_time_mean_s=45.0, pit_time_std_s=2.0, n_cars=5)
    return RaceConfig(name="imsa 3h", series_code="imsa",
                      duration_s=3 * 3600.0, classes=[fast, slow])


def refusal(exc_type, fn, *args, **kwargs) -> str:
    """Call `fn`, require `exc_type`, and hand back the message.

    Stands in for `pytest.raises(..., match=...)`, which the fallback runner's
    shim does not support. Where a refusal exists to tell somebody *why*, the
    wording is part of the behaviour and is worth asserting on.
    """
    try:
        fn(*args, **kwargs)
    except exc_type as e:
        return str(e)
    raise AssertionError(f"expected {exc_type.__name__}, nothing was raised")


# ----------------------------------------------------------------------
# Amendment 25 - never_pit, built rather than only recorded
# ----------------------------------------------------------------------
def test_never_pit_is_in_the_roster_and_not_in_the_background():
    """The distinction the two mappings exist to keep.

    `freeze_background` looks names up in `BASELINES`, so a control that
    drifted into that mapping would become part of the field it exists to be
    measured against - and a whole field asking for nothing is a different
    experiment, not a background.
    """
    assert "never_pit" in ROSTER
    assert "never_pit" not in BASELINES


def test_never_pit_still_stops_because_the_rules_make_it():
    """The point of it: asking for nothing is not the same as not stopping.

    `_must_pit` takes a full service the moment the tank cannot cover a lap,
    and it applies to this class exactly as it applies to the five strategies
    and to the agent. A `never_pit` car finishing with no stops would mean the
    forced-stop rule had stopped applying to somebody, and that rule is what
    holds the agent to the same terms as the humans.
    """
    cfg = config()
    result = run_race(cfg, strategies=assign_strategy(cfg, NeverPit()), seed=1)
    stops = result.classification()["stops"]
    assert (stops > 0).all(), "a car that never asks to stop still has to refuel"


def test_never_pit_is_not_the_null_wearing_another_name():
    """A control indistinguishable from the null would catch nothing.

    **The first version of this asserted the wrong thing** - that `never_pit`
    takes strictly fewer stops than `fuel_window` - and it failed at three
    against three. The two differ by where in the tank they stop, not by how
    often: the null keeps about a lap and a half in hand and this keeps none,
    which is a fraction of a stint. Over a twenty-four-hour race that fraction
    accumulates into about half a stop; over the three hours this file runs it
    accumulates into nothing at all. A count was never the right statistic.

    What must hold is weaker and is the actual property: it cannot stop *more*
    than the null, since it stops only when the rules make it, and it must not
    be the null, since the null's own deltas are identically zero by
    construction and a control that produced zeros would be scoring the null
    against itself under a second name.
    """
    cfg = config()
    rows = harness.compare_roster(cfg, SEEDS, freeze_background(cfg)).rows
    mine = rows[rows["strategy"] == "never_pit"]
    null = rows[rows["strategy"] == "fuel_window"]

    # One column at a time, the way `evaluate.py` and `test_harness.py` do it.
    # Handing a *tuple* of names to `DataFrame.__getitem__` is not a multiple
    # selection - pandas reads it as one label and raises - and the first
    # version of this test did exactly that.
    for col in DELTAS:
        assert (null[col] == 0).all(), f"the null moved on {col}; nothing reads"

    assert mine["stops"].median() <= null["stops"].median()
    assert any((mine[col] != 0).any() for col in DELTAS), (
        "never_pit produced the null's race on every seed, so the roster has "
        "two nulls and no control")


def test_never_pit_is_scored_by_the_same_lines_as_the_others():
    """It arrives through `compare_roster` or it is not a control.

    The row this class was first measured on came from a script outside the
    roster, which is exactly why it could not catch the next degenerate policy:
    it was never on the held-out bank and never in a saved table. This asserts
    it is now an ordinary row.
    """
    cfg = config()
    summary = harness.compare_roster(cfg, SEEDS, freeze_background(cfg)
                                     ).summarise()
    assert "never_pit" in set(summary["strategy"])
    assert len(summary) == len(ROSTER)


def test_every_roster_member_takes_no_constructor_arguments():
    """Parameter-freeness has a shape: nowhere to put a tuning."""
    for cls in ROSTER.values():
        cls()


# ----------------------------------------------------------------------
# Amendment 28 - the set-value path
# ----------------------------------------------------------------------
def test_a_multiplier_cannot_move_a_dial_that_sits_at_zero():
    """The defect that made amendment 23's dial unsweepable.

    This is the falsifier for `set_dials` existing at all: if it ever fails,
    the default has moved off zero and the set path may no longer be needed
    for this dial.
    """
    cfg = config()
    assert cfg.classes[0].pit_transit_caution_discount == 0.0
    for multiplier in (0.25, 1.0, 3.0, 1e6):
        moved = scale_dials(cfg, pit_transit_caution_discount=multiplier)
        assert moved.classes[0].pit_transit_caution_discount == 0.0


def test_set_dials_moves_it_and_moves_the_fingerprint():
    """A swept race has to be a different race, visibly.

    Amendment 21's complaint was a change that moved every race and no
    fingerprint. A dial is the good case and must not have that property.
    """
    cfg = config()
    moved = set_dials(cfg, pit_transit_caution_discount=0.4)
    assert all(c.pit_transit_caution_discount == 0.4 for c in moved.classes)
    assert dials_fingerprint(moved) != dials_fingerprint(cfg)


def test_set_dials_refuses_to_flatten_classes_that_disagree():
    """Setting writes one number onto every class; scaling does not.

    `base_pace_s` differs by fifteen seconds between these two, so a set would
    quietly make them the same car. The message has to say which dial, because
    the caller either meant a dial that is shared or did not mean this.
    """
    cfg = config()
    assert len({c.base_pace_s for c in cfg.classes}) > 1
    message = refusal(ValueError, set_dials, cfg, base_pace_s=100.0)
    assert "flatten" in message and "base_pace_s" in message


def test_the_two_paths_agree_where_they_can():
    """Scaling by `m` and setting to `old * m` give the same config.

    The property that says these are two ways of writing one operation rather
    than two operations that happen to sit near each other.
    """
    cfg = config()
    was = cfg.classes[0].pit_caution_discount
    assert len({c.pit_caution_discount for c in cfg.classes}) == 1
    assert (dials_fingerprint(scale_dials(cfg, pit_caution_discount=1.5))
            == dials_fingerprint(set_dials(cfg, pit_caution_discount=was * 1.5)))


def test_neither_path_touches_the_original():
    """`scale_dials` has always promised this; the shared path must keep it."""
    cfg = config()
    before = cfg.classes[0].pit_caution_discount
    scale_dials(cfg, pit_caution_discount=3.0)
    set_dials(cfg, pit_caution_discount=0.9)
    assert cfg.classes[0].pit_caution_discount == before


def test_an_unknown_dial_is_refused_on_both_paths():
    cfg = config()
    for move in (scale_dials, set_dials):
        with pytest.raises(AttributeError):
            move(cfg, not_a_dial=1.0)


def test_the_caution_rate_clamp_holds_on_both_paths():
    """`scale_dials` has always held the rate below 1; setting must too.

    A rate at or above 1 is a race that is entirely caution, and the draw does
    not terminate on it.
    """
    assert all(c.caution_rate <= 0.95
               for c in scale_dials(config(), caution_rate=9.0).classes)
    assert all(c.caution_rate <= 0.95
               for c in set_dials(config(), caution_rate=0.99).classes)


def test_sweep_dial_wants_exactly_one_of_the_two():
    cfg, field = config(), freeze_background(config())
    for kwargs in ({"multipliers": (1.0,), "values": (0.4,)}, {}):
        message = refusal(ValueError, harness.sweep_dial, cfg, [11], field,
                          "pit_caution_discount", **kwargs)
        assert "not both" in message


def test_sweep_dial_by_value_reaches_the_zero_dial():
    """End to end: the dial amendment 23 added can now be swept.

    `multiplier` is empty on this path deliberately - back-computing one from a
    zero default would invent a number - and `value` carries what the race saw.
    """
    cfg = config()
    sweep = harness.sweep_dial(cfg, SEEDS[:2], freeze_background(cfg),
                               "pit_transit_caution_discount",
                               values=(0.0, 0.2, 0.4))
    assert sorted(sweep["value"].unique()) == [0.0, 0.2, 0.4]
    assert sweep["multiplier"].isna().all()


def test_sweep_dial_by_multiplier_is_unchanged():
    """The published sweeps went through this path and must still.

    `dial` and `multiplier` are the two columns `viz.plot_sweep_response` and
    every saved sweep table read.
    """
    cfg = config()
    sweep = harness.sweep_dial(cfg, SEEDS[:2], freeze_background(cfg),
                               "pit_caution_discount", (0.5, 1.0))
    assert sorted(sweep["multiplier"].unique()) == [0.5, 1.0]
    assert set(sweep["dial"]) == {"pit_caution_discount"}


# ----------------------------------------------------------------------
# Amendment 29 - the rules fingerprint
# ----------------------------------------------------------------------
def test_the_rules_fingerprint_is_a_different_number():
    """Two questions, two answers, so a mismatch says which one moved."""
    assert rules_fingerprint() != dials_fingerprint(config())
    assert len(rules_fingerprint()) == 16


def test_the_rules_fingerprint_moves_when_a_rules_version_does():
    """The falsifier. A fingerprint that cannot move is not a tripwire.

    Bumping the version is the deliberate act a rules change is meant to come
    with, and this asserts the act has an effect. Set and restored by hand
    rather than through `monkeypatch`, which the fallback runner has not got.
    """
    from endurance import assets

    before = assets.rules_fingerprint()
    was = assets.PITSTOP_RULES_VERSION
    try:
        assets.PITSTOP_RULES_VERSION = was + 1
        assert assets.rules_fingerprint() != before
    finally:
        assets.PITSTOP_RULES_VERSION = was
    assert assets.rules_fingerprint() == before


def test_a_missing_rules_fingerprint_is_not_a_failure():
    """Every artefact on disk predates this check, and none of them is invalid.

    `None` means "frozen before the check existed", which the methods page
    reports rather than treating as a pass. A strict check would have refused
    the whole project on its first run.
    """
    assert rules_mismatch(None) is None
    assert rules_mismatch(rules_fingerprint()) is None
    message = rules_mismatch("0000000000000000", what="the IMSA bank")
    assert message is not None
    assert "the IMSA bank" in message
    assert rules_fingerprint() in message


def test_a_new_field_records_it():
    """Recorded from now on, so next time the check has something to check."""
    cfg = config()
    field = freeze_background(cfg)
    assert field.provenance["rules_fingerprint"] == rules_fingerprint()
    assert field.provenance["dials_fingerprint"] == dials_fingerprint(cfg)


def test_an_artefact_frozen_before_the_check_still_loads(tmp_path=None):
    """The compatibility the shape was chosen for: that is every file today."""
    import tempfile

    tmp_path = Path(tmp_path or tempfile.mkdtemp())
    field = freeze_background(config())
    field.provenance.pop("rules_fingerprint")
    path = tmp_path / "field.json"
    field.save(path)
    reloaded = BackgroundField.load(path)
    assert rules_mismatch(reloaded.provenance.get("rules_fingerprint")) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
