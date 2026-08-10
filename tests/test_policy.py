"""03b's verification gate, and the card that makes a checkpoint checkable.

The blueprint gives 03b a boundary constraint and no gate, which 02c already
named as a gate nobody can fail. Two conditions are asserted here.

**One: the plumbing, actually exercised.** 03a's gate compares
`RunToFuelWindow()` against `lambda car, state: RunToFuelWindow()(car, state)`
and its docstring says the agent's plumbing sits in the middle - the
observation built, the action chosen, the decision rebuilt. It does not: the
echo is a bare callable that touches neither `observe` nor `to_decision` nor
`PolicyStrategy`, so what that gate actually asserts is that `run_focal` does
not care whether a callable is a class instance or a function. The condition
below closes that gap rather than restating it: a `PolicyStrategy` returning
a constant action must reproduce, bit for bit, the same constant decision
passed in as an ordinary strategy - with `observe` and `to_decision` in the
path both times on one side and neither on the other.

**Two: the checkpoint is the policy that was measured.** The deliverable is a
saved policy, and the failure only 03b can produce is an artefact that has
drifted from the thing the table was computed on - a `.zip` that reloads to a
different policy, or an `.onnx` export that disagrees with its `.zip`. 04
loads the export and 05 hosts it, so a divergence here puts the app on
numbers nobody produced. These run only when a checkpoint is on disk;
everything about the *card* is checked unconditionally, because that is the
part that catches a policy scored against the wrong race.

Both conditions carry a falsifier. A gate that cannot fail is not a gate,
which is the rule this suite has applied since 02b.

Running without pytest: `tests/run_tests_nopytest.py` supplies only `raises`
and `fixture`, so there is no `skip` or `importorskip` here. The optional
tests return early and say why.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from endurance import ClassDials, RaceConfig  # noqa: E402
from endurance.assets import (  # noqa: E402
    SeedBank,
    dials_fingerprint,
    draw_seed_bank,
    freeze_background,
)
from endurance import harness  # noqa: E402
from endurance.gym_env import (  # noqa: E402
    FLAG_KEEP,
    FLAG_TYRES,
    FULL_KEEP,
    FULL_TYRES,
    N_ACTIONS,
    N_OBS,
    STAY,
    PolicyStrategy,
    to_decision,
)
from endurance.policy import (  # noqa: E402
    PolicyCard,
    agent_roster,
    bank_fingerprint,
    load_policy,
)
from endurance.strategies import ROSTER  # noqa: E402


SEEDS = [11, 12, 13]
ACTIONS = (STAY, FULL_TYRES, FULL_KEEP, FLAG_TYRES, FLAG_KEEP)
POLICIES = ROOT / "outputs" / "policies"


def config(series="imsa", duration_s=3 * 3600.0) -> RaceConfig:
    """The stand-in's shape, shortened so the suite stays quick."""
    quick = ClassDials(
        series_code=series,
        class_name="GTP" if series == "imsa" else "HYPERCAR",
        base_pace_s=97.5, deg_slope_s_per_lap=0.012, pace_spread_s=0.5,
        lap_noise_s=0.5, caution_rate=0.20, caution_mean_dur_s=600.0,
        green_stint_laps=30.0, fuel_per_lap=1 / 30,
        fuel_per_lap_caution=0.6 / 30, tyre_life_laps=60.0,
        pit_time_mean_s=47.0, pit_time_std_s=2.0, n_cars=8)
    gt = ClassDials(
        series_code=series,
        class_name="GTD" if series == "imsa" else "LMGT3",
        base_pace_s=112.0, deg_slope_s_per_lap=0.02, pace_spread_s=0.9,
        lap_noise_s=0.6, caution_rate=0.20, caution_mean_dur_s=600.0,
        green_stint_laps=28.0, fuel_per_lap=1 / 28,
        fuel_per_lap_caution=0.6 / 28, tyre_life_laps=56.0,
        pit_time_mean_s=45.0, pit_time_std_s=2.0, n_cars=10)
    return RaceConfig(name=f"{series} test", series_code=series,
                      duration_s=duration_s, classes=[quick, gt])


def constant_policy(action: int) -> PolicyStrategy:
    """The agent's path: observe, choose, rebuild the decision."""
    return PolicyStrategy(lambda obs, deterministic=True: (int(action), None))


def constant_strategy(action: int):
    """The same answer as an ordinary strategy, with no wrapper in the way."""
    return lambda car, state: to_decision(int(action), car, state)


COLUMNS = ("laps", "race_time_s", "stops", "pit_time_s", "traffic_time_s",
           "class_pos", "overall_pos", "caution_laps")


# ----------------------------------------------------------------------
# Gate one: the plumbing, with the wrapper actually in the path
# ----------------------------------------------------------------------
def test_the_agents_path_reproduces_the_same_decision_taken_directly():
    """Every action, both series, bit for bit.

    Run through `PolicyStrategy` the observation is built and thrown away
    and the decision is rebuilt from an action; run directly it is not. The
    race must not be able to tell, or the number 03b reports is about the
    wrapper as much as about the policy.
    """
    for series in ("imsa", "wec"):
        cfg = config(series)
        field = freeze_background(cfg)
        for action in ACTIONS:
            for seed in SEEDS:
                focal = harness.focal_car(cfg, seed)
                through = harness.run_focal(cfg, seed, focal,
                                            constant_policy(action), field)
                direct = harness.run_focal(cfg, seed, focal,
                                           constant_strategy(action), field)
                for col in COLUMNS:
                    assert through[col] == direct[col], (series, action, seed, col)


def test_that_gate_would_notice_a_policy_choosing_something_else():
    """The falsifier. Two different constants must not agree."""
    cfg = config()
    field = freeze_background(cfg)
    seed, focal = SEEDS[0], harness.focal_car(config(), SEEDS[0])

    stay = harness.run_focal(cfg, seed, focal, constant_policy(STAY), field)
    stopping = harness.run_focal(cfg, seed, focal,
                                 constant_policy(FULL_TYRES), field)
    assert stay["stops"] != stopping["stops"]


def test_the_observation_is_built_on_the_way_through():
    """The gate above would still pass if `observe` were never called.

    Asserted directly, because that is the specific hole in 03a's version:
    a `PolicyStrategy` whose `predict` ignores its argument cannot show that
    the argument was ever computed.
    """
    seen = []

    def predict(obs, deterministic=True):
        seen.append(np.asarray(obs))
        return STAY, None

    cfg = config()
    field = freeze_background(cfg)
    harness.run_focal(cfg, SEEDS[0], harness.focal_car(cfg, SEEDS[0]),
                      PolicyStrategy(predict), field)

    assert len(seen) > 50, "the policy was never asked"
    stacked = np.stack(seen)
    assert stacked.shape[1] == N_OBS
    assert ((stacked >= 0.0) & (stacked <= 1.0)).all(), "an observation left the box"
    assert stacked.std(axis=0).max() > 0, "every observation was identical"


# ----------------------------------------------------------------------
# The roster insertion
# ----------------------------------------------------------------------
def test_the_agent_goes_in_as_a_sixth_member_and_not_over_a_human():
    roster = agent_roster(ROSTER, constant_policy(FULL_TYRES))
    assert set(roster) == set(ROSTER) | {"agent"}
    with pytest.raises(KeyError):
        agent_roster(roster, constant_policy(STAY))      # already there
    with pytest.raises(KeyError):
        agent_roster(ROSTER, constant_policy(STAY), name="fuel_window")


def test_the_roster_shares_one_policy_rather_than_reloading_it():
    """`compare_roster` calls each value once per race.

    A factory that rebuilt the policy would reload a checkpoint two hundred
    times a series, and - worse - any per-instance state would silently
    reset between races while looking like it persisted.
    """
    strategy = constant_policy(STAY)
    factory = agent_roster(ROSTER, strategy)["agent"]
    assert factory() is factory() is strategy


def test_the_agent_row_comes_out_of_the_same_call_the_humans_do():
    cfg = config()
    roster = agent_roster(ROSTER, constant_policy(FULL_TYRES))
    rows = harness.compare_roster(cfg, SEEDS, freeze_background(cfg),
                                  roster=roster).rows

    assert set(rows["strategy"]) == set(roster)
    agent = rows[rows["strategy"] == "agent"]
    assert len(agent) == len(SEEDS)
    for col in ("d_class_pos", "d_laps", "d_race_time_s"):
        assert col in agent.columns
    # And the null is still the null with a sixth member in the table.
    null = rows[rows["strategy"] == "fuel_window"]
    assert (null["d_class_pos"] == 0).all()


# ----------------------------------------------------------------------
# Gate two, part one: the card, which needs no checkpoint
# ----------------------------------------------------------------------
def card_for(cfg, bank, **over) -> PolicyCard:
    base = dict(series_code=cfg.series_code, algorithm="MaskablePPO",
                dials_fingerprint=dials_fingerprint(cfg),
                bank_fingerprint=bank_fingerprint(bank),
                train_seed=0, total_timesteps=1000, checkpoint="test.zip")
    base.update(over)
    return PolicyCard(**base)


def test_a_card_round_trips(tmp_path=None):
    import tempfile

    tmp_path = Path(tmp_path or tempfile.mkdtemp())
    cfg = config()
    card = card_for(cfg, draw_seed_bank(cfg))
    checkpoint = tmp_path / "test.zip"
    card.save(checkpoint)
    assert PolicyCard.load(checkpoint) == card


def test_a_policy_trained_against_other_dials_is_refused():
    """The tripwire that matters. Same seeds, different races.

    A policy trained on one set of dials and scored on another produces a
    complete, plausible, meaningless table - which is the failure mode
    `dials_fingerprint` was introduced for, applied to the artefact most
    likely to outlive a config change.
    """
    cfg = config()
    bank = draw_seed_bank(cfg)
    card = card_for(cfg, bank)
    card.check(cfg, bank)                       # must not raise

    twisted = config()
    twisted.classes[0].base_pace_s += 0.5
    with pytest.raises(ValueError):
        card.check(twisted, bank)


def test_a_policy_from_the_other_series_is_refused():
    """Two series, two tables, and never a policy crossing between them."""
    imsa, wec = config("imsa"), config("wec")
    card = card_for(imsa, draw_seed_bank(imsa))
    with pytest.raises(ValueError):
        card.check(wec, draw_seed_bank(wec))


def test_a_redrawn_bank_is_refused():
    """The held-out fifty are only held out relative to a particular draw."""
    cfg = config()
    bank = draw_seed_bank(cfg)
    other = draw_seed_bank(cfg, draw_seed=1)
    assert bank_fingerprint(bank) != bank_fingerprint(other)
    with pytest.raises(ValueError):
        card_for(cfg, bank).check(cfg, other)


def test_a_checkpoint_with_no_card_cannot_be_loaded(tmp_path=None):
    """An unlabelled checkpoint cannot be checked, so it is not usable.

    Refused rather than loaded with a warning: a warning in a notebook is a
    line of output nobody reads, and the whole point of the card is that the
    mismatch is caught before a table exists to argue about.
    """
    import tempfile

    tmp_path = Path(tmp_path or tempfile.mkdtemp())
    orphan = tmp_path / "orphan.zip"
    orphan.write_bytes(b"not really a checkpoint")
    with pytest.raises(FileNotFoundError):
        load_policy(orphan, config=config())


def test_an_unknown_format_is_refused(tmp_path=None):
    import tempfile

    tmp_path = Path(tmp_path or tempfile.mkdtemp())
    weird = tmp_path / "policy.pt"
    weird.write_bytes(b"")
    card_for(config(), draw_seed_bank(config())).save(weird)
    with pytest.raises(ValueError):
        load_policy(weird, config=config())


# ----------------------------------------------------------------------
# Gate two, part two: the checkpoint on disk, when there is one
# ----------------------------------------------------------------------
def _trained(series_code: str):
    """The frozen assets and a checkpoint, or None if either is absent.

    These tests are about artefacts a training run produces, so they cannot
    be preconditions for the suite passing on a fresh clone. They are
    conditions on the *deliverable*, and 03b's notebook runs them where the
    deliverable exists.
    """
    try:
        from freeze_assets import load_assets
    except ImportError:
        return None
    checkpoint = POLICIES / f"{series_code}_maskable_ppo.zip"
    if not checkpoint.exists():
        return None
    try:
        cfg, bank, field = load_assets(series_code)
    except FileNotFoundError:
        return None
    return cfg, bank, field, checkpoint


def test_the_reloaded_checkpoint_scores_what_the_trained_one_scored():
    """Gate two. The artefact is the policy the table was computed on.

    Loaded twice rather than compared against a live model, because the
    thing being guarded is the file: two loads of the same `.zip` must
    produce identical classification rows on the same races, or the
    checkpoint is not deterministic and no number taken from it reproduces.
    """
    for series_code in ("imsa", "wec"):
        assets = _trained(series_code)
        if assets is None:
            print(f"  (skipped {series_code}: no checkpoint on disk)")
            continue
        cfg, bank, field, checkpoint = assets

        a = load_policy(checkpoint, config=cfg, bank=bank)
        b = load_policy(checkpoint, config=cfg, bank=bank)
        for seed in bank.headline[:5]:
            focal = harness.focal_car(cfg, seed)
            ra = harness.run_focal(cfg, seed, focal, a, field)
            rb = harness.run_focal(cfg, seed, focal, b, field)
            for col in COLUMNS:
                assert ra[col] == rb[col], (series_code, seed, col)


def test_the_export_agrees_with_the_checkpoint_over_a_whole_race():
    """The `.onnx` 04 loads must be the `.zip` 03b measured.

    `train.py` checks this on thirty-two probe observations before writing
    the file. This is the version that matters: the observations a real race
    actually visits, which is a much narrower and stranger part of the unit
    box than uniform noise, and the part any divergence would hide in.
    """
    for series_code in ("imsa", "wec"):
        assets = _trained(series_code)
        if assets is None:
            print(f"  (skipped {series_code}: no checkpoint on disk)")
            continue
        cfg, bank, field, checkpoint = assets
        export = checkpoint.with_suffix(".onnx")
        if not export.exists():
            print(f"  (skipped {series_code}: no .onnx export)")
            continue

        zipped = load_policy(checkpoint, config=cfg, bank=bank)
        onnx = load_policy(export, config=cfg, bank=bank)

        disagreements, asked = 0, 0
        seen = []

        def spy(strategy):
            def wrapped(car, state):
                from endurance.gym_env import observe
                seen.append(observe(car, state))
                return strategy(car, state)
            return wrapped

        seed = bank.headline[0]
        harness.run_focal(cfg, seed, harness.focal_car(cfg, seed),
                          spy(zipped), field)
        assert seen, "the race asked nothing"

        for obs in seen:
            asked += 1
            if zipped.predict(obs, deterministic=True)[0] != \
                    onnx.predict(obs, deterministic=True)[0]:
                disagreements += 1
        assert disagreements == 0, (
            f"{series_code}: the export disagrees with the checkpoint on "
            f"{disagreements} of {asked} observations a real race visited")


def test_that_gate_would_notice_a_different_policy():
    """The falsifier for gate two, without needing a second training run.

    Two constant policies choosing different actions must produce different
    races, which is what makes "the two agree" a statement about the export
    rather than about the comparison being blind.
    """
    cfg = config()
    field = freeze_background(cfg)
    seed = SEEDS[0]
    focal = harness.focal_car(cfg, seed)

    rows = {a: harness.run_focal(cfg, seed, focal, constant_policy(a), field)
            for a in ACTIONS}
    fingerprints = {(r["laps"], r["stops"], round(r["pit_time_s"], 3))
                    for r in rows.values()}
    assert len(fingerprints) >= 3, fingerprints


def test_the_action_space_the_card_assumes_has_not_moved():
    """A checkpoint's output layer is `N_ACTIONS` wide.

    Stated here because the failure is quiet: adding a sixth action would
    make every existing checkpoint load and predict nonsense rather than
    raise, and the card records no shape.
    """
    assert N_ACTIONS == 5
    assert len(ACTIONS) == N_ACTIONS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
