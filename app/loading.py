"""What the app loads, and what it refuses to load.

Presentation only, like everything under `app/`. Nothing here computes a
race quantity; it reads four kinds of file - a config, a bank, a field and a
checkpoint - and it checks that they are about the same race before the page
gets to draw anything.

**Streamlit is optional here on purpose.** `cache` falls back to
`functools.lru_cache` when Streamlit is absent, so the loaders can be
exercised by the gates and by any script without a browser. The alternative
is a module the tests cannot import, which means loading is the one part of
04 nothing checks.

Why the consistency check exists
--------------------------------
Every artefact in this project already carries a `dials_fingerprint`, and
each of them is checked at the point it is *used* - `PolicyCard.check` on the
policy, `NullRuns` on the cache key. Nothing checked that the three files the
app opens together are the same generation, and at 04 that gap produced a
real one: a seed bank and a background field for a three-hour IMSA race
sitting beside a config for the twenty-four-hour one, frozen ninety-seven
minutes *later*, with no config of their own anywhere in the tree. Loaded by
name, they would have produced a complete and plausible race that no other
stage could reproduce. So the three are checked against each other here,
once, before anything reads them.

The rule about the sliders
--------------------------
04's headline feature moves dials; 03b's card refuses a policy whose dials
are not the ones it trained on. Both are right and they collide.

The resolution is that the card exists to stop a *table* being published
against the wrong race, and the app does not publish tables off-nominal. So:
the policy may be driven at any dial setting, because watching it fail on a
race it was not trained for is the demonstration; but `nominal` is False the
moment any slider moves, and no comparison number may be shown while it is.
`load_agent` returns that flag rather than deciding it, because deciding it
is the page's job and hiding it would be the quiet version of the failure.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

from endurance.assets import BackgroundField, SeedBank, dials_fingerprint
from endurance.params import ASSUMED_FIELDS, RaceConfig, scale_dials
from endurance.policy import PolicyCard, load_policy

def cache(fn):
    """`st.cache_resource` where there is a Streamlit, `lru_cache` where not.

    `cache_resource` rather than `cache_data` because these objects are
    shared, read-only and expensive - a config, a bank, an onnxruntime
    session. Nothing returned by this module is mutated; the controller
    copies the config before it scales it and never writes to the field.
    """
    try:
        import streamlit as st
    except ImportError:
        return functools.lru_cache(maxsize=None)(fn)
    return st.cache_resource(fn)


ROOT = Path(__file__).resolve().parents[1]

# Where things are, found rather than assumed. The blueprint's module map
# names the package and the notebooks and says nothing about where the
# frozen artefacts sit, and in this tree they are not under `assets/` -
# there is a `data/` and an `outputs/` with policies inside it. An app that
# hard-codes a folder either finds nothing or, worse, finds a stale copy of
# it. So each directory is resolved in three steps: an environment variable,
# then the conventional names, then a search of the tree for the files
# themselves. `where()` reports what it settled on, and Home prints it,
# because a path resolved silently is a path nobody can debug.
ASSET_ENV = "ENDURANCE_ASSETS"
CHECKPOINT_ENV = "ENDURANCE_CHECKPOINTS"
RESULTS_ENV = "ENDURANCE_RESULTS"

# The frozen three, under every name they may legitimately wear.
#
# **`_banked` replaces `_standin`.** That file held a six-hour invented race
# before 00 was re-run; since then it is a byte-for-byte copy of the real
# calibrated config. The old name is still read so an unmigrated tree loads.
#
# **`seed_bank_{code}` and `background_field_{code}` are no longer read.** They
# are an older convention, and the only files still wearing them describe a
# three-hour IMSA race whose config is not in the tree - amendment 19's live
# instance. Accepting them meant the app avoided that race by the *order* of a
# tuple rather than by a check, and only for IMSA, because no WEC pair was
# ever written. `scripts/check_artefacts.py` reports such a file as an orphan
# and this list no longer offers it a way in.
CONFIG_NAMES = ("{code}.json", "{code}_banked.json", "{code}_standin.json")
SEED_NAMES = ("{code}_seeds.json",)
FIELD_NAMES = ("{code}_field.json",)

_SKIP = {".git", ".venv", "venv", "__pycache__", "node_modules", ".ipynb_checkpoints"}


def _walkable(path: Path) -> bool:
    return not any(part in _SKIP or part.startswith(".") for part in path.parts)


def _named(code: str, patterns: tuple[str, ...], folder: Path) -> Path | None:
    for pattern in patterns:
        candidate = folder / pattern.format(code=code)
        if candidate.exists():
            return candidate
    return None


def _series_in(folder: Path) -> list[str]:
    """Which series have a complete set of three files in this folder."""
    codes = set()
    for pattern in SEED_NAMES:
        stem = pattern.replace("{code}", "*")
        for hit in folder.glob(stem):
            before, _, after = pattern.partition("{code}")
            code = hit.name[len(before):len(hit.name) - len(after)]
            if (_named(code, CONFIG_NAMES, folder)
                    and _named(code, FIELD_NAMES, folder)):
                codes.add(code)
    return sorted(codes)


@cache
def asset_dir() -> Path:
    """The folder holding the frozen three, or a raised error naming the search."""
    override = os.environ.get(ASSET_ENV)
    if override:
        return Path(override).expanduser().resolve()

    conventional = [ROOT / "assets", ROOT / "data" / "processed",
                    ROOT / "data", ROOT / "outputs" / "assets", ROOT / "outputs"]
    for folder in conventional:
        if folder.is_dir() and _series_in(folder):
            return folder

    found = sorted({p.parent for pattern in SEED_NAMES
                    for p in ROOT.rglob(pattern.replace("{code}", "*"))
                    if _walkable(p.relative_to(ROOT))})
    for folder in found:
        if _series_in(folder):
            return folder

    raise FileNotFoundError(
        f"no folder under {ROOT} holds a config, a seed bank and a "
        f"background field for the same series. Looked for "
        f"{', '.join(SEED_NAMES)} beside {', '.join(CONFIG_NAMES)}. Set "
        f"{ASSET_ENV} to point at them.")


@cache
def checkpoint_dir() -> Path | None:
    """The folder holding the `.onnx` exports, if there is one."""
    override = os.environ.get(CHECKPOINT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    conventional = [ROOT / "checkpoints", ROOT / "outputs" / "policies",
                    ROOT / "outputs" / "checkpoints", ROOT / "outputs"]
    for folder in conventional:
        if folder.is_dir() and any(folder.glob("*.onnx")):
            return folder
    found = sorted(p.parent for p in ROOT.rglob("*.onnx")
                   if _walkable(p.relative_to(ROOT)))
    return found[0] if found else None


@cache
def results_dir() -> Path | None:
    """The folder holding 02c's saved roster table, if it was saved."""
    override = os.environ.get(RESULTS_ENV)
    if override:
        return Path(override).expanduser().resolve()
    for folder in [ROOT / "outputs" / "evaluation", ROOT / "results",
                   ROOT / "outputs"]:
        if folder.is_dir() and any(folder.glob("*_headline_rows.csv")):
            return folder
    found = sorted(p.parent for p in ROOT.rglob("*_headline_rows.csv")
                   if _walkable(p.relative_to(ROOT)))
    return found[0] if found else None


def where() -> dict[str, str]:
    """What the app resolved, for the page to print and the reader to check."""
    try:
        assets = str(asset_dir())
    except FileNotFoundError as e:
        assets = f"not found - {e}"
    checkpoints = checkpoint_dir()
    results = results_dir()
    return {"assets": assets,
            "checkpoints": str(checkpoints) if checkpoints else "none found",
            "results": str(results) if results else "none found"}


# ----------------------------------------------------------------------
# The frozen three
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Assets:
    """One series' config, bank and background field, checked to agree."""

    series_code: str
    config: RaceConfig
    bank: SeedBank
    field: BackgroundField

    @property
    def fingerprint(self) -> str:
        return dials_fingerprint(self.config)


def available_series() -> list[str]:
    """Series with a complete set of three files present."""
    return _series_in(asset_dir())


@cache
def load_assets(series_code: str) -> Assets:
    """The three files, refused unless they are the same generation."""
    folder = asset_dir()
    config = RaceConfig.load(_named(series_code, CONFIG_NAMES, folder))
    bank = SeedBank.load(_named(series_code, SEED_NAMES, folder))
    field = BackgroundField.load(_named(series_code, FIELD_NAMES, folder))

    want = dials_fingerprint(config)
    for name, got in (("seed bank", bank.provenance.get("dials_fingerprint")),
                      ("background field",
                       field.provenance.get("dials_fingerprint"))):
        if got != want:
            raise ValueError(
                f"{series_code}: the {name} was frozen against dials {got!r} "
                f"and {series_code}.json is {want!r}. These are different "
                f"races wearing the same seed numbers - refreeze, or point "
                f"the app at the generation that matches.")
    if config.series_code != series_code or bank.series_code != series_code:
        raise ValueError(f"{series_code}: the files disagree about which "
                         f"series they are for")
    return Assets(series_code, config, bank, field)


# ----------------------------------------------------------------------
# The policy
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Agent:
    """A loaded policy, or the reason there isn't one.

    A missing checkpoint is a state the page has to render rather than an
    exception it has to catch: 04 is legible with the agent panel switched
    off and a line saying why, and unusable if it will not start.
    """

    strategy: object | None
    card: PolicyCard | None
    reason: str = ""
    nominal: bool = True

    def __bool__(self) -> bool:
        return self.strategy is not None


def checkpoint_for(series_code: str) -> Path | None:
    """The `.onnx` export, which is the artefact 04 was given.

    The `.zip` is the artefact of record and needs Stable-Baselines3 and
    torch; the export exists so the hosted app does not. If only a `.zip` is
    present the app says so rather than importing half a training stack.
    """
    folder = checkpoint_dir()
    if folder is None:
        return None
    found = sorted(folder.glob(f"{series_code}*.onnx"))
    if len(found) > 1:
        raise ValueError(
            f"{len(found)} exports match {series_code}*.onnx: "
            f"{[p.name for p in found]}. Which one 03b measured is not "
            f"something this app can guess - leave one.")
    return found[0] if found else None


@cache
def load_agent(series_code: str) -> Agent:
    """The policy for this series, checked against the nominal race.

    Checked against *nominal* dials, always - the check is what says this
    checkpoint belongs to this race at all, and it is not skipped merely
    because a slider has moved. What a slider moves is `nominal`, which
    `for_config` below recomputes and which the page uses to decide whether
    a number may be shown.
    """
    path = checkpoint_for(series_code)
    if path is None:
        folder = checkpoint_dir()
        return Agent(None, None,
                     f"no {series_code}*.onnx in "
                     f"{folder if folder else 'any folder searched'}; the "
                     f"agent panel is off")
    assets = load_assets(series_code)
    try:
        card = PolicyCard.load(path)
        strategy = load_policy(path, assets.config, assets.bank)
    except (FileNotFoundError, ValueError, ImportError) as e:
        return Agent(None, None, f"{path.name} was refused: {e}")
    return Agent(strategy, card, "", True)


def agent_for_config(agent: Agent, config: RaceConfig) -> Agent:
    """The same policy, told whether the race in front of it is its own."""
    if not agent or agent.card is None:
        return agent
    nominal = dials_fingerprint(config) == agent.card.dials_fingerprint
    return Agent(agent.strategy, agent.card, agent.reason, nominal)


# ----------------------------------------------------------------------
# The levers
# ----------------------------------------------------------------------
# What the sidebar exposes. Every one of these is either an assumed dial -
# in which case moving it is the sweep 02b and 03b already ran, made
# interactive - or one of the two caution dials that decide how much of the
# race is spent under yellow, which is the lever the roster is most
# sensitive to.
#
# `fuel_per_lap` is deliberately absent. `scale_dials` multiplies one field
# at a time, and fuel per lap is set so that a tank lasts exactly
# `green_stint_laps`; a slider on one of the pair silently breaks that
# relationship and the page would be showing a stint length nobody
# calibrated. Changing stint length properly means recalibrating, which is
# stage 00's job and not a slider's.
LEVERS = (
    "caution_rate",
    "caution_mean_dur_s",
    "caution_pace_multiplier",
    "pit_caution_discount",
    "pit_transit_frac",
    "pit_tyre_frac",
    "tyre_life_laps",
    "caution_pits_open_delay_laps",
    "traffic_window_frac",
    "traffic_penalty_s",
)

# What the page must print beside each one. `caution_rate` and
# `caution_mean_dur_s` are measured; everything else on the list is assumed,
# and two of the assumed ones now have measured counterparts that disagree
# with them (amendment 15), which is a fact about the model the page owes
# the reader rather than a footnote for the write-up.
MEASURED_COUNTERPART = {
    "caution_pace_multiplier": "assumed 1.6; observed 1.87 in IMSA, 1.79 in WEC",
    # This used to claim 0.51-0.66 from a low quantile of real pit times.
    # Retired at 05: that quantile is not an estimator of a fixed cost, and
    # `scripts/estimate_pit_transit.py` shows why in three ways - it is biased
    # high by about +0.14 against a planted truth, it ranges from 0.50 to 0.93
    # across the seven classes of these two races, and the regression that
    # would decide the question has no leverage because endurance cars fill to
    # the brim. There is no measured counterpart, and saying so is the finding.
    "pit_transit_frac": "assumed 0.25; NOT MEASURABLE from lap timing - "
                        "separating what a stop costs to enter from what it "
                        "costs to fill needs stops that took different amounts "
                        "of fuel, and 62-90% of stops in these two races are "
                        "followed by a near-full tank",
}


def lever_kind(dial: str) -> str:
    return "assumed" if dial in ASSUMED_FIELDS else "measured"


def apply_levers(config: RaceConfig, multipliers: dict[str, float]) -> RaceConfig:
    """`scale_dials`, with the multipliers that do nothing dropped.

    Dropping the ones at 1.0 keeps the fingerprint identical to nominal when
    every slider is home, so `nominal` is a property of the dials rather
    than of whether a widget has been touched.
    """
    live = {k: v for k, v in multipliers.items() if v != 1.0}
    return scale_dials(config, **live) if live else config


def lever_warnings(config: RaceConfig) -> list[str]:
    """Constraints a slider combination can break, checked rather than assumed.

    One so far, from amendment 15: transit plus a tyre change cannot exceed
    the whole stop, or `stop_cost` is solving a negative refuel duration and
    clamping it, and the class quietly stops pricing fuel at all. It is
    reachable from the sliders, so the page has to say when it has been
    reached.
    """
    out = []
    for c in config.classes:
        total = c.pit_transit_frac + c.pit_tyre_frac
        if total > 1.0:
            out.append(f"{c.class_name}: pit_transit_frac + pit_tyre_frac = "
                       f"{total:.2f} > 1, so a full service costs less than "
                       f"its parts and fuel is no longer priced")
    return out


# ----------------------------------------------------------------------
# 02c's table, if it was saved
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SavedComparison:
    """02c's table as it was written, rather than as it would be recomputed.

    Three files: the paired rows, the summary and the provenance. The summary
    is read rather than rebuilt from the rows, because 03b's version carries
    bootstrap intervals on `gained`, `lost` and the median that
    `harness.summarise` does not produce - recomputing would quietly drop
    them and the page would show point estimates where an interval exists.
    """

    series_code: str
    bank: str                 # "headline" or "held_out"
    rows: object              # pandas DataFrame
    summary: object | None
    provenance: dict


def saved_banks(series_code: str) -> list[str]:
    folder = results_dir()
    if folder is None:
        return []
    return [bank for bank in ("headline", "held_out")
            if (folder / f"{series_code}_{bank}_rows.csv").exists()]


@cache
def saved_comparison(series_code: str, bank: str = "headline"):
    """The saved table, or `None`.

    Measured on this machine, a live roster pass costs 5.26 s a seed in IMSA
    and 2.61 s in WEC, so the two hundred headline races are eighteen and
    nine minutes. That is not a click. The page therefore reads what 03b
    wrote and offers a small live run only where nothing was written.

    Returned with its provenance so the fingerprint can be checked before
    anything is drawn: a saved table is exactly the artefact most likely to
    outlive the dials it was measured on.
    """
    import json

    import pandas as pd

    folder = results_dir()
    if folder is None:
        return None
    rows_path = folder / f"{series_code}_{bank}_rows.csv"
    if not rows_path.exists():
        return None

    summary_path = folder / f"{series_code}_{bank}_summary.csv"
    meta_path = folder / f"{series_code}_{bank}_provenance.json"
    return SavedComparison(
        series_code=series_code, bank=bank,
        rows=pd.read_csv(rows_path),
        summary=pd.read_csv(summary_path) if summary_path.exists() else None,
        provenance=json.loads(meta_path.read_text()) if meta_path.exists() else {},
    )
