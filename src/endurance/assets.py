"""The things 02b freezes so that 02c and 03 are measuring the same races.

Decision 6 asks for artefacts rather than a notebook: 03 has to evaluate on
the races the human strategies were scored on, or the comparison is between
two different questions. Two of those artefacts are made here - the seed
banks and the frozen background field - because 02b is the first stage that
needs them, and everything after inherits them unchanged.

Nothing in here is clever. Its whole job is to be boring and to stay the
same, and the provenance block is the part that makes that checkable: a bank
records which dials it was drawn against, so a bank quietly reused with a
recalibrated engine is a mismatch that can be detected rather than a
discrepancy someone argues about six months later.

The banks, and why they are nested the way they are
---------------------------------------------------
Decision 10 sets the budget: two hundred paired seeds per strategy per
series for headline claims, fifty for sweeps, and a disjoint fifty held out
for anything the roster gets selected on.

The sweep fifty are the *first fifty of the headline two hundred* rather
than a separate draw. That is deliberate. A sweep is asking how a claim
moves as a dial moves, and the cleanest version of that question compares
against the same races the headline claim was made on. A disjoint sweep bank
would add sampling noise to a comparison that does not need any.

The held-out fifty are genuinely disjoint, for the opposite reason: they
exist to answer whether a roster chosen on the headline races generalises,
and a held-out set overlapping the selection set answers nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .caution import RULES_VERSION as CAUTION_RULES_VERSION
from .pitstop import RULES_VERSION as PITSTOP_RULES_VERSION

# Imported at the top rather than inside the function, unlike `strategies`
# below: `pitstop` and `caution` import nothing from this package, so there is
# no cycle to avoid, and a fingerprint that could fail at call time rather
# than at import time is a tripwire with a soft spot in it.

HEADLINE_N = 200
SWEEP_N = 50
HELD_OUT_N = 50

# Where seeds are drawn from. Wide enough that collision is not a concern and
# small enough to stay readable in a JSON file somebody has to eyeball.
SEED_SPACE = 1_000_000


@dataclass
class SeedBank:
    """The races every later stage is scored on, written down rather than derived.

    The seeds are stored as explicit lists and not as a rule for generating
    them. A rule is shorter and invites exactly one failure: somebody
    changes the generator, the lists move, and two stages are silently
    scored on different races.
    """

    series_code: str
    headline: list[int]
    sweep: list[int]
    held_out: list[int]
    provenance: dict = field(default_factory=dict)

    def check(self) -> None:
        """The properties the banks are only useful if they have."""
        if len(set(self.headline)) != len(self.headline):
            raise ValueError("headline bank repeats a seed")
        if len(set(self.held_out)) != len(self.held_out):
            raise ValueError("held-out bank repeats a seed")
        if self.sweep != self.headline[:len(self.sweep)]:
            raise ValueError("sweep bank is not a prefix of the headline bank")
        overlap = set(self.headline) & set(self.held_out)
        if overlap:
            raise ValueError(f"held-out bank overlaps the headline bank: "
                             f"{sorted(overlap)[:5]}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SeedBank":
        bank = cls(**d)
        bank.check()
        return bank

    def save(self, path: str | Path) -> None:
        self.check()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SeedBank":
        return cls.from_dict(json.loads(Path(path).read_text()))


def dials_fingerprint(config) -> str:
    """A short hash of the dials a bank or field was built against.

    Not a security measure - a tripwire. If this does not match when a bank
    is loaded, the engine has been recalibrated since and any comparison
    across the two is between different races wearing the same seed numbers.
    """
    payload = json.dumps(config.to_dict(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def rules_fingerprint() -> str:
    """A short hash of the rule logic the dials are run through.

    Amendment 21's gap, closed beside `dials_fingerprint` rather than inside
    it. `dials_fingerprint` hashes `config.to_dict()`, which is parameters
    only - so `stop_cost` could be rewritten, every race in the project would
    change, and every bank, field, card and saved table would still match. The
    two fingerprints answer different questions and are kept apart so that a
    mismatch says which one moved.

    A property of the code and not of any config, so it takes no argument.

    **What this does not cover.** `engine.py`'s own arithmetic, which includes
    the floor clamp in `_apply_pit`. Putting the engine in here would mean
    bumping a version on every edit to the largest and most frequently touched
    module in the project, and a version nobody bumps honestly is worse than
    an absent one. The rules layers are covered; the engine is not, and that
    is stated rather than discovered.
    """
    payload = json.dumps({"caution": CAUTION_RULES_VERSION,
                          "pitstop": PITSTOP_RULES_VERSION},
                         sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def rules_mismatch(recorded: str | None, what: str = "this artefact") -> str | None:
    """The message to raise or print, or `None` if there is nothing to say.

    Checked **only where present**, which is deliberate. Every artefact
    currently on disk was frozen before this existed and carries no rules
    fingerprint, so a strict check would refuse the whole project on its first
    run. `None` in means "this artefact predates the check", and that is
    reported on the methods page rather than treated as a pass.
    """
    if recorded is None:
        return None
    current = rules_fingerprint()
    if recorded == current:
        return None
    return (f"{what} was built against rule logic {recorded!r} and this tree "
            f"is {current!r}. The dials may well match - they are hashed "
            f"separately - but `pitstop.py` or `caution.py` has changed since, "
            f"so the same seed is a different race.")


def dials_source(config) -> str:
    """Where these dials came from, derived rather than typed.

    The literal `"STAND-IN, not calibrated"` was written into `train.py`,
    `evaluate.py` and `sweep_pit_transit.py` when the dials genuinely were a
    six-hour invented race. It stopped being true when 00's re-run landed, and
    it went on being stamped onto policy cards and evaluation provenance for
    four more generations - by then reading `"STAND-IN, calibrated"`, which is
    not a description of anything.

    So it is computed from the config, which knows. A string that has been
    wrong in three places at once should exist in one.
    """
    if not config.classes:
        return "unknown: this config has no classes"
    source = config.classes[0].source_event
    if not source:
        return "unknown: no source_event on the dials"
    if config.classes[0].n_laps_observed:
        return (f"calibrated from {source}, "
                f"{config.classes[0].n_laps_observed} laps observed")
    return f"calibrated from {source}"


def dials_source(config) -> str:
    """Where these dials came from, read off the config rather than typed.

    Four files wrote this as a string literal. It said `"STAND-IN, not
    calibrated"` for four generations after 00's re-run made it false, was
    stamped onto every policy card and every evaluation provenance file in
    that time, and was then changed to `"STAND-IN, calibrated"`, which
    describes nothing. A card that lies about its dials is worse than a card
    with no note on them.

    `source_event` is written by `calibrate.build_race_config` and names the
    session, so this cannot drift from the dials it describes: they move
    together or not at all.

    The pooled case is reported rather than smoothed over. Several classes
    naming several events is what a scoping fault looks like from here, and 00
    spent a whole re-run on one.
    """
    events = sorted({c.source_event for c in config.classes if c.source_event})
    if not events:
        return "unknown - this config records no source event"
    if len(events) > 1:
        return "POOLED across " + ", ".join(events)
    return f"calibrated from {events[0]}"


def draw_seed_bank(config, draw_seed: int = 20260806,
                   headline_n: int = HEADLINE_N, sweep_n: int = SWEEP_N,
                   held_out_n: int = HELD_OUT_N) -> SeedBank:
    """Draw the three banks once. `draw_seed` is the only thing to keep.

    Drawn without replacement across all three so that no held-out race can
    coincide with a headline one.
    """
    rng = np.random.default_rng(draw_seed)
    total = headline_n + held_out_n
    seeds = rng.choice(SEED_SPACE, size=total, replace=False)
    headline = [int(s) for s in seeds[:headline_n]]
    held_out = [int(s) for s in seeds[headline_n:]]

    bank = SeedBank(
        series_code=config.series_code,
        headline=headline,
        sweep=headline[:sweep_n],
        held_out=held_out,
        provenance={
            "drawn_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "draw_seed": draw_seed,
            "seed_space": SEED_SPACE,
            "race": config.name,
            "duration_s": config.duration_s,
            "dials_fingerprint": dials_fingerprint(config),
            "rules_fingerprint": rules_fingerprint(),
            "sweep_is_prefix_of_headline": True,
            "held_out_disjoint": True,
        },
    )
    bank.check()
    return bank


@dataclass
class BackgroundField:
    """What every car that is not the focal car runs, frozen.

    Held as a per-car map rather than a single strategy name even though
    every value is identical today. Decision 2 makes the background choice
    an assumed parameter that gets swept, and one of the sweeps worth
    running is a *mixed* field - 02a's traffic work only bites when the
    field is out of phase, and a single-strategy background pits in lockstep
    and hides it entirely. A map costs nothing now and means that sweep is a
    change of values rather than a change of format.
    """

    strategies: dict[str, str]
    provenance: dict = field(default_factory=dict)

    def resolve(self, focal: str | None = None) -> dict:
        """Turn the names into callables, leaving the focal car to its plan."""
        from .strategies import BASELINES

        out = {}
        for car_id, name in self.strategies.items():
            if focal is not None and car_id == focal:
                continue
            if name not in BASELINES:
                raise KeyError(f"{name!r} is not a baseline strategy")
            out[car_id] = BASELINES[name]()
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BackgroundField":
        return cls(**d)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BackgroundField":
        return cls.from_dict(json.loads(Path(path).read_text()))


def freeze_background(config, strategy: str = "fuel_window") -> BackgroundField:
    """Give every car in the race the same background strategy.

    `fuel_window` by default because decision 10 makes it the null every
    paired delta is measured from, so the field a strategy is measured
    against and the plan it is measured against are the same idea.
    """
    from .strategies import BASELINES

    if strategy not in BASELINES:
        raise KeyError(f"{strategy!r} is not a baseline strategy")

    strategies = {}
    for cls in config.classes:
        for i in range(cls.n_cars):
            strategies[f"{cls.class_name}-{i + 1:02d}"] = strategy

    return BackgroundField(
        strategies=strategies,
        provenance={
            "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "race": config.name,
            "series_code": config.series_code,
            "uniform_strategy": strategy,
            "dials_fingerprint": dials_fingerprint(config),
            "rules_fingerprint": rules_fingerprint(),
            "note": ("assumed parameter per decision 2; belongs in "
                     "ASSUMED_FIELDS and gets swept"),
        },
    )
