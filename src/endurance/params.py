"""The dials the race engine runs on.

Notebook 00 ended by naming five things a minimal endurance simulator has to
reproduce: a degradation slope, a caution pattern, a stint length, a pit cost
and a traffic density. This module is where those five live as data, one set
per class per series, so that every later notebook, the RL agent and the app
all read the same numbers from the same place.

Two notes on what these numbers are and are not:

* Everything in `ClassDials` is either measured from timing data by
  `calibrate.py` or explicitly marked as assumed. `assumed_fields()` returns
  the assumed ones, so nothing can quietly get mistaken for a measured
  quantity later.
* Fuel is held in **normalised tank units**, not litres. The lap data has no
  fuel column, so a real consumption figure is not recoverable. Instead a
  full tank is defined as 1.0 and `fuel_per_lap` is set so that a tank lasts
  exactly as many laps as the green stints actually observed. That gives an
  honest, data-anchored fuel lever without pretending to know litres.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


# Fields that no amount of lap timing data can identify, and which are
# therefore set by assumption and swept rather than trusted. Kept in one
# place so the notebook can print them and the app can expose them as
# clearly-labelled levers.
ASSUMED_FIELDS = (
    "caution_pace_multiplier",
    "pit_caution_discount",
    "traffic_window_frac",
    "traffic_penalty_s",
    "tyre_life_laps",
    "pit_transit_frac",
    "pit_tyre_frac",
    "pit_transit_caution_discount",
    "caution_pits_open_delay_laps",
    "caution_queue_gap_s",
    "caution_close_frac",
)


@dataclass
class ClassDials:
    """One class of one series: everything the engine needs to run its cars."""

    series_code: str
    class_name: str

    # --- Dial 1: pace and degradation -----------------------------------
    base_pace_s: float                 # median clean green lap, seconds
    deg_slope_s_per_lap: float         # seconds added per lap of tyre age
    pace_spread_s: float               # car-to-car spread in base pace
    lap_noise_s: float                 # lap-to-lap random variation

    # --- Dial 2: cautions (series-wide, stored per class for convenience)
    caution_rate: float                # share of race time under caution
    caution_mean_dur_s: float          # mean length of one caution period
    caution_pace_multiplier: float = 1.6   # ASSUMED: caution lap vs green lap
    # How the field bunches up behind the safety car. Neither is measurable
    # from lap timing - the data records that a lap was slow, never how close
    # the car in front was - so both are ASSUMED and both get swept.
    caution_queue_gap_s: float = 2.0   # ASSUMED: spacing in the queue, seconds
    caution_close_frac: float = 0.5    # ASSUMED: share of the excess gap closed per lap

    # --- Dial 3: stint length, via fuel ---------------------------------
    green_stint_laps: float = 30.0     # observed green stint length, laps
    fuel_per_lap: float = 1.0 / 30.0   # normalised: full tank = 1.0
    fuel_per_lap_caution: float = 0.6 / 30.0   # cautions burn less fuel
    tyre_life_laps: float = 60.0       # ASSUMED: laps before tyres force a stop

    # --- Dial 4: pit cost ------------------------------------------------
    pit_time_mean_s: float = 45.0
    pit_time_std_s: float = 3.0
    pit_caution_discount: float = 0.4  # ASSUMED: share of *service* saved under caution
    # ASSUMED: the same, for the lane transit, and separate from the line above
    # because the two are different claims. The service discount says a slow
    # field makes the time in the box matter less. Applying it to the transit
    # as well says the drive down the lane is cheaper too - arguable, and it is
    # what produced amendment 14's floor violation, because at 0.4 a caution
    # stop came out below the cost of simply driving the length of the lane.
    # Default 0.0: the transit is not discounted. Set it equal to
    # `pit_caution_discount` to reproduce the pre-amendment engine exactly,
    # which is what the test of that name asserts.
    pit_transit_caution_discount: float = 0.0
    # How a stop divides up. Shares of the measured mean rather than absolute
    # seconds, so they cannot go negative on a fast class: a full tank plus
    # tyres still costs `pit_time_mean_s` whatever these are set to, and they
    # decide only what a partial stop saves. Both ASSUMED - lap timing records
    # how long a stop took, never what happened during it.
    pit_transit_frac: float = 0.25     # ASSUMED: lane transit, no service
    pit_tyre_frac: float = 0.35        # ASSUMED: a tyre change, car stationary
    # ASSUMED: caution laps before the pits are declared open. The rulebooks
    # tie this to the field forming up behind the safety car; the class
    # staging that follows it is regulation and lives in `pitstop.py`.
    caution_pits_open_delay_laps: float = 1.0

    # --- Dial 5: traffic -------------------------------------------------
    n_cars: int = 10                   # cars of this class on track
    traffic_window_frac: float = 0.02  # ASSUMED: how close counts as "in traffic"
    traffic_penalty_s: float = 0.8     # ASSUMED: seconds lost per car held up behind

    # --- provenance -------------------------------------------------------
    source_event: str = ""
    n_laps_observed: int = 0

    def fuel_laps(self) -> float:
        """How many green laps a full tank covers."""
        return 1.0 / self.fuel_per_lap

    @classmethod
    def assumed_fields(cls) -> tuple[str, ...]:
        return ASSUMED_FIELDS

    def measured_fields(self) -> list[str]:
        return [f.name for f in fields(self)
                if f.name not in ASSUMED_FIELDS
                and f.name not in ("series_code", "class_name",
                                   "source_event", "n_laps_observed")]


@dataclass
class RaceConfig:
    """One race: how long it runs and which classes are in it."""

    name: str
    series_code: str
    duration_s: float
    classes: list[ClassDials] = field(default_factory=list)

    # Regulation-ish constraints, kept deliberately simple for now.
    max_driver_stint_s: float = 4 * 3600.0   # forces a driver change

    def class_by_name(self, class_name: str) -> ClassDials:
        for c in self.classes:
            if c.class_name == class_name:
                return c
        raise KeyError(f"no class {class_name!r} in race {self.name!r}")

    @property
    def total_cars(self) -> int:
        return sum(c.n_cars for c in self.classes)

    # --- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RaceConfig":
        classes = [ClassDials(**c) for c in d.get("classes", [])]
        rest = {k: v for k, v in d.items() if k != "classes"}
        return cls(classes=classes, **rest)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RaceConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _adjusted(config: RaceConfig, changes: dict[str, float],
              multiply: bool) -> RaceConfig:
    """The one place a twisted copy of the dials is made.

    `scale_dials` and `set_dials` differ by one line and are otherwise the
    same operation - copy, check the name, write the field, hold the caution
    rate below 1. Written once because a sweep that took a different route to
    a config would be a second definition of what a lever does, and the two
    would agree right up until one of them was edited.
    """
    new = RaceConfig.from_dict(config.to_dict())

    if not multiply:
        # Setting writes the *same* number onto every class, while scaling
        # moves each class's own value. That is fine for a dial the classes
        # share and wrong for one they do not - `base_pace_s` differs by
        # fifteen seconds between GTP and GTD, and flattening it would be a
        # different race presented as a lever. Refused rather than reported:
        # the caller either meant a dial that is shared, or did not mean this.
        for dial in changes:
            seen = {getattr(c, dial) for c in new.classes
                    if hasattr(c, dial)}
            if len(seen) > 1:
                raise ValueError(
                    f"the classes hold {len(seen)} different values for "
                    f"{dial!r} ({sorted(seen)}), so setting one number would "
                    f"flatten them. Scale it instead, or set it per class.")

    for c in new.classes:
        for dial, value in changes.items():
            if not hasattr(c, dial):
                raise AttributeError(f"{dial!r} is not a dial on ClassDials")
            setattr(c, dial, getattr(c, dial) * value if multiply else value)
        c.caution_rate = min(c.caution_rate, 0.95)
    return new


def scale_dials(config: RaceConfig, **multipliers: float) -> RaceConfig:
    """Return a copy of `config` with named dials multiplied.

    This is the mechanism behind every lever in the app: rather than editing
    the engine, you hand it a twisted copy of the dials. For example
    `scale_dials(cfg, caution_rate=3.0)` gives a race three times as
    caution-heavy, everything else held fixed.
    """
    return _adjusted(config, multipliers, multiply=True)


def set_dials(config: RaceConfig, **values: float) -> RaceConfig:
    """Return a copy of `config` with named dials set to a value.

    Multiplying cannot move a dial that sits at zero, and one now does:
    `pit_transit_caution_discount` defaults to 0.0, so every multiplier
    leaves it there. Amendment 23 says that dial is swept like any other
    assumption, and until this existed it could not be swept at all.

    Use this for a dial whose default is zero, or wherever the question is
    "what if this were 0.4" rather than "what if this were twice what it is".
    Scaling remains the default everywhere else, because a multiplier is
    comparable across classes and a value is not.
    """
    return _adjusted(config, values, multiply=False)
