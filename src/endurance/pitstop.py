"""What a pit stop costs, and whether you are allowed to take one.

Notebook 01 priced every stop at a single `pit_time_mean_s`, which made a
30% splash cost exactly what a full service costs. That is not a missing
feature so much as a missing strategy: splash-and-dash is unimplementable
until a stop has a shape, and the caution gambler is only gambling if the
lane can be shut when it arrives.

Three things live here as of amendment 14 - the shape of a stop, the lane
status, and **what a caution does to the price**, which used to be a line in
`engine._apply_pit` and is a statement about pit regulations rather than about
the race loop. Following decision 4/7, this encodes
just the rules that change the shape of a stop - the fixed part, the refuel
duration, and whether tyres may be worked on while fuel goes in - plus the
lane-status rule the blueprint adds on top. Driver-time regulations stay
out.

**The level is measured; only the shape is regulated.** `pit_time_mean_s`
comes from timing data and is the one thing here that is not a guess, so
the model is anchored to it: a full tank plus tyres costs exactly the
measured mean, by construction. Everything the layer does is therefore a
statement about stops that are *not* full service, which is the only place
a rulebook can tell you something lap times cannot. A pleasant consequence
is that turning the layer on changes nothing at all for 01's baselines,
which never ask for anything but a full tank and fresh tyres - so any
movement in the numbers is the lane, not the arithmetic.

Where the two series genuinely differ
-------------------------------------
IMSA permits four people over the wall including the refueller, air jack
and tyre changes (art. 34.1.1), so tyres come off while fuel goes in and a
stop costs the longer of the two jobs. WEC forbids tools during the
refuelling phase (art. 12), so the jobs are sequential and a stop costs
their sum. This is the point at which the two series stop being one model
with different constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Amendment 21, closed as far as it can be closed here. `dials_fingerprint`
# hashes parameters only, so a change to the *logic* in this module changes
# every race in the project while leaving every seed bank, field, card and
# saved table matching perfectly - the tripwire that has caught four artefact
# collisions would report nothing. This number is what `assets.rules_fingerprint`
# hashes, and it is recorded beside the dials fingerprint from now on.
#
# **Bump it when the arithmetic changes, not when a comment does.** History:
#   1  02a - the layer as first written; the discount lived in `_apply_pit`
#   2  amendment 23 - the discount moved here and split in two, and the lane
#      transit became a floor a stop cannot be priced below
RULES_VERSION = 2


# Classes are released from a closed pit lane in category order, prototypes
# first. Mapping the calibrated class names onto those groups is a lookup
# rather than a rule, so it lives here where it can be seen and corrected
# rather than being buried in a comparison.
_PROTOTYPE = ("GTP", "LMP2", "HYPERCAR", "LMP1", "LMDH")
_GT = ("GTD PRO", "GTD", "LMGT3", "GTE", "GTE PRO", "GTE AM", "GT3")


@dataclass(frozen=True)
class PitRules:
    """One series' pit regulations, as far as they change a stop.

    `unknown` is not a series - it is what a config gets when its
    `series_code` matches no rulebook, as the synthetic test configs do. It
    leaves the lane open always, because closing it would mean asserting a
    regulation nobody wrote.
    """

    series_code: str

    # --- the shape of a stop ---------------------------------------------
    tyres_during_refuel: bool = True

    # --- the lane, under caution -----------------------------------------
    lane_closes_under_caution: bool = True
    # Caution laps before each group is released, counted from the moment the
    # pits are declared open. How long *that* takes is not a rulebook fact -
    # both books tie it to the field forming up behind the safety car, which
    # is not simulated - so it is an assumed dial, `caution_pits_open_delay_laps`,
    # and it is swept rather than trusted. IMSA releases
    # prototypes, then GTs, then anyone (art. 46.3.1). WEC releases everyone
    # together once the closure expires (art. 14.6.5).
    stagger_laps: tuple[float, float, float] = (1.0, 2.0, 3.0)
    # A caution that never offers a stop at all. IMSA's Short FCY covers any
    # caution inside this window of the start, or of a restart (art. 46.3.3),
    # and art. 46.3.2 disapplies the standard procedure in the last half hour
    # too - read here as the same outcome, which is an interpretation rather
    # than a quotation.
    never_opens_window_s: float = 0.0
    never_opens_after_restart_s: float = 0.0

    @classmethod
    def for_series(cls, series_code: str) -> "PitRules":
        code = (series_code or "").strip().lower()
        if code == "imsa":
            return cls(
                series_code="imsa",
                tyres_during_refuel=True,          # art. 34.1.1, four over the wall
                stagger_laps=(1.0, 2.0, 3.0),      # art. 46.3.1
                never_opens_window_s=1800.0,       # art. 46.3.3 and 46.3.2
                never_opens_after_restart_s=900.0,
            )
        if code == "wec":
            return cls(
                series_code="wec",
                tyres_during_refuel=False,         # art. 12, no tools while fuelling
                stagger_laps=(3.0, 3.0, 3.0),      # art. 14.6.5, everyone together
            )
        return cls(series_code="unknown", lane_closes_under_caution=False)

    # -- the shape of a stop ----------------------------------------------
    def group_index(self, class_name: str) -> int:
        name = (class_name or "").strip().upper()
        if name in _PROTOTYPE:
            return 0
        if name in _GT:
            return 1
        return 2


def transit_s(cls_dials) -> float:
    """The floor: driving the length of the pit lane, with nobody touching the car.

    Amendment 14's finding was that a stop came out at 12.85 s against a
    transit of 22.45 s. **A stop cannot cost less than this**, whatever the
    caution discount and whatever the noise draw, and `_apply_pit` clamps to it
    after noise rather than before - a negative tail on the noise can violate a
    floor just as easily as a discount can.
    """
    return cls_dials.pit_time_mean_s * cls_dials.pit_transit_frac


def stop_cost(cls_dials, rules: PitRules, fuel_added: float,
              change_tyres: bool, *, under_caution: bool = False,
              legacy: bool = False) -> float:
    """The mean cost of one stop, before noise.

    `fuel_added` is in normalised tank units, so a full tank is 1.0 and a
    splash is whatever fraction the strategy asked for.

    **The caution discount is applied here as of amendment 14**, and applied to
    the transit and the service separately, because they are separate claims
    about what a caution is worth. It used to be applied by `_apply_pit` to the
    whole stop including the transit, which is how a caution stop came to cost
    less than driving down the lane. Moving it here is not tidiness: the
    discount changes what a stop costs, and what a stop costs is what this
    module is.

    One consequence, deliberate and worth stating. The discount used to
    multiply the *noise* too, because `_apply_pit` scaled the total after
    adding it. It no longer does: a caution stop now has the same spread as a
    green one. Nobody decided the old behaviour - it was a side effect of the
    order of two lines - and a discount on the variance is a different claim
    from a discount on the cost.
    """
    mean = cls_dials.pit_time_mean_s
    if legacy:
        return mean

    transit = mean * cls_dials.pit_transit_frac
    tyre_s = mean * cls_dials.pit_tyre_frac if change_tyres else 0.0

    # Solve the refuel rate from the measured mean rather than inventing a
    # litres-per-second figure the data cannot support: a full tank plus
    # tyres has to come to exactly `pit_time_mean_s`.
    if rules.tyres_during_refuel:
        full_refuel_s = mean - transit          # the longer job sets the cost
    else:
        full_refuel_s = mean - transit - mean * cls_dials.pit_tyre_frac
    full_refuel_s = max(full_refuel_s, 0.0)

    refuel_s = full_refuel_s * max(fuel_added, 0.0)
    service = max(refuel_s, tyre_s) if rules.tyres_during_refuel else refuel_s + tyre_s

    if under_caution:
        transit *= 1.0 - cls_dials.pit_transit_caution_discount
        service *= 1.0 - cls_dials.pit_caution_discount

    return transit + service


# ----------------------------------------------------------------------
# The lane
# ----------------------------------------------------------------------
@dataclass
class LaneStatus:
    """Whether the lane is open, and if not, why not - the reason is the point.

    A strategy that finds the lane shut needs to know whether waiting will
    help. Under a Short FCY it never will.
    """

    open: bool
    reason: str = ""
    opens_at_s: float | None = None


_OPEN = LaneStatus(open=True)


def lane_status(rules: PitRules, cautions, t: float, class_name: str,
                caution_lap_s: float, duration_s: float,
                open_delay_laps: float = 1.0) -> LaneStatus:
    """Is this class allowed into the pits at time `t`?

    Reopening is counted in caution laps because that is the unit both
    rulebooks use, and converted here rather than at the call site so there
    is one place where the conversion can be wrong.
    """
    if not rules.lane_closes_under_caution:
        return _OPEN

    episode = _episode_at(cautions, t)
    if episode is None:
        return _OPEN

    start, end = episode
    if _never_opens(rules, cautions, start, duration_s):
        return LaneStatus(False, "short caution, lane stays shut", None)

    laps = open_delay_laps + rules.stagger_laps[rules.group_index(class_name)]
    opens_at = start + laps * max(caution_lap_s, 1e-6)
    if t >= opens_at:
        return _OPEN
    if opens_at >= end:
        return LaneStatus(False, "lane shut for this caution", None)
    return LaneStatus(False, "lane not yet open to this class", opens_at)


def _episode_at(cautions, t: float):
    for start, end in cautions.periods:
        if start <= t < end:
            return (start, end)
    return None


def _never_opens(rules: PitRules, cautions, start: float, duration_s: float) -> bool:
    """Short FCY: a caution too near the start, a restart, or the finish."""
    if rules.never_opens_window_s <= 0.0:
        return False
    if start < rules.never_opens_window_s:
        return True
    if start > duration_s - rules.never_opens_window_s:
        return True
    prev_end = max((e for s, e in cautions.periods if e <= start), default=None)
    if prev_end is not None and start - prev_end < rules.never_opens_after_restart_s:
        return True
    return False
