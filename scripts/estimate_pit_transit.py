"""What fraction of a pit stop is the lane, and what fraction is the tyres?

    python scripts/estimate_pit_transit.py
    python scripts/estimate_pit_transit.py --laps path/to/laps.csv

Writes `data/processed/pit_transit_estimate.json` and changes no dial. It is a
measurement of two assumed quantities, offered beside them rather than
substituted for them - amendment 15's rule, which is that an assumed dial with
a measured counterpart that disagrees is left alone and the disagreement is
shown.

Why a fifth percentile is not the answer
----------------------------------------
`calibrate.calibrate_pit` returns `pit_lane_transit_s` as the fifth percentile
of the trimmed green-flag sample, and `pit_transit_frac`'s measured
counterpart is that over the median. It is the best thing available without a
model, and it rests on an assumption nothing checks: that the cheapest one stop
in twenty had *almost no service in it*. It might instead be a short splash, a
stop that overlapped a caution, or a car that pitted and retired. A quantile
cannot tell those apart, because it is a rank rather than a decomposition.

The mismatch that prompted this makes the point. `data/processed/laps_profile.md`
reports a fifth percentile of 22.57 s against a median of 101.76 s, which
divides out to 0.22 - close enough to the assumed 0.25 to look like
confirmation. It is not the same statistic: the profile queries `laps` with no
predicate at all, so it pools four series, six seasons and every session type,
and 462,651 of its rows are practice. `calibrate_pit` scopes to one race
session, one class, green flags only, trimmed. Same arithmetic, different
population, and only the second is a measurement of anything.

**Neither number should be quoted without its population, and the sentence in
the profile saying the fifth percentile is what would become
`pit_transit_frac` was written before the scoped estimator existed. It is now
wrong and the file should say so.**

What this does instead
----------------------
A stop is a fixed part and a variable part. The fixed part is the lane: entry,
the speed limit, exit, and it is paid whatever the car came in for. The
variable part is the service, and the largest piece of it is fuel, which is not
recorded anywhere in the file.

**The length of the stint that follows a stop is a proxy for how much fuel went
in**, because a car that fills more runs longer before it has to come back.
That gives a line to fit:

    pit_time = transit + refuel_rate * (green laps in the next stint)
                       + tyre_cost * (tyres were changed)

and the intercept is the thing the quantile was guessing at. The tyre term is
in the same fit rather than a separate one because the two are confounded -
a stop with tyres costs more, and if long fills come with fresh tyres the
intercept absorbs the difference. Fitting both gives `pit_transit_frac` and
`pit_tyre_frac`, which are two of the assumed dials, from one measurement.

Tyre changes are read from `est_tire_age` resetting across the stop, which is
the same column `calibrate_pace` uses for degradation, so this cannot disagree
with that fit about which stops changed tyres.

When it cannot answer
---------------------
Following `calibrate_pace`'s convention, this reports `identified: false`
rather than a number it cannot support. Two ways that happens:

* **Every stop is a full fill.** Then the next stint is the same length every
  time, the regressor has no spread, and any intercept fits. A race where
  nobody splashes cannot tell the lane from the tank.
* **Nobody double-stints.** Then the tyre indicator is constant and its
  coefficient is not identified, exactly as `deg_identified` is false for a
  class that changes tyres at every stop.

What to do with the answer
--------------------------
Not recalibrate. Two things:

**Put it in the note.** `app/loading.MEASURED_COUNTERPART` carries the
disagreement for `pit_transit_frac`, and it should carry one number produced
one way rather than a range whose method nobody can chase.

**Check it against the shape of `stop_cost`.** A full service is anchored at
the measured mean and split into shares, so `pit_transit_frac + pit_tyre_frac`
cannot exceed 1.0 without a full stop costing something other than what was
measured. `scripts/sweep_pit_transit.py` caps its sweep at 0.65 for that
reason. If this estimate puts the two together at or above 1.0, the finding is
not about the value of a dial - it is that the current shape cannot express a
real pit lane, which is a bigger thing and is what that script's docstring
already warns about.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def find_project_root(marker: str = "src/endurance") -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(
        f"could not find {marker!r} at or above {here.parent}.\n\n"
        "This script belongs in `scripts/`, beside `src/` and `notebooks/`.")


ROOT = find_project_root()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

from endurance import calibrate                                 # noqa: E402
from endurance.params import RaceConfig                         # noqa: E402

LAPS = ROOT / "data" / "raw" / "laps.csv"
OUT = ROOT / "data" / "processed" / "pit_transit_estimate.json"

# The sessions the frozen configs were calibrated from. Read from the config's
# own `source_event` where possible rather than restated, so this cannot end
# up measuring a different race from the one the dials describe.
FROZEN = {"imsa": 682, "wec": 1000}

GREEN = "GF"
# The same trim `calibrate_pit` uses, and for the same reason: within one
# properly scoped race the column still carries hour-long repairs beside
# eighty-second stops. Kept identical so the two are comparable.
TRIM_FACTOR = 3.0
# Below this many usable stops a per-class fit is not worth reporting.
MIN_STOPS = 30
# The regressor needs spread to identify an intercept. Expressed as the
# interquartile range of the next-stint length, in laps.
MIN_STINT_SPREAD_LAPS = 2.0
# And it needs to *explain* something. A proxy carrying no information gives a
# flat line, and a flat line hands its entire budget to the intercept - which
# then equals the mean stop cost and looks like a transit that costs almost a
# whole stop. The first run of this script did exactly that on IMSA GTD and
# reported 0.95 as identified. Refused on the slope's t statistic rather than
# on R2, because R2 answers "how much of the spread is explained" and the
# question here is the narrower one: is this slope distinguishable from zero.
MIN_SLOPE_T = 2.0
# And the fills have to actually differ. A spread in the regressor is not the
# same as a spread in *fuel*: a stint cut short by a caution is short for a
# reason that has nothing to do with what went in the tank. If nearly every
# stop is a full fill, the design has no leverage on the split however wide
# the stint lengths look.
NEAR_FULL = 0.90        # of a tankful counts as a full fill
MAX_NEAR_FULL_SHARE = 0.60


def find(name: str) -> Path | None:
    for hit in sorted(ROOT.rglob(name)):
        if ".ipynb_checkpoints" not in hit.parts:
            return hit
    return None


def session_for(code: str) -> int:
    """The session id the frozen config names, or the recorded default."""
    path = find(f"{code}.json")
    if path is None:
        return FROZEN[code]
    config = RaceConfig.load(path)
    for cls in config.classes:
        # `source_event` reads like "Daytona 2026 (imsa session 682)".
        if "session" in cls.source_event:
            tail = cls.source_event.rsplit("session", 1)[1]
            digits = "".join(c for c in tail if c.isdigit())
            if digits:
                return int(digits)
    return FROZEN[code]


# ----------------------------------------------------------------------
# The stops, and what followed each one
# ----------------------------------------------------------------------
def stops_with_next_stint(con, session_id: int, class_name: str) -> pd.DataFrame:
    """One row per green pit stop: what it cost and what it bought.

    `next_green_laps` counts green laps from the stop to the car's next stop,
    which is the proxy for fuel taken. `tyres` is `est_tire_age` falling across
    the stop, read rather than inferred from the stint counter - `stint_number`
    steps on driver changes, which is the defect that reported fifty-eight-lap
    green stints at Daytona.

    The last stop of each car is dropped: there is no following stint, and a
    stop with no observable purchase carries no information about the price.
    """
    laps = con.execute(f"""
        SELECT car, lap, pit_time, flags, est_tire_age, driver_name
        FROM laps
        WHERE session = 'race' AND class = '{class_name}'
          AND session_id = {session_id}
        ORDER BY car, lap
    """).df()
    if laps.empty:
        return laps

    laps["pit_time"] = pd.to_numeric(laps["pit_time"], errors="coerce")
    laps["est_tire_age"] = pd.to_numeric(laps["est_tire_age"], errors="coerce")

    rows = []
    for car, sub in laps.groupby("car", sort=False):
        sub = sub.reset_index(drop=True)
        stop_at = sub.index[sub["pit_time"].notna()].tolist()
        for i, here in enumerate(stop_at):
            if i + 1 >= len(stop_at):
                break                       # no following stint to measure
            nxt = stop_at[i + 1]
            window = sub.iloc[here + 1:nxt + 1]
            after = sub.iloc[here + 1] if here + 1 < len(sub) else None
            rows.append({
                "car": car,
                "lap": int(sub.at[here, "lap"]),
                "pit_time": float(sub.at[here, "pit_time"]),
                "flags": sub.at[here, "flags"],
                "next_green_laps": int((window["flags"] == GREEN).sum()),
                # Counted separately rather than lumped in: a caution lap
                # burns `fuel_per_lap_caution`, and at a race a third of which
                # is yellow the distinction is most of the proxy.
                "next_caution_laps": int((window["flags"] != GREEN).sum()),
                # Tyre age falling across the stop is a change. `None` on
                # either side means the column cannot say, and those rows are
                # dropped rather than assumed either way.
                "tyres": (None if after is None
                          or pd.isna(after["est_tire_age"])
                          or pd.isna(sub.at[here, "est_tire_age"])
                          else float(after["est_tire_age"])
                          < float(sub.at[here, "est_tire_age"])),
                "driver_change": (None if after is None else
                                  after["driver_name"] != sub.at[here, "driver_name"]),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------
def estimate(stops: pd.DataFrame, median_stop_s: float,
             caution_fuel_ratio: float = 1.0,
             tank_laps: float = 0.0) -> dict:
    """Least squares on the three-term model, with its own refusals.

    Written out with `numpy.linalg.lstsq` rather than reached for through a
    fitting library, because the whole quantity of interest is the intercept
    and it is worth being able to see exactly what is being regressed on what.
    """
    green = stops[(stops["flags"] == GREEN) & stops["pit_time"].notna()]
    kept = green[green["pit_time"] <= TRIM_FACTOR * median_stop_s]
    usable = kept[kept["tyres"].notna()]

    out = {
        "n_stops_green": int(len(green)),
        "n_trimmed_out": int(len(green) - len(kept)),
        "n_usable": int(len(usable)),
        "median_stop_s": round(median_stop_s, 3),
        "identified": False,
        "why_not": "",
    }
    if len(usable) < MIN_STOPS:
        out["why_not"] = (f"{len(usable)} usable stops, fewer than the "
                          f"{MIN_STOPS} this is willing to fit")
        return out

    # Fuel taken, as well as this can be proxied. A green lap burns
    # `fuel_per_lap` and a caution lap burns `fuel_per_lap_caution`, so at
    # Daytona - a third of it yellow - counting green laps alone is not a
    # proxy for the tank at all: a full fill followed by a long caution
    # covers far more laps than the same fill under green, and the difference
    # has nothing to do with what went in.
    laps = (usable["next_green_laps"].astype(float)
            + caution_fuel_ratio * usable["next_caution_laps"].astype(float)
            ).to_numpy()
    spread = float(np.percentile(laps, 75) - np.percentile(laps, 25))
    out["next_stint_iqr_fuel_laps"] = round(spread, 2)
    out["caution_fuel_ratio"] = round(caution_fuel_ratio, 4)

    # **Is the proxy tracking the tank at all?** This decides how the bound
    # below should be read, and it is the check the first two versions of this
    # script lacked entirely.
    #
    # A regressor measured with error biases its own slope towards zero - so a
    # real refuel component seen through a noisy proxy and no refuel component
    # at all produce the same near-zero slope, and the intercept swells to
    # cover the difference. The two are indistinguishable from the fit.
    #
    # They are distinguishable from the proxy's own distribution. A car cannot
    # cover more than a tankful between stops, so if the observed fuel-laps sit
    # under the calibrated tank the proxy is behaving like fuel; if a large
    # share run past it, it is measuring something else - a stint that spans a
    # stop this script did not see, most likely - and the bound is soft.
    near_full = 0.0
    if tank_laps > 0.0:
        over = float((laps > tank_laps * 1.05).mean())
        near_full = float((laps >= NEAR_FULL * tank_laps).mean())
        out.update({
            "proxy_median_fuel_laps": round(float(np.median(laps)), 2),
            "proxy_p90_fuel_laps": round(float(np.percentile(laps, 90)), 2),
            "tank_laps": round(tank_laps, 2),
            "share_over_a_tank": round(over, 3),
            "share_near_full": round(near_full, 3),
            "proxy_credible": bool(over < 0.10),
        })

    # **The refusal that supersedes the bound.** An earlier version of this
    # script asked only whether the proxy exceeded a tankful - a sanity check
    # on measurement - and passed every IMSA class at 1-3%. That is the wrong
    # question. What decides whether the split is identifiable is whether the
    # *fills differ*, and they do not: the median stop is a full tank in all
    # seven classes and the ninetieth percentile is at or past it. Nearly
    # every stop is brim-full.
    #
    # So the regressor's spread is entirely stints cut short by cautions, by
    # driver changes, or by the flag - none of which says anything about fuel
    # taken. A slope fitted on that has no leverage, and a bound extrapolated
    # from it runs far outside anything the data covers. It was reporting
    # floors of 0.84 to 0.98 on exactly this basis, and those were not
    # measurements of anything.
    if near_full > MAX_NEAR_FULL_SHARE:
        out["identified"] = False
        out["why_not"] = (
            f"{near_full:.0%} of stops are followed by a near-full tank "
            f"({NEAR_FULL:.0%} or more of {tank_laps:.0f} laps), so this race "
            f"is run on full fills and the fill size barely varies. The split "
            f"between what a stop costs to enter and what it costs to fill is "
            f"not identifiable from a sample where almost nothing but a full "
            f"fill was ever taken - not by this fit, and not by a quantile "
            f"either.")
        return out
    if spread < MIN_STINT_SPREAD_LAPS:
        out["why_not"] = (f"every stint after a stop is about the same length "
                          f"(interquartile range {spread:.1f} laps), so the "
                          f"fill size never varies and no intercept is "
                          f"identified - this race has no splashes in it")
        return out

    tyres = usable["tyres"].astype(float).to_numpy()
    out["tyre_change_share"] = round(float(tyres.mean()), 3)
    fit_tyres = 0.02 < tyres.mean() < 0.98
    if not fit_tyres:
        out["tyre_note"] = ("tyres changed at effectively every stop or none, "
                            "so the tyre term is not identified and is left "
                            "out of the fit - the same condition "
                            "`deg_identified` reports for degradation")

    drivers = usable["driver_change"].fillna(False).astype(float).to_numpy()
    fit_drivers = 0.02 < drivers.mean() < 0.98
    out["driver_change_share"] = round(float(drivers.mean()), 3)

    cost = usable["pit_time"].astype(float).to_numpy()
    columns = ([np.ones_like(laps), laps]
               + ([tyres] if fit_tyres else [])
               + ([drivers] if fit_drivers else []))
    design = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(design, cost, rcond=None)

    predicted = design @ beta
    residual = cost - predicted
    total = cost - cost.mean()
    r2 = 1.0 - float(residual @ residual) / float(total @ total)

    # The slope's standard error, from the usual (X'X)^-1 s^2. Written out
    # rather than reached for, because this one number decides whether any of
    # the rest is reportable.
    dof = max(len(cost) - design.shape[1], 1)
    sigma2 = float(residual @ residual) / dof
    covariance = np.linalg.pinv(design.T @ design) * sigma2
    slope_se = float(np.sqrt(max(covariance[1, 1], 0.0)))

    transit_s = float(beta[0])
    refuel_s_per_lap = float(beta[1])
    tyre_s = float(beta[2]) if fit_tyres else None
    driver_s = float(beta[2 + int(fit_tyres)]) if fit_drivers else None
    slope_t = refuel_s_per_lap / slope_se if slope_se > 0 else 0.0

    out.update({
        "identified": True,
        "transit_s": round(transit_s, 3),
        "refuel_s_per_green_lap": round(refuel_s_per_lap, 4),
        "tyre_s": None if tyre_s is None else round(tyre_s, 3),
        "driver_change_s": None if driver_s is None else round(driver_s, 3),
        "r2": round(r2, 4),
        "slope_t": round(slope_t, 2),
        "residual_sd_s": round(float(residual.std(ddof=len(beta))), 3),
        "pit_transit_frac": round(transit_s / median_stop_s, 4),
        "pit_tyre_frac": (None if tyre_s is None
                          else round(tyre_s / median_stop_s, 4)),
    })

    # The refusal that matters most, and the one the first version lacked. A
    # slope indistinguishable from zero means the proxy carries no
    # information, and an intercept fitted alongside it is not a measurement
    # of anything - it is the mean stop cost with a regression drawn through
    # it. That is a fraction near 1.0, which looks like a startling finding
    # and is an artefact.
    if abs(slope_t) < MIN_SLOPE_T:
        out["identified"] = False
        out["why_not"] = (
            f"the fill proxy explains nothing: the slope is {refuel_s_per_lap:+.2f} "
            f"s a lap with t = {slope_t:+.1f} and R2 = {r2:.3f}, so it cannot be "
            f"told from zero. The intercept then absorbs the whole cost of an "
            f"average stop, which is why it comes out at "
            f"{transit_s / median_stop_s:.2f} of one - that number is the mean, "
            f"not the lane")
        out["slope_ci95"] = [round(refuel_s_per_lap - 1.96 * slope_se, 3),
                             round(refuel_s_per_lap + 1.96 * slope_se, 3)]
    elif transit_s <= 0.0:
        out["identified"] = False
        out["why_not"] = (f"the fit puts the fixed part of a stop at "
                          f"{transit_s:.1f} s, which is not a pit lane. Either "
                          f"the next-stint proxy is not tracking fuel in this "
                          f"race, or something other than service dominates "
                          f"the column")
    elif refuel_s_per_lap <= 0.0:
        out["identified"] = False
        out["why_not"] = (f"the fit says a longer stint costs less time in the "
                          f"box ({refuel_s_per_lap:+.2f} s a lap), so the proxy "
                          f"is measuring something other than fuel taken")
    return out


# `bound()` used to live here. It turned a failed fit into a one-sided
# statement - "at least this much of a stop is fixed" - by multiplying the top
# of the slope's interval by a whole tank. It has now produced two numbers
# that had to be withdrawn, and the second withdrawal is the instructive one.
#
# The bound assumed that a short stint means a small fill. It does not. A
# stint ends early for three reasons - a small fill, a caution, or the flag -
# and nothing in the lap data says which. Endurance racing then supplies
# mostly full fills plus caution-truncated stints, so the short stints in the
# sample are the ambiguous ones and the long ones are all the same length.
# Extrapolating a slope fitted on that across a whole tank reaches far outside
# anything the data covers, and it produced floors of 0.84 to 0.98 that were
# not measurements of anything.
#
# What remains is the interval on the slope itself, which is reported and is
# honest because it is quoted over the range the data actually spans.
#
# The general lesson, and the one worth carrying: a refusal that offers a
# number anyway is a refusal nobody heeds.


def against_the_model(result: dict, assumed_transit: float,
                      assumed_tyre: float) -> dict:
    """What the estimate means for `stop_cost`'s shape, not just for a dial."""
    if not result.get("identified"):
        return {}
    transit = result["pit_transit_frac"]
    tyre = result.get("pit_tyre_frac")
    note = {
        "assumed_pit_transit_frac": assumed_transit,
        "assumed_pit_tyre_frac": assumed_tyre,
        "measured_pit_transit_frac": transit,
        "measured_pit_tyre_frac": tyre,
    }
    if tyre is not None:
        total = transit + tyre
        note["measured_shares_sum"] = round(total, 4)
        if total > 1.0:
            note["verdict"] = (
                "the two measured shares exceed 1.0, so no setting of these "
                "dials reproduces them: `stop_cost` anchors a full service at "
                "the measured mean and splits it, and a lane plus a tyre "
                "change already costing more than a whole stop cannot be "
                "expressed by shares of one. This is a finding about the "
                "shape of the pit model rather than about the value of a "
                "dial, and it is the one sweep_pit_transit.py warns may be "
                "waiting at the top of its range.")
        elif total > 0.9:
            note["verdict"] = (
                "the two measured shares leave under a tenth of a stop for "
                "refuelling, which is at the very edge of what the current "
                "shape can express. The sweep's 0.65 cap on "
                "`pit_transit_frac` binds before the measurement does.")
        else:
            note["verdict"] = ("both shares sit inside what `stop_cost` can "
                               "express; the disagreement is about values.")
    return note


# ----------------------------------------------------------------------
def run(laps_path: Path) -> dict:
    con = calibrate.connect(str(laps_path))
    report = {"laps": str(laps_path),
              "estimated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "method": ("least squares of green pit time on the following "
                         "green stint length and a tyre-change indicator, per "
                         "class, trimmed at 3x the class median"),
              "series": {}}

    for code in ("imsa", "wec"):
        session_id = session_for(code)
        config_path = find(f"{code}.json")
        config = RaceConfig.load(config_path) if config_path else None
        print(f"\n=== {code.upper()} — session {session_id} ===")
        report["series"][code] = {"session_id": session_id, "classes": {}}

        classes = ([c.class_name for c in config.classes] if config else [])
        if not classes:
            print("  no frozen config; nothing to scope to")
            continue

        for class_name in classes:
            cls = config.class_by_name(class_name)
            stops = stops_with_next_stint(con, session_id, class_name)
            if stops.empty:
                print(f"  {class_name:8} no stops found")
                continue

            green = stops[stops["flags"] == GREEN]["pit_time"].dropna()
            if green.empty:
                print(f"  {class_name:8} no green stops")
                continue

            ratio = (cls.fuel_per_lap_caution / cls.fuel_per_lap
                     if cls.fuel_per_lap else 1.0)
            result = estimate(stops, float(green.median()),
                              caution_fuel_ratio=ratio,
                              tank_laps=cls.fuel_laps())
            result["quantile_estimate_pit_transit_frac"] = round(
                float(green[green <= TRIM_FACTOR * green.median()].quantile(0.05))
                / float(green.median()), 4)
            result.update(against_the_model(result, cls.pit_transit_frac,
                                            cls.pit_tyre_frac))
            report["series"][code]["classes"][class_name] = result

            if result["identified"]:
                tyre = result["pit_tyre_frac"]
                print(f"  {class_name:8} transit {result['transit_s']:6.1f} s "
                      f"= {result['pit_transit_frac']:.2f} of a stop  |  "
                      f"tyres {'--' if tyre is None else f'{tyre:.2f}'}  |  "
                      f"refuel {result['refuel_s_per_green_lap']:.2f} s/lap "
                      f"(t {result['slope_t']:+.1f})  |  "
                      f"R2 {result['r2']:.2f}  ({result['n_usable']} stops)")
                print(f"  {'':8} the fifth percentile says "
                      f"{result['quantile_estimate_pit_transit_frac']:.2f}; "
                      f"the dial is set to {cls.pit_transit_frac:.2f}")
                if result.get("verdict"):
                    print(f"  {'':8} {result['verdict']}")
            else:
                print(f"  {class_name:8} not identified - {result['why_not']}")
                if result.get("share_over_a_tank") is not None:
                    print(f"  {'':8} proxy: median "
                          f"{result['proxy_median_fuel_laps']:.1f} fuel-laps, "
                          f"90th pct {result['proxy_p90_fuel_laps']:.1f}, "
                          f"against a {result['tank_laps']:.0f}-lap tank; "
                          f"{result['share_over_a_tank']:.0%} run past a "
                          f"tankful "
                          f"({'consistent with the tank' if result['proxy_credible'] else 'MORE THAN A TANK - stops are missing from the pit column'})")
                if result.get("slope_ci95"):
                    lo, hi = result["slope_ci95"]
                    print(f"  {'':8} the refuel rate is somewhere in "
                          f"[{lo:+.2f}, {hi:+.2f}] s a fuel-lap, quoted over "
                          f"the range these stops actually cover and not "
                          f"extrapolated across a tank")
                print(f"  {'':8} the dial is set to "
                      f"{cls.pit_transit_frac:.2f}; the fifth percentile says "
                      f"{result['quantile_estimate_pit_transit_frac']:.2f}, "
                      f"and nothing here supports either")
    return report


def self_check(seed: int = 0) -> int:
    """Recover a planted transit from stops built to a known truth.

    The estimator has to be checked against something before it is pointed at
    a column nobody can independently verify, and a synthetic race is the only
    thing that can do that - the same argument `tests/make_fixture.py` makes
    for the calibration as a whole. Needs no lap data and no DuckDB.

    It also settles the question this script exists for. With the truth known,
    the fifth-percentile estimate can be scored too, and it comes out high:
    the cheapest one stop in twenty is not a bare lane transit, it is a short
    fill, so the quantile prices the lane at whatever the smallest real
    service cost. That is a bias in a knowable direction, which is worth more
    than knowing only that two numbers disagree.
    """
    transit, refuel, tyre, noise = 45.0, 3.0, 30.0, 3.0
    rng = np.random.default_rng(seed)
    n = 300
    laps = rng.integers(4, 30, n).astype(float)
    tyres = (rng.random(n) < 0.5).astype(float)
    cost = (transit + refuel * laps + tyre * tyres
            + rng.normal(0.0, noise, n))

    stops = pd.DataFrame({"car": 1, "lap": np.arange(n), "pit_time": cost,
                          "flags": GREEN, "next_green_laps": laps,
                          "next_caution_laps": 0,
                          "tyres": tyres.astype(bool), "driver_change": False})
    median = float(np.median(cost))
    result = estimate(stops, median)

    print("planted:   transit {:.1f} s, refuel {:.2f} s/lap, tyres {:.1f} s"
          .format(transit, refuel, tyre))
    if not result["identified"]:
        print(f"FAILED to identify a planted truth: {result['why_not']}")
        return 1
    print("recovered: transit {:.1f} s, refuel {:.2f} s/lap, tyres {:.1f} s"
          .format(result["transit_s"], result["refuel_s_per_green_lap"],
                  result["tyre_s"]))

    truth_frac = transit / median
    kept = cost[cost <= TRIM_FACTOR * median]
    quantile_frac = float(np.quantile(kept, 0.05)) / median
    print(f"\n  true transit fraction        {truth_frac:.3f}")
    print(f"  this estimator says          {result['pit_transit_frac']:.3f}")
    print(f"  a fifth percentile would say {quantile_frac:.3f}")

    errors = [abs(result["transit_s"] - transit) > 3.0 * noise,
              abs(result["tyre_s"] - tyre) > 3.0 * noise,
              abs(result["refuel_s_per_green_lap"] - refuel) > 0.5]
    if any(errors):
        print("\n  the fit did not recover the planted values")
        return 1
    print(f"\n  Recovered within noise. The quantile is out by "
          f"{quantile_frac - truth_frac:+.3f}, and in the direction that "
          f"matters:\n  the cheapest stops in a real sample still have "
          f"service in them, so pricing\n  the lane off them overstates it.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--laps", default=str(LAPS))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--self-check", action="store_true",
                        help="recover a planted truth; needs no lap data")
    args = parser.parse_args()

    if args.self_check:
        return self_check()

    laps_path = Path(args.laps)
    if not laps_path.exists():
        print(f"no lap data at {laps_path}. This is the one thing that cannot "
              f"be regenerated from what ships in this repository.")
        return 2

    report = run(laps_path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    print("No dial has been changed. Amendment 15: an assumed dial with a "
          "measured counterpart\nthat disagrees stays at its assumed value, "
          "and the disagreement is shown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
