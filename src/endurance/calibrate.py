"""Turn the raw lap data into the dials the engine runs on.

This is notebook 00's recon, promoted from a set of exploratory queries into
one reusable step. Every number it produces traces back to a query in this
file, so the provenance question ("where did that figure come from?") always
has an answer.

The connection is injected rather than created here, which keeps this module
testable without a live DuckDB and lets a notebook reuse a connection it has
already set up.

**Scoped to one race session.** The previous version selected with an ILIKE
pattern on `event`, which carries no edition, so every dial that needed one
running of the race was computed across all of them: durations added, counts
summed, stints concatenated. Selection is now by `session_id`, which the data
recon established is unique to a session and never spans an event, a year, a
series or a session type. The pattern path is gone rather than deprecated - a
reachable pooling path is how the fault survived two stages.

**`car` is read as text, not as an integer.** Both series field entries whose
numbers differ only by a leading zero, and the pair is sometimes in the same
class: `#7` Toyota and `#007` Aston in Hypercar at Le Mans, `#21` and `#021`
in GTD at Daytona. Parsed as an integer they collapse onto one identifier and
two cars' laps merge, which doubled every duration and undercounted every
grid. `connect` forces the column to VARCHAR; nothing here may parse it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .params import ClassDials, RaceConfig

GREEN = "GF"

# What counts as a caution. The previous version took "anything that is not
# GF", which swept in the chequered-flag lap and, elsewhere in the file, a
# literal 'nan'. Small in the races calibrated here - a tenth of a per cent -
# but wrong, and wrong in a way that inflates rather than cancels.
CAUTION_FLAGS = ("FCY", "SF", "RF")

CLEAN_QUARTILES = (1, 2)

# A stop longer than this multiple of the class median is a repair, a penalty
# or a red-flag hold rather than service. Within one properly scoped race the
# pit column still runs to six thousand seconds against a median of eighty,
# so a mean and a standard deviation over it describe nothing. 3.0 is a
# modelling choice and the notebook shows the dials across 2.0 to 5.0.
PIT_TRIM_FACTOR = 3.0

# Above this correlation between tyre age and laps-since-fuel, the two cannot
# be told apart and the degradation slope is a net within-stint trend rather
# than a tyre effect. Reported, never silently absorbed.
DEG_COLLINEAR_ABOVE = 0.90


def connect(laps_csv: str, drivers_csv: str | None = None):
    """Open a DuckDB connection with the lap data as a view called `laps`."""
    import duckdb

    # quote is forced rather than left to auto-detection: the sniffer samples
    # a prefix of the file and, on this dataset, decides there's no quote
    # character - then splits team names like "Lone Star Racing, LLC" on
    # their internal comma, producing a ragged row further down the file.
    #
    # `car` is forced to VARCHAR for the leading-zero reason in the module
    # docstring. This is not cosmetic: without it, `#7` and `#007` are one car.
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW laps AS SELECT * FROM read_csv_auto('{laps_csv}', "
        f"quote='\"', escape='\"', types={{'car': 'VARCHAR'}})")
    if drivers_csv:
        con.execute(f"CREATE VIEW drivers AS SELECT * FROM read_csv_auto('{drivers_csv}', quote='\"', escape='\"')")
    return con


# ----------------------------------------------------------------------
# Scope
# ----------------------------------------------------------------------
def _scope(session_ids: int | Sequence[int]) -> str:
    """The SQL fragment selecting a race.

    A sequence is accepted only so the verification gate's falsifier can widen
    the scope deliberately and require the gate to fail. Nothing in normal use
    passes more than one id.
    """
    if isinstance(session_ids, (int, np.integer)):
        ids = [int(session_ids)]
    else:
        ids = [int(s) for s in session_ids]
    if not ids:
        raise ValueError("no session ids given")
    return "session_id IN (" + ", ".join(str(i) for i in ids) + ")"


def list_races(con, series_code: str, event: str | None = None) -> pd.DataFrame:
    """Every race session for a series, one row each, oldest first.

    The notebook uses this twice: to pick an edition with the numbers in
    front of it, and to show how far a dial moves between adjacent runnings.
    """
    where = f"session = 'race' AND series_code = '{series_code}'"
    if event is not None:
        where += f" AND event ILIKE '{event}'"
    return con.execute(f"""
        SELECT session_id,
               MIN(year)  AS year,
               MIN(event) AS event,
               COUNT(DISTINCT car) AS cars,
               COUNT(*) AS laps_recorded,
               MAX(session_time) - MIN(session_time) AS duration_s
        FROM laps
        WHERE {where}
        GROUP BY session_id
        ORDER BY year, session_id
    """).df()


def find_race(con, series_code: str, event: str, year: int | None = None) -> dict:
    """Resolve one race session, latest by default.

    Raises rather than guessing when the answer is not unique: two race
    sessions under one event and year is a real pattern in this file - the
    Asian Le Mans double-headers - and picking one silently is the failure
    this whole re-run exists to remove.
    """
    races = list_races(con, series_code, event)
    if year is not None:
        races = races[races["year"] == year]
    if races.empty:
        raise ValueError(f"no race session for {series_code}/{event}"
                         + (f"/{year}" if year else ""))
    if year is not None and len(races) > 1:
        raise ValueError(
            f"{len(races)} race sessions for {series_code}/{event}/{year}: "
            f"{races['session_id'].tolist()} - name one explicitly")
    row = races.iloc[-1]
    return {"session_id": int(row["session_id"]), "year": int(row["year"]),
            "event": str(row["event"]), "series_code": series_code,
            "label": f"{row['event']} {int(row['year'])} "
                     f"({series_code} session {int(row['session_id'])})"}


def describe_race(con, session_ids: int | Sequence[int]) -> str:
    """A one-line provenance label, stored on every ClassDials."""
    d = con.execute(f"""
        SELECT MIN(series_code) AS series_code, MIN(event) AS event,
               MIN(year) AS year, COUNT(DISTINCT session_id) AS n_sessions
        FROM laps WHERE session = 'race' AND {_scope(session_ids)}
    """).df().iloc[0]
    if int(d["n_sessions"]) != 1:
        return (f"{d['event']} - {int(d['n_sessions'])} sessions POOLED "
                f"({d['series_code']})")
    sid = session_ids if isinstance(session_ids, (int, np.integer)) \
        else list(session_ids)[0]
    return (f"{d['event']} {int(d['year'])} "
            f"({d['series_code']} session {int(sid)})")


# ----------------------------------------------------------------------
# Fuel stints, derived from the pit records
# ----------------------------------------------------------------------
def _fuel_stints(df: pd.DataFrame) -> pd.DataFrame:
    """Label every lap with the fuel stint it belongs to and its position in it.

    `stint_number` in the raw file is **not** a fuel stint. The recon showed
    every step of that counter coinciding with a driver change and none with a
    stop: two to three fuel stints per step, which is why the old dial reported
    fifty-eight-lap green stints where the winner averaged twenty-three.

    A stint is therefore laps between consecutive pit records. The stop is
    counted as the last lap of the stint it ends, so `fuel_lap` counts laps
    completed on the tank.
    """
    df = df.sort_values(["session_id", "car", "lap"]).copy()
    stopped = df["pit_time"].notna().astype(int)
    key = [df["session_id"], df["car"]]
    # Stops strictly before this lap: the pit lap stays in its own stint.
    df["fuel_stint"] = stopped.groupby(key).cumsum() - stopped
    df["fuel_lap"] = df.groupby(["session_id", "car", "fuel_stint"]).cumcount() + 1
    return df


def _stint_frame(con, session_ids, class_name: str) -> pd.DataFrame:
    return con.execute(f"""
        SELECT session_id, car, lap, flags, pit_time
        FROM laps
        WHERE session = 'race' AND class = '{class_name}' AND {_scope(session_ids)}
        ORDER BY session_id, car, lap
    """).df()


# ----------------------------------------------------------------------
# Dial 1 - pace and degradation
# ----------------------------------------------------------------------
def calibrate_pace(con, session_ids, class_name: str) -> dict:
    """Base pace, degradation slope and spread, in the field-relative frame.

    Lap time is taken relative to that lap's class median, which removes the
    part of the variation that hits every car equally - track evolution, a
    slow zone, the weather. Laps with a non-null `pit_time` are dropped: an in
    or out lap can carry a clean quartile flag while being inflated by the stop.

    **Degradation is fitted jointly against tyre age and fuel load.** Within a
    stint the car gets lighter as the tyres get older, and the two effects have
    opposite signs, so a slope on tyre age alone is their sum. The
    field-relative frame does not save it: cars in a class stagger their stops,
    so at a given lap number they are at different points of the tank and fuel
    is not common-mode. Where a class changes tyres at every stop the two
    regressors are the same number and nothing can separate them; that case is
    reported through `deg_identified` rather than resolved.
    """
    raw = con.execute(f"""
        SELECT session_id, lap, car, est_tire_age, lap_time, flags, pit_time,
               bpillar_quartile
        FROM laps
        WHERE session = 'race' AND class = '{class_name}' AND {_scope(session_ids)}
        ORDER BY session_id, car, lap
    """).df()
    if raw.empty:
        raise ValueError(f"no laps for {class_name} in this scope")

    raw = _fuel_stints(raw)
    clean = raw[(raw["flags"] == GREEN)
                & (raw["bpillar_quartile"].isin(CLEAN_QUARTILES))
                & (raw["pit_time"].isna())].copy()
    if clean.empty:
        raise ValueError(f"no clean green laps for {class_name}")

    field_med = clean.groupby(["session_id", "lap"])["lap_time"].transform("median")
    clean["delta"] = clean["lap_time"] - field_med

    age = pd.to_numeric(clean["est_tire_age"], errors="coerce")
    fuel = clean["fuel_lap"].astype(float)
    ok = age.notna() & clean["delta"].notna()
    age, fuel, delta = age[ok].to_numpy(float), fuel[ok].to_numpy(float), \
        clean.loc[ok, "delta"].to_numpy(float)

    simple = float(np.polyfit(age, delta, 1)[0]) if len(age) > 2 else 0.0

    corr = float(np.corrcoef(age, fuel)[0, 1]) if len(age) > 2 and \
        age.std() > 0 and fuel.std() > 0 else 1.0
    identified = abs(corr) < DEG_COLLINEAR_ABOVE

    if identified:
        X = np.column_stack([age, fuel, np.ones_like(age)])
        beta = np.linalg.lstsq(X, delta, rcond=None)[0]
        slope, fuel_slope = float(beta[0]), float(beta[1])
    else:
        # Not separable. The dial then holds the net within-stint trend, which
        # is what the engine will reproduce, and `deg_identified` says so.
        slope, fuel_slope = simple, float("nan")

    per_car = clean.groupby("car")["lap_time"].median()

    return {
        "base_pace_s": float(clean["lap_time"].median()),
        "deg_slope_s_per_lap": slope,
        "deg_slope_simple_s_per_lap": simple,
        "fuel_slope_s_per_lap": fuel_slope,
        "deg_identified": bool(identified),
        "age_fuel_corr": corr,
        "pace_spread_s": float(per_car.std()) if len(per_car) > 1 else 0.0,
        "lap_noise_s": float(clean["delta"].std()),
        "n_laps_observed": int(len(clean)),
    }


# ----------------------------------------------------------------------
# Dial 2 - cautions
# ----------------------------------------------------------------------
def calibrate_cautions(con, session_ids,
                       base_pace_s: float | None = None,
                       legacy: bool = False,
                       red_flag_lap_factor: float = 5.0) -> dict:
    """Caution share and typical caution length, both in seconds of race time.

    A caution is a property of the race, not of a car, so this uses a single
    reference car - the one with the most laps - rather than pooling the field,
    which would count the same caution once per car. With `car` read as text
    that reference is one car; parsed as an integer it was sometimes two cars'
    laps interleaved, and the run-length segmentation below was then meaningless.

    **Both quantities are measured in seconds, not laps.** The engine's caution
    timeline lives in race time, and a caution lap takes appreciably longer than
    a green one, so a share of laps and a share of time are different numbers.

    Red-flag periods are a problem for a lap-time-based measure: the race clock
    runs while the cars sit still, so a single lap record can absorb hours. Laps
    longer than `red_flag_lap_factor` times the reference car's median are
    excluded and reported separately rather than silently averaged in.
    """
    ref_car = con.execute(f"""
        SELECT car FROM laps
        WHERE session = 'race' AND {_scope(session_ids)}
        GROUP BY car ORDER BY COUNT(*) DESC, car ASC LIMIT 1
    """).fetchone()[0]

    seq = con.execute(f"""
        SELECT session_id, lap, flags AS flag, lap_time, pit_time
        FROM laps
        WHERE session = 'race' AND {_scope(session_ids)} AND car = '{ref_car}'
        ORDER BY session_id, lap
    """).df()

    census = (seq["flag"].astype("string").fillna("<null>")
              .value_counts().to_dict())

    # Green and caution are named sets. Anything else - the chequered lap, a
    # null - leaves both the numerator and the denominator.
    is_caution = seq["flag"].isin(CAUTION_FLAGS)
    is_green = seq["flag"] == GREEN
    seq = seq[is_caution | is_green].copy()
    seq["state"] = seq["flag"].isin(CAUTION_FLAGS).astype(int)
    seq["grp"] = (seq["state"].diff().fillna(0) != 0).cumsum()

    if legacy:
        if base_pace_s is None:
            raise ValueError("legacy caution calibration needs base_pace_s")
        ep_lens = seq[seq["state"] == 1].groupby("grp").size()
        return {
            "caution_rate": float(seq["state"].mean()),
            "caution_mean_dur_s": (float(ep_lens.mean()) if len(ep_lens) else 0.0) * base_pace_s,
            "n_caution_episodes": int(len(ep_lens)),
            "reference_car": ref_car,
            "flag_census": census,
            "units": "legacy (lap share, episode laps x green pace)",
        }

    med = float(seq["lap_time"].median())
    stopped = seq["lap_time"] > red_flag_lap_factor * med

    usable = seq[~stopped]
    total_s = float(usable["lap_time"].sum())
    caution_s = float(usable.loc[usable["state"] == 1, "lap_time"].sum())
    caution_rate = caution_s / total_s if total_s else 0.0

    ep_secs = usable[usable["state"] == 1].groupby("grp")["lap_time"].sum()
    ep_laps = usable[usable["state"] == 1].groupby("grp").size()

    # The caution pace multiplier is an assumed dial. It is also directly
    # observable here, so the observation is returned as a check on the
    # assumption rather than quietly replacing it.
    clean = usable[usable["pit_time"].isna()]
    green_mean = clean.loc[clean["state"] == 0, "lap_time"].mean()
    caut_mean = clean.loc[clean["state"] == 1, "lap_time"].mean()
    observed_mult = (float(caut_mean / green_mean)
                     if green_mean and pd.notna(caut_mean) else None)

    return {
        "caution_rate": caution_rate,
        "caution_mean_dur_s": float(ep_secs.mean()) if len(ep_secs) else 0.0,
        "caution_mean_dur_laps": float(ep_laps.mean()) if len(ep_laps) else 0.0,
        "n_caution_episodes": int(len(ep_secs)),
        "caution_lap_share": float(usable["state"].mean()),
        "observed_caution_multiplier": observed_mult,
        "n_red_flag_laps": int(stopped.sum()),
        "red_flag_s": float(seq.loc[stopped, "lap_time"].sum()),
        "reference_car": ref_car,
        "flag_census": census,
        "units": "seconds of race time",
    }


# ----------------------------------------------------------------------
# Dial 3 - stint length, expressed as fuel
# ----------------------------------------------------------------------
def calibrate_stints(con, session_ids, class_name: str) -> dict:
    """Green stint length in laps, converted into a normalised fuel burn.

    Endurance stints are fuel-limited far more often than tyre-limited, so the
    engine triggers stops on fuel rather than on a lap count. This schema has
    no fuel column, so a full tank is defined as 1.0 and `fuel_per_lap` is set
    so that a tank lasts exactly the observed stint.

    Stints come from the pit records, not from `stint_number` - see
    `_fuel_stints`. Green laps only: a stint interrupted by a long caution
    covers fewer green laps than its fuel allowed.

    A car's last stint is dropped. It ends at the chequered flag rather than at
    the bottom of the tank, so it is truncated by an amount nobody can recover.
    """
    df = _stint_frame(con, session_ids, class_name)
    if df.empty:
        raise ValueError(f"no laps for {class_name} in this scope")
    df = _fuel_stints(df)

    last = df.groupby(["session_id", "car"])["fuel_stint"].transform("max")
    df = df[df["fuel_stint"] < last]

    stints = (df[df["flags"] == GREEN]
              .groupby(["session_id", "car", "fuel_stint"])
              .size().rename("green_laps").reset_index())
    stints = stints[stints["green_laps"] >= 3]
    if stints.empty:
        raise ValueError(f"no complete stints for {class_name}")

    # The upper quartile is a better estimate of tank range than the mean:
    # many stints end early for reasons other than running dry - a caution, a
    # driver change, damage - but very few run longer than the fuel allows.
    green_stint_laps = float(stints["green_laps"].quantile(0.75))

    return {
        "green_stint_laps": green_stint_laps,
        "fuel_per_lap": 1.0 / green_stint_laps,
        "fuel_per_lap_caution": 0.6 / green_stint_laps,
        "mean_stint_laps": float(stints["green_laps"].mean()),
        "median_stint_laps": float(stints["green_laps"].median()),
        "n_stints": int(len(stints)),
    }


# ----------------------------------------------------------------------
# Dial 4 - pit cost
# ----------------------------------------------------------------------
def calibrate_pit(con, session_ids, class_name: str,
                  trim_factor: float = PIT_TRIM_FACTOR) -> dict:
    """Green-flag pit cost, robust to the repairs sharing the column.

    Green stops only. A stop taken under caution is a different, cheaper thing,
    and how much cheaper is not identifiable from lap data - that is the
    `pit_caution_discount` assumption, swept rather than fitted.

    **`pit_time_mean_s` holds a median, and the field name is unchanged.**
    Within one properly scoped race the column still carries hour-long repairs
    beside eighty-second stops, so an arithmetic mean sat two to four times its
    own median and the standard deviation sat above the mean. The dial is the
    median of green stops; the spread is the standard deviation of the sample
    trimmed at `trim_factor` times that median.

    `pit_lane_transit_s` is the fifth percentile of the trimmed sample - a stop
    with almost no service in it, which is the lane transit delta. It is
    returned for the `pit_transit_frac` measurement and is not itself a dial.
    """
    v = con.execute(f"""
        SELECT pit_time FROM laps
        WHERE session = 'race' AND class = '{class_name}'
          AND {_scope(session_ids)}
          AND pit_time IS NOT NULL AND flags = '{GREEN}'
    """).df()["pit_time"].astype(float).dropna()
    if v.empty:
        raise ValueError(f"no green pit stops for {class_name}")

    med = float(v.median())
    kept = v[v <= trim_factor * med]
    if len(kept) < 3:
        kept = v

    return {
        "pit_time_mean_s": med,
        "pit_time_std_s": float(kept.std()) if len(kept) > 1 else 2.0,
        "pit_lane_transit_s": float(kept.quantile(0.05)),
        "pit_time_raw_mean_s": float(v.mean()),
        "n_pit_stops": int(len(v)),
        "n_pit_stops_trimmed_out": int(len(v) - len(kept)),
        "trim_factor": trim_factor,
    }


# ----------------------------------------------------------------------
# Dial 5 - traffic
# ----------------------------------------------------------------------
def calibrate_traffic(con, session_ids) -> pd.DataFrame:
    """Cars per class in this race - the raw material for traffic."""
    return con.execute(f"""
        SELECT class, COUNT(DISTINCT car) AS cars
        FROM laps
        WHERE session = 'race' AND {_scope(session_ids)}
        GROUP BY class
        ORDER BY cars DESC
    """).df()


# ----------------------------------------------------------------------
# Race length
# ----------------------------------------------------------------------
def calibrate_duration(con, session_ids) -> float:
    """Race length in seconds, as elapsed session time.

    Was the longest car's summed lap time, which is the same number only when
    one car number means one car. It did not: a collided number summed two
    cars' races and reported forty-eight hours for a twenty-four-hour event.

    Summing per session rather than taking one span across the selection is
    deliberate. It makes a widened scope report the racing it actually
    contains, so the gate's falsifier fails on duration as it should.
    """
    row = con.execute(f"""
        SELECT SUM(span) AS duration_s FROM (
            SELECT session_id, MAX(session_time) - MIN(session_time) AS span
            FROM laps
            WHERE session = 'race' AND {_scope(session_ids)}
            GROUP BY session_id
        ) t
    """).fetchone()
    return float(row[0])


# ----------------------------------------------------------------------
# Put it together
# ----------------------------------------------------------------------
def caution_report(con, session_ids, base_pace_s: float) -> pd.DataFrame:
    """The old and new caution calibrations side by side, for the record."""
    old = calibrate_cautions(con, session_ids, base_pace_s, legacy=True)
    new = calibrate_cautions(con, session_ids)
    return pd.DataFrame([
        {"quantity": "caution share", "legacy (laps)": old["caution_rate"],
         "measured (seconds)": new["caution_rate"]},
        {"quantity": "mean episode, s", "legacy (laps)": old["caution_mean_dur_s"],
         "measured (seconds)": new["caution_mean_dur_s"]},
        {"quantity": "episodes", "legacy (laps)": old["n_caution_episodes"],
         "measured (seconds)": new["n_caution_episodes"]},
        {"quantity": "caution pace multiplier (assumed 1.6)",
         "legacy (laps)": None,
         "measured (seconds)": new["observed_caution_multiplier"]},
        {"quantity": "red-flag laps excluded", "legacy (laps)": 0,
         "measured (seconds)": new["n_red_flag_laps"]},
    ])


def build_race_config(con, series_code: str, session_ids, name: str | None = None,
                      classes: list[str] | None = None,
                      min_cars: int = 3,
                      legacy_cautions: bool = False,
                      trim_factor: float = PIT_TRIM_FACTOR) -> RaceConfig:
    """Calibrate every class of one race session and return a runnable config.

    `session_ids` is one id. A sequence is accepted only for the gate's
    falsifier, which widens the scope on purpose and requires the gate to fail.
    """
    source = describe_race(con, session_ids)
    traffic = calibrate_traffic(con, session_ids)
    if classes is None:
        classes = traffic[traffic["cars"] >= min_cars]["class"].tolist()

    duration_s = calibrate_duration(con, session_ids)
    cautions = calibrate_cautions(con, session_ids, legacy=legacy_cautions) \
        if not legacy_cautions else None

    class_dials: list[ClassDials] = []
    for class_name in classes:
        pace = calibrate_pace(con, session_ids, class_name)
        # Cautions are a property of the race, so they are calibrated once and
        # shared. The old code called this inside the class loop, which is why
        # seven classes of one race carried seven different episode lengths.
        c = cautions if cautions is not None else calibrate_cautions(
            con, session_ids, pace["base_pace_s"], legacy=True)
        stints = calibrate_stints(con, session_ids, class_name)
        pit = calibrate_pit(con, session_ids, class_name, trim_factor)
        n_cars = int(traffic.loc[traffic["class"] == class_name, "cars"].iloc[0])

        class_dials.append(ClassDials(
            series_code=series_code,
            class_name=class_name,
            base_pace_s=pace["base_pace_s"],
            deg_slope_s_per_lap=pace["deg_slope_s_per_lap"],
            pace_spread_s=pace["pace_spread_s"],
            lap_noise_s=pace["lap_noise_s"],
            caution_rate=c["caution_rate"],
            caution_mean_dur_s=c["caution_mean_dur_s"],
            green_stint_laps=stints["green_stint_laps"],
            fuel_per_lap=stints["fuel_per_lap"],
            fuel_per_lap_caution=stints["fuel_per_lap_caution"],
            # Tyres are assumed to last twice a fuel stint unless told
            # otherwise, which makes fuel the binding constraint - the
            # endurance norm.
            tyre_life_laps=2.0 * stints["green_stint_laps"],
            pit_time_mean_s=pit["pit_time_mean_s"],
            pit_time_std_s=pit["pit_time_std_s"],
            n_cars=n_cars,
            source_event=source,
            n_laps_observed=pace["n_laps_observed"],
        ))

    return RaceConfig(name=name or source, series_code=series_code,
                      duration_s=duration_s, classes=class_dials)


def dials_table(config: RaceConfig) -> pd.DataFrame:
    """The five dials as one readable table - notebook 00's shopping list, filled in."""
    rows = []
    for c in config.classes:
        rows.append({
            "series": c.series_code,
            "class": c.class_name,
            "base_pace_s": round(c.base_pace_s, 2),
            "deg_s_per_lap": round(c.deg_slope_s_per_lap, 4),
            "caution_rate": round(c.caution_rate, 3),
            "caution_dur_s": round(c.caution_mean_dur_s, 0),
            "stint_laps": round(c.green_stint_laps, 1),
            "pit_s": round(c.pit_time_mean_s, 1),
            "pit_sd_s": round(c.pit_time_std_s, 1),
            "cars": c.n_cars,
        })
    return pd.DataFrame(rows)


def degradation_table(con, session_ids, classes: Sequence[str]) -> pd.DataFrame:
    """Every class's degradation fit, with the identifiability verdict beside it.

    Separate from `dials_table` because the interesting column is not the slope
    but whether the slope means what its name says.
    """
    rows = []
    for class_name in classes:
        p = calibrate_pace(con, session_ids, class_name)
        rows.append({
            "class": class_name,
            "deg_s_per_lap": round(p["deg_slope_s_per_lap"], 5),
            "simple_slope": round(p["deg_slope_simple_s_per_lap"], 5),
            "fuel_s_per_lap": (round(p["fuel_slope_s_per_lap"], 5)
                               if p["deg_identified"] else None),
            "age_fuel_corr": round(p["age_fuel_corr"], 3),
            "identified": p["deg_identified"],
            "n_laps": p["n_laps_observed"],
        })
    return pd.DataFrame(rows)
