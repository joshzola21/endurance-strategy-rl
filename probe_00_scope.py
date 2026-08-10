"""Read-only reconnaissance for the stage 00 re-run.

Answers the questions the fix depends on, against the real `laps.csv`, and
writes a markdown report. It imports nothing from `src/endurance` except
optionally `calibrate`, and it modifies no file in the project.

    python probe_00_scope.py --laps data/raw/laps.csv --out probe_report.md

Every section is numbered to match the question it settles, so the report can
be pasted back whole.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# The two events the demonstration is about. Kept as a constant rather than a
# flag: the probe is about these races, and a general-purpose census is a
# different tool.
TARGETS = [("imsa", "Daytona"), ("wec", "Le Mans")]

# The projection worth materialising. `microsectors_json` is dropped on
# purpose - it is the widest column in the file and nothing here reads it.
COLS = """series_code, year, start_date, event, race_label, session, session_id,
          car, class, driver_id, driver_name, lap, lap_time, session_time,
          pit_time, flags, stint_number, stint_lap, est_tire_age,
          bpillar_quartile"""

GREEN = "GF"
CLEAN_QUARTILES = (1, 2)

_lines: list[str] = []


def say(text: str = "") -> None:
    """Print and record, so the terminal and the report agree."""
    print(text)
    _lines.append(text)


def table(df: pd.DataFrame, floatfmt: str = "%.3f") -> None:
    if df is None or df.empty:
        say("_(empty)_")
        return
    say("```")
    with pd.option_context("display.width", 200, "display.max_columns", 50,
                           "display.float_format", lambda v: floatfmt % v):
        say(df.to_string(index=False))
    say("```")


# ----------------------------------------------------------------------
# Section A - is `session_id` a clean edition key?
# ----------------------------------------------------------------------
def section_a(con, laps_csv: str) -> None:
    say("\n## A. Is `session_id` a clean edition key?\n")

    # One scan of the whole file, race sessions and everything else, because
    # a session id that spans two events would break the key silently.
    purity = con.execute(f"""
        SELECT COUNT(*) AS n_session_ids,
               SUM(CASE WHEN n_event > 1 THEN 1 ELSE 0 END)   AS spans_events,
               SUM(CASE WHEN n_year > 1 THEN 1 ELSE 0 END)    AS spans_years,
               SUM(CASE WHEN n_series > 1 THEN 1 ELSE 0 END)  AS spans_series,
               SUM(CASE WHEN n_session > 1 THEN 1 ELSE 0 END) AS spans_session_types
        FROM (
            SELECT session_id,
                   COUNT(DISTINCT event)       AS n_event,
                   COUNT(DISTINCT year)        AS n_year,
                   COUNT(DISTINCT series_code) AS n_series,
                   COUNT(DISTINCT session)     AS n_session
            FROM read_csv_auto('{laps_csv}', quote='"', escape='"')
            GROUP BY session_id
        ) t
    """).df()
    table(purity, "%.0f")
    say("\nAnything other than zero in the `spans_` columns means `session_id` "
        "is not a per-session key and the scope key must be composite.\n")

    multi = con.execute("""
        SELECT series_code, event, year, COUNT(DISTINCT session_id) AS race_sessions
        FROM l
        GROUP BY series_code, event, year
        HAVING COUNT(DISTINCT session_id) > 1
        ORDER BY race_sessions DESC, series_code, event, year
        LIMIT 25
    """).df()
    say("### Race sessions per (series, event, year) where there is more than one\n")
    table(multi, "%.0f")
    say("\nEmpty means `(series_code, event, year)` identifies a race on its own "
        "and either key works. Non-empty means it does not, and `session_id` is "
        "the only safe scope - **or** those rows are the same race ingested "
        "twice, which section B decides.\n")


# ----------------------------------------------------------------------
# Section B - is any race session ingested more than once?
# ----------------------------------------------------------------------
def section_b(con) -> None:
    say("\n## B. Duplicated ingestion\n")

    dupes = con.execute("""
        SELECT COUNT(*) AS race_sessions,
               SUM(CASE WHEN dup_rows > 0 THEN 1 ELSE 0 END) AS sessions_with_dupes,
               SUM(dup_rows) AS total_duplicate_car_lap_rows
        FROM (
            SELECT session_id,
                   COUNT(*) - COUNT(DISTINCT CAST(car AS VARCHAR) || ':' ||
                                             CAST(lap AS VARCHAR)) AS dup_rows
            FROM l GROUP BY session_id
        ) t
    """).df()
    say("### Repeated (car, lap) inside one race session\n")
    table(dupes, "%.0f")

    # Two sessions carrying identical (event, year, car, lap, lap_time) is the
    # signature of the same race loaded twice under different ids, which is the
    # one explanation that would let a car's summed lap time exceed 24 hours.
    cross = con.execute("""
        SELECT series_code, event, year, COUNT(*) AS shared_lap_records,
               COUNT(DISTINCT session_id) AS sessions
        FROM (
            SELECT series_code, event, year, session_id, car, lap, lap_time,
                   COUNT(*) OVER (PARTITION BY series_code, event, year, car,
                                               lap, lap_time) AS n_copies
            FROM l
        ) t
        WHERE n_copies > 1
        GROUP BY series_code, event, year
        ORDER BY shared_lap_records DESC
        LIMIT 15
    """).df()
    say("\n### Identical lap records under more than one session id\n")
    table(cross, "%.0f")
    say("\nThis is the section that decides whether scoping alone can fix the "
        "duration. If it is empty, it cannot, and the 216 hours has another "
        "cause.\n")


# ----------------------------------------------------------------------
# Section C - the editions, and where the 216 hours comes from
# ----------------------------------------------------------------------
def editions(con, series_code: str, event: str) -> pd.DataFrame:
    return con.execute(f"""
        WITH scoped AS (
            SELECT * FROM l
            WHERE series_code = '{series_code}' AND event = '{event}'
        ),
        per_session AS (
            SELECT session_id,
                   MIN(year) AS year,
                   MIN(race_label) AS race_label,
                   COUNT(DISTINCT car) AS cars,
                   MAX(lap) AS max_lap,
                   COUNT(*) AS laps_recorded,
                   (MAX(session_time) - MIN(session_time)) / 3600.0 AS elapsed_h
            FROM scoped GROUP BY session_id
        ),
        per_car AS (
            SELECT session_id, MAX(car_sum) / 3600.0 AS max_car_sum_h
            FROM (
                SELECT session_id, car, SUM(lap_time) AS car_sum
                FROM scoped GROUP BY session_id, car
            ) t GROUP BY session_id
        )
        SELECT p.session_id, p.year, p.race_label, p.cars, p.max_lap,
               p.laps_recorded, p.elapsed_h, c.max_car_sum_h
        FROM per_session p JOIN per_car c USING (session_id)
        ORDER BY p.year
    """).df()


def section_c(con) -> dict[tuple[str, str], pd.DataFrame]:
    say("\n## C. The editions, and the duration reconciliation\n")
    found = {}
    for series_code, event in TARGETS:
        ed = editions(con, series_code, event)
        found[(series_code, event)] = ed
        say(f"### {series_code} / {event}\n")
        table(ed, "%.2f")
        if not ed.empty:
            say(f"\nEditions in file: **{len(ed)}**. "
                f"Sum of per-edition max car totals: "
                f"**{ed['max_car_sum_h'].sum():.1f} h**.")
            say("`imsa.json` records 216.2 h and `wec.json` 120.4 h. If the sum "
                "above does not reach those, pooling is not the whole story and "
                "`calibrate_duration` is inflating on its own account.\n")
            say("`max_lap` against a single running - roughly 800 at Daytona, 380 "
                "at Le Mans - says whether `lap` resets per session.\n")
    return found


# ----------------------------------------------------------------------
# Section D - what does `stint_number` count?
# ----------------------------------------------------------------------
def section_d(con, session_id: int, label: str) -> None:
    say(f"\n### {label}\n")

    per_car = con.execute(f"""
        SELECT class,
               AVG(stints)  AS mean_distinct_stint_numbers,
               AVG(stops)   AS mean_pit_records,
               AVG(drivers) AS mean_distinct_drivers,
               AVG(CAST(stops AS DOUBLE) / NULLIF(stints, 0)) AS stops_per_stint
        FROM (
            SELECT class, car,
                   COUNT(DISTINCT stint_number) AS stints,
                   SUM(CASE WHEN pit_time IS NOT NULL THEN 1 ELSE 0 END) AS stops,
                   COUNT(DISTINCT driver_name) AS drivers
            FROM l WHERE session_id = {session_id}
            GROUP BY class, car
        ) t GROUP BY class ORDER BY class
    """).df()
    table(per_car, "%.2f")
    say("\n`stops_per_stint` near 1 means `stint_number` counts fuel stints, "
        "which is what `calibrate_stints` assumes. Near 3 means it counts "
        "driver stints, and the stint dial is wrong independently of scoping.\n")

    # The decisive form: does the counter step exactly where the driver changes?
    align = con.execute(f"""
        SELECT SUM(CASE WHEN sn <> psn THEN 1 ELSE 0 END) AS stint_steps,
               SUM(CASE WHEN dn <> pdn THEN 1 ELSE 0 END) AS driver_changes,
               SUM(CASE WHEN sn <> psn AND dn <> pdn THEN 1 ELSE 0 END) AS both,
               SUM(CASE WHEN sn <> psn AND ppit IS NOT NULL THEN 1 ELSE 0 END)
                   AS stint_steps_after_a_pit_lap
        FROM (
            SELECT stint_number AS sn, driver_name AS dn,
                   LAG(stint_number) OVER w AS psn,
                   LAG(driver_name)  OVER w AS pdn,
                   LAG(pit_time)     OVER w AS ppit
            FROM l WHERE session_id = {session_id}
            WINDOW w AS (PARTITION BY car ORDER BY lap)
        ) t WHERE psn IS NOT NULL
    """).df()
    say("Where the counter steps:\n")
    table(align, "%.0f")
    say("\n`both` close to `stint_steps` means the counter follows the driver. "
        "`stint_steps_after_a_pit_lap` close to `stint_steps` means it follows "
        "the stop. Whichever it tracks is what the dial is actually measuring.\n")


# ----------------------------------------------------------------------
# Sections E-H - one scoped race, in pandas
# ----------------------------------------------------------------------
def deep_dive(con, session_id: int, label: str) -> None:
    df = con.execute(f"SELECT * FROM l WHERE session_id = {session_id}").df()
    say(f"\n### {label}  (session_id {session_id}, {len(df):,} lap records)\n")

    # -- E. the flag census ------------------------------------------------
    flags = (df["flags"].astype("string").fillna("<null>")
             .value_counts().rename_axis("flag").reset_index(name="laps"))
    flags["share"] = flags["laps"] / len(df)
    flags["counted_as_caution_today"] = flags["flag"] != GREEN
    say("**E. Flag census.** `calibrate_cautions` treats everything that is not "
        "`GF` as a caution lap.\n")
    table(flags, "%.4f")

    # -- F. what `pit_time` measures --------------------------------------
    green_med = (df[df["flags"] == GREEN].groupby("car")["lap_time"]
                 .median().rename("green_med"))
    pit = df[df["pit_time"].notna()].join(green_med, on="car")
    pit = pit.assign(lap_excess=pit["lap_time"] - pit["green_med"])
    ok = pit["lap_excess"].notna() & (pit["lap_excess"] > 0)
    ratio = (pit.loc[ok, "pit_time"] / pit.loc[ok, "lap_excess"])
    say("\n**F. What `pit_time` measures.** Ratio of `pit_time` to the lap's "
        "excess over that car's median green lap.\n")
    table(pd.DataFrame([{
        "n_pit_laps": int(ok.sum()),
        "ratio_p25": ratio.quantile(0.25),
        "ratio_median": ratio.median(),
        "ratio_p75": ratio.quantile(0.75),
    }]))
    say("\nA median near 1.0 means `pit_time` is the whole time lost, lane to "
        "lane, and the low-quantile route to `pit_transit_frac` is coherent. "
        "Much below 1.0 means it is stationary time only, and a low quantile of "
        "it is not the transit delta.\n")

    # -- G. the pit distribution, green stops only, as calibrate sees it ---
    say("**G. The pit column on green stops, per class.**\n")
    gp = df[(df["pit_time"].notna()) & (df["flags"] == GREEN)]
    rows = []
    for cls, g in gp.groupby("class"):
        v = g["pit_time"]
        rows.append({"class": cls, "n": len(v), "mean": v.mean(), "sd": v.std(),
                     "sd_over_mean": v.std() / v.mean() if v.mean() else np.nan,
                     "p05": v.quantile(0.05), "p50": v.median(),
                     "p95": v.quantile(0.95), "max": v.max()})
    table(pd.DataFrame(rows).sort_values("n", ascending=False), "%.2f")
    say("\nGate condition two requires `sd` no greater than `mean`. `p05` is the "
        "candidate for `pit_transit_frac`, and it is only meaningful if F came "
        "back near 1.0.\n")

    # -- H. degradation, three ways ---------------------------------------
    say("**H. Degradation.** The frame `calibrate_pace` uses, then the same "
        "slope with a per-car-per-stint intercept removed.\n")
    rows = []
    clean_all = df[(df["flags"] == GREEN)
                   & (df["bpillar_quartile"].isin(CLEAN_QUARTILES))
                   & (df["pit_time"].isna())]
    for cls, g in clean_all.groupby("class"):
        if len(g) < 200:
            continue
        g = g.copy()
        g["delta"] = g["lap_time"] - g.groupby("lap")["lap_time"].transform("median")
        pooled = np.polyfit(g["est_tire_age"], g["delta"], 1)[0]

        # Within stint: fuel load is a per-car, per-stint effect and it falls as
        # tyre age rises, so leaving it in biases the slope downwards. Removing
        # a per-(car, stint) intercept is what separates the two.
        key = ["car", "stint_number"]
        d = g["delta"] - g.groupby(key)["delta"].transform("mean")
        a = g["est_tire_age"] - g.groupby(key)["est_tire_age"].transform("mean")
        within = np.polyfit(a, d, 1)[0] if a.std() > 0 else np.nan

        # And without the clean-quartile filter, which truncates slow laps and
        # bites hardest where the mean is highest - late in a stint.
        h = df[(df["flags"] == GREEN) & (df["pit_time"].isna())
               & (df["class"] == cls)].copy()
        h["delta"] = h["lap_time"] - h.groupby("lap")["lap_time"].transform("median")
        unfiltered = np.polyfit(h["est_tire_age"], h["delta"], 1)[0]

        rows.append({"class": cls, "n_clean": len(g),
                     "slope_as_calibrated": pooled,
                     "slope_within_stint": within,
                     "slope_no_quartile_filter": unfiltered,
                     "max_tire_age": int(g["est_tire_age"].max())})
    table(pd.DataFrame(rows), "%.5f")
    say("\nIf `slope_as_calibrated` is negative and `slope_within_stint` is "
        "positive, the sign problem is fuel load leaking through the "
        "field-relative frame and the scoping fix will not touch it. If both are "
        "negative, condition four has a different cause again.\n")

    # -- the stint dial, computed the way calibrate does, on one race ------
    st = (df[df["flags"] == GREEN].groupby(["class", "car", "stint_number"])
          .size().rename("green_laps").reset_index())
    st = st[st["green_laps"] >= 3]
    say("\n**Stint length on one scoped race**, grouped exactly as "
        "`calibrate_stints` groups it.\n")
    table(st.groupby("class")["green_laps"]
          .agg(n_stints="size", mean="mean", q75=lambda v: v.quantile(0.75),
               max="max").reset_index(), "%.1f")

    # -- the observed classification, for gate condition two ---------------
    cls_tbl = (df.groupby(["class", "car"])
               .agg(laps=("lap", "max"),
                    stops=("pit_time", lambda v: int(v.notna().sum())))
               .reset_index())
    winners = (cls_tbl.sort_values("laps", ascending=False)
               .groupby("class").head(1)
               .assign(green_laps_per_stop=lambda d: d["laps"] / d["stops"]))
    say("\n**The observed classification** - what gate condition two compares "
        "against.\n")
    table(winners, "%.1f")


# ----------------------------------------------------------------------
# Section I - the class mix by edition, for the edition decision
# ----------------------------------------------------------------------
def section_i(con) -> None:
    say("\n## I. Cars per class per edition\n")
    for series_code, event in TARGETS:
        mix = con.execute(f"""
            SELECT year, class, COUNT(DISTINCT car) AS cars
            FROM l WHERE series_code = '{series_code}' AND event = '{event}'
            GROUP BY year, class ORDER BY year, cars DESC
        """).df()
        if mix.empty:
            continue
        wide = mix.pivot(index="year", columns="class", values="cars").fillna(0)
        wide["TOTAL"] = wide.sum(axis=1)
        say(f"### {series_code} / {event}\n")
        table(wide.reset_index(), "%.0f")
    say("\nThis is the table the edition decision should be made on: a grid of "
        "three classes is a thinner demonstration than one of five, and "
        "recency trades against that.\n")


# ----------------------------------------------------------------------
# Section J - is the caution dial nondeterministic?
# ----------------------------------------------------------------------
def section_j(con, src: Path | None) -> None:
    say("\n## J. Is `caution_mean_dur_s` reproducible?\n")
    if src is None or not (src / "endurance" / "calibrate.py").exists():
        say("_Skipped: pass `--src path/to/src` to run this._\n")
        return
    import sys
    sys.path.insert(0, str(src))
    from endurance import calibrate  # noqa: E402

    rows = []
    for series_code, event in TARGETS:
        for i in range(5):
            c = calibrate.calibrate_cautions(con, series_code, f"%{event}%")
            rows.append({"series": series_code, "call": i + 1,
                         "caution_rate": c["caution_rate"],
                         "mean_dur_s": c["caution_mean_dur_s"],
                         "episodes": c["n_caution_episodes"],
                         "reference_car": c["reference_car"]})
    out = pd.DataFrame(rows)
    table(out, "%.6f")
    spread = out.groupby("series")["mean_dur_s"].agg(["min", "max", "std"])
    say("\nSpread across five identical calls:\n")
    table(spread.reset_index(), "%.3f")
    say("\nFive identical calls returning five different episode lengths "
        "confirms the tie-ordering diagnosis, and explains the seven different "
        "values already sitting in `imsa.json`. A flat column means "
        "`imsa.json` was built by a different `calibrate.py` than the one in "
        "the tree, which is worth knowing before anything else.\n")


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", default="data/raw/laps.csv")
    ap.add_argument("--out", default="probe_report.md")
    ap.add_argument("--src", default="src",
                    help="path to src/, to run section J")
    args = ap.parse_args()

    import duckdb
    con = duckdb.connect()

    laps_csv = args.laps
    say(f"# Stage 00 scope probe\n\n- file: `{laps_csv}`")

    # Race sessions only, and only the columns anything here reads. Everything
    # after this point is one in-memory table rather than a rescan of 582 MB.
    con.execute(f"""
        CREATE TABLE l AS
        SELECT {COLS}
        FROM read_csv_auto('{laps_csv}', quote='"', escape='"')
        WHERE session = 'race'
    """)
    n = con.execute("SELECT COUNT(*) FROM l").fetchone()[0]
    say(f"- race-session lap records: {n:,}\n")

    # `calibrate.py` (used by section J) queries a view called `laps`; alias
    # it to the in-memory table rather than rescanning the CSV.
    con.execute("CREATE VIEW laps AS SELECT * FROM l")

    section_a(con, laps_csv)
    section_b(con)
    found = section_c(con)
    section_i(con)

    say("\n## D. What `stint_number` counts\n")
    picks: list[tuple[int, str]] = []
    for (series_code, event), ed in found.items():
        if ed.empty:
            continue
        # The latest edition, and the one before it - the second is the
        # falsifier's partner and the agreement check.
        for _, row in ed.tail(2).iterrows():
            picks.append((int(row["session_id"]),
                          f"{series_code} / {event} {int(row['year'])}"))
    for sid, label in picks:
        section_d(con, sid, label)

    say("\n## E-H. One scoped race at a time\n")
    for sid, label in picks:
        deep_dive(con, sid, label)

    section_j(con, Path(args.src) if args.src else None)

    Path(args.out).write_text("\n".join(_lines))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
