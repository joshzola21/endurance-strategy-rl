"""Profile the real `laps.csv` without moving it anywhere.

    python scripts/profile_laps.py
    python scripts/profile_laps.py --laps path/to/laps.csv

Writes `data/processed/laps_profile.md` - a few kilobytes of schema, counts
and cardinalities, which is what the stage 00 re-run needs to start. The
582 MB file stays where it is; DuckDB reads it off disk and only aggregates
come out.

What this is looking for
------------------------
One question decides the shape of the whole thread: **does this file contain
anything that identifies which running of an event a lap belongs to?** The
calibrated dials describe 9.009 runnings of the Daytona 24 and 5.015 of Le
Mans, which says the scoping predicate selects every edition at once. If
there is a year, a date, a session id or an event id in here, the fix is a
query. If there is not, 00 has a data problem before it has a query problem.

Everything else below is corroboration: whether lap and stint numbering
resets, how many distinct cars an event pattern selects, and whether the pit
column looks like service time or like something else.

Nothing here writes to the raw file and nothing here is destructive.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def find_project_root(marker: str = "src/endurance") -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"could not find {marker!r} at or above {here.parent}")


ROOT = find_project_root()
DEFAULT_LAPS = ROOT / "data" / "raw" / "laps.csv"
OUT = ROOT / "data" / "processed" / "laps_profile.md"

# Columns worth listing values for. Anything with more distinct values than
# this is summarised by its cardinality instead, so the profile stays small.
MAX_VALUES = 40

# Names that would answer the question outright.
DISCRIMINATOR_HINTS = ("year", "date", "time_utc", "timestamp", "event_id",
                       "session_id", "round", "edition", "meeting", "start")


def q(con, sql: str):
    return con.execute(sql).df()


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--laps", default=str(DEFAULT_LAPS))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv[1:])

    laps = Path(args.laps)
    if not laps.exists():
        raise SystemExit(f"no laps file at {laps}")

    import duckdb

    con = duckdb.connect()
    src = f"read_csv_auto('{laps.as_posix()}', quote='\"', escape='\"')"
    con.execute(f"CREATE VIEW laps AS SELECT * FROM {src}")

    lines: list[str] = []
    add = lines.append
    add(f"# `laps.csv` profile\n")
    add(f"- file: `{laps}`")
    add(f"- size: {laps.stat().st_size / 1e6:.0f} MB")

    schema = q(con, "DESCRIBE SELECT * FROM laps")
    total = int(q(con, "SELECT count(*) AS n FROM laps")["n"].iloc[0])
    add(f"- rows: {total:,}\n")

    # --- schema -------------------------------------------------------
    add("## Schema\n")
    add("| column | type |")
    add("|---|---|")
    for r in schema.itertuples():
        add(f"| `{r.column_name}` | {r.column_type} |")
    add("")

    columns = list(schema["column_name"])
    lower = {c.lower(): c for c in columns}

    # --- the question -------------------------------------------------
    add("## Is there an edition discriminator?\n")
    hits = [c for c in columns
            if any(h in c.lower() for h in DISCRIMINATOR_HINTS)]
    add(f"Columns whose name suggests one: "
        f"{', '.join(f'`{c}`' for c in hits) if hits else '**none**'}\n")
    for c in hits:
        d = q(con, f'SELECT count(DISTINCT "{c}") AS n, min("{c}") AS lo, '
                   f'max("{c}") AS hi FROM laps')
        add(f"- `{c}`: {int(d['n'].iloc[0])} distinct, "
            f"range {d['lo'].iloc[0]!r} to {d['hi'].iloc[0]!r}")
    add("")

    # A year hiding inside the event string would answer it too.
    if "event" in lower:
        ev = lower["event"]
        vals = q(con, f'SELECT DISTINCT "{ev}" AS v FROM laps LIMIT 200')["v"]
        with_year = [v for v in vals if isinstance(v, str)
                     and re.search(r"(19|20)\d{2}", v)]
        add(f"Event strings containing a four-digit year: "
            f"**{len(with_year)} of {len(vals)}** sampled")
        if with_year:
            add(f"  examples: {', '.join(repr(v) for v in with_year[:5])}")
        add("")

    # --- low-cardinality columns --------------------------------------
    add("## Values\n")
    for name in ("series_code", "event", "session", "class", "flags"):
        c = lower.get(name)
        if not c:
            continue
        n = int(q(con, f'SELECT count(DISTINCT "{c}") AS n FROM laps')["n"].iloc[0])
        add(f"### `{c}` — {n} distinct\n")
        if n <= MAX_VALUES:
            d = q(con, f'SELECT "{c}" AS v, count(*) AS rows FROM laps '
                       f'GROUP BY 1 ORDER BY rows DESC')
            add("| value | rows |")
            add("|---|---|")
            for r in d.itertuples():
                add(f"| `{r.v}` | {r.rows:,} |")
        else:
            d = q(con, f'SELECT "{c}" AS v, count(*) AS rows FROM laps '
                       f'GROUP BY 1 ORDER BY rows DESC LIMIT 15')
            add(f"Too many to list; the fifteen largest:\n")
            add("| value | rows |")
            add("|---|---|")
            for r in d.itertuples():
                add(f"| `{r.v}` | {r.rows:,} |")
        add("")

    # --- what a scoping pattern actually selects ----------------------
    add("## What the current scoping selects\n")
    add("`build_race_config` scopes with an ILIKE pattern on `event`. This is "
        "what those patterns pick up.\n")
    if "event" in lower and "car" in lower:
        ev, car = lower["event"], lower["car"]
        sess = lower.get("session")
        add("| pattern | rows | distinct cars | distinct events |")
        add("|---|---|---|---|")
        for pattern in ("%daytona%", "%le mans%", "%sebring%", "%spa%"):
            where = f'lower("{ev}") LIKE \'{pattern}\''
            if sess:
                where += f" AND lower(\"{sess}\") LIKE '%race%'"
            d = q(con, f'SELECT count(*) AS rows, count(DISTINCT "{car}") AS cars, '
                       f'count(DISTINCT "{ev}") AS events FROM laps WHERE {where}')
            add(f"| `{pattern}` | {int(d['rows'].iloc[0]):,} | "
                f"{int(d['cars'].iloc[0])} | {int(d['events'].iloc[0])} |")
        add("")
        add("A single Daytona 24 grid is roughly 60 cars and a single Le Mans "
            "roughly 62. Substantially more than that is the pooling.\n")

    # --- does the numbering reset? ------------------------------------
    add("## Does lap and stint numbering reset per event?\n")
    if "event" in lower and "car" in lower and "lap" in lower:
        ev, car, lap = lower["event"], lower["car"], lower["lap"]
        # Aggregated per event, not per car: a per-car frame is thousands of
        # rows and the profile has to stay small enough to paste.
        inner = [f'max("{lap}") AS max_lap']
        outer = ["max(max_lap) AS max_lap"]
        for extra, alias in (("stint_number", "max_stint_number"),
                             ("stint_lap", "max_stint_lap")):
            if extra in lower:
                inner.append(f'max("{lower[extra]}") AS {alias}')
                outer.append(f"max({alias}) AS {alias}")
        d = q(con, f'SELECT "{ev}" AS event, count(*) AS cars, '
                   f'{", ".join(outer)} '
                   f'FROM (SELECT "{ev}", "{car}", {", ".join(inner)} '
                   f'FROM laps GROUP BY 1, 2) t '
                   f'GROUP BY 1 ORDER BY max_lap DESC LIMIT 15')
        add("Per event, across its cars, the largest values seen "
            "(fifteen highest):\n")
        add("```")
        add(d.to_string(index=False))
        add("```")
        add("")
        add("A 24-hour race is roughly 800 laps at Daytona and 380 at Le Mans. "
            "A max lap far above that means the numbering runs across "
            "editions, which is what produced 199-lap 'green stints'.\n")

    # --- the pit column ------------------------------------------------
    add("## The pit column\n")
    if "pit_time" in lower:
        pt = lower["pit_time"]
        d = q(con, f'SELECT count(*) AS n, avg("{pt}") AS mean, '
                   f'stddev("{pt}") AS sd, min("{pt}") AS lo, '
                   f'quantile_cont("{pt}", 0.05) AS p05, '
                   f'quantile_cont("{pt}", 0.50) AS p50, '
                   f'quantile_cont("{pt}", 0.95) AS p95, '
                   f'max("{pt}") AS hi FROM laps WHERE "{pt}" IS NOT NULL')
        add("```")
        add(d.round(2).to_string(index=False))
        add("```")
        add("")
        add("The calibrated dials report a standard deviation two to five "
            "times the mean. If that shows here too, the column is capturing "
            "something other than service time — red flags, penalties, or "
            "non-race sessions. The 5th percentile is the number that would "
            "become `pit_transit_frac` if it turns out to be a stop with no "
            "service in it.\n")

    # --- a few rows ----------------------------------------------------
    add("## Sample rows\n")
    add("```")
    add(q(con, "SELECT * FROM laps LIMIT 5").to_string(index=False))
    add("```")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out.relative_to(ROOT)} — {out.stat().st_size / 1024:.0f} KB")
    print("That file is small enough to share. The 582 MB one stays put.")


if __name__ == "__main__":
    main(sys.argv)
