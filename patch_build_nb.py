"""Add `build_00` to build_nb.py, and repair what the 00 re-run broke.

Run from the project root:

    python patch_build_nb.py

**Idempotent per change, not per run.** Every edit is skipped if it is already
in place, so running it twice is a no-op, and running it against a tree an
earlier version already patched tops up only what is missing. It edits four
things and nothing else.

1. **Adds `build_00`** and its TARGETS entry, so `build_nb.py 00` works and
   the standing rule that notebook edits go in `build_nb.py` can be honoured
   for this stage. The 00 notebook predates the convention and had no builder.

2. **Repairs 01's call sites.** `build_race_config` takes a session id where
   it took an ILIKE pattern, so 01 would not run as written. The anchors now
   resolve through `find_race`, and Part 6's own SQL is scoped the same way.

3. **Repairs 02a's Parts 8.3 and 11**, which call `calibrate_cautions` and
   query `laps` with the same patterns.

4. **Adds the stint diagnostic** at the foot of 00's Part 6. Condition two's
   stint rows fail in opposite directions in the two series, so the cause has
   to be located before the condition is rewritten.

Two of 01's and 02a's queries are corrected while in there, both for defects
the recon found rather than for the signature:

* `MAX(stint_number)` as a stop count is a driver-stint count, roughly a
  third of the stops. Replaced by counting pit records.
* `flags <> 'GF'` as "under caution" sweeps in the chequered lap. Replaced by
  the named caution set.

**01's and 02a's numbers will move.** That is expected and is on the 00
handover's invalidation list; say so in the notebook rather than quietly
regenerating.
"""

from pathlib import Path

BUILD_00 = '''

def build_00():
    cells = []

    cells.append(md("""# Data reconnaissance (00) - re-run

**This notebook has been rebuilt.** Its first version was recorded as done
with the calibration unverified against the real `laps.csv`. 03b was the
first stage to read the frozen dials rather than assume them, refused them,
and used a stand-in instead. What follows is the re-run, and it is the stage
that decides whether this project demonstrates Daytona and Le Mans or an
invented six-hour race.

**Boundary constraint.** Constants only; no simulation code. No change to the
`ClassDials` schema or to `ASSUMED_FIELDS` - a new or renamed field
invalidates every saved config, bank and checkpoint in the project. No engine
changes, no strategy, benchmark or agent work.

**Verification gate.** The stage was specified without one, which is how the
fault survived. Four conditions, all of which could fail and one of which is
required to: Part 6.

## What was wrong

`imsa.json` described a 216-hour race with 149 cars carrying DPi, GTLM,
GTDPRO and GTP in one field. Two faults, and the second was not suspected:

1. **The queries were not scoped to one running.** `build_race_config` took an
   ILIKE pattern on `event`, and `event` carries a circuit and no edition. Six
   Daytonas and three Le Mans went into one config: durations added, counts
   summed, stints concatenated.
2. **`car` was read as an integer, so a leading zero was lost.** `#7` Toyota
   and `#007` Aston are both Hypercar at Le Mans; `#4` Corvette and `#04`
   CrowdStrike are both at Daytona. Collapsed onto one identifier, two cars'
   laps merged - which is why every edition reported 48 hours of running for a
   24-hour race, and why Le Mans 2026's Hypercar "winner" showed 62 stops.

The two compose exactly. Summed across the Le Mans editions as the old code
pooled them: 24.1 + 48.1 + 48.2 = 120.4 hours, against the 120.4 in
`wec.json`. Daytona reconciles the same way at 216.2. The near-integer
multiple of 24 hours that looked like a clean diagnosis was an integer number
of *car-races*, not of editions.

Three further defects were found that scoping does not fix, and all three are
corrected here: `stint_number` counts driver stints rather than fuel stints;
the pit column carries hour-long repairs beside 80-second stops within a
single race; and the caution flag test swept in the chequered lap."""))

    cells.append(md("""### Setup"""))

    cells.append(code(\'\'\'import sys
from pathlib import Path


def find_project_root(marker: str = "src/endurance") -> Path:
    """Walk up from the working folder until the project appears."""
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"Could not find {marker!r} at or above {here}.")


ROOT = find_project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                          # noqa: E402
from endurance import calibrate                              # noqa: E402
from endurance import gate00                                 # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

LAPS_CSV    = ROOT / "data" / "raw" / "laps.csv"
DRIVERS_CSV = ROOT / "data" / "raw" / "drivers.csv"
PARAMS_DIR  = ROOT / "data" / "processed"
PARAMS_DIR.mkdir(parents=True, exist_ok=True)

# The two races the demonstration is about, and the class each is headlined
# by. The edition is resolved from the data in Part 1 rather than named here.
ANCHORS = {
    "imsa": {"event": "%daytona%", "headline_class": "GTP",      "name": "Daytona 24"},
    "wec":  {"event": "%le mans%", "headline_class": "HYPERCAR", "name": "Le Mans 24"},
}

con = calibrate.connect(str(LAPS_CSV), str(DRIVERS_CSV))
print(f"loaded {con.execute(\\\'SELECT COUNT(*) FROM laps\\\').fetchone()[0]:,} laps")\'\'\'))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 1 - which race, and which running of it

`event` names a circuit. The edition discriminator is `session_id`: 1013
values, none of which spans an event, a year, a series or a session type.
`(series, event, year)` is *not* sufficient - the Asian Le Mans double-headers
put two races under one key - which is why the scope is the session and not
the year.

The table below is the one the edition decision was taken on. Note what it
says about caution share: it moves by nearly a factor of two between adjacent
Daytonas, so a dial calibrated from one running is a sample of one. Freezing
the latest edition is the decision; showing the spread beside it is what stops
the number being read as a property of the race rather than of one race."""))

    cells.append(code(\'\'\'RACES = {}
for series, anchor in ANCHORS.items():
    editions = calibrate.list_races(con, series, anchor["event"])
    RACES[series] = calibrate.find_race(con, series, anchor["event"])
    print(f"--- {series} {anchor[\\\'name\\\']}: {len(editions)} editions in file, "
          f"taking {RACES[series][\\\'label\\\']}")
    print(editions.to_string(index=False))\'\'\'))

    cells.append(code(\'\'\'# How far a dial moves between runnings of the same race. The caution share
# is the one that moves most, and it is the one every strategy result is
# sensitive to.
spread = []
for series, anchor in ANCHORS.items():
    for _, ed in calibrate.list_races(con, series, anchor["event"]).iterrows():
        sid = int(ed["session_id"])
        try:
            c = calibrate.calibrate_cautions(con, sid)
        except Exception as exc:                       # a partial edition
            print(f"{series} {int(ed[\\\'year\\\'])}: skipped ({exc})")
            continue
        spread.append({"series": series, "year": int(ed["year"]),
                       "cars": int(ed["cars"]),
                       "duration_h": round(ed["duration_s"] / 3600, 2),
                       "caution_rate": round(c["caution_rate"], 3),
                       "caution_dur_s": round(c["caution_mean_dur_s"]),
                       "episodes": c["n_caution_episodes"],
                       "frozen": sid == RACES[series]["session_id"]})
spread = pd.DataFrame(spread)
spread\'\'\'))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 2 - the five dials

One query per dial per class, every number traceable to a named function in
`src/endurance/calibrate.py`. Three of the five changed in this re-run:

- **Stint length** comes from the pit records, not from `stint_number`. Every
  step of that counter coincides with a driver change and none with a stop -
  two to three fuel stints to the step - so reading it as a fuel stint
  reported 58-lap green stints at Daytona where the winner averaged 23.
- **Pit cost** is a median, under an unchanged field name. Within one scoped
  race the column still runs to 6,800 seconds against a median of 80, so the
  arithmetic mean sat two to four times its own median and the standard
  deviation sat above the mean. The spread is the standard deviation of the
  sample trimmed at three times the median.
- **Cautions** are calibrated once for the race rather than once per class.
  They are a property of the race; calling the function inside the class loop
  is what left seven classes of one race carrying seven different episode
  lengths."""))

    cells.append(code(\'\'\'configs = {}
for series, anchor in ANCHORS.items():
    race = RACES[series]
    configs[series] = calibrate.build_race_config(
        con, series, race["session_id"], f"{anchor[\\\'name\\\']} {race[\\\'year\\\']}")
    cfg = configs[series]
    print(f"{series}: {len(cfg.classes)} classes, {cfg.total_cars} cars, "
          f"{cfg.duration_s / 3600:.2f} h - {cfg.classes[0].source_event}")

pd.concat([calibrate.dials_table(cfg) for cfg in configs.values()],
          ignore_index=True)\'\'\'))

    cells.append(md("""### What is measured, and what is assumed

Unchanged in this re-run, and printed here so nothing gets quietly promoted
from guess to fact. `pit_transit_frac` is on the assumed list and Part 5
measures a candidate for it without moving it."""))

    cells.append(code(\'\'\'from endurance import ClassDials                             # noqa: E402

sample = configs["imsa"].classes[0]
pd.DataFrame(
    [{"field": f, "value": round(getattr(sample, f), 4), "status": "measured"}
     for f in sample.measured_fields()]
    + [{"field": f, "value": getattr(sample, f), "status": "ASSUMED"}
       for f in ClassDials.assumed_fields()])\'\'\'))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 3 - degradation, and what this data can support

Degradation is fitted jointly against tyre age and laps since the last fill.
Within a stint the car gets lighter as the tyres get older and the two effects
have opposite signs, so a slope on tyre age alone is their sum. The
field-relative frame does not rescue it: cars in a class stagger their stops,
so at a given lap number they are at different points of the tank and fuel is
not common-mode.

Where a class changes tyres at every stop the two regressors are the same
number and nothing can separate them. That is reported rather than resolved:
`identified` is the column to read before the slope."""))

    cells.append(code(\'\'\'for series, cfg in configs.items():
    print(f"--- {series}")
    print(calibrate.degradation_table(
        con, RACES[series]["session_id"],
        [c.class_name for c in cfg.classes]).to_string(index=False))\'\'\'))

    cells.append(md("""**Read the `identified` column first.** A negative slope
in a class where it is false is not a defect to be tuned away; it is the net
within-stint pace trend, which is what the engine will reproduce, and it says
that this file cannot separate tyre wear from fuel burn for that class. A
negative slope where `identified` is true would be a second defect and gate
condition four would be right to fail on it."""))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 4 - the caution units, old and new

Carried forward from 02a. The share and the episode length are measured in
seconds of race time rather than laps, and the observed caution pace
multiplier is reported beside the assumed one - a measured counterpart to an
assumed dial, shown rather than silently substituted."""))

    cells.append(code(\'\'\'for series, cfg in configs.items():
    print(f"--- {series}")
    print(calibrate.caution_report(
        con, RACES[series]["session_id"],
        cfg.classes[0].base_pace_s).to_string(index=False))\'\'\'))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 5 - `pit_transit_frac`, measured

03b established that 02c's `splash_and_dash` result depends on this dial:
gained falls monotonically as it runs 0.25 to 0.65 while no other roster row
moves monotonically. It is in `ASSUMED_FIELDS` at 0.25 and had never been
swept when 02c relied on it.

A low quantile of `pit_time` for one properly scoped race is a stop with
almost no service in it, which is the lane transit delta itself. Two things
had to be true for that to be worth measuring, and both now are: the column
had to be lane-to-lane time lost rather than stationary time - the recon put
the ratio of `pit_time` to the lap's excess over a green lap at 1.01 to 1.16
at the lower quartile - and the outliers had to be gone.

The regulations supply a cross-check and not the answer. The pit lane speed
limit is 60 km/h in both series (IMSA art. 32.3, WEC art. 12.1.4) and neither
rulebook carries the lane length, which is a circuit fact.

**This does not move the dial.** Promoting a field out of `ASSUMED_FIELDS` is
a blueprint amendment and a separate decision; what follows is the evidence
for taking it."""))

    cells.append(code(\'\'\'rows = []
for series, cfg in configs.items():
    for c in cfg.classes:
        pit = calibrate.calibrate_pit(con, RACES[series]["session_id"], c.class_name)
        rows.append({
            "series": series, "class": c.class_name,
            "median_stop_s": round(pit["pit_time_mean_s"], 1),
            "raw_mean_s": round(pit["pit_time_raw_mean_s"], 1),
            "trimmed_out": pit["n_pit_stops_trimmed_out"],
            "n": pit["n_pit_stops"],
            "transit_s_p05": round(pit["pit_lane_transit_s"], 1),
            "implied_frac": round(pit["pit_lane_transit_s"] / pit["pit_time_mean_s"], 3),
            "assumed_frac": c.pit_transit_frac})
pd.DataFrame(rows)\'\'\'))

    cells.append(md("""`raw_mean_s` beside `median_stop_s` is the argument for
decision 6 in one column: the mean is what the old dial reported and it is not
a stop anyone made.

`implied_frac` against `assumed_frac` is the finding. Where it lands well
above 0.25, 02c's `splash_and_dash` result was computed at the wrong end of
the dial it is most sensitive to, and 03b's sweep already tells us which
direction that moves it."""))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 6 - the verification gate

Four conditions. Condition three is the falsifier: it widens the scope to two
adjacent editions on purpose and **requires** conditions one and two to fail.
Adjacent rather than distant, because two runnings of the same race are
genuinely similar and are the hardest case for a gate to catch.

Two things the gate does not do, stated rather than discovered:

- **Grid size does not detect pooling.** Car numbers recur between editions -
  six Daytonas carry 91 numbers, not 360 - so the count grows by far less than
  the racing does. That check earns its place against the leading-zero
  collision and against a class list assembled from somewhere other than the
  scope. Duration and lap counts are what catch pooling.
- **Nor does the pit dial.** Since it became a median it is robust to pooling
  in the same way base pace always was: the edition with more stops carries
  the statistic. Both are better dials for it and neither is a detector."""))

    cells.append(code(\'\'\'gates = {}
for series in ANCHORS:
    editions = calibrate.list_races(con, series, ANCHORS[series]["event"])
    others = [int(s) for s in editions["session_id"]
              if int(s) != RACES[series]["session_id"]]
    print(f"===== {series}: {RACES[series][\\\'label\\\']}, "
          f"falsifier pools with session {others[-1]}")
    gates[series] = gate00.run_gate(con, series, RACES[series]["session_id"],
                                    others[-1], cfg=configs[series])
    print()\'\'\'))

    cells.append(code(\'\'\'# The falsifier's own detail: every one of these is required to fail.
for series, g in gates.items():
    detail = g[g["condition"].str.startswith("three: falsifier (")]
    print(f"--- {series}: {int((~detail[\\\'passed\\\']).sum())} of {len(detail)} "
          f"pooled checks failed, as required")
    print(detail[["check", "value", "threshold", "passed"]].to_string(index=False))
    print()\'\'\'))

    cells.append(code(\'\'\'for series, g in gates.items():
    print(f"{series}: gate {\\\'PASSES\\\' if gate00.gate_passes(g) else \\\'FAILS\\\'}")\'\'\'))

    # ------------------------------------------------------------------
    cells.append(md("""## Part 7 - freeze

`data/processed/{series}.json` is what every later stage reads. Writing it is
the last thing this notebook does, and it happens only if the gate passed -
a config that fails its own gate must not be able to reach 02b's bank or
03b's training run.

Everything downstream re-runs from here. The seed lists do not move -
`draw_seed_bank` draws from `draw_seed` alone - but the races they name do,
because a seed plus different dials is a different race."""))

    cells.append(code(\'\'\'for series, cfg in configs.items():
    if not gate00.gate_passes(gates[series]):
        print(f"{series}: gate failed, NOT written")
        continue
    path = PARAMS_DIR / f"{series}.json"
    cfg.save(path)
    print(f"wrote {path}  ({cfg.classes[0].source_event})")\'\'\'))

    cells.append(md("""## Where this leaves us

**The stage.** Both configs describe one running of one race, scoped by
`session_id`, with `car` read as text so a leading zero is a different car.

**What must re-run, in order.** `scripts/freeze_assets.py --force`; both
policies retrained, which `PolicyCard.check` will insist on against the new
`dials_fingerprint`; 02c's roster table; 03b's tables; and 01 Part 6 and 02a
Part 6, outstanding since 02a and now runnable against real dials.

**What does not.** Nothing in 03b's findings: every number it produced is
labelled stand-in, and the degenerate reward, the `pit_transit_frac` exploit
and the verification gates are about the apparatus rather than about the
dials.

**What is still open.** Where `identified` is false in Part 3, this file
cannot separate tyre wear from fuel burn and the dial holds a net trend. That
is a documented limitation, not a defect, and it is the honest outcome the
re-run was allowed to have."""))

    return cells

'''

TARGETS_OLD = '''TARGETS = {
    "01": ("01_race_engine.ipynb", build_01),'''

TARGETS_NEW = '''TARGETS = {
    "00": ("00_data_recon.ipynb", build_00),
    "01": ("01_race_engine.ipynb", build_01),'''

# --- 01: resolve the edition, then scope on it -------------------------
O1_CONFIGS_OLD = '''    cells.append(code(\'\'\'configs = {}
for series, anchor in ANCHORS.items():
    configs[series] = calibrate.build_race_config(
        con, series, anchor["event"], anchor["name"]
    )'''

O1_CONFIGS_NEW = '''    cells.append(code(\'\'\'# One running of one race. `find_race` resolves the latest edition of the
# event and refuses when the answer is not unique; passing its session id on
# is what keeps every dial below scoped to a single race. 00 has the detail.
RACES = {s: calibrate.find_race(con, s, a["event"]) for s, a in ANCHORS.items()}

configs = {}
for series, anchor in ANCHORS.items():
    configs[series] = calibrate.build_race_config(
        con, series, RACES[series]["session_id"],
        f"{anchor[\\\'name\\\']} {RACES[series][\\\'year\\\']}"
    )'''

O1_PART6_OLD = '''    real = con.execute(f"""
        SELECT MAX(laps) AS winner_laps, AVG(stops) AS mean_stops
        FROM (
            SELECT car, MAX(lap) AS laps, MAX(stint_number) AS stops
            FROM laps
            WHERE session='race' AND series_code='{series_code}'
              AND class='{hc}' AND event ILIKE '{anchor["event"]}'
            GROUP BY car
        ) t
    """).df().iloc[0]'''

O1_PART6_NEW = '''    sid = RACES[series_code]["session_id"]

    # Stops are counted from the pit records. `MAX(stint_number)` counted
    # driver stints - roughly a third of the stops - which is 00's finding.
    real = con.execute(f"""
        SELECT MAX(laps) AS winner_laps, AVG(stops) AS mean_stops
        FROM (
            SELECT car, MAX(lap) AS laps,
                   SUM(CASE WHEN pit_time IS NOT NULL THEN 1 ELSE 0 END) AS stops
            FROM laps
            WHERE session='race' AND series_code='{series_code}'
              AND class='{hc}' AND session_id = {sid}
            GROUP BY car
        ) t
    """).df().iloc[0]'''

O1_REF_OLD = '''    ref_car = con.execute(f"""
        SELECT car FROM laps
        WHERE session='race' AND series_code='{series_code}'
          AND event ILIKE '{anchor["event"]}'
        GROUP BY car ORDER BY COUNT(*) DESC LIMIT 1
    """).fetchone()[0]

    real_caution = con.execute(f"""
        SELECT AVG(CASE WHEN flags <> 'GF' THEN 1.0 ELSE 0.0 END) AS share
        FROM laps
        WHERE session='race' AND series_code='{series_code}'
          AND event ILIKE '{anchor["event"]}' AND car = '{ref_car}'
    """).df().iloc[0]["share"]'''

O1_REF_NEW = '''    ref_car = con.execute(f"""
        SELECT car FROM laps
        WHERE session='race' AND series_code='{series_code}'
          AND session_id = {sid}
        GROUP BY car ORDER BY COUNT(*) DESC LIMIT 1
    """).fetchone()[0]

    # Caution is a named set. `flags <> 'GF'` swept in the chequered lap.
    real_caution = con.execute(f"""
        SELECT AVG(CASE WHEN flags IN ('FCY', 'SF', 'RF') THEN 1.0 ELSE 0.0 END)
                   AS share
        FROM laps
        WHERE session='race' AND series_code='{series_code}'
          AND session_id = {sid} AND car = '{ref_car}'
          AND flags IN ('GF', 'FCY', 'SF', 'RF')
    """).df().iloc[0]["share"]'''

# --- 02a: two loops over (series, pattern) -----------------------------
O2A_83_OLD = '''    con = calibrate.connect(str(DATA))
    for series_code, pattern in (("imsa", "%daytona%"), ("wec", "%le mans%")):
        cfg = RaceConfig.load(PARAMS_DIR / f"{series_code}.json")
        cls = cfg.classes[0]
        report = calibrate.calibrate_cautions(con, series_code, pattern,
                                              cls.base_pace_s)'''

O2A_83_NEW = '''    con = calibrate.connect(str(DATA))
    for series_code, pattern in (("imsa", "%daytona%"), ("wec", "%le mans%")):
        cfg = RaceConfig.load(PARAMS_DIR / f"{series_code}.json")
        cls = cfg.classes[0]
        # Scoped to the edition the frozen dials came from, per 00's re-run.
        sid = calibrate.find_race(con, series_code, pattern)["session_id"]
        report = calibrate.calibrate_cautions(con, sid, cls.base_pace_s)'''

O2A_11_OLD = '''        real = con.execute(
            "SELECT "
            "SUM(CASE WHEN flags <> 'GF' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), "
            "SUM(CASE WHEN flags <> 'GF' THEN lap_time ELSE 0 END) "
            "/ SUM(lap_time) "
            "FROM laps "
            f"WHERE series_code = '{series_code}' "
            f"AND event ILIKE '{pattern}' "
            "AND session = 'race' AND pit_time IS NULL").fetchone()'''

O2A_11_NEW = '''        sid = calibrate.find_race(con, series_code, pattern)["session_id"]
        real = con.execute(
            "SELECT "
            "SUM(CASE WHEN flags IN ('FCY','SF','RF') THEN 1 ELSE 0 END) "
            "* 1.0 / COUNT(*), "
            "SUM(CASE WHEN flags IN ('FCY','SF','RF') THEN lap_time ELSE 0 END) "
            "/ SUM(lap_time) "
            "FROM laps "
            f"WHERE series_code = '{series_code}' "
            f"AND session_id = {sid} "
            "AND session = 'race' AND pit_time IS NULL "
            "AND flags IN ('GF','FCY','SF','RF')").fetchone()'''

DOCSTRING_OLD = '"""Regenerate notebooks/01_race_engine.ipynb. Edit here, not the .ipynb."""'
DOCSTRING_NEW = ('"""Regenerate the notebooks. Edit here, not the .ipynb.\\n\\n'
                 'Targets: 00, 01, 02a, 02b, 02c, 03a, 03b. No argument builds all.\\n"""')

# --- the stint diagnostic, at the foot of Part 6 -----------------------
# Inserted separately from `build_00` so this script can top up a build_nb.py
# that an earlier run of it already patched.
DIAG_ANCHOR = '''    # ------------------------------------------------------------------
    cells.append(md("""## Part 7 - freeze'''

DIAG_CELLS = '''    cells.append(md("""### Where the stint discrepancy lives

Condition two's stint rows fail in **opposite directions** in the two series -
IMSA long, WEC short - which rules out a single systematic cause. Three
quantities side by side: what the file's stints look like, what the dial took
from them, and what the engine then does with it.

`dial_green_stint_laps` above `file_green_max` would be a tank nobody ever
emptied, and the upper-quartile choice would be wrong for this purpose.
`sim_laps_per_stop` far from the dial puts the discrepancy in the engine's
stopping rule or in `fuel_per_lap_caution` - an assumed dial, never swept -
rather than in the calibration. Those have different owners, and the gate
should not be rewritten until it is known which one this is.

Nothing here can fail. It is not a gate condition and `run_gate` does not call
it. It sits before Part 7 because the freeze is the last thing this notebook
does, and a diagnostic read after the decision it informs is decoration."""))

    cells.append(code(\'\'\'for series, cfg in configs.items():
    print(f"--- {series}")
    print(gate00.stint_diagnostic(
        con, cfg, RACES[series]["session_id"]).to_string(index=False))
    print()\'\'\'))

'''


EDITS = [
    ("module docstring", DOCSTRING_OLD, DOCSTRING_NEW),
    ("TARGETS", TARGETS_OLD, TARGETS_NEW),
    ("01 config build", O1_CONFIGS_OLD, O1_CONFIGS_NEW),
    ("01 Part 6 real classification", O1_PART6_OLD, O1_PART6_NEW),
    ("01 Part 6 caution share", O1_REF_OLD, O1_REF_NEW),
    ("02a Part 8.3", O2A_83_OLD, O2A_83_NEW),
    ("02a Part 11", O2A_11_OLD, O2A_11_NEW),
]


def apply_edits(src: str) -> tuple[str, list[str]]:
    """Apply each edit that is not already in place. Skipping is not failing."""
    applied = []
    for label, old, new in EDITS:
        if new in src:
            continue
        if src.count(old) != 1:
            raise SystemExit(
                f"could not apply {label!r}: found {src.count(old)} matches, "
                "expected exactly 1. build_nb.py has moved on; patch by hand.")
        src = src.replace(old, new)
        applied.append(label)
    return src, applied


def insert_build_00(src: str) -> tuple[str, list[str]]:
    if "def build_00" in src:
        return src, []
    anchor = "\ndef write(name, cells):"
    if src.count(anchor) != 1:
        raise SystemExit("could not find `def write` to insert build_00 before")
    return src.replace(anchor, BUILD_00 + anchor), ["build_00"]


def insert_diagnostic(src: str) -> tuple[str, list[str]]:
    """Add the stint diagnostic to the foot of 00's Part 6.

    A separate step from `build_00` so a tree patched by an earlier run of
    this script gets the diagnostic without the whole builder going back in.
    """
    if "stint_diagnostic" in src:
        return src, []
    if "def build_00" not in src:
        raise SystemExit("build_00 must exist before the diagnostic can go in it")
    if src.count(DIAG_ANCHOR) != 1:
        raise SystemExit(
            "could not find Part 7's heading to insert the diagnostic before "
            f"({src.count(DIAG_ANCHOR)} matches, expected 1)")
    return src.replace(DIAG_ANCHOR, DIAG_CELLS + DIAG_ANCHOR), ["stint diagnostic"]


def main() -> None:
    path = Path("build_nb.py")
    if not path.exists():
        raise SystemExit("run this from the project root, beside build_nb.py")

    src = path.read_text()
    done: list[str] = []
    for step in (apply_edits, insert_build_00, insert_diagnostic):
        src, applied = step(src)
        done += applied

    if not done:
        print("build_nb.py is already up to date - nothing to do")
        return

    path.write_text(src)
    print(f"patched {path}: {len(done)} change(s)")
    for label in done:
        print(f"  - {label}")
    print("\nnow run:  python build_nb.py 00")


if __name__ == "__main__":
    main()
