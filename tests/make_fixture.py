"""Build a small synthetic laps.csv in the real schema.

Used to exercise the calibration end to end without shipping real timing data
around. The generator plants known values - a chosen degradation slope, stint
length and pit cost - so the calibration can be checked against the truth it
was built from, which a real file cannot do.

**Two editions per series, deliberately different.** The old fixture had one
running of each race, so a query that pooled editions passed every test while
the real output was wrong. Each series now gets two sessions with different
planted values: the tests recover the target edition's truth, and the gate's
falsifier widens the scope across both and requires the recovery to fail.

**Three things the old fixture could not express**, each of which hid a defect
that the real data has:

* A car number and the same number with a leading zero, **in the same class**.
  Read as an integer they collapse into one car. `#7` and `#007` in Hypercar
  at Le Mans is the real instance.
* `stint_number` stepping on driver changes rather than on stops, at roughly
  three fuel stints to the driver stint.
* Pit records carrying the occasional hour-long repair beside normal service.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# The edition each test calibrates against. `other` is the second running,
# planted differently so pooling is detectable rather than merely present.
# Each session carries two classes. The second changes tyres every other
# stop, so tyre age and fuel load decorrelate and the degradation fit is
# identified; the first changes them at every stop, where it is not. Both
# branches then have a race in the fixture to be tested against. LMGT3 at Le
# Mans is the real instance of the double-stinted case.
TRUTH = {
    "imsa": dict(event="Daytona", cls="GTP", year=2026, session_id=11,
                 base_pace=97.5, deg=0.012, pit=47.0, n_cars=8, stint=30,
                 sc_p=0.010, sc_dur=6, laps=(650, 700), caution_flag="FCY", number_offset=0,
                 second=dict(cls="GTD", n_cars=6, base_pace=110.0, deg=0.020,
                             pit=52.0, stint=26, tyre_stints=2)),
    "wec": dict(event="Le Mans", cls="HYPERCAR", year=2026, session_id=21,
                base_pace=222.0, deg=0.030, pit=58.0, n_cars=9, stint=12,
                sc_p=0.012, sc_dur=9, laps=(340, 380), caution_flag="SF", number_offset=0,
                second=dict(cls="LMGT3", n_cars=7, base_pace=245.0, deg=0.045,
                            pit=61.0, stint=11, tyre_stints=3)),
}

OTHER = {
    "imsa": dict(event="Daytona", cls="GTP", year=2025, session_id=10,
                 base_pace=99.0, deg=0.020, pit=62.0, n_cars=6, stint=18,
                 sc_p=0.020, sc_dur=9, laps=(600, 640), caution_flag="FCY", number_offset=10,
                 second=dict(cls="GTD", n_cars=5, base_pace=112.0, deg=0.030,
                             pit=68.0, stint=15, tyre_stints=2)),
    "wec": dict(event="Le Mans", cls="HYPERCAR", year=2025, session_id=20,
                base_pace=225.0, deg=0.040, pit=74.0, n_cars=7, stint=20,
                sc_p=0.008, sc_dur=5, laps=(300, 330), caution_flag="SF", number_offset=10,
                second=dict(cls="LMGT3", n_cars=6, base_pace=248.0, deg=0.060,
                            pit=66.0, stint=17, tyre_stints=2)),
}

# Fuel stints per driver stint, so `stint_number` is wrong in the fixture in
# exactly the way it is wrong in the file.
STINTS_PER_DRIVER = 3

# A car whose number collides with another's once the leading zero is lost.
# Same class on purpose: a composite (car, class) key would not separate these.
COLLIDING_PAIR = ("7", "007")


def _car_numbers(n_cars: int, base: int, offset: int,
                 with_collision: bool) -> list[str]:
    """`n_cars` numbers for one class of one session.

    The colliding pair goes in one class only - two cars sharing a number
    across two classes of one race does not happen, and a fixture that says it
    does would let a composite `(car, class)` key look sufficient.

    `offset` shifts the block between editions, so the grid changes year to
    year as a real one does and pooling grows the distinct-number count. Only
    somewhat: numbers recur, which is why six Daytonas hold ninety-one numbers
    rather than three hundred and sixty, and why grid size is a weak detector
    of pooling and duration is a strong one.
    """
    nums = list(COLLIDING_PAIR) if with_collision else []
    nums += [str(base + offset + i) for i in range(max(n_cars - len(nums), 0))]
    return nums[:n_cars]


def _class_specs(t: dict) -> list[dict]:
    """The session's classes: the headline one, then the double-stinting one."""
    head = dict(cls=t["cls"], n_cars=t["n_cars"], base_pace=t["base_pace"],
                deg=t["deg"], pit=t["pit"], stint=t["stint"], tyre_stints=1)
    return [head, dict(t["second"])]


def _one_session(rows: list[dict], series: str, t: dict, rng) -> None:
    for i, spec in enumerate(_class_specs(t)):
        spec["base"] = 30 + 20 * i
        spec["with_collision"] = (i == 0)
        _one_class(rows, series, t, spec, rng)


def _one_class(rows, series: str, t: dict, spec: dict, rng) -> None:
    for car in _car_numbers(spec["n_cars"], spec["base"], t["number_offset"],
                            spec["with_collision"]):
        lap = stint_lap = tire_age = fuel_lap = 0
        stint_number = 1
        stops = 0
        clock = 0.0
        caution_left = 0
        n_laps = int(rng.integers(*t["laps"]))
        target = int(spec["stint"] + rng.integers(-2, 3))
        driver = 0

        for i in range(n_laps):
            lap += 1
            stint_lap += 1
            fuel_lap += 1
            tire_age += 1

            if caution_left > 0:
                flag, caution_left = t["caution_flag"], caution_left - 1
            elif rng.random() < t["sc_p"]:
                flag = t["caution_flag"]
                caution_left = int(rng.geometric(1 / t["sc_dur"])) - 1
            else:
                flag = "GF"

            # The chequered lap is neither green nor a caution. The old
            # calibration counted it as a caution; the new one drops it.
            if i == n_laps - 1:
                flag = "FF"

            pit_time = None
            do_pit = fuel_lap >= target and flag == "GF" and i < n_laps - 1

            if flag == "GF":
                # Tyre wear pushes the lap time up, fuel burn-off pulls it
                # down. Planted with wear the larger of the two, so the
                # recovered slope is positive and the sign test can fail.
                lap_time = (spec["base_pace"] + spec["deg"] * tire_age
                            - 0.3 * spec["deg"] * fuel_lap + rng.normal(0, 0.35))
                bpillar = 1 if rng.random() < 0.6 else 2
            else:
                lap_time = spec["base_pace"] * 1.6 + rng.normal(0, 1.0)
                bpillar = 3

            if do_pit:
                pit_time = round(spec["pit"] + rng.normal(0, 2.0), 2)
                # One car in twenty picks up a repair. Rare enough not to move
                # a median, ruinous to a mean - which is the point.
                if rng.random() < 0.02:
                    pit_time = round(pit_time + rng.uniform(600, 3000), 2)
                lap_time += pit_time

            clock += lap_time

            rows.append({
                "series_code": series, "series": f"{series}-{t['year']}",
                "year": t["year"], "event": t["event"],
                "session": "race", "session_id": t["session_id"],
                "session_time": round(clock, 3),
                "class": spec["cls"], "car": car,
                "driver_name": f"{car}-driver-{driver}",
                "stint_number": stint_number, "stint_lap": stint_lap, "lap": lap,
                "lap_time": round(lap_time, 3), "pit_time": pit_time,
                "flags": flag, "bpillar_quartile": bpillar,
                "est_tire_age": tire_age,
            })

            if do_pit:
                stops += 1
                fuel_lap = 0
                # Tyres last `tyre_stints` fuel stints. Where that is 1 the
                # two counters are the same number and no fit can separate
                # them; where it is more, they decorrelate.
                if stops % spec["tyre_stints"] == 0:
                    tire_age = 0
                # The driver, and with them `stint_number`, changes every third
                # stop rather than every stop.
                if stops % STINTS_PER_DRIVER == 0:
                    stint_number += 1
                    stint_lap = 0
                    driver += 1


def build(out_dir, seed: int = 7) -> Path:
    """Write data/raw/{laps,drivers}.csv under `out_dir` and return the laps path."""
    out_dir = Path(out_dir)
    (out_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    for series in TRUTH:
        _one_session(rows, series, OTHER[series], rng)
        _one_session(rows, series, TRUTH[series], rng)

    path = out_dir / "data" / "raw" / "laps.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    pd.DataFrame({"driver_id": ["d1"], "driver_name": ["Test Driver"],
                  "license": ["P"]}).to_csv(
        out_dir / "data" / "raw" / "drivers.csv", index=False)
    return path


if __name__ == "__main__":
    import tempfile
    print(build(Path(tempfile.mkdtemp())))
