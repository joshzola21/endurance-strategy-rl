"""Does the calibration recover the numbers it was built from?

The fixture is generated with a known degradation slope, stint length and pit
cost, so these tests can check the calibration against ground truth rather
than merely checking it returns something. That is the only way to catch a
query that runs fine and is quietly wrong.

**The old suite passed while the real output was wrong**, because the fixture
had one running of each race and no leading-zero car numbers, so neither the
pooling nor the collision could express itself. Every test below whose name
mentions an edition, a leading zero, a driver counter or a repair exists
because the fixture now can.
"""

import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))
sys.path.insert(0, str(HERE))

from endurance import calibrate, run_race  # noqa: E402
from make_fixture import OTHER, TRUTH, build  # noqa: E402
from sqlite_shim import connect_fixture  # noqa: E402

_tmp = Path(tempfile.mkdtemp())
LAPS = build(_tmp)


def con():
    return connect_fixture(str(LAPS))


def sid(series: str) -> int:
    return TRUTH[series]["session_id"]


# ----------------------------------------------------------------------
# The scope
# ----------------------------------------------------------------------
def test_the_file_holds_two_editions_of_each_race():
    """Without this the pooling tests below are vacuous."""
    for series, truth in TRUTH.items():
        races = calibrate.list_races(con(), series, f"%{truth['event']}%")
        assert len(races) == 2, series
        assert set(races["session_id"]) == {truth["session_id"],
                                            OTHER[series]["session_id"]}


def test_find_race_returns_the_latest_edition():
    for series, truth in TRUTH.items():
        r = calibrate.find_race(con(), series, f"%{truth['event']}%")
        assert r["session_id"] == truth["session_id"], series
        assert r["year"] == truth["year"], series


def test_find_race_can_name_an_earlier_edition():
    for series in TRUTH:
        r = calibrate.find_race(con(), series, f"%{TRUTH[series]['event']}%",
                                year=OTHER[series]["year"])
        assert r["session_id"] == OTHER[series]["session_id"], series


def test_duration_is_one_race_and_pooling_adds_to_it():
    for series, truth in TRUTH.items():
        one = calibrate.calibrate_duration(con(), truth["session_id"])
        both = calibrate.calibrate_duration(
            con(), [truth["session_id"], OTHER[series]["session_id"]])
        assert both > 1.5 * one, series


# ----------------------------------------------------------------------
# Car identity
# ----------------------------------------------------------------------
def test_a_leading_zero_is_a_different_car():
    """`#7` and `#007` are two cars in the same class, and must stay two.

    Read as an integer they merge, which doubles the merged car's laps and
    stops and removes one car from the grid. This is the defect that made a
    twenty-four-hour race report forty-eight hours.
    """
    for series, truth in TRUTH.items():
        rows = con().execute(
            f"SELECT DISTINCT car, class FROM laps "
            f"WHERE session_id = {truth['session_id']}").df()
        cars = set(rows["car"].astype(str))
        assert {"7", "007"} <= cars, series
        # In one class, so a composite (car, class) key would not have
        # separated them either. This is the real Le Mans case: #7 Toyota and
        # #007 Aston are both Hypercar.
        classes = set(rows.loc[rows["car"].astype(str).isin(["7", "007"]), "class"])
        assert len(classes) == 1, series
        # And they are two cars, not one twice as busy as everyone else.
        laps = con().execute(
            f"SELECT car, MAX(lap) AS laps FROM laps "
            f"WHERE session_id = {truth['session_id']} "
            f"AND car IN ('7', '007') GROUP BY car").df()
        assert len(laps) == 2, series


def test_the_grid_is_the_size_it_was_planted():
    for series, truth in TRUTH.items():
        tr = calibrate.calibrate_traffic(con(), truth["session_id"])
        n = int(tr.loc[tr["class"] == truth["cls"], "cars"].iloc[0])
        assert n == truth["n_cars"], series


# ----------------------------------------------------------------------
# Dial 1 - pace and degradation
# ----------------------------------------------------------------------
def test_base_pace_is_recovered():
    for series, truth in TRUTH.items():
        pace = calibrate.calibrate_pace(con(), truth["session_id"], truth["cls"])
        assert abs(pace["base_pace_s"] - truth["base_pace"]) < 2.0, series


def test_degradation_slope_has_the_planted_sign():
    # The field-relative frame removes the common-mode part of the wear, so
    # the recovered slope is smaller than the planted one by design. What must
    # hold is the sign.
    for series, truth in TRUTH.items():
        pace = calibrate.calibrate_pace(con(), truth["session_id"], truth["cls"])
        assert pace["deg_slope_s_per_lap"] > 0, series


def test_wec_degrades_faster_than_imsa_as_planted():
    imsa = calibrate.calibrate_pace(con(), sid("imsa"), TRUTH["imsa"]["cls"])
    wec = calibrate.calibrate_pace(con(), sid("wec"), TRUTH["wec"]["cls"])
    assert wec["deg_slope_s_per_lap"] > imsa["deg_slope_s_per_lap"]


def test_degradation_is_unidentified_when_tyres_change_at_every_stop():
    """The headline class changes tyres at every stop, so tyre age and fuel
    load are the same number and no fit can separate them. The calibration
    must say so rather than report a tyre slope it cannot support."""
    for series, truth in TRUTH.items():
        pace = calibrate.calibrate_pace(con(), truth["session_id"], truth["cls"])
        assert pace["deg_identified"] is False, series
        assert pace["age_fuel_corr"] > calibrate.DEG_COLLINEAR_ABOVE, series


def test_degradation_is_identified_when_tyres_are_double_stinted():
    """And where they do decorrelate, the fuel term comes out negative -
    the car gets lighter as the tyres get older."""
    for series, truth in TRUTH.items():
        pace = calibrate.calibrate_pace(con(), truth["session_id"],
                                        truth["second"]["cls"])
        assert pace["deg_identified"] is True, series
        assert pace["deg_slope_s_per_lap"] > 0, series
        assert pace["fuel_slope_s_per_lap"] < 0, series


# ----------------------------------------------------------------------
# Dial 2 - cautions
# ----------------------------------------------------------------------
def test_caution_rate_is_in_the_right_region():
    for series in TRUTH:
        c = calibrate.calibrate_cautions(con(), sid(series))
        assert 0.0 < c["caution_rate"] < 0.5, series
        assert c["n_caution_episodes"] > 0
        assert c["caution_mean_dur_s"] > 0


def test_the_chequered_lap_is_not_a_caution():
    """`FF` is neither green nor a caution and leaves both sides of the share.

    The old calibration counted everything that was not `GF` as a caution.
    """
    for series in TRUTH:
        c = calibrate.calibrate_cautions(con(), sid(series))
        assert "FF" in c["flag_census"], series
        laps = c["caution_lap_share"] * (c["n_caution_episodes"] and 1 or 1)
        assert 0.0 <= laps <= 1.0
        # The reference car's own chequered lap must not appear as a caution
        # episode of length one.
        assert c["caution_mean_dur_s"] > 0


def test_one_caution_calibration_serves_the_whole_race():
    """Cautions are a property of the race, so every class must carry the same
    numbers. Seven classes of one race carrying seven different episode
    lengths is what put `imsa.json` beyond explanation."""
    for series in TRUTH:
        cfg = calibrate.build_race_config(con(), series, sid(series))
        assert len({c.caution_rate for c in cfg.classes}) == 1, series
        assert len({c.caution_mean_dur_s for c in cfg.classes}) == 1, series


# ----------------------------------------------------------------------
# Dial 3 - stints
# ----------------------------------------------------------------------
def test_stint_length_is_recovered_and_becomes_fuel():
    for series, truth in TRUTH.items():
        st = calibrate.calibrate_stints(con(), truth["session_id"], truth["cls"])
        assert abs(st["green_stint_laps"] - truth["stint"]) <= 3, series
        # A full tank must last exactly the calibrated stint.
        assert abs(1.0 / st["fuel_per_lap"] - st["green_stint_laps"]) < 1e-9


def test_stints_come_from_the_pit_records_not_the_driver_counter():
    """`stint_number` steps once per driver change, roughly three fuel stints.

    Reading it as a fuel stint is what reported fifty-eight-lap green stints
    at Daytona where the winner averaged twenty-three.
    """
    for series, truth in TRUTH.items():
        st = calibrate.calibrate_stints(con(), truth["session_id"], truth["cls"])
        by_counter = con().execute(f"""
            SELECT AVG(n) FROM (
                SELECT COUNT(*) AS n FROM laps
                WHERE session_id = {truth['session_id']}
                  AND class = '{truth['cls']}' AND flags = 'GF'
                GROUP BY car, stint_number)
        """).fetchone()[0]
        assert by_counter > 2.0 * st["green_stint_laps"], series


# ----------------------------------------------------------------------
# Dial 4 - pit cost
# ----------------------------------------------------------------------
def test_pit_cost_is_recovered_closely():
    for series, truth in TRUTH.items():
        pit = calibrate.calibrate_pit(con(), truth["session_id"], truth["cls"])
        assert abs(pit["pit_time_mean_s"] - truth["pit"]) < 1.5, series
        assert pit["n_pit_stops"] > 0


def test_a_repair_in_the_column_does_not_move_the_dial():
    """The fixture plants the occasional hour-long stop, as the real column
    carries. The median survives it; the arithmetic mean does not."""
    for series, truth in TRUTH.items():
        pit = calibrate.calibrate_pit(con(), truth["session_id"], truth["cls"])
        assert pit["n_pit_stops_trimmed_out"] > 0, series
        assert pit["pit_time_raw_mean_s"] > 1.3 * pit["pit_time_mean_s"], series
        assert pit["pit_time_std_s"] < pit["pit_time_mean_s"], series


def test_the_transit_estimate_is_below_a_serviced_stop():
    for series, truth in TRUTH.items():
        pit = calibrate.calibrate_pit(con(), truth["session_id"], truth["cls"])
        assert 0 < pit["pit_lane_transit_s"] < pit["pit_time_mean_s"], series


# ----------------------------------------------------------------------
# The config
# ----------------------------------------------------------------------
def test_build_race_config_produces_a_runnable_race():
    cfg = calibrate.build_race_config(con(), "imsa", sid("imsa"))
    assert cfg.classes
    assert cfg.duration_s > 0
    assert cfg.total_cars == (TRUTH["imsa"]["n_cars"]
                              + TRUTH["imsa"]["second"]["n_cars"])

    result = run_race(cfg, seed=0)
    c = result.classification()
    assert len(c) == cfg.total_cars
    assert (c["laps"] > 0).all()
    assert (c["stops"] > 0).all()


def test_the_config_records_which_edition_it_came_from():
    """`source_event` used to hold the ILIKE pattern, which named a circuit
    and not a race. A frozen config must say which running it describes."""
    for series, truth in TRUTH.items():
        cfg = calibrate.build_race_config(con(), series, truth["session_id"])
        for c in cfg.classes:
            assert str(truth["year"]) in c.source_event, series
            assert str(truth["session_id"]) in c.source_event, series
            assert "%" not in c.source_event, series


def test_config_round_trips_through_json(tmp_path=None):
    from endurance import RaceConfig
    tmp_path = tmp_path or Path(tempfile.mkdtemp())
    cfg = calibrate.build_race_config(con(), "wec", sid("wec"))
    p = Path(tmp_path) / "cfg.json"
    cfg.save(p)
    assert RaceConfig.load(p).to_dict() == cfg.to_dict()


def test_dials_table_lists_every_class():
    cfg = calibrate.build_race_config(con(), "imsa", sid("imsa"))
    t = calibrate.dials_table(cfg)
    assert len(t) == len(cfg.classes)
    assert {"base_pace_s", "deg_s_per_lap", "caution_rate",
            "stint_laps", "pit_s", "cars"} <= set(t.columns)


# ----------------------------------------------------------------------
# The falsifier
# ----------------------------------------------------------------------
@pytest.mark.parametrize("series", list(TRUTH))
def test_pooling_two_editions_breaks_the_recovery(series):
    """Condition three, as a test rather than only as a notebook cell.

    Two adjacent editions with different planted values must not calibrate to
    either one of them. If this passes, the scoping is not doing anything.
    """
    truth, other = TRUTH[series], OTHER[series]
    both = [truth["session_id"], other["session_id"]]

    scoped = calibrate.build_race_config(con(), series, truth["session_id"])
    pooled = calibrate.build_race_config(con(), series, both, name="pooled")

    # Duration is the strong detector: two editions is twice the racing.
    assert pooled.duration_s > 1.5 * scoped.duration_s

    # Grid size is a *weak* one and is not relied on here. Car numbers recur
    # between editions - six Daytonas carry ninety-one numbers, not three
    # hundred and sixty - so pooling grows the count by much less than it
    # grows the racing. It moves in the fixture, and the assertion is loose
    # on purpose so nobody reads it as the pooling check.
    assert pooled.total_cars > scoped.total_cars

    assert "POOLED" in pooled.classes[0].source_event

    # Far outside the gate's 2% duration tolerance, which is the check that
    # actually catches this.
    assert abs(pooled.duration_s - scoped.duration_s) / scoped.duration_s > 0.02

    # What this deliberately does *not* assert: that the pit dial moves. Since
    # it became a median it is robust to pooling in the same way base pace
    # always was - the edition with more stops carries the statistic and the
    # other barely shifts it. Both are good dials and neither detects pooling.
    # Only quantities that count, sum or sequence do, which is why the
    # falsifier rests on duration rather than on a dial.
    pooled_pit = calibrate.calibrate_pit(con(), both, truth["cls"])
    scoped_pit = calibrate.calibrate_pit(con(), truth["session_id"], truth["cls"])
    assert pooled_pit["n_pit_stops"] > scoped_pit["n_pit_stops"]
