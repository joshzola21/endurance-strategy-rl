"""Fingerprint the engine as it stands, before 02a changes it.

Run this against the pre-change engine; paste the output into
test_cautions.py's GOLDEN_01. That makes the legacy gate a comparison
against the old engine rather than against the new engine's own opinion
of itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from endurance import ClassDials, RaceConfig, run_race  # noqa: E402


def small_config(duration_s=3600.0, caution_rate=0.1):
    fast = ClassDials(
        series_code="test", class_name="FAST",
        base_pace_s=100.0, deg_slope_s_per_lap=0.02,
        pace_spread_s=0.8, lap_noise_s=0.3,
        caution_rate=caution_rate, caution_mean_dur_s=300.0,
        green_stint_laps=20.0, fuel_per_lap=1 / 20, fuel_per_lap_caution=0.6 / 20,
        tyre_life_laps=40.0, pit_time_mean_s=40.0, pit_time_std_s=2.0, n_cars=6,
    )
    slow = ClassDials(
        series_code="test", class_name="SLOW",
        base_pace_s=115.0, deg_slope_s_per_lap=0.03,
        pace_spread_s=1.0, lap_noise_s=0.4,
        caution_rate=caution_rate, caution_mean_dur_s=300.0,
        green_stint_laps=22.0, fuel_per_lap=1 / 22, fuel_per_lap_caution=0.6 / 22,
        tyre_life_laps=44.0, pit_time_mean_s=42.0, pit_time_std_s=2.0, n_cars=5,
    )
    return RaceConfig(name="Test 1h", series_code="test",
                      duration_s=duration_s, classes=[fast, slow])


def fingerprint(result) -> tuple:
    """Laps, stops and finishing time per car, in classification order.

    Times are rounded to a millisecond: enough to catch a changed draw,
    loose enough not to trip on the last bit of a float.
    """
    c = result.classification()
    return tuple(
        (row.car_id, int(row.laps), int(row.stops), round(float(row.race_time_s), 3))
        for row in c.itertuples()
    )


if __name__ == "__main__":
    print("GOLDEN_01 = {")
    for seed in (0, 3, 7):
        fp = fingerprint(run_race(small_config(), seed=seed,
                                  legacy_cautions=True, split_streams=False))
        print(f"    {seed}: {fp!r},")
    print("}")
