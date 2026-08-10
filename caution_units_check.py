"""Does the engine reproduce the caution load it was calibrated to?

`calibrate_cautions` measures cautions in *laps* off the reference car's flag
column. `CautionTimeline.draw` consumes those numbers as *seconds of race
time*. A caution lap takes `caution_pace_multiplier` times as long as a green
one, so the two quantities are not the same thing and the conversion between
them is missing.

This script measures the size of the gap rather than arguing about it: it
runs the engine on a set of dials, then asks the finished race the same
question notebook 00 asked the data - what share of this car's laps were run
under caution - and compares that with the share that went in.

No lap data is needed. The dials below are plausible Daytona GTP figures,
not measured ones; the point is the ratio between input and output, which
does not depend on them being exactly right.
"""

import sys

sys.path.insert(0, "/home/claude/src")

import numpy as np
import pandas as pd

from endurance import ClassDials, RaceConfig, run_race
from endurance.engine import CautionTimeline


# Plausible Daytona 24 GTP dials. Substitute the real frozen ones from
# data/processed/imsa.json when running this for the record.
CAUTION_LAP_SHARE = 0.30       # what calibrate_cautions would return
CAUTION_EPISODE_LAPS = 8.0     # mean episode length, in laps
BASE_PACE_S = 100.0
MULTIPLIER = 1.6               # caution_pace_multiplier, as assumed


def make_config(caution_rate: float, caution_mean_dur_s: float) -> RaceConfig:
    dials = ClassDials(
        series_code="imsa",
        class_name="GTP",
        base_pace_s=BASE_PACE_S,
        deg_slope_s_per_lap=0.02,
        pace_spread_s=0.4,
        lap_noise_s=0.8,
        caution_rate=caution_rate,
        caution_mean_dur_s=caution_mean_dur_s,
        caution_pace_multiplier=MULTIPLIER,
        green_stint_laps=32.0,
        fuel_per_lap=1.0 / 32.0,
        fuel_per_lap_caution=0.6 / 32.0,
        tyre_life_laps=64.0,
        pit_time_mean_s=45.0,
        pit_time_std_s=3.0,
        n_cars=10,
    )
    return RaceConfig(name="Daytona 24", series_code="imsa",
                      duration_s=24 * 3600.0, classes=[dials])


def realised(config: RaceConfig, seeds=range(12)) -> pd.DataFrame:
    """Run the engine and measure what actually came out."""
    rows = []
    for seed in seeds:
        res = run_race(config, seed=seed)
        laps = res.laps
        # The same measurement calibrate_cautions makes on the data: one
        # reference car, share of its lap records flagged non-green.
        ref = laps["car_id"].value_counts().idxmax()
        ref_laps = laps[laps["car_id"] == ref]

        rows.append({
            "seed": seed,
            "lap_share": ref_laps["under_caution"].mean(),
            "time_share": res.cautions.total_caution_s() / config.duration_s,
            "episodes": len(res.cautions.periods),
            "mean_episode_s": np.mean([e - s for s, e in res.cautions.periods])
                              if res.cautions.periods else 0.0,
        })
    return pd.DataFrame(rows)


def analytic_lap_share(time_share: float, m: float) -> float:
    """Lap share implied by a given time share, at caution multiplier m."""
    return (time_share / m) / ((time_share / m) + (1.0 - time_share))


def analytic_time_share(lap_share: float, m: float) -> float:
    """Time share implied by a given lap share - the missing conversion."""
    return (lap_share * m) / (lap_share * m + (1.0 - lap_share))


if __name__ == "__main__":
    pd.set_option("display.width", 100)

    # --- as calibrate.py feeds it today ------------------------------------
    dur_s_now = CAUTION_EPISODE_LAPS * BASE_PACE_S          # green pace used
    cfg_now = make_config(CAUTION_LAP_SHARE, dur_s_now)
    now = realised(cfg_now)

    print("What calibrate_cautions hands the engine today")
    print(f"  caution_rate        {CAUTION_LAP_SHARE:.3f}   "
          f"(a share of LAPS, consumed as a share of TIME)")
    print(f"  caution_mean_dur_s  {dur_s_now:.0f} s   "
          f"({CAUTION_EPISODE_LAPS:.0f} laps x green pace {BASE_PACE_S:.0f} s)")
    print()
    print("What the engine then produces, over 12 seeds")
    print(now.describe().loc[["mean", "std"]].round(3).to_string())
    print()

    print(f"  observed caution lap share (target)  {CAUTION_LAP_SHARE:.3f}")
    print(f"  simulated caution lap share          {now['lap_share'].mean():.3f}"
          f"   <- Part 6 of notebook 01 compares these two")
    print(f"  ratio                                "
          f"{now['lap_share'].mean() / CAUTION_LAP_SHARE:.2f}x")
    print()
    print(f"  analytic prediction for the lap share: "
          f"{analytic_lap_share(CAUTION_LAP_SHARE, MULTIPLIER):.3f}")
    print()

    # --- with the conversion put in ----------------------------------------
    target_time_share = analytic_time_share(CAUTION_LAP_SHARE, MULTIPLIER)
    dur_s_fixed = CAUTION_EPISODE_LAPS * BASE_PACE_S * MULTIPLIER
    cfg_fixed = make_config(target_time_share, dur_s_fixed)
    fixed = realised(cfg_fixed)

    print("With the lap->time conversion applied")
    print(f"  caution_rate        {target_time_share:.3f}  (share of time)")
    print(f"  caution_mean_dur_s  {dur_s_fixed:.0f} s  "
          f"({CAUTION_EPISODE_LAPS:.0f} caution laps at caution pace)")
    print()
    print(fixed.describe().loc[["mean", "std"]].round(3).to_string())
    print()
    print(f"  simulated caution lap share          {fixed['lap_share'].mean():.3f}"
          f"   vs target {CAUTION_LAP_SHARE:.3f}")
    print()

    # --- what it does to episode structure ---------------------------------
    print("Episode structure, which is what a strategy actually reacts to")
    print(f"  episodes per race    {now['episodes'].mean():.1f}  ->  "
          f"{fixed['episodes'].mean():.1f}")
    print(f"  mean episode length  {now['mean_episode_s'].mean():.0f} s  ->  "
          f"{fixed['mean_episode_s'].mean():.0f} s")
    print(f"  episode in caution laps  "
          f"{now['mean_episode_s'].mean() / (BASE_PACE_S * MULTIPLIER):.1f}  ->  "
          f"{fixed['mean_episode_s'].mean() / (BASE_PACE_S * MULTIPLIER):.1f}"
          f"   (observed: {CAUTION_EPISODE_LAPS:.0f})")
