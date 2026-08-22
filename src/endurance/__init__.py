"""Endurance race strategy sandbox.

An interactive model of WEC and IMSA racing: real levers you can twist
(cautions, stint length, fuel, traffic), a whole field of cars, and a common
interface that human-style strategies and a reinforcement learning agent
both plug into on equal terms.

Layout
------
`params`      the five dials, as data, plus the lever mechanism
`calibrate`   turns real lap data into those dials
`engine`      runs the race
`pitstop`     what a stop costs, and whether the lane is open
`caution`     what a caution does to the field
`strategies`  how a car decides when to stop - background and roster
`benchmark`   the per-race reference, clairvoyant and causal
`assets`      the frozen seed banks and background field
`harness`     the paired comparison, the sweeps and 02c's gate
`viz`         race charts and comparison charts
`gym_env`     the gymnasium adapter, and nothing else

Typical use:

    from endurance import calibrate, run_race

    con = calibrate.connect("data/raw/laps.csv")
    cfg = calibrate.build_race_config(con, "imsa", "%daytona%", "Daytona 24")
    result = run_race(cfg, seed=0)
    result.classification()
"""

from .params import ClassDials, RaceConfig, scale_dials, set_dials
from .pitstop import PitRules, lane_status, stop_cost
from .engine import (
    CarState,
    CautionTimeline,
    Compat,
    PitDecision,
    RaceEngine,
    RaceResult,
    RaceState,
    run_race,
)
from .strategies import (
    BASELINES,
    ROSTER,
    CautionGambler,
    FixedLapStint,
    LapDownDefender,
    NeverPit,
    OpportunistUnderCaution,
    RunToFuelWindow,
    SplashAndDashPlanner,
    TrackPositionDefender,
    assign_strategy,
    fuel_to_the_flag,        # NEW in 03a: shared by the planner and the agent
)
from .harness import (
    Comparison,
    NullRuns,
    attach_benchmark,
    compare_roster,
    focal_car,
    headline_class,
    null_is_the_null,
    rotate_pace_rank,
    summarise,
    sweep_dial,
    sweep_grid,
)
from .gym_env import (
    EnduranceEnv,
    PolicyStrategy,
    action_mask,
    observe,
    to_decision,
)

__all__ = [
    "ClassDials", "RaceConfig", "scale_dials", "set_dials",
    "PitRules", "lane_status", "stop_cost",
    "CarState", "CautionTimeline", "Compat", "PitDecision",
    "RaceEngine", "RaceResult", "RaceState", "run_race",
    "BASELINES", "FixedLapStint", "OpportunistUnderCaution",
    "RunToFuelWindow", "assign_strategy",
     # 02c: the roster
    "ROSTER", "CautionGambler", "TrackPositionDefender",
    "SplashAndDashPlanner", "LapDownDefender",
    # amendment 25: the null for a learner, not a plan
    "NeverPit",
    # 02c: the comparison
    "Comparison", "NullRuns", "compare_roster", "summarise",
    "focal_car", "headline_class", "null_is_the_null",
    "sweep_dial", "sweep_grid", "rotate_pace_rank", "attach_benchmark",
    # 03a: the gym wrapper
    "EnduranceEnv", "PolicyStrategy",
    "observe", "action_mask", "to_decision",
    # 03a: lifted out of SplashAndDashPlanner so the agent can ask it too
    "fuel_to_the_flag",
]
