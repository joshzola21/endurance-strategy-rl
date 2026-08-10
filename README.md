# Endurance race strategy sandbox

An interactive model of WEC and IMSA racing. Twist the real levers —
cautions, stint length, fuel, traffic — and watch a 24-hour race change.
Human-style strategies and a reinforcement learning agent plug into the
same race on equal terms.

## Getting started

Unzip this into your project folder, then:

```bash
pip install -r requirements.txt
```

Put `laps.csv` and `drivers.csv` in `data/raw/`, then open
`notebooks/01_race_engine.ipynb` and run it top to bottom.

```bash
pytest tests/          # or: python tests/run_tests_nopytest.py
```

33 tests should pass. If anything goes wrong, see `SETUP.md`.

## Using the engine directly

```python
import sys; sys.path.insert(0, "src")
from endurance import calibrate, run_race, scale_dials

con = calibrate.connect("data/raw/laps.csv")
cfg = calibrate.build_race_config(con, "imsa", "%daytona%", "Daytona 24")

result = run_race(cfg, seed=0)
result.classification()

# Twist a lever: three times as many cautions, everything else held fixed.
chaotic = run_race(scale_dials(cfg, caution_rate=3.0), seed=0)
```

A full 24-hour race with a 62-car field runs in about half a second.

## Layout

| Path | What it is |
|---|---|
| `notebooks/00_data_recon.ipynb` | first look at real lap data; where the dials came from |
| `notebooks/01_race_engine.ipynb` | calibrate, run a race, watch it, twist the levers, validate |
| `src/endurance/params.py` | the five dials as data; `scale_dials()` is the lever mechanism |
| `src/endurance/calibrate.py` | real lap data → dials |
| `src/endurance/engine.py` | the race itself |
| `src/endurance/strategies.py` | the strategy interface and starter baselines |
| `src/endurance/viz.py` | race charts |
| `tests/` | invariants, calibration checks, a notebook smoke run |
| `build_nb.py` | regenerates the notebook — edit this, not the `.ipynb` |
| `SETUP.md` | folder layout and troubleshooting |
| `HANDOVER.md` | current state, decisions taken, what to do next |

## The five dials

From notebook 00: a degradation slope, a caution rate and duration, a stint
length (held as fuel), a pit cost, and a traffic density — one set per class
per series. `calibrate.py` derives each from timing data; anything it cannot
measure is listed in `params.ASSUMED_FIELDS` and shown as assumed in the
notebook.

## Status

00 and 01 are done. Next is 02 (human-style strategies), then 03 (the RL
agent), then the Streamlit app. See `HANDOVER.md`.
