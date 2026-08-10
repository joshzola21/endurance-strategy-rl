# Setting this up on your Mac

## Just unzip it

The zip contains a folder called `endurance-strategy-rl/` laid out exactly
like your project. Unzip it next to your existing folder and merge, or
unzip it somewhere else and drag the pieces across. Nothing in it will
overwrite your data or your `00_data_recon.ipynb`.

```
endurance-strategy-rl/
├── data/
│   ├── raw/                    put laps.csv, drivers.csv here
│   └── processed/              calibrated dials get written here
├── notebooks/
│   └── 01_race_engine.ipynb    (00_data_recon.ipynb is yours, keep it)
├── outputs/
│   └── figures/
├── src/
│   └── endurance/              the package - all six files matter
│       ├── __init__.py
│       ├── params.py
│       ├── calibrate.py
│       ├── engine.py
│       ├── strategies.py
│       └── viz.py
├── tests/
├── build_nb.py
├── requirements.txt
├── README.md
├── HANDOVER.md
└── SETUP.md
```

Then:

```bash
cd endurance-strategy-rl
pip install -r requirements.txt
pytest tests/
```

33 tests should pass. Then open `notebooks/01_race_engine.ipynb`.

## What went wrong last time

`ModuleNotFoundError: No module named 'endurance'` had two causes:

1. **The package was never on your machine.** I handed over only some of
   the files, and the package needs all six. That was my mistake.
2. **The notebook looked in the wrong place.** It used
   `Path.cwd() / "src"`, and since the notebook runs from `notebooks/`,
   that resolved to `notebooks/src`, which does not exist. The data paths
   had the same problem.

The notebook now walks up the folder tree until it finds `src/endurance`,
so it works from `notebooks/`, from the project root, or from wherever your
editor decides to start. There is a test (`tests/run_notebook.py`) that
runs the notebook from a `notebooks/` folder specifically to catch this
class of bug.

## If the import still fails

The first cell prints the project root it found. If that line does not
appear, or points somewhere unexpected, the layout is not what the notebook
expects — check that `src/endurance/__init__.py` exists at the path it
names.

Two things worth knowing:

- **Restart the kernel** after moving files about. Jupyter caches
  `sys.path` and imported modules, so an old failed import can persist even
  once the files are right.
- `src/endurance/` must contain `__init__.py`. Without it Python will not
  treat the folder as a package.

If you would rather not rely on the path trick at all, the alternative is
to install the package into your environment with `pip install -e .`, which
needs a `pyproject.toml` this project does not have yet. The root-finding
approach was chosen because it needs no install step — say the word and I
will add proper packaging instead.

## A note on the figures

`SAVE_FIGURES = False` in the settings cell. Set it to `True` and every
chart is written to `outputs/figures/` as well as shown inline, which is
what that folder was for.

## Running the tests without pytest

```bash
python tests/run_tests_nopytest.py    # the unit tests
python tests/run_notebook.py          # executes the notebook end to end
```

Both work without pytest or DuckDB installed — the second builds a
synthetic `laps.csv` and stands in for DuckDB with sqlite, so it exercises
the real SQL without needing the real data.
