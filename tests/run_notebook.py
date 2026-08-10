"""Execute a notebook's code cells against a replica of the real project.

    python run_notebook.py            # every target, in order
    python run_notebook.py 03b        # just this one
    python run_notebook.py 01 02a     # a couple

Builds a throwaway copy of the real project layout - data/raw, notebooks,
scripts, src, tests - and runs the notebook with the working folder set to
`notebooks/`, which is exactly where Jupyter puts you. That is what catches
path bugs: a notebook that only works when launched from the project root
will fail here.

Notebooks run in order into the same sandbox, because later ones read what
earlier ones write. Running one alone is still valid - the data-dependent
cells detect what is missing and say so - but it exercises less.

Targets match `build_nb.py`'s, so the two lists cannot drift apart without
somebody noticing.

What the sandbox has and what it deliberately does not
------------------------------------------------------
`scripts/freeze_assets.py` is *run* in the sandbox, because 02c, 03a and 03b
all resolve their race through `load_assets` and a sandbox without the three
frozen artefacts fails in Part 1 for a reason that has nothing to do with
the notebook. It is fast - two banks and two fields - and it exercises the
freezing path as a side effect.

Trained checkpoints are **copied if they exist and not produced**. Training
in a smoke test would take minutes and prove nothing about the notebook, and
the 03b notebook is written to degrade cleanly without them: the gate-two
cells print a skip line, the tables raise a message naming the command that
produces them. So a clean clone exercises the structure and a working tree
exercises the gates as well. Which of the two you got is printed at the end,
because "03b passed" means different things in the two cases.
"""

import json
import os
import runpy
import shutil
import sys
import tempfile
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from make_fixture import build  # noqa: E402
from sqlite_shim import SqliteDuckShim  # noqa: E402


NOTEBOOKS = {
    "01": "01_race_engine.ipynb",
    "02a": "02a_engine_corrections.ipynb",
    "02b": "02b_benchmark.ipynb",
    "02c": "02c_human_strategies.ipynb",
    "03a": "03a_gym_wrapper.ipynb",
    "03b": "03b_rl_training.ipynb",
}

# Targets that resolve their race through `scripts/freeze_assets.py`.
NEEDS_ASSETS = {"02c", "03a", "03b"}

wanted = sys.argv[1:] or list(NOTEBOOKS)
unknown = [t for t in wanted if t not in NOTEBOOKS]
if unknown:
    raise SystemExit(f"unknown target {unknown}; choose from {list(NOTEBOOKS)}")


# --- build a replica of the real project layout -------------------------
# In a temp folder, so repeated runs never leave anything behind in the repo.
sandbox = Path(tempfile.mkdtemp(prefix="endurance_nb_"))

build(sandbox)                                   # data/raw/*.csv
shutil.copytree(REPO / "src", sandbox / "src")   # src/endurance/
shutil.copytree(HERE, sandbox / "tests", dirs_exist_ok=True)
if (REPO / "scripts").exists():
    shutil.copytree(REPO / "scripts", sandbox / "scripts")
(sandbox / "notebooks").mkdir()

for name in NOTEBOOKS.values():
    src_nb = REPO / "notebooks" / name
    if src_nb.exists():
        shutil.copy(src_nb, sandbox / "notebooks" / name)

missing_nb = [t for t in wanted if not (sandbox / "notebooks" / NOTEBOOKS[t]).exists()]
if missing_nb:
    raise SystemExit(f"no notebook on disk for {missing_nb}; "
                     f"run `python build_nb.py {' '.join(missing_nb)}` first")

# Trained artefacts, if the working tree has any. Copied rather than built.
checkpoints = REPO / "outputs" / "policies"
have_checkpoints = False
if checkpoints.exists():
    shutil.copytree(checkpoints, sandbox / "outputs" / "policies")
    have_checkpoints = any((sandbox / "outputs" / "policies").glob("*.zip"))

evaluation = REPO / "outputs" / "evaluation"
have_tables = False
if evaluation.exists():
    shutil.copytree(evaluation, sandbox / "outputs" / "evaluation")
    have_tables = any((sandbox / "outputs" / "evaluation").glob("*_summary.csv"))

# Stand in for duckdb so calibrate.connect works unchanged.
duckdb_stub = types.ModuleType("duckdb")
duckdb_stub.connect = lambda: SqliteDuckShim()
sys.modules["duckdb"] = duckdb_stub

# Freeze the assets the later notebooks resolve their race through. Run as a
# script rather than reimplemented, so the sandbox gets the same three files
# a real tree gets and a fault in the freezing shows up here.
have_assets = False
if set(wanted) & NEEDS_ASSETS and (sandbox / "scripts" / "freeze_assets.py").exists():
    print("--- scripts/freeze_assets.py ---")
    argv, cwd = sys.argv[:], Path.cwd()
    try:
        sys.argv = ["freeze_assets.py"]
        os.chdir(sandbox)
        runpy.run_path(str(sandbox / "scripts" / "freeze_assets.py"),
                       run_name="__main__")
        have_assets = True
    except SystemExit as e:
        if e.code:
            raise
        have_assets = True
    finally:
        sys.argv, _ = argv, os.chdir(cwd)

# Run from notebooks/, the way Jupyter would.
os.chdir(sandbox / "notebooks")

spaces: dict = {}
for target in wanted:
    name = NOTEBOOKS[target]
    print(f"\n--- {name} ---")
    nb = json.loads((sandbox / "notebooks" / name).read_text())
    ns: dict = {}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            exec(compile(src, f"<{target} cell {i}>", "exec"), ns)
            plt.close("all")
            print(f"  ok   {target} cell {i:>2}")
        except Exception as e:
            print(f"  FAIL {target} cell {i:>2}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    spaces[target] = ns

print("\nALL NOTEBOOK CELLS EXECUTED OK (launched from notebooks/)")
print("sandbox was:", sandbox, "\n")

if "01" in spaces:
    ns = spaces["01"]
    pd = ns["pd"]
    print("dials:")
    print(pd.concat([ns["calibrate"].dials_table(c) for c in ns["configs"].values()],
                    ignore_index=True).to_string(index=False))
    print("\nvalidation vs real:")
    print(ns["checks"].to_string(index=False))
    print("\nlevers:")
    print(ns["lever_table"].to_string(index=False))
    print("\nstrategy comparison:")
    print(ns["comparison"].to_string(index=False))

if "02a" in spaces:
    ns = spaces["02a"]
    print("\nthe engine against the model:")
    print(ns["against_engine"].to_string(index=False))
    print("\nthe four corrections:")
    print(ns["corrections"].to_string(index=False))

if "03b" in spaces:
    # Say which of the two runs this was. "03b passed" means the gates held
    # if there were artefacts to hold them against, and means only that the
    # notebook survives their absence if there were not - and those are very
    # different claims to leave a reader to guess between.
    # Built outside the f-strings: a nested quote inside one only compiles
    # on 3.12, and this project targets 3.11.
    assets_note = "yes" if have_assets else "NO"
    ckpt_note = ("yes" if have_checkpoints
                 else "no - gate two and Part 7 were skipped")
    table_note = ("yes" if have_tables
                  else "no - Part 6 reported them missing")
    print("\n03b ran with:")
    print(f"  frozen assets:      {assets_note}")
    print(f"  trained checkpoint: {ckpt_note}")
    print(f"  evaluation tables:  {table_note}")
    if not have_checkpoints:
        print("  This run exercised the notebook's structure, not its gates.")
