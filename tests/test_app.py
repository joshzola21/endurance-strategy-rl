"""04's verification gates.

The blueprint gives 04 a boundary constraint and a feature list and no gate,
which is the condition 02c named as a gate nobody can fail and which 00 and
03b both had one supplied for. Two conditions, each with a falsifier, and
neither of them imports Streamlit - a gate that needs a browser is a gate
nobody runs.

**Gate A - the app steps the race everything else was measured on.** A race
driven through `RaceController`, with no human override, reproduces
`harness.run_focal`'s classification bit for bit. This is the failure only
this stage can produce: a UI that quietly steps the race differently from
the thing every number in 02c and 03b came from, and which shows a complete,
plausible race that nobody else would get. Its second half covers the one
optimisation the controller makes - that a seeked race equals a driven one,
so the decision log really is the race and the live generator really is only
a cache.

**Gate B - presentation only, mechanically.** The boundary constraint is a
sentence until something checks it. Three properties an import and an AST
walk can decide: the app steps a race in exactly one place, it never reaches
for the engine's private working, and the package does not import the app or
Streamlit. It cannot decide "no physics" in general; it can decide the ways
physics has actually arrived in this project before.
"""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Two inserts rather than the usual one, and this is the only test file that
# needs the first. Every other file imports `endurance` and puts `src` on the
# path for itself; this one also imports `app`, which lives at the project
# root. There is no `conftest.py` to do it - deliberately, because
# `run_tests_nopytest.py` has to work in an environment without pytest and a
# conftest would not run there - so the file does it, like its neighbours.
#
# `pip install -e .` makes the second insert redundant and the first still
# necessary: `app/` is an entry point rather than an installed package, which
# is the same reason `app/__init__.py` opens with a raw insert of the root.
ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from app.controller import RaceController, roster_seat  # noqa: E402
from endurance import harness  # noqa: E402

APP_DIR = ROOT / "app"

N_GATE = 5          # races per series; the fault this catches is not subtle
SERIES = ("imsa", "wec")
COLS = ("laps", "race_time_s", "stops", "pit_time_s", "traffic_time_s",
        "class_pos", "overall_pos", "caution_laps")


@pytest.fixture(scope="module")
def assets():
    """The frozen three, per series. Loaded once; nothing here writes."""
    from app.loading import load_assets

    out = {}
    for code in SERIES:
        a = load_assets(code)          # resolved, and checked to agree
        out[code] = (a.config, a.bank, a.field)
    return out


# ----------------------------------------------------------------------
# Gate A
# ----------------------------------------------------------------------
def _drive(cfg, field, seed, seat_name, log=None):
    ctrl = RaceController(config=cfg, field=field, seed=seed,
                          seat=roster_seat(seat_name), seat_name=seat_name,
                          log=dict(log or {}))
    result = ctrl.finish()
    row = result.classification().set_index("car_id").loc[ctrl.focal]
    return ctrl, {c: row[c] for c in COLS}


@pytest.mark.parametrize("code", SERIES)
def test_gate_a_controller_reproduces_the_harness(assets, code):
    """The app's race is the harness's race."""
    cfg, bank, field = assets[code]
    for seed in bank.headline[:N_GATE]:
        focal = harness.focal_car(cfg, seed)
        ctrl, mine = _drive(cfg, field, seed, "fuel_window")
        assert ctrl.focal == focal
        theirs = harness.run_focal(cfg, seed, focal,
                                   harness.ROSTER["fuel_window"](), field)
        for c in COLS:
            assert mine[c] == theirs[c], f"{code} seed {seed}: {c} differs"


@pytest.mark.parametrize("code", SERIES)
def test_gate_a_falsifier_two_seats_disagree(assets, code):
    """Or the comparison above is asserting that a race agrees with itself.

    `splash_and_dash` rather than the more obvious `caution_gambler`, and
    *every* seed rather than any: measured on the first five headline races
    of each series, the gambler and the null finish identically on two of
    five in WEC - art. 14.6.5 releases the whole field at once, so far fewer
    caution stops are reachable and the gambler frequently has nothing to
    do. A falsifier that is inert two times in five is barely a falsifier.
    The planner and the track-position defender both differ on five of five
    in both series.
    """
    cfg, bank, field = assets[code]
    for seed in bank.headline[:N_GATE]:
        _, a = _drive(cfg, field, seed, "fuel_window")
        _, b = _drive(cfg, field, seed, "splash_and_dash")
        assert any(a[c] != b[c] for c in COLS), \
            f"{code} seed {seed}: two different seats produced identical races"


@pytest.mark.parametrize("code", SERIES)
def test_gate_a_a_seeked_race_is_a_driven_race(assets, code):
    """The log is the race; the generator is only a cache.

    Stepped forward, seeked backwards, stepped again - and the finish has to
    be the one the straight drive produced. This is the half of the gate that
    covers the controller's own optimisation, which is the thing 04 added.
    """
    cfg, bank, field = assets[code]
    for seed in bank.headline[:N_GATE]:
        _, straight = _drive(cfg, field, seed, "caution_gambler")

        ctrl = RaceController(config=cfg, field=field, seed=seed,
                              seat=roster_seat("caution_gambler"),
                              seat_name="caution_gambler")
        ctrl.seek(40)
        ctrl.seek(12)          # backwards: a replay from lap zero
        ctrl.seek(55)
        result = ctrl.finish()
        row = result.classification().set_index("car_id").loc[ctrl.focal]
        for c in COLS:
            assert row[c] == straight[c], f"{code} seed {seed}: seek moved {c}"


@pytest.mark.parametrize("code", SERIES)
def test_gate_a_falsifier_an_override_changes_the_race(assets, code):
    """A log that is ignored would make the test above pass trivially."""
    cfg, bank, field = assets[code]
    seed = bank.headline[0]
    _, plain = _drive(cfg, field, seed, "fuel_window")
    # Stop on every one of the first forty laps. Whatever else this is, it is
    # not the fuel-window plan.
    _, forced = _drive(cfg, field, seed, "fuel_window",
                       log={lap: 1 for lap in range(1, 41)})
    assert any(plain[c] != forced[c] for c in COLS), \
        f"{code}: the decision log changed nothing"


def test_gate_a_state_round_trips(assets):
    """A session restored from JSON is stepping the same race.

    Streamlit re-executes on every interaction and drops anything it cannot
    hold, so this is the path the app is actually on after a hot reload.
    """
    cfg, bank, field = assets["imsa"]
    seed = bank.headline[0]
    ctrl = RaceController(config=cfg, field=field, seed=seed,
                          seat=roster_seat("splash_and_dash"),
                          seat_name="splash_and_dash")
    ctrl.seek(30)
    ctrl.step(action=1)
    ctrl.seek(60)

    restored = RaceController.from_dict(ctrl.to_dict(), cfg, field,
                                        roster_seat("splash_and_dash"))
    assert restored.lap == ctrl.lap
    assert restored.log == ctrl.log
    assert restored.frame.t == ctrl.frame.t
    assert restored.frame.car.fuel == ctrl.frame.car.fuel
    assert restored.frame.obs.tolist() == ctrl.frame.obs.tolist()


def test_gate_a_state_refuses_a_different_race(assets):
    """The fingerprint on the session is a tripwire, not a note."""
    cfg, bank, field = assets["imsa"]
    ctrl = RaceController(config=cfg, field=field, seed=bank.headline[0],
                          seat=roster_seat("fuel_window"))
    payload = ctrl.to_dict()
    payload["dials_fingerprint"] = "0" * 16
    with pytest.raises(ValueError, match="different race"):
        RaceController.from_dict(payload, cfg, field, roster_seat("fuel_window"))


# ----------------------------------------------------------------------
# Gate B
# ----------------------------------------------------------------------
# How a race gets stepped. Allowed in the controller and nowhere else, so
# that "there is exactly one race loop" survives contact with a UI.
_STEPPING = {"RaceEngine", "run_race", "run_stream"}

# The engine's working. An app module reaching for any of these is computing
# something it should have read.
_PRIVATE = {"_next_lap", "_apply_pit", "_must_pit", "_progress", "_take_wave",
            "_caution_lap", "_build_field", "_set_lap", "_seed_streams",
            "stop_cost", "lane_status", "_traffic_penalty", "_queue_ahead"}

# Charts come from viz.py unchanged; a chart built in the app is one the
# notebooks cannot use.
_CHARTING = {"matplotlib", "plotly", "altair"}

_STEPPING_ALLOWED = {"controller.py"}


def scan(source: str, filename: str) -> list[str]:
    """What this file does that a presentation layer may not.

    An AST walk rather than a substring search: `# never call run_race` is
    not a violation and `getattr(engine, "run_" + "race")` is not caught by
    either, which is the honest limit of a check like this. It decides the
    ways physics has actually arrived in this project before, which is what
    a regression gate is for.
    """
    problems = []
    tree = ast.parse(source, filename=filename)
    allowed = Path(filename).name in _STEPPING_ALLOWED

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _PRIVATE:
            problems.append(f"{filename}: reaches for {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in _PRIVATE:
            problems.append(f"{filename}: reaches for .{node.attr}")
        elif isinstance(node, ast.Name) and node.id in _STEPPING and not allowed:
            problems.append(f"{filename}: steps a race via {node.id}; "
                            f"only the controller may")
        elif isinstance(node, ast.Attribute) and node.attr in _STEPPING and not allowed:
            problems.append(f"{filename}: steps a race via .{node.attr}; "
                            f"only the controller may")

        root = None
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in _CHARTING:
                    problems.append(f"{filename}: imports {root}; charts "
                                    f"belong in viz.py")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in _CHARTING:
                problems.append(f"{filename}: imports {root}; charts belong "
                                f"in viz.py")

    return problems


def test_gate_b_the_app_is_presentation_only():
    """Every module under `app/`, walked."""
    files = sorted(APP_DIR.rglob("*.py"))
    assert files, f"no app modules found under {APP_DIR}"
    problems = []
    for path in files:
        problems += scan(path.read_text(), str(path.relative_to(APP_DIR.parent)))
    assert not problems, "04's boundary constraint is broken:\n" + "\n".join(problems)


def test_gate_b_falsifier_the_scanner_catches_all_three():
    """A scanner that finds nothing passes on an app that does everything."""
    bad = ("import matplotlib.pyplot as plt\n"
           "from endurance.engine import RaceEngine\n"
           "def go(cfg, car, cls):\n"
           "    return RaceEngine(cfg)._next_lap(car, 0.0)\n")
    found = scan(bad, "pages/bad.py")
    assert any("matplotlib" in p for p in found)
    assert any("RaceEngine" in p for p in found)
    assert any("_next_lap" in p for p in found)
    # And the controller is allowed the one it is allowed.
    assert not any("RaceEngine" in p for p in scan(bad, "app/controller.py"))


def _subprocess_env() -> dict:
    """A child that can find the tree, whether or not it has been installed.

    Gate B asks what `import endurance` drags in, and it has to ask in a fresh
    interpreter because this session has already imported both. That child
    starts with the working directory on its path and nothing else, so in an
    uninstalled tree it cannot find `endurance` at all and the gate fails for
    a reason that has nothing to do with what it is testing.

    Both paths go in on purpose. Making `app` *importable* is what gives the
    gate its teeth: it then asserts that importing the package does not pull
    in an app that was sitting right there to be pulled in.
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src")] + ([existing] if existing else []))
    return env


def test_gate_b_the_package_does_not_import_the_app():
    """`import endurance` must not drag in Streamlit.

    In a subprocess, because this test session has already imported both and
    `sys.modules` in here would answer a different question.
    """
    code = ("import sys, importlib; importlib.import_module('endurance'); "
            "bad = [m for m in sys.modules "
            "if m.split('.')[0] in {'streamlit', 'app'}]; "
            "print(','.join(sorted(bad)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True, env=_subprocess_env())
    assert not out.stdout.strip(), \
        f"importing endurance pulled in {out.stdout.strip()}"


def test_gate_b_the_controller_needs_no_streamlit():
    """The gates run headless because the stepping does."""
    code = ("import sys, importlib; importlib.import_module('app.controller'); "
            "print('streamlit' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True, env=_subprocess_env())
    assert out.stdout.strip() == "False"
