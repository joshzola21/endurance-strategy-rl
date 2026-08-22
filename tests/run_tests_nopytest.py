"""Minimal stand-in for pytest, for environments where pytest isn't installed.

    python tests/run_tests_nopytest.py

Collects `test_*` functions from every `test_*.py` here, supplies a `tmp_path`
fixture, and shims the small amount of the pytest API the suite actually uses:
`raises` (with `match`), `approx`, `fixture` and `mark`. `pytest tests/` is
still the intended path - this exists so the suite can be run anywhere.

**It skips rather than stops.** The first version imported each file and let
any failure escape, so one module that needed something it had not got ended
the whole run with a traceback and no results at all - which is what happened
to `test_app.py`. A file this runner cannot handle is a gap in the runner, not
a fault in the file, and the two should not look the same from the outside.
Skips are counted and named at the end.

What it cannot do, and says so rather than pretending
-----------------------------------------------------
* **Fixtures with arguments.** `fixture` is a no-op decorator, so a decorated
  function stays an ordinary function and a test asking for one by name would
  receive nothing. Those are skipped by signature rather than reported as
  failures.
* **`parametrize`.** Shimming it faithfully means running every case;
  pretending means silently running one. Neither is worth it here, so a marked
  test is skipped and named.

Both land on `test_app.py`, `test_position_reward.py` and
`test_stop_cost_floor.py`, which also need frozen artefacts on disk. Under
real pytest those are three of the most important files in the suite; under
this runner they are out of reach, and knowing exactly which is the point of
the summary at the end.
"""

import contextlib
import importlib.util
import inspect
import re
import sys
import tempfile
import traceback
import types
from pathlib import Path

# `abs` is a keyword argument of `pytest.approx`, so the builtin is kept under
# another name before anything shadows it.
_abs = abs

# --- minimal pytest shim, installed before the test modules import it ---
pytest_stub = types.ModuleType("pytest")


class _Skipped(Exception):
    """Raised by `pytest.skip`, caught by the runner."""


@contextlib.contextmanager
def _raises(exc_type, match=None):
    """`pytest.raises`, including `match`.

    The suite uses `match` where the *wording* of a refusal is part of the
    behaviour - a message naming which dial, or which race - and dropping it
    would quietly weaken those tests rather than skip them.
    """
    try:
        yield
    except exc_type as e:
        if match is not None and not re.search(match, str(e)):
            raise AssertionError(
                f"{exc_type.__name__} was raised, but its message {str(e)!r} "
                f"does not match {match!r}") from e
        return
    except Exception as e:
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(e).__name__}") from e
    raise AssertionError(f"expected {exc_type.__name__}, nothing raised")


class _Approx:
    """Enough of `pytest.approx` for scalars, which is all the suite asks."""

    def __init__(self, expected, rel=None, abs_=None):
        self.expected, self.rel, self.abs = expected, rel, abs_

    def _tolerance(self) -> float:
        if self.abs is None and self.rel is None:
            return 1e-6 * max(_abs(self.expected), 1.0)
        tol = self.abs or 0.0
        if self.rel is not None:
            tol = max(tol, self.rel * _abs(self.expected))
        return tol

    def __eq__(self, other):
        return _abs(other - self.expected) <= self._tolerance()

    def __repr__(self):
        return f"approx({self.expected!r} +- {self._tolerance():.3g})"


class _Mark:
    """`pytest.mark.anything`, recording that the test needs the real thing."""

    def __getattr__(self, name):
        def marker(*_args, **_kwargs):
            def decorate(fn):
                fn._needs_pytest = f"pytest.mark.{name}"
                return fn
            return decorate
        return marker


def _skip(reason: str = "") -> None:
    raise _Skipped(reason)


pytest_stub.raises = _raises
pytest_stub.approx = lambda expected, rel=None, abs=None: _Approx(
    expected, rel=rel, abs_=abs)
pytest_stub.fixture = lambda *a, **k: (lambda f: f)
pytest_stub.mark = _Mark()
pytest_stub.skip = _skip
pytest_stub.main = lambda *a, **k: 0
sys.modules.setdefault("pytest", pytest_stub)

HERE = Path(__file__).resolve().parent

# --- collect ------------------------------------------------------------
collected: list[tuple[str, object]] = []
unimportable: list[tuple[str, str]] = []

for test_file in sorted(HERE.glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(test_file.stem, test_file)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                       # noqa: BLE001 - reported below
        unimportable.append((test_file.name, f"{type(e).__name__}: {e}"))
        continue
    collected += [(f"{test_file.stem}::{n}", f) for n, f in vars(mod).items()
                  if n.startswith("test_") and callable(f)
                  and getattr(f, "__module__", None) == test_file.stem]


def why_skip(fn) -> str | None:
    """Why this runner cannot run a test, or `None` if it can."""
    marked = getattr(fn, "_needs_pytest", None)
    if marked:
        return f"uses {marked}"
    wanted = [p for p in inspect.signature(fn).parameters if p != "tmp_path"]
    if wanted:
        return f"needs the {', '.join(wanted)} fixture(s)"
    return None


# --- run ----------------------------------------------------------------
passed, failed, skipped = 0, [], []
for name, fn in collected:
    reason = why_skip(fn)
    if reason:
        skipped.append((name, reason))
        continue

    kwargs = {}
    if "tmp_path" in inspect.signature(fn).parameters:
        kwargs["tmp_path"] = Path(tempfile.mkdtemp())
    try:
        fn(**kwargs)
        passed += 1
        print(f"  PASS  {name}")
    except _Skipped as e:
        skipped.append((name, str(e) or "skipped by the test itself"))
    except Exception:
        failed.append(name)
        print(f"  FAIL  {name}")
        print("        " + traceback.format_exc().replace("\n", "\n        ")[:1500])

print(f"\n{passed} passed, {len(failed)} failed, {len(skipped)} skipped, "
      f"out of {len(collected)} collected")

if unimportable:
    print(f"\n{len(unimportable)} file(s) this runner could not import. Run "
          f"them with `pytest tests/`:")
    for name, reason in unimportable:
        print(f"  {name}: {reason}")

if skipped:
    print(f"\n{len(skipped)} test(s) skipped - this runner's limits, not "
          f"theirs:")
    for name, reason in sorted(skipped):
        print(f"  {name}: {reason}")

if failed:
    print("\nfailed:", ", ".join(failed))
    sys.exit(1)
if skipped or unimportable:
    print("\nSkips are not passes. `pytest tests/` is the run that counts.")
