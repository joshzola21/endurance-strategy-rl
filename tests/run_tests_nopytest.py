"""Minimal stand-in for pytest, for environments where pytest isn't installed.

Collects `test_*` functions from every test_*.py here, supplies a `tmp_path`
fixture and a `pytest.raises` context manager, and reports pass/fail.
`pytest tests/` is still the intended path - this exists so the suite can be
run anywhere.
"""

import contextlib
import importlib.util
import inspect
import sys
import tempfile
import traceback
import types
from pathlib import Path

# --- minimal pytest shim, installed before the test modules import it ---
pytest_stub = types.ModuleType("pytest")


@contextlib.contextmanager
def _raises(exc_type):
    try:
        yield
    except exc_type:
        return
    except Exception as e:
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(e).__name__}") from e
    raise AssertionError(f"expected {exc_type.__name__}, nothing raised")


pytest_stub.raises = _raises
pytest_stub.fixture = lambda *a, **k: (lambda f: f)
sys.modules.setdefault("pytest", pytest_stub)

HERE = Path(__file__).resolve().parent

tests = []
for test_file in sorted(HERE.glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(test_file.stem, test_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tests += [(f"{test_file.stem}::{n}", f) for n, f in vars(mod).items()
              if n.startswith("test_") and callable(f)
              and getattr(f, "__module__", None) == test_file.stem]

passed, failed = 0, []
for name, fn in tests:
    kwargs = {}
    if "tmp_path" in inspect.signature(fn).parameters:
        kwargs["tmp_path"] = Path(tempfile.mkdtemp())
    try:
        fn(**kwargs)
        passed += 1
        print(f"  PASS  {name}")
    except Exception:
        failed.append(name)
        print(f"  FAIL  {name}")
        print("        " + traceback.format_exc().replace("\n", "\n        ")[:1500])

print(f"\n{passed} passed, {len(failed)} failed out of {len(tests)}")
if failed:
    print("failed:", ", ".join(failed))
    sys.exit(1)
