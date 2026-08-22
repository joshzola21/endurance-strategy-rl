"""04's presentation layer.

`app` is a package so the gates can `from app.controller import ...` without
a browser, and so the pages import each other by a name that means one
thing. It re-exports nothing on purpose: pulling `loading` in here would
drag Streamlit into `import app`, and gate B checks that it does not.

Two `sys.path` problems, fixed in two places
--------------------------------------------
`streamlit run app/Home.py` puts **the entry script's own directory** on
`sys.path` and nothing else. That breaks two imports, and they have to be
fixed in different places because the first one gates the second.

*`app` itself.* Nothing here can fix it - importing this file is the thing
that fails - so each entry point opens with the raw insert of the project
root. Four files, two lines each.

*`endurance`.* The package lives under `src/`, which a notebook or a `pytest`
run puts on the path for itself and `streamlit run` does not. By the time
this file executes the root is on the path, so it can be fixed here, once,
for every module in the app.

**The permanent fix is `pip install -e .`**, which puts `endurance` on the
path for every entry point including this one. The insert below is what
makes the app work in a tree where that has not been done; it is deliberately
conditional, so an installed package always wins and this cannot shadow it
with a stale working copy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

if importlib.util.find_spec("endurance") is None:      # not installed
    _src = Path(__file__).resolve().parents[1] / "src"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
