"""A DuckDB-shaped connection backed by sqlite3.

Only here so `calibrate.py` can be exercised in environments without DuckDB.
It runs the real SQL text from that module, so it catches genuine query
mistakes rather than re-implementing the logic and testing the
re-implementation. It supports the small amount of DuckDB the calibration
actually uses: `read_csv_auto` views, `ILIKE`, and `STDDEV`.

`car` is read as text here for the same reason `connect` forces it to VARCHAR
in DuckDB: `#7` and `#007` are two cars and an integer parse makes them one.
The type override in the real SQL text is a `read_csv_auto` argument the
regex below tolerates and cannot act on, so it is applied here instead. If
these two ever disagree the tests pass against a car identity the production
path does not have, which is the exact shape of the fault this re-run fixed.
"""

import fnmatch
import re
import sqlite3

import numpy as np
import pandas as pd

# Columns whose text form carries information an inferred dtype would destroy.
TEXT_COLUMNS = ("car",)


class _Stddev:
    def __init__(self):
        self.vals = []

    def step(self, value):
        if value is not None:
            self.vals.append(value)

    def finalize(self):
        return float(np.std(self.vals, ddof=1)) if len(self.vals) > 1 else None


def _ilike(value, pattern):
    if value is None:
        return 0
    return 1 if fnmatch.fnmatch(str(value).lower(),
                                pattern.replace("%", "*").lower()) else 0


def _read_csv(path: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    dtypes = {c: "string" for c in TEXT_COLUMNS if c in header.columns}
    return pd.read_csv(path, dtype=dtypes or None)


class _Result:
    def __init__(self, rows, columns):
        self._rows, self._cols = rows, columns

    def df(self):
        return pd.DataFrame(self._rows, columns=self._cols)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class SqliteDuckShim:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.create_aggregate("STDDEV", 1, _Stddev)
        self.conn.create_function("ILIKE_MATCH", 2, _ilike)

    def execute(self, sql: str) -> _Result:
        m = re.match(
            r"\s*CREATE VIEW (\w+)\s+AS SELECT \* FROM read_csv_auto\("
            r"'([^']+)'[^)]*\)",   # tolerate quote=/escape=/types= and friends
            sql, re.IGNORECASE)
        if m:
            _read_csv(m.group(2)).to_sql(m.group(1), self.conn,
                                         index=False, if_exists="replace")
            return _Result([], [])

        sql = re.sub(r"(\w+)\s+ILIKE\s+'([^']*)'", r"ILIKE_MATCH(\1, '\2')", sql)
        cur = self.conn.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return _Result(rows, cols)


def connect_fixture(laps_csv: str) -> SqliteDuckShim:
    con = SqliteDuckShim()
    con.execute(f"CREATE VIEW laps AS SELECT * FROM read_csv_auto('{laps_csv}')")
    return con
