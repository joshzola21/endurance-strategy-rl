"""What is measured, what is assumed, and what is known to be wrong.

The four statements in full, the dial table with every row labelled, and the
provenance of every artefact the app has open. This is the page the other two
link to, and the strip at the top of them is its short form.

It is deliberately the least interactive page in the app. A limitation
rendered as a widget is a limitation somebody can collapse.
"""

from __future__ import annotations

# Streamlit puts the entry script's directory on sys.path, not the project root, so
# `import app` fails before anything else can. See app/__init__.py.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app import statements
from app.loading import (MEASURED_COUNTERPART, available_series, load_agent,
                         load_assets)
from endurance.params import ASSUMED_FIELDS

st.set_page_config(page_title="Methods", layout="wide")

st.title("What is measured and what is assumed")
st.markdown(
    "Every number in this app comes from one of two places: a quantity fitted "
    "to real lap timing, or a quantity nobody can fit to lap timing and which "
    "was therefore chosen. The second kind are marked *assumed* throughout, "
    "and the sliders exist because the honest response to an assumed number "
    "is to move it and see whether the conclusion survives."
)

st.divider()
st.header("Four things to know before you read any number here")
for s in statements.STATEMENTS:
    st.subheader(s.short)
    st.write(s.full)
    st.caption(f"Settled at {s.source}.")
    st.write("")

st.divider()
st.header("The dials")

series = st.selectbox("Series", available_series(),
                      format_func=lambda c: c.upper())
try:
    assets = load_assets(series)
except ValueError as e:
    st.error(str(e))
    st.stop()

rows = []
for cls in assets.config.classes:
    for name in cls.measured_fields() + list(ASSUMED_FIELDS):
        rows.append({
            "class": cls.class_name,
            "dial": name,
            "value": getattr(cls, name),
            "kind": "assumed" if name in ASSUMED_FIELDS else "measured",
            "note": MEASURED_COUNTERPART.get(name, ""),
        })
frame = pd.DataFrame(rows)

kinds = st.multiselect("Show", ["measured", "assumed"],
                       default=["measured", "assumed"])
st.dataframe(frame[frame["kind"].isin(kinds)], hide_index=True,
             use_container_width=True)

st.caption(
    "One assumed dial has a measured counterpart that disagrees with it, and "
    "it is left at its assumed value rather than quietly corrected — changing "
    "a dial changes every number measured against it, and that is a "
    "recalibration rather than an edit. A second, `pit_transit_frac`, turns "
    "out not to be measurable from lap timing at all: telling what a stop "
    "costs to enter from what it costs to fill needs stops that took "
    "different amounts of fuel, and 62% to 90% of stops in these two races "
    "are followed by a near-full tank. Both notes are in the note column."
)

st.divider()
st.header("Where each class came from")
st.dataframe(
    pd.DataFrame([{"class": c.class_name, "source": c.source_event,
                   "laps observed": c.n_laps_observed,
                   "cars": c.n_cars, "base pace (s)": round(c.base_pace_s, 3)}
                  for c in assets.config.classes]),
    hide_index=True, use_container_width=True)

st.caption(
    "One event per series, one running of it. See the first statement above: "
    "the caution share alone moves by a factor of 2.8 between two adjacent "
    "Daytonas."
)

st.divider()
st.header("The artefacts this app has open")

agent = load_agent(series)
provenance = {
    "config": {"name": assets.config.name,
               "dials fingerprint": assets.fingerprint,
               "duration (h)": round(assets.config.duration_s / 3600, 2)},
    "seed bank": {"headline races": len(assets.bank.headline),
                  "sweep races": len(assets.bank.sweep),
                  "held out": len(assets.bank.held_out),
                  **assets.bank.provenance},
    "background field": assets.field.provenance,
    "policy": ({"checkpoint": agent.card.checkpoint or "—",
                "algorithm": agent.card.algorithm,
                "trained against dials": agent.card.dials_fingerprint,
                "trained against bank": agent.card.bank_fingerprint,
                "timesteps": agent.card.total_timesteps,
                "trained at": agent.card.trained_at}
               if agent and agent.card else {"loaded": False,
                                             "reason": agent.reason}),
}
st.json(provenance)

st.caption(
    "The held-out fifty are disjoint from the headline two hundred and are "
    "not used anywhere in this app. They exist to answer whether a design "
    "chosen on the headline races generalises, and a set that has been "
    "looked at cannot answer that."
)
