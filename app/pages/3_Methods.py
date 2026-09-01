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

st.set_page_config(page_title="Methods · RL in Endurance Race Strategy", page_icon="★", layout="wide")


def _readable(key: str) -> str:
    """A provenance key as a person would say it.

    The keys come from files this page does not own, so an unknown one has
    to render as something rather than raise. Underscores out, first letter
    up, and nothing else - a lookup table here would go stale the moment a
    freezing script added a field.
    """
    return key.replace("_", " ").strip().capitalize()


def _plain(value) -> str:
    """A provenance value as text, with lists spelled out rather than bracketed."""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "—"
    return str(value)

st.title("What's Measured and What's Assumed")
st.markdown(
    "Every number in this app comes from one of two places. Either it was "
    "fitted to real lap timing, or nobody can fit it to lap timing and "
    "somebody picked it. The second kind are marked *assumed* wherever they "
    "appear, and they're on sliders for a reason: the honest thing to do "
    "with a number somebody picked is move it and see whether the result "
    "survives."
)

st.divider()
st.header("Four things to know before you read any number here")
for s in statements.STATEMENTS:
    st.subheader(s.short)
    st.write(s.full)
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
    "One assumed dial has a measured counterpart that disagrees with it. It's "
    "been left where it is rather than quietly corrected, because changing a "
    "dial changes every number measured against it, and that's a "
    "recalibration rather than an edit. A second one, `pit_transit_frac`, "
    "turns out not to be measurable from lap timing at all: telling what a "
    "stop costs to enter apart from what it costs to fill needs stops that "
    "took different amounts of fuel, and 62% to 90% of stops in these two "
    "races are followed by a near-full tank. Both notes are in the last "
    "column."
)

st.divider()
st.header("Everything else this model assumes")
st.markdown(
    "The dials above are assumptions you can move. These are the ones built "
    "into the shape of the simulator, which no slider reaches — the things it "
    "doesn't represent at all. None of them is hidden anywhere else in this "
    "app, and the honest way to read any result here is to ask which of them "
    "the result depends on."
)

for group in statements.ASSUMPTION_GROUPS:
    st.subheader(group)
    for a in statements.ASSUMPTIONS:
        if a.group == group:
            st.markdown(f"- {a.text}")

st.caption(
    "This list is the same one quoted in the write-up's closing section, and "
    "it lives in one place in the code so the two can't drift apart."
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
    "One event per series, and one running of it. See the first statement "
    "above: the caution share alone moves by a factor of 2.8 between two "
    "adjacent Daytonas."
)

st.divider()
st.header("Where each number in this app came from")

agent = load_agent(series)
provenance = [
    ("Race", assets.config.name),
    ("Length", f"{assets.config.duration_s / 3600:.0f} hours"),
    ("Dials", assets.fingerprint),
    ("Races used for every published figure", f"{len(assets.bank.headline)}"),
    ("Races used for sweeping a dial", f"{len(assets.bank.sweep)}"),
    ("Races held back and never selected on", f"{len(assets.bank.held_out)}"),
]
provenance += [(_readable(k), _plain(v))
               for k, v in assets.bank.provenance.items()]
provenance += [(f"Background field: {_readable(k)}", _plain(v))
               for k, v in assets.field.provenance.items()]

if agent and agent.card:
    provenance += [
        ("Agent", agent.card.checkpoint or "—"),
        ("Trained with", agent.card.algorithm),
        ("Trained on dials", agent.card.dials_fingerprint),
        ("Trained on races", agent.card.bank_fingerprint),
        ("Steps of training", f"{agent.card.total_timesteps:,}"),
        ("Trained on", agent.card.trained_at),
    ]
else:
    provenance += [("Agent", f"not loaded — {agent.reason}")]

st.dataframe(
    pd.DataFrame(provenance, columns=["what", "value"]),
    hide_index=True, use_container_width=True)

st.caption(
    "The fifty held-back races are a separate set, and nothing in this app "
    "runs on them. They're there to answer whether something chosen on the "
    "two hundred still holds somewhere it was never fitted to - and a set "
    "that's been looked at can't answer that."
)
