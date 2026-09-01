"""What this is, before anyone clicks anything.

The landing page carries the four statements in short form and the one
sentence about what the model is, because a visitor who lands here and reads
nothing else should still leave with the right idea of what they have been
shown. Everything interactive is on the three pages beside it.

06 removed the "Where the app is reading from" panel. It printed resolved
absolute paths, which on the hosted app are the deployment's own internals
and are of no use to anybody who cannot see that filesystem. `loading.where()`
still exists and is still the way to answer the question; it is now answered
from a terminal rather than on the landing page.
"""

from __future__ import annotations

# `streamlit run app/Home.py` puts app/ on sys.path, not the project root, so
# `import app` fails before anything else can. See app/__init__.py.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app import statements
from app.loading import available_series, load_agent, load_assets
from endurance.strategies import ROSTER

st.set_page_config(page_title="Endurance strategy sandbox", layout="wide")

st.title("RL in Endurance Race Strategy")
st.markdown(
    "Twenty-four-hour races are won in the pit lane. This is a simulator of "
    "one, built from real lap timing, with five human pit strategies, a "
    "control that never calls a stop, and an agent trained by reinforcement "
    "learning, all driving the same car. Pick a race, change a dial, make a "
    "call yourself, and see what the agent would have done."
)

st.subheader("Four things to know before you read any number here")
for s in statements.STATEMENTS:
    with st.expander(s.short):
        st.write(s.full)

st.divider()
st.subheader("who's calling the strategy")
st.caption(
    "Six plans, none of which has a number you can tune. Each works its "
    "thresholds out from the dials, so none of them can be quietly fitted to "
    "the races it's scored on."
)

undescribed, orphaned = statements.roster_gaps(ROSTER.keys())
if undescribed or orphaned:
    st.warning(
        f"The roster and these descriptions have drifted apart. "
        f"Not described: {undescribed or 'none'}. "
        f"Described but not in the roster: {orphaned or 'none'}.", icon="⚠️")

for note in statements.STRATEGY_NOTES:
    with st.expander(f"**{note.title}** - {note.one_line}"):
        st.write(note.detail)

st.divider()
st.subheader("what's loaded")

for code in available_series():
    try:
        assets = load_assets(code)
    except ValueError as e:
        st.error(f"{code.upper()}: {e}")
        continue

    agent = load_agent(code)
    cols = st.columns([2, 1, 1, 2])
    cols[0].markdown(f"**{code.upper()}** - {assets.config.name}")
    cols[1].markdown(f"{assets.config.duration_s / 3600:.0f} h, "
                     f"{assets.config.total_cars} cars")
    cols[2].markdown(f"dials `{assets.fingerprint}`")
    cols[3].markdown("agent loaded" if agent else f"no agent — {agent.reason}")

st.caption(
    "That code is a hash of the dials the race is set up with. "
)

st.divider()
st.page_link("pages/1_Race.py", label="watch a race, lap by lap", icon="🏁")
st.page_link("pages/2_Comparison.py", label="see how the different strategies compare", icon="📊")
st.page_link("pages/3_Methods.py", label="understand what's measured and what's assumed",
             icon="📐")
