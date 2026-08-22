"""What this is, before anyone clicks anything.

The landing page carries the four statements in short form and the one
sentence about what the model is, because a visitor who lands here and reads
nothing else should still leave with the right idea of what they have been
shown. Everything interactive is on the three pages beside it.
"""

from __future__ import annotations

# `streamlit run app/Home.py` puts app/ on sys.path, not the project root, so
# `import app` fails before anything else can. See app/__init__.py.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app import statements
from app.loading import available_series, load_agent, load_assets, where

st.set_page_config(page_title="Endurance strategy sandbox", layout="wide")

st.title("Endurance race strategy")
st.markdown(
    "A simulator of WEC and IMSA endurance racing, calibrated from real lap "
    "timing, with five human-style pit strategies, a control that takes no "
    "decisions at all, and a trained reinforcement learning policy — all "
    "plugged into it through the same interface. Move a dial, watch a race, "
    "take a decision yourself, and see what the policy would have done."
)

st.subheader("Four things to know before you read any number here")
for s in statements.STATEMENTS:
    with st.expander(s.short):
        st.write(s.full)
        st.caption(f"Settled at {s.source}.")

st.divider()
st.subheader("What is loaded")

for code in available_series():
    try:
        assets = load_assets(code)
    except ValueError as e:
        st.error(f"{code.upper()}: {e}")
        continue

    agent = load_agent(code)
    cols = st.columns([2, 1, 1, 2])
    cols[0].markdown(f"**{code.upper()}** — {assets.config.name}")
    cols[1].markdown(f"{assets.config.duration_s / 3600:.0f} h, "
                     f"{assets.config.total_cars} cars")
    cols[2].markdown(f"`{assets.fingerprint}`")
    cols[3].markdown("policy loaded" if agent else f"no policy — {agent.reason}")

st.caption(
    "The fingerprint is a hash of the dials. Every artefact in this project "
    "carries one, and anything that does not match is refused rather than "
    "quietly used."
)

with st.expander("Where the app is reading from"):
    st.json(where())
    st.caption(
        "Resolved rather than hard-coded: an environment variable if one is "
        "set, then the conventional folder names, then a search of the tree. "
        "Set ENDURANCE_ASSETS, ENDURANCE_CHECKPOINTS or ENDURANCE_RESULTS to "
        "override any of them."
    )

st.divider()
st.page_link("pages/1_Race.py", label="Watch a race, lap by lap", icon="🏁")
st.page_link("pages/2_Comparison.py", label="How the strategies compare", icon="📊")
st.page_link("pages/3_Methods.py", label="What is measured and what is assumed",
             icon="📐")
