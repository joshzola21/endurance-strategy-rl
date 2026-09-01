"""How the strategies compare, and the agent beside them.

Presentation only: the table comes from `harness.summarise` and the chart
from `viz.plot_paired_deltas`, both unchanged. This page runs no race loop of
its own; where it needs races it asks `harness.compare_roster`, which is the
same function 02c's number came from.

Three rules this page keeps
---------------------------
**No benchmark row, anywhere.** 02b's reference is degenerate on real dials -
the clairvoyant arm extracts zero foreknowledge on every seed and is beaten
by its own control - so there is nothing to compare against and a column
would only invite the comparison.

**No headline claim from a live run.** A roster pass costs 5.26 s a seed in
IMSA and 2.61 s in WEC, so the two hundred headline races are eighteen and
nine minutes. The page reads 02c's saved rows where they exist and otherwise
offers a handful of races, said out loud to be a handful of races.

**No agent row off-nominal.** The policy may be watched at any dial setting
on the race page, because watching it fail on a race it was not trained for
is the demonstration. A *number* is a different thing: `PolicyCard` refuses
a policy scored against dials it did not train on, and that refusal is the
whole reason the card exists.
"""

from __future__ import annotations

# Streamlit puts the entry script's directory on sys.path, not the project root, so
# `import app` fails before anything else can. See app/__init__.py.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app import statements
from app.loading import (Assets, agent_for_config, apply_levers,
                         available_series, LEVERS, lever_warnings, load_agent,
                         load_assets, saved_banks, saved_comparison)
from endurance import harness, viz
from endurance.assets import dials_fingerprint
from endurance.policy import agent_roster

st.set_page_config(page_title="Comparison", layout="wide")
st.caption("  ·  ".join(statements.strip_lines()))
st.divider()

st.title("How the Strategies Compare")
st.markdown(
    "Every strategy is scored on the same race, run twice: once with the "
    "strategy calling the shots and once with a plain fuel-window plan. The "
    "difference between the two runs is what that strategy was worth on that "
    "race, with the cautions, the traffic and the other sixty cars held "
    "identical. Positive is better in every column."
)

with st.sidebar:
    series = st.selectbox("Series", available_series(),
                          format_func=lambda c: c.upper())
    try:
        assets: Assets = load_assets(series)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.header("the levers")
    multipliers = {d: st.slider(d, 0.25, 3.0, 1.0, 0.05) for d in LEVERS}

config = apply_levers(assets.config, multipliers)
nominal = dials_fingerprint(config) == assets.fingerprint
for warning in lever_warnings(config):
    st.sidebar.warning(warning)

agent = agent_for_config(load_agent(series), config)

# ----------------------------------------------------------------------
# Where the rows come from
# ----------------------------------------------------------------------
banks = saved_banks(series) if nominal else []
saved = None

if banks:
    bank = st.radio(
        "which races", banks, horizontal=True,
        format_func=lambda b: {"headline": "the 200 headline races",
                               "held_out": "the 50 held-out races"}[b],
        help="The fifty are a separate set of races, kept back and never "
             "used to choose anything. They're the check on whether a "
             "result holds up away from the races it was found on.")
    saved = saved_comparison(series, bank)
    if saved and saved.provenance.get("dials_fingerprint") not in (
            None, assets.fingerprint):
        st.error(f"The saved table was measured on dials "
                 f"{saved.provenance['dials_fingerprint']!r} and these are "
                 f"{assets.fingerprint!r}. Not shown — it's about a "
                 f"different race.")
        saved = None

if saved is not None:
    rows, summary, provenance = saved.rows, saved.summary, saved.provenance
    headline = saved.bank == "headline"
    n_races = int(provenance.get("n_seeds", rows["seed"].nunique()))
else:
    if not nominal:
        st.info("You've moved the sliders, so no saved table covers this "
                "race. Run a few below.")
    else:
        st.info("No saved table here. Run a few races below.")

    n = st.select_slider("Races to run", [3, 5, 10, 25], value=5,
                         help="About 5 seconds a race at Daytona and 3 at Le "
                              "Mans, for every strategy at once.")
    include_agent = bool(agent) and nominal
    if bool(agent) and not nominal:
        st.caption("The agent sits this one out. Scoring it on dials it "
                   "didn't train on gives a complete, plausible, "
                   "meaningless number.")

    if st.button(f"Run {n} races", type="primary"):
        roster = dict(harness.ROSTER)
        if include_agent:
            roster = agent_roster(roster, agent.strategy)
        with st.spinner(f"Running {n} races per strategy…"):
            st.session_state[f"comparison_{series}"] = harness.compare_roster(
                config, assets.bank.headline[:n], assets.field, roster=roster)

    comparison = st.session_state.get(f"comparison_{series}")
    if comparison is None:
        st.stop()
    rows, provenance = comparison.rows, comparison.provenance
    summary, headline = None, False
    n_races = int(provenance.get("n_seeds", 0))

if summary is None:
    summary = harness.summarise(rows)

# ----------------------------------------------------------------------
# The table
# ----------------------------------------------------------------------
with st.expander("What each strategy in the table actually does"):
    for note in statements.STRATEGY_NOTES:
        st.markdown(f"**{note.title}** (`{note.key}`) — {note.one_line}")

st.subheader("how often, not how much on average")
st.caption(
    "**`never_pit` is a control, not a plan.** It never calls a stop; the "
    "rules drag it in when the tank runs dry. So it scores what you get for "
    "taking no decisions at all, and a row that can't beat it hasn't learned "
    "anything. That's the row to read the agent against."
)
st.caption(
    "\"Gains a place in 40% of races, loses one in 12%\" is a stronger claim "
    "than an average of 4.7, and it survives the odd race a strategy throws "
    "away entirely. There's no mean here on purpose. Where the saved table "
    "has them, `_lo` and `_hi` are the range you'd expect these figures to "
    "move in if you ran the whole thing again."
)
st.dataframe(
    summary.rename(columns={
        "gained": "gains a place", "level": "level", "lost": "loses a place",
        "median_d_pos": "median Δ position", "p10_d_pos": "10th pct",
        "p90_d_pos": "90th pct", "median_d_laps": "median Δ laps",
        "median_d_pit_s": "median Δ pit time (s)",
        "median_stops": "median stops"}),
    hide_index=True, use_container_width=True)

if not headline:
    st.warning(
        f"This is {n_races} races. Every published figure in this project "
        f"is two hundred, and a handful can't separate strategies whose "
        f"effects are a fraction of a place.", icon="⚠️")

if "agent" in set(rows["strategy"]):
    st.error(statements.agent_caveat().full, icon="⚠️")

st.subheader("race by race")
st.pyplot(viz.plot_paired_deltas(rows))
