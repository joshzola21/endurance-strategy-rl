"""The race, stepped. 04's main page.

Presentation only: every number here was read from a `Frame` that
`RaceController` got from the engine, and every chart comes from `viz.py`
unchanged. Nothing on this page computes a race quantity, and the gate in
`tests/test_app.py` checks that mechanically rather than on trust.

Session state, and why it is what it is
---------------------------------------
Streamlit re-executes this file top to bottom on every click, so nothing can
be assumed to survive except what is in `st.session_state`. What lives there
is the controller's `to_dict()` - a seed, a seat, a dials fingerprint and a
map of lap to overridden action - which is JSON and rebuilds the race
exactly, because every random number was drawn before the race started. The
live controller sits beside it as a cache under a key built from the same
identity; when the key moves, or a hot reload has dropped the object, the
controller is rebuilt from the log and nothing is lost but a second.
"""

from __future__ import annotations

# Streamlit puts the entry script's directory on sys.path, not the project root, so
# `import app` fails before anything else can. See app/__init__.py.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app import panels, statements
from app.controller import ACTION_NAMES, RaceController, policy_seat, roster_seat
from app.loading import (LEVERS, MEASURED_COUNTERPART, Assets, agent_for_config,
                         apply_levers, available_series, lever_kind,
                         lever_warnings, load_agent, load_assets)
from endurance.assets import dials_fingerprint
from endurance.strategies import ROSTER

st.set_page_config(page_title="Race · RL in Endurance Race Strategy", page_icon="★", layout="wide")

# The four statements, on every screen rather than only on the page nobody
# clicks. Short forms here; the methods page carries them in full.
with st.container():
    st.caption("  ·  ".join(statements.strip_lines()))
st.divider()


# ----------------------------------------------------------------------
# Sidebar: the race, the seat, the levers
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("the race")
    series = st.selectbox("Series", available_series(),
                          format_func=lambda c: c.upper())
    try:
        assets: Assets = load_assets(series)
    except ValueError as e:            # the generation check in `load_assets`
        st.error(str(e))
        st.stop()

    st.caption(f"{assets.config.name} · {assets.config.duration_s / 3600:.0f} h "
               f"· {assets.config.total_cars} cars")

    seed = st.selectbox("Seed", assets.bank.headline[:50],
                        help="One of the two hundred races every number in this "
                             "project was measured on.")

    seat_name = st.selectbox(
        "Chosen Race Strategy",
        [*ROSTER.keys(), "agent"],
        help="One of the five human strategies, the never-pit control, or "
             "the trained agent. Whoever's calling it, the other sixty cars "
             "run the same plan every time.")

    note = statements.NOTE_BY_KEY.get(seat_name)
    if note:
        st.caption(f"**{note.title}.** {note.one_line}")
        with st.expander("more on this one"):
            st.write(note.detail)
    elif seat_name == "agent":
        st.caption("**The agent.** Trained by reinforcement learning. No "
                   "number is reported for it, see the strip along the top.")

    st.header("the levers")
    st.caption("Each slider multiplies a dial. The ones marked *assumed* are "
               "numbers nobody can measure from lap timing, so somebody "
               "picked them. Move one and see whether the result survives.")
    multipliers = {}
    for dial in LEVERS:
        note = MEASURED_COUNTERPART.get(dial, "")
        multipliers[dial] = st.slider(
            f"{dial} ({lever_kind(dial)})", 0.25, 3.0, 1.0, 0.05,
            help=note or None)

config = apply_levers(assets.config, multipliers)
nominal = dials_fingerprint(config) == assets.fingerprint
for warning in lever_warnings(config):
    st.sidebar.warning(warning)


# ----------------------------------------------------------------------
# The seat
# ----------------------------------------------------------------------
agent = agent_for_config(load_agent(series), config)

if seat_name == "agent":
    if not agent:
        st.error(f"The agent can't take this seat: {agent.reason}")
        st.stop()
    seat, seat_label = policy_seat(agent.strategy), "agent"
else:
    seat, seat_label = roster_seat(seat_name), seat_name

if agent and not nominal:
    st.warning(
        "You've moved the sliders, so this isn't the race the agent was "
        "trained on. It'll still make calls and you can watch what it does, "
        "but no comparison number appears once the dials have moved. It "
        "would be a number about a different race.", icon="⚠️")


# ----------------------------------------------------------------------
# The controller, cached against a rerun
# ----------------------------------------------------------------------
# Two identities, not one. **The race** is the series and the seed: change
# either and the decision log is about laps that no longer exist, so it goes.
# **The setting** adds the seat and the dials: change one of those and the log
# still means something - it is a list of calls taken on particular laps - so
# it is kept and replayed, and the page says that is what happened.
#
# The earlier version had a single identity and silently dropped the log on
# any change, which meant nudging a slider quietly threw away everything the
# reader had done. Silently discarding somebody's work reads as a crash.
race_key = (series, int(seed))
setting_key = (*race_key, seat_label, dials_fingerprint(config))


def controller() -> RaceController:
    held = st.session_state.get("controller")
    if held is not None and st.session_state.get("setting") == setting_key:
        return held

    same_race = st.session_state.get("race") == race_key
    saved = st.session_state.get("saved") if same_race else None
    log = {int(k): int(v) for k, v in (saved or {}).get("log", {}).items()}

    built = RaceController(config=config, field=assets.field, seed=int(seed),
                           seat=seat, seat_name=seat_label, log=log)
    if saved:
        built.seek(int(saved.get("lap", 0)))
        if log and st.session_state.get("setting") is not None:
            st.session_state["replayed"] = (
                f"The race changed under you, so your {len(log)} "
                f"{'call' if len(log) == 1 else 'calls'} were replayed against "
                f"it from lap zero. Same laps; what happens on them is not "
                f"the same.")
    elif st.session_state.get("race") is not None and not same_race:
        st.session_state.pop("last_override", None)
        st.session_state["replayed"] = (
            "A different seed is a different race, so your calls were "
            "cleared.")

    st.session_state["controller"] = built
    st.session_state["race"] = race_key
    st.session_state["setting"] = setting_key
    st.session_state["saved"] = built.to_dict()
    return built


def remember(ctrl: RaceController) -> None:
    st.session_state["saved"] = ctrl.to_dict()


ctrl = controller()
frame = ctrl.frame

replayed = st.session_state.pop("replayed", None)
if replayed:
    st.info(replayed, icon="↻")

# ----------------------------------------------------------------------
# Controls
# ----------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
if c1.button("step one lap", use_container_width=True):
    ctrl.step(); remember(ctrl); st.rerun()
if c2.button("run to the next decision", use_container_width=True,
             help="Stops when a caution comes out, when the fuel window "
                  "opens, when the pit lane opens to this class, or when "
                  "the rules have already called the car in."):
    ctrl.run(); remember(ctrl); st.rerun()
if c3.button("run to the flag", use_container_width=True):
    ctrl.finish(); remember(ctrl); st.rerun()
if c4.button("go back ten laps", use_container_width=True,
             help="Replays the race from the start, which takes about a "
                  "second. Your own calls are kept."):
    ctrl.seek(max(ctrl.lap - 10, 0)); remember(ctrl); st.rerun()
if c5.button("start again", use_container_width=True):
    ctrl.log.clear(); ctrl.reset(); remember(ctrl)
    st.session_state.pop("last_override", None)
    st.rerun()

# ----------------------------------------------------------------------
# The race, as it stands
# ----------------------------------------------------------------------
if ctrl.finished:
    st.success("Chequered flag.")
    row = ctrl.result.classification().set_index("car_id").loc[ctrl.focal]
    m = st.columns(5)
    m[0].metric("Class position", int(row["class_pos"]))
    m[1].metric("Laps", int(row["laps"]))
    m[2].metric("Stops", int(row["stops"]))
    m[3].metric("In the pits", f"{row['pit_time_s'] / 60:.1f} min")
    m[4].metric("In traffic", f"{row['traffic_time_s'] / 60:.1f} min")

    from endurance import viz            # charts come from viz.py unchanged
    st.pyplot(viz.plot_race(ctrl.result, class_name=ctrl.class_name))
    st.pyplot(viz.plot_stint_pace(ctrl.result, class_name=ctrl.class_name))
else:
    head = panels.headline(frame)
    m = st.columns(6)
    m[0].metric("Lap", head["lap"])
    m[1].metric("Clock", head["clock"])
    m[2].metric("Flag", head["flag"])
    m[3].metric("Fuel", head["fuel"])
    m[4].metric("Tyre age", head["tyre_age"])
    m[5].metric("Stops", head["stops"])

    g = st.columns(3)
    g[0].metric("Gap ahead", head["gap_ahead"])
    g[1].metric("Gap behind", head["gap_behind"])
    g[2].metric("Pit lane", head["lane"])

    for pause in frame.pauses:
        st.info(f"**{pause.kind.replace('_', ' ')}** — {pause.detail}")

    # ------------------------------------------------------------------
    # Explainability, and the override
    # ------------------------------------------------------------------
    left, right = st.columns([3, 2])

    with left:
        st.subheader("what the agent sees")
        st.caption("The ten numbers it's handed at this point in the race, "
                   "with the quantity each was built from beside it.")
        st.dataframe(panels.observation_rows(frame), hide_index=True,
                     use_container_width=True)

    with right:
        st.subheader("what it would do")
        probabilities = None
        if agent:
            try:
                from endurance.policy import action_probabilities
                probabilities = action_probabilities(agent.strategy, frame.obs)
            except (AttributeError, ImportError) as e:
                st.caption(f"No probabilities to show: {e}")
        else:
            st.caption(agent.reason)

        st.caption("These are probabilities, not values. The agent ranks the "
                   "five calls; it doesn't say how much better one is than "
                   "another, and there's no way to ask it.")
        st.dataframe(
            [{"action": r.name,
              "P(a|s)": "—" if r.probability is None else f"{r.probability:.1%}",
              "the rules allow it": "yes" if r.available else "no"}
             for r in panels.action_ranking(frame, probabilities)],
            hide_index=True, use_container_width=True)

        note = panels.mask_note(frame)
        if note:
            st.caption(note)

    st.subheader("make the call yourself")
    chosen = st.radio("your call on this lap", range(len(ACTION_NAMES)),
                      format_func=lambda i: ACTION_NAMES[i], horizontal=True)
    if st.button("make it and step"):
        # Stamped with the lap it was taken on. The note used to be stored
        # bare, so it sat under later laps it no longer described - in the one
        # panel whose whole job is to be unambiguous.
        note = None
        if probabilities is not None and nominal:
            note = panels.override_comparison(frame, chosen,
                                              probabilities).note
        st.session_state["last_override"] = (frame.lap, int(chosen), note)
        ctrl.step(action=int(chosen))
        remember(ctrl)
        st.rerun()

    last = st.session_state.get("last_override")
    if last:
        lap, action, note = last
        st.caption(f"**Lap {lap}** — you called {ACTION_NAMES[action]!r}."
                   + (f" {note}" if note else ""))
        if note:
            st.caption(statements.agent_caveat().short)
