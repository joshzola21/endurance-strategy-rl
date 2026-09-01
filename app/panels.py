"""What the panels show, computed as data so it can be tested without a browser.

No Streamlit and no charts here: this module turns a `Frame` into rows and
the pages render them. Splitting it that way is not tidiness - a panel whose
content only exists inside a `st.` call is a panel no gate can look at, and
the explainability panel is the one part of 04 that can be quietly wrong
while looking right.

**Everything is read.** The observation rows come from `gym_env.observe`, so
the panel shows the policy the vector it was actually given rather than a
reconstruction of it. The action ranking is a softmax of the exported
logits, through `policy.action_probabilities`, so the panel and the decision
come out of one session and cannot disagree.

Three things the panel has to say out loud
------------------------------------------
**Probabilities, not values.** 03b chose MaskablePPO because the mask
decides it, and the cost is that there is no Q(s,a) - only P(a|s). The
ranking is a ranking. A panel wanting a magnitude is a retraining job.

**The mask is a training artefact.** It is drawn beside the ranking rather
than applied to it. `PolicyStrategy` carries no mask, which is how the agent
was scored, so an agent asking to stay out on an empty tank genuinely is
asking to stay out - and the engine makes it stop anyway. The panel says
both.

**Class position is shown mid-race, and used not to be.** The rule this
module kept was that position is derived from laps and then time, and
deriving it here would be a second implementation of
`RaceResult.positions`. That reasoning was right and the conclusion is now
obsolete: amendment 24 put `class_position` in the observation, so the
engine offers it at the line through `RaceState.class_position` and the
panel reads it like everything else. The rule was never "do not show
position" - it was "do not compute it here", and that still holds.

The headline strip above the panel still leaves it out, because those are
the quantities the engine offers about the *car* rather than about the
policy's view of the race, and the finishing position comes from
`classification()` at the flag like every other number in the project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from endurance.gym_env import N_ACTIONS

from .controller import ACTION_NAMES, Frame

# What each observation row means in the unit the reader thinks in. The
# normalisation is a modelling choice and gets measured - 03a's convention -
# so the panel prints the raw quantity beside the normalised one wherever
# the engine can supply it.
ROW_HELP = {
    "race_progress": "how much of the race has gone",
    "fuel": "fuel left, where 1.0 is a full tank",
    "tyre_age": "laps on these tyres, against how long a set is meant to last",
    "gap_ahead": "gap to the car ahead in class, counted in pit stops; "
                 "1.0 means there's nobody there",
    "gap_behind": "gap to the car behind in class, counted in pit stops; "
                  "1.0 means there's nobody there",
    "under_caution": "is the race under a full-course yellow",
    "stint_laps": "laps since the last stop, against a 40-lap scale",
    "laps_down": "laps behind the class leader, against a 3-lap scale",
    "pit_lane_open": "can this class pit right now",
    "class_position": "where it sits in class, where 0.0 is leading and "
                      "1.0 is last",
}


@dataclass(frozen=True)
class ActionRow:
    index: int
    name: str
    probability: float | None      # None when no policy is loaded
    available: bool                # the training mask, drawn not applied
    chosen: bool


def observation_rows(frame: Frame) -> list[dict]:
    """The ten rows the policy sees, with the raw quantity beside them.

    Ten since amendment 24. The tenth, `class_position`, is the row the reward
    was defined on while the observation could not see it - the nearest proxy
    the nine had, `laps_down`, correlates 0.091 with actual class position
    because it is clipped at three laps and a front-runner sits at zero all
    race.
    """
    cls = frame.state.config.class_by_name(frame.car.class_name)
    obs = frame.observation
    raw = {
        "race_progress": f"{frame.t / frame.state.duration_s:.1%} of {frame.state.duration_s / 3600:.1f} h",
        "fuel": f"{frame.car.fuel:.2f} tanks, about {frame.car.fuel / max(cls.fuel_per_lap, 1e-9):.0f} green laps",
        "tyre_age": f"{frame.car.tyre_age} laps of {cls.tyre_life_laps:.0f}",
        "gap_ahead": _gap_text(frame.state.gap_ahead_s(frame.car)),
        "gap_behind": _gap_text(frame.state.gap_behind_s(frame.car)),
        "under_caution": "yellow" if frame.state.under_caution else "green",
        "stint_laps": f"{frame.car.stint_laps} laps",
        "laps_down": f"{frame.state.laps_down(frame.car)} laps",
        "pit_lane_open": frame.lane.reason or "open",
        "class_position": f"P{frame.state.class_position(frame.car)} of "
                          f"{cls.n_cars} in {frame.car.class_name}",
    }
    return [{"row": k, "value": v, "raw": raw[k], "means": ROW_HELP[k]}
            for k, v in obs.items()]


def _gap_text(gap: float | None) -> str:
    return "nobody within a lap" if gap is None else f"{gap:.1f} s"


def action_ranking(frame: Frame, probabilities: np.ndarray | None = None
                   ) -> list[ActionRow]:
    """The five actions, ranked, with the mask beside them rather than on them.

    `probabilities` is passed in rather than fetched, so this module never
    touches a checkpoint and the page decides whether there is a policy to
    ask. With none, the rows still render - the mask alone is worth showing,
    because it says what the rules have already decided.
    """
    if probabilities is not None and len(probabilities) != N_ACTIONS:
        raise ValueError(f"expected {N_ACTIONS} probabilities, "
                         f"got {len(probabilities)}")
    top = int(np.argmax(probabilities)) if probabilities is not None else -1
    rows = [ActionRow(i, ACTION_NAMES[i],
                      None if probabilities is None else float(probabilities[i]),
                      bool(frame.mask[i]), i == top)
            for i in range(N_ACTIONS)]
    return sorted(rows, key=lambda r: (-(r.probability or 0.0), r.index))


def mask_note(frame: Frame) -> str | None:
    """Why an action is greyed, and what happens if the policy picks it anyway.

    The honest version, per 03b: the mask is a training convenience and the
    policy is scored without it, so the ranking may well be led by an action
    the mask has removed. That is not a bug in the panel.
    """
    if frame.forced:
        return (f"The rules have already taken this decision — {frame.forced} — "
                f"so the car is coming in whatever you pick below. The mask is "
                f"a training aid and the agent isn't scored with it, so it may "
                f"still rank staying out first.")
    if not frame.lane.open:
        return (f"The pit lane is shut to this class: {frame.lane.reason}. A "
                f"stop you choose is refused; a stop the rules force is taken "
                f"and recorded.")
    return None


@dataclass(frozen=True)
class Override:
    """A human decision beside the policy's, at one crossing."""

    lap: int
    human: int
    human_name: str
    agent: int | None
    agent_name: str
    agreed: bool | None
    agent_probability: float | None
    note: str


def override_comparison(frame: Frame, human_action: int,
                        probabilities: np.ndarray | None = None) -> Override:
    """What the human did, and what the policy would have done.

    The policy's answer here is **unmasked**, which is the same answer it is
    scored on. So "the agent would have stayed out" on an empty tank is a
    correct reading of the policy and not a fault in the comparison.
    """
    human = int(human_action)
    if not 0 <= human < N_ACTIONS:
        raise ValueError(f"action {human} outside 0..{N_ACTIONS - 1}")

    if probabilities is None:
        return Override(frame.lap, human, ACTION_NAMES[human], None, "",
                        None, None,
                        "No agent is loaded, so there's nothing to compare "
                        "against.")

    agent = int(np.argmax(probabilities))
    agreed = agent == human
    note = ("You and the agent made the same call." if agreed else
            f"The agent would have chosen {ACTION_NAMES[agent]!r}, at "
            f"{probabilities[agent]:.0%} against {probabilities[human]:.0%} "
            f"for yours.")
    if frame.forced:
        note += (f" Neither call matters here: {frame.forced}, so the rules "
                 f"take the decision.")
    return Override(frame.lap, human, ACTION_NAMES[human], agent,
                    ACTION_NAMES[agent], agreed, float(probabilities[human]),
                    note)


def headline(frame: Frame) -> dict:
    """The numbers across the top of the race page.

    No class position here, though the observation panel below does show it:
    these are the quantities the engine offers about the *car* at the line,
    and position is about the race. See the module docstring.
    """
    return {
        "lap": frame.lap,
        "clock": f"{frame.t / 3600:.2f} h of "
                 f"{frame.state.duration_s / 3600:.1f} h",
        "flag": "yellow" if frame.state.under_caution else "green",
        "fuel": f"{frame.car.fuel:.2f}",
        "tyre_age": frame.car.tyre_age,
        "stops": frame.car.n_stops,
        "laps_down": frame.state.laps_down(frame.car),
        "gap_ahead": _gap_text(frame.state.gap_ahead_s(frame.car)),
        "gap_behind": _gap_text(frame.state.gap_behind_s(frame.car)),
        "lane": frame.lane.reason or "open",
    }
