"""The four things 04 must say on the page, written once.

Settled at the 00 re-run and not optional. They appear in two places - a
persistent strip on the race page and the methods page in full - and they
are held here rather than typed into each, because a caveat that exists in
two copies is a caveat that will shortly exist in two versions. The strip
shows `short` and the page shows `full`.

The framing of the third is the diagnosed one and must stay that way. "The
agent lost" invites the reader to conclude that reinforcement learning does
not work on endurance strategy, which is not what was shown. What was shown
is that a single training run measures nothing here, and that four separate
properties of this simulator mislead a learner in ways that were measured on
the engine rather than inferred from a policy's score. Those are results
about the simulator, and they are the more interesting ones.

**`source` is no longer rendered, and is kept anyway.** It was drawn as a
"Settled at ..." caption under each statement until 06 removed it: a citation
to an internal decision record is furniture to a visitor, who cannot follow
it. It stays on the record because the point of the field is that a caveat
can be chased back to where it was settled, and that is a property of the
project rather than of the page.

That framing has already been rewritten once. An earlier version of the third
statement said the policy had found a cost function pricing a stop below the
cost of driving down the pit lane - true at the time, superseded when the
five-seed sweep showed the run it rested on was not a measurement. Anything
rewriting it again should check that the replacement is not itself resting on
one run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Statement:
    key: str
    short: str          # the strip: one line, no room for nuance
    full: str           # the methods page
    source: str         # where it was settled, so it can be chased


STATEMENTS = (
    Statement(
        key="sample_of_one",
        short="Every dial starts from one running of one race",
        full=("The dials come from a single running of a single event per "
              "series → Daytona 2026 for IMSA, Le Mans 2026 for WEC. They're "
              "not an average of that race, and they're certainly not an "
              "average of the series. Between two adjacent Daytonas the share "
              "of the race spent under caution moves by a factor of 2.8, so "
              "the width of each slider tells you more about the uncertainty "
              "than its starting value does. Every dial has been checked to "
              "regenerate exactly from the raw timing it was fitted to - "
              "which was briefly not true of the two caution dials, frozen "
              "from a run made before a determinism fix landed and 0.7% away "
              "from what the fixed code returns."),
        source="00 re-run sections 3 and 6; re-run post-fix and confirmed by "
               "scripts/check_dials_provenance.py",
    ),
    Statement(
        key="degradation",
        short="'Degradation' isn't tyre wear",
        full=("The degradation dial is a straight line fitted to lap time "
              "against tyre age. The raw data at Le Mans showed a line that appeared to slope the wrong way "
              "for Hypercar and LMP2, i.e. the cars were getting quicker as the stint goes "
              "on, which can't just be tyres wearing out. It's in fact also the fuel burning off, "
              "a track rubbering in and traffic thinning, and the fit can't "
              "pull any of that apart from the tyre. The engine applies it as "
              "though it were wear, so on those two classes the tyre slider "
              "is doing something the data never measured."),
        source="00 re-run; blueprint section 7D",
    ),
    Statement(
        key="no_agent_result",
        short="There's no score for the agent, one training run doesn't "
              "measure anything",
        full=("You can watch the trained agents take decisions here, but no "
              "number is reported for them, because there isn't one to "
              "report. Every agent figure this project produced came from a "
              "single training run. Training five instead, changing nothing "
              "but the seed, showed that in WEC the seed alone moves the "
              "headline statistic by 0.45 where two hundred paired races can "
              "only resolve 0.03 - one run of the five gained a place in 45% "
              "of races and three of the others gained one in none. In IMSA "
              "all five came out identical, which looked like stability and "
              "wasn't: they'd all converged on never asking to pit, and they "
              "score exactly what a car that takes no decisions scores. That "
              "car is now a strategy in its own right, `never_pit`, on the "
              "comparison page, so you can read the two rows side by side, "
              "and on the fifty held-out races they agree to three decimal "
              "places. What the project did find is four measured ways this "
              "simulator misleads a learner: a stop priced below the cost of "
              "driving down the pit lane, caution compression handing back "
              "70% of the time a stop costs, an agent that stops looking "
              "before a caution stop has paid off, and a reward "
              "defined on a quantity the agent couldn't see. Each was "
              "measured on the engine rather than inferred from an agent's "
              "score, so each stands on its own."),
        source="post-04 pass; decision record amendments 23-27",
    ),
    Statement(
        key="no_benchmark",
        short="There's no benchmark row, the reference doesn't work on the "
              "real dials",
        full=("The plan was to score a strategy against what was actually "
              "achievable on a race rather than only against its rivals. That "
              "means two references: one arm that knows when the cautions "
              "are coming and one that doesn't, with the gap between them "
              "measuring what foreknowledge is worth. On the real dials it "
              "falls over. The arm that can see the future extracts nothing "
              "from it on any seed, and gets beaten by its own control. "
              "Rather than print a number that doesn't mean what it looks "
              "like it means, there's no benchmark row anywhere in this app."),
        source="02b; blueprint section 3",
    ),
)

BY_KEY = {s.key: s for s in STATEMENTS}


# ----------------------------------------------------------------------
# The roster, in words
# ----------------------------------------------------------------------
# Written once here rather than in the three pages that show them, for the
# same reason the four statements are: a description kept in two places is a
# description that will shortly disagree with itself.
#
# `key` must match `endurance.strategies.ROSTER`. It is not imported and
# checked here, because this module deliberately imports nothing - the
# checking is `roster_gaps`, which the pages call with the real roster's
# names. A strategy added to the roster and not described here should be
# visible rather than silently unlabelled.


@dataclass(frozen=True)
class StrategyNote:
    key: str
    title: str          # a human name for a snake_case key
    one_line: str       # beside the selector
    detail: str         # in the expander


STRATEGY_NOTES = (
    StrategyNote(
        key="fuel_window",
        title="fuel window",
        one_line="Runs the tank dry every stint, then stops. The baseline.",
        detail=("The simplest plan there is: stay out until the fuel forces a "
                "stop, then take a full one. It's the plan every other "
                "strategy is measured against, and it's what the other sixty "
                "cars are running, so a strategy is compared to the same idea "
                "it's racing against. Its own row scores exactly zero by "
                "construction, which is the check that the comparison is "
                "working at all."),
    ),
    StrategyNote(
        key="caution_gambler",
        title="caution gambler",
        one_line="Stops under a yellow, but only when the stop is free.",
        detail=("Behind a safety car the whole field is slow, so time lost in "
                "the pits costs less. This strategy waits for that. The catch "
                "is that it only takes the stop when doing so doesn't add a "
                "stop to the rest of the race → it counts stops rather than "
                "pricing seconds, which is why it needs nothing the benchmark "
                "measures. It also checks whether the pit lane is actually "
                "open, because under IMSA's Short Full-Course-Yellow it never "
                "opens at all, and a strategy that can't see that isn't "
                "gambling, it's being lucky."),
    ),
    StrategyNote(
        key="track_position",
        title="track position",
        one_line="Won't take a stop that hands a rival the place.",
        detail=("Refuses a stop it could choose to take if a car currently "
                "behind would come out ahead, accepting a worse fuel window "
                "to hold the position. The rival is whoever is in that "
                "position, never a named car. Working out who that is has to "
                "go through the engine's own position rule: at the moment of "
                "the decision the car is a lap ahead of most of its slower "
                "rivals, so comparing raw race times would be comparing "
                "different laps, and the defence would fire on nonsense about "
                "a quarter of the time while still looking like it worked."),
    ),
    StrategyNote(
        key="splash_and_dash",
        title="splash and dash",
        one_line="Plans the last stop backwards from the flag.",
        detail=("Works out how much fuel the race still needs and takes "
                "exactly that, skipping tyres if the current set will last to "
                "the end. A short final stop can be worth a place on its own. "
                "The arithmetic is shared with the agent's action space, so "
                "there's one implementation of \"fill to the flag\" rather "
                "than two that could drift apart. Note that this strategy's "
                "results move steadily as the pit transit slider does, so part "
                "of what you see is an assumption rather than a measurement."),
    ),
    StrategyNote(
        key="lap_down",
        title="lap-down defender",
        one_line="Protects its place in the wave-around queue.",
        detail=("Two rules. It won't take a stop that would concede a whole "
                "lap to the class leader. And once it's a lap down under "
                "caution it won't take any stop it could avoid, because being "
                "waved back onto the lead lap depends on where the car is "
                "round the lap, and every stop pushes it backwards. Worth "
                "knowing that this one barely fires in either series: WEC runs "
                "one wave rather than two, and in IMSA the field bunches so "
                "hard under caution that the car is almost never a lap down in "
                "the first place."),
    ),
    StrategyNote(
        key="never_pit",
        title="never pit",
        one_line="Never calls a stop at all. A control, not a plan.",
        detail=("It always stays out, so every stop it takes is one the rules "
                "forced on it when the tank ran dry. It isn't trying to be "
                "good. It's here to answer one question: what do you score for "
                "taking no decisions? Anything that can't beat it hasn't "
                "learned anything, which is exactly the comparison the agent "
                "row needs → and on the fifty held-out races the trained IMSA "
                "agent and this control agree to three decimal places."),
    ),
)

NOTE_BY_KEY = {n.key: n for n in STRATEGY_NOTES}


def roster_gaps(roster_names) -> tuple[list[str], list[str]]:
    """Which strategies have no description, and which describe nothing.

    Returns (undescribed, orphaned). Both should be empty. A page that finds
    either should say so on screen rather than rendering a shorter list than
    the roster it is describing, which is the failure that looks like nothing
    at all.
    """
    names, described = set(roster_names), set(NOTE_BY_KEY)
    return sorted(names - described), sorted(described - names)


# ----------------------------------------------------------------------
# Everything the model assumes
# ----------------------------------------------------------------------
# The assumed *dials* are a separate thing and have their own table on the
# methods page, with a slider each. These are the assumptions built into the
# shape of the simulator, which no slider can move: the things it does not
# represent at all.
#
# Every entry traces to a decision or an amendment in the project record
# rather than to somebody's memory of what the engine does.


@dataclass(frozen=True)
class Assumption:
    group: str
    text: str


ASSUMPTIONS = (
    Assumption("the cars and the people in them",
               "Every car in a class has the same pace, give or take a fixed "
               "offset for where it started. There's no such thing as a "
               "quicker driver here."),
    Assumption("the cars and the people in them",
               "Nobody gets tired, and driver changes cost nothing beyond the "
               "stop they happen at. Driver-time regulations aren't modelled, "
               "so there's no strategy around who's in the car."),
    Assumption("the cars and the people in them",
               "Nothing ever breaks. No mechanical failures, no damage, no "
               "contact, and no penalties."),

    Assumption("the race",
               "The weather never changes, and it's never wet."),
    Assumption("the race",
               "There's no overtaking model. Position comes out of accumulated "
               "race time, and cars never pass each other as an event. Traffic "
               "is a general cost of being among other cars, not a queue of "
               "specific ones."),
    Assumption("the race",
               "Cautions are drawn before the race starts, at a constant rate. "
               "Nothing a car does can cause one, extend one, or bring one out "
               "at a convenient moment. That's what makes the comparison fair, "
               "and it also means the simulator can't represent a caution "
               "caused by a crash."),
    Assumption("the race",
               "Under caution the field bunches up, and how hard it bunches is "
               "an assumption that hasn't been validated against real timing. "
               "It may close the field harder than a real safety car does."),
    Assumption("the race",
               "Only full-course yellows exist. No local yellows, no slow "
               "zones."),

    Assumption("tyres and fuel",
               "Tyres wear off in a straight line with age. The calibration "
               "fits a straight line, so the engine can only apply one. If "
               "real degradation curves, this model can't see it."),
    Assumption("tyres and fuel",
               "Fuel is what decides stint length; tyres last about twice as "
               "long. "),
    Assumption("tyres and fuel",
               "There's no choice of how much fuel to take beyond the five "
               "actions on offer. A free choice of fill level would be a "
               "tuning knob, and none of the strategies is allowed one."),

    Assumption("the pit stop",
               "A full service always costs the same. The sliders change how "
               "that cost divides between driving down the lane, changing "
               "tyres and refuelling, so raising one lowers another. This is "
               "the wrong shape for a real pit lane, where getting in and out "
               "is most of the cost whatever you stopped for."),
    Assumption("the pit stop",
               "Both rulebooks are implemented where they change the shape of "
               "a stop: pit lane closures, the order classes are released in, "
               "how many people may work on the car, and wave-arounds. "
               "Everything else in a 200-page rulebook is not here."),

    Assumption("the field",
               "Every car that isn't yours runs the fuel-window plan, so the "
               "whole field pits in lockstep. Real fields don't, and traffic "
               "only really bites when stints are out of step, so this "
               "probably understates it."),
    Assumption("the field",
               "Your car starts fifth in its class. That's where a strategy's "
               "effect shows up most clearly: the leader can only lose places, "
               "and a car at the back is too far from anyone to be affected."),

    Assumption("the numbers underneath",
               "Every dial comes from one running of one race per series. Not "
               "an average of that race, and certainly not an average of the "
               "series."),
    Assumption("the numbers underneath",
               "Pit times are trimmed medians, because the real data has "
               "hour-long repairs sitting in the same column as ordinary "
               "service and nothing distinguishes them."),
    Assumption("the numbers underneath",
               "At Le Mans the fitted tyre degradation for Hypercar and LMP2 "
               "comes out backwards, and the engine applies it anyway. On "
               "those two classes the tyre slider is doing something the data "
               "never measured."),
)

ASSUMPTION_GROUPS = tuple(dict.fromkeys(a.group for a in ASSUMPTIONS))


def strip_lines() -> list[str]:
    """The persistent strip on the race page."""
    return [s.short for s in STATEMENTS]


def agent_caveat() -> Statement:
    """The one that has to sit next to anything the agent does, wherever it
    appears. Named for what it was, a caveat on a result, and kept under that
    name because every call site means the same thing by it: whatever the
    policy just did on screen, no number goes with it."""
    return BY_KEY["no_agent_result"]
