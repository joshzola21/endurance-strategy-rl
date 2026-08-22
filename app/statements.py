"""The four things 04 must say on the page, written once.

Settled at the 00 re-run and not optional. They appear in two places - a
persistent strip on the race page and the methods page in full - and they
are held here rather than typed into each, because a caveat that exists in
two copies is a caveat that will shortly exist in two versions. The strip
shows `short`; the page shows `full` and `source`.

The framing of the third is the diagnosed one and must stay that way. "The
agent lost" invites the reader to conclude that reinforcement learning does
not work on endurance strategy, which is not what was shown. What was shown
is that a single training run measures nothing here, and that four separate
properties of this simulator mislead a learner in ways that were measured on
the engine rather than inferred from a policy's score. Those are results
about the simulator, and they are the more interesting ones.

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
        short="Every slider starts from one running of one race.",
        full=("The dials are calibrated from a single running of a single "
              "event per series - Daytona 2026 for IMSA, Le Mans 2026 for "
              "WEC. They are not an average of that race, and they are "
              "certainly not an average of the series. The caution share "
              "moves by a factor of 2.8 between two adjacent Daytonas, so "
              "the starting point of every slider on this page is a sample "
              "of one, and the width of the sliders is a better guide to "
              "the uncertainty than their starting value is. Every dial "
              "on this page has been checked to regenerate exactly from the "
              "raw timing it was calibrated from — which was briefly not true "
              "of the two caution dials, frozen from a run made before a "
              "determinism fix landed and 0.7% away from what the fixed code "
              "returns."),
        source="00 re-run sections 3 and 6; re-run post-fix and confirmed by "
               "scripts/check_dials_provenance.py",
    ),
    Statement(
        key="degradation",
        short="Hypercar and LMP2 'degradation' at Le Mans is a within-stint "
              "trend, not tyre wear.",
        full=("The degradation dial is fitted as a linear slope of lap time "
              "against tyre age. At Le Mans that slope comes out negative "
              "for Hypercar and LMP2, which cannot be tyre wear. It is a net "
              "within-stint trend - fuel burning off, a track rubbering in, "
              "traffic thinning - and the fit cannot separate those from the "
              "tyre. The engine applies it as though it were degradation, so "
              "on those two classes the tyre lever is doing something the "
              "data did not measure."),
        source="00 re-run; blueprint section 7D",
    ),
    Statement(
        key="no_agent_result",
        short="There is no agent result: a single training run measures "
              "nothing here.",
        full=("The trained policies are shown on this page and can be watched "
              "taking decisions, but no number is reported for them, because "
              "there is not one to report. Every agent figure this project "
              "produced was a single training run. Running five instead, with "
              "everything else held fixed, showed that in WEC the training "
              "seed alone moves the headline statistic by 0.45 where two "
              "hundred paired races can only resolve 0.03 - one run of five "
              "gained a place in 45% of races and three of the others gained "
              "one in none. In IMSA all five runs came out identical, which "
              "looked like stability and was not: they had converged on never "
              "asking to pit at all, and score exactly what a car that takes "
              "no decisions scores. That car is now a strategy in its own "
              "right - `never_pit` on the comparison page - so the two rows "
              "can be read side by side, and on the fifty held-out races they "
              "agree to three decimal places. What the project did find is "
              "four "
              "measured ways this simulator misleads a learner - a stop "
              "priced below the cost of driving down the pit lane, caution "
              "compression refunding 70% of the time a stop costs, a credit "
              "horizon shorter than the window in which a caution stop pays "
              "back, and a reward defined on a quantity the agent could not "
              "see. Each was measured on the engine rather than inferred from "
              "a policy's score, so each stands on its own."),
        source="post-04 pass; decision record amendments 23-27",
    ),
    Statement(
        key="no_benchmark",
        short="There is no benchmark row: the reference is degenerate on real "
              "dials.",
        full=("02b built a two-stage reference - a clairvoyant arm that knows "
              "the caution timeline in advance and a causal arm that does "
              "not - so that a strategy could be scored against what was "
              "actually achievable rather than only against its rivals. On "
              "real dials it is degenerate: the clairvoyant arm extracts "
              "zero foreknowledge on every seed and the benchmark is beaten "
              "by its own control. Rather than show a number that does not "
              "mean what it appears to, there is no benchmark row anywhere "
              "in this app."),
        source="02b; blueprint section 3",
    ),
)

BY_KEY = {s.key: s for s in STATEMENTS}


def strip_lines() -> list[str]:
    """The persistent strip on the race page."""
    return [s.short for s in STATEMENTS]


def agent_caveat() -> Statement:
    """The one that has to sit next to anything the agent does, wherever it
    appears. Named for what it was - a caveat on a result - and kept under that
    name because every call site means the same thing by it: whatever the
    policy just did on screen, no number goes with it."""
    return BY_KEY["no_agent_result"]
