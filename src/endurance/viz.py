"""Views of a race, and of a comparison.

Kept out of the notebooks so that the same charts can be dropped straight
into the app later. Every function returns a bare matplotlib figure -
nothing here knows or cares whether it is being drawn in Jupyter or in
Streamlit, and nothing here is allowed to compute anything a table does not
already contain.

Two families live here now. The 01 and 02a plots take a `RaceResult` and
show one race. The 02c plots take the frames `harness.py` produces and show
a comparison across a seed bank: the paired-delta distribution decision 10
asks for, the gap to 02b's benchmark, and decision 11's sweep responses.

Three rules the 02c plots keep, each of them a decision rather than a
preference:

* **One series per figure.** Decision 10's budget is per strategy per
  series, and the rulebooks differ in ways that make a lever live in one and
  dead in the other. `_one_series` raises rather than quietly pooling.
* **No means of a position delta.** Decision 10 asks for a distribution.
  These plots draw shares and quantiles, and there is no argument to make
  them draw a mean.
* **The benchmark is optional.** 02b's per-seed plans cost 50-130 s a seed,
  so the gap plot draws a labelled placeholder when the cache is absent
  rather than raising. A notebook has to run end to end before the
  benchmark does.

The palette, and where it does and does not reach
-------------------------------------------------
This module is the app's charts *and* the notebooks' charts, so the colours
below apply in both places and `.streamlit/config.toml` reaches neither of
them: `st.pyplot` renders a figure that was drawn before Streamlit saw it.
The two files carry the same hexes and have to be edited together.

Style is applied per figure through `@_styled` rather than by writing to
`plt.rcParams` at import. A module that restyles matplotlib the moment it is
imported changes every other figure in whatever notebook imported it, which
is a side effect nobody asked for and nobody can see.
"""

from __future__ import annotations

import functools

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from .engine import RaceResult


# ----------------------------------------------------------------------
# The palette
# ----------------------------------------------------------------------
# Contrast ratios are against GROUND and are measured, not judged. A mark
# needs 3:1 to be a legible graphical object and text needs 4.5:1, which is
# why ACCENT draws bars and ACCENT_TEXT writes words.
INK = "#2A2730"          # 13.6:1 - axis labels, titles, tick text
GROUND = "#F7F6FA"       # the page the app draws on
SILVER_TEXT = "#6E6A78"  #  4.9:1 - annotations, secondary labels
SILVER_MARK = "#8A8694"  #  3.3:1 - percentile ranges, rules, guides
SILVER_RULE = "#DCD8E4"  # grid lines only
ACCENT = "#8968CD"       #  4.0:1 - fills and marks, never text
ACCENT_TEXT = "#7A57BC"  #  5.4:1 - the darkened counterpart, for words

# Signed so that green is better in every plot below. A position delta of
# +1 is a place gained, which is a *lower* class_pos - getting that the
# wrong way round produces a chart that reads correctly and says the
# opposite of what happened.
#
# **Reconsidered at 06 and kept, deliberately.** The objection is real:
# roughly one man in twelve cannot separate green from red, and this is the
# figure carrying the strongest result the project has. A purple-and-orange
# pair was built and rejected on the grounds that green-for-good is the
# convention every reader of a motorsport chart already has, and that
# spending it buys accessibility for a figure whose meaning is also carried
# by the bar order and the axis label. Recorded here so the next person to
# notice the problem finds the decision rather than re-making it.
#
# `_LEVEL` did move, from a bare `"0.75"` grey to the house silver, because
# nothing depended on that one.
_GAINED = "tab:green"
_LEVEL = "#C3BFCB"
_LOST = "tab:red"

# Yellow means yellow. Muted from matplotlib's `gold`, which at alpha 0.25
# over GROUND was the brightest thing on a chart about something else.
_CAUTION = "#E8C56A"

# For per-car lines: up to eight cars in `plot_race`, so hue has to do the
# separating and a purple ramp would be unreadable. The accent leads and the
# rest are chosen to stay apart in greyscale as well as in colour.
_CYCLE = ("#8968CD", "#3E8A8F", "#C8663C", "#4A6BB5",
          "#9C5A8C", "#7A8C3F", "#B8963C", "#6E6A78")

# The sweep grid's surface. Light where the statistic is low, so the cell
# labels can be dark where the cell is pale - see `_label_colour`.
_SURFACE = LinearSegmentedColormap.from_list("endurance_purples", [
    "#F2EEF9", "#E2D9F2", "#CFC0E8", "#BBA6DE", "#A88DD4",
    "#9575CA", "#7F5CB6", "#68489A", "#52397A", "#3C2A5A"])

# Matplotlib cannot fetch a webfont, so it draws with whatever is installed.
# Plex is named first because a development machine with it installed then
# matches the app's page type exactly; the deployed host almost certainly
# falls back to DejaVu, and a chart in DejaVu beside a page in Plex is a
# tolerable mismatch. Making them match on the host means committing the
# font file and registering it here, which is a binary in the repository for
# a difference nobody has complained about.
HOUSE_RC = {
    "figure.facecolor": GROUND,
    "figure.dpi": 110,
    "savefig.facecolor": GROUND,
    "savefig.bbox": "tight",
    "axes.facecolor": GROUND,
    "axes.edgecolor": SILVER_MARK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": plt.cycler(color=list(_CYCLE)),
    "text.color": INK,
    "xtick.color": SILVER_MARK,
    "ytick.color": SILVER_MARK,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.color": SILVER_RULE,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "font.family": "sans-serif",
    "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans", "sans-serif"],
}


_INK_LUMINANCE = 0.0216   # `_luminance(INK)`, held as a constant so that
                          # `_label_colour` is not recomputing it per cell


def _styled(fn):
    """Draw this figure under the house style and leave rcParams alone.

    The context has to wrap the whole function rather than just the call to
    `subplots`: matplotlib reads rcParams when each artist is created, so a
    context closed after the axes exist would style the axes and none of the
    lines on them.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.rc_context(HOUSE_RC):
            return fn(*args, **kwargs)
    return wrapper


def _luminance(colour) -> float:
    """WCAG relative luminance, for choosing what can be read on what."""
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
               for c in to_rgb(colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _label_colour(rgba) -> str:
    """Ink or paper, whichever can actually be read on this cell.

    The old grid wrote every label in white, including over the pale end of
    the surface where white on near-white is invisible. That is worse than
    an unlabelled grid, because the label is there and the reader assumes
    they have seen it.

    Both contrasts are computed and the better one wins, rather than a
    threshold being picked by eye. A guessed threshold is wrong in the
    middle of the ramp, which is exactly where the cells are hardest to
    read and where a guess feels safest.
    """
    cell = _luminance(rgba[:3])
    on_ink = (max(cell, _INK_LUMINANCE) + 0.05) / (min(cell, _INK_LUMINANCE) + 0.05)
    on_paper = (1.0 + 0.05) / (cell + 0.05)
    return INK if on_ink >= on_paper else "#FFFFFF"


def _caution_bands(ax, result: RaceResult, hours: bool = True) -> None:
    scale = 3600.0 if hours else 1.0
    for i, (start, end) in enumerate(result.cautions.periods):
        ax.axvspan(start / scale, end / scale, color=_CAUTION, alpha=0.35,
                   linewidth=0, label="caution" if i == 0 else None)


@_styled
def plot_race(result: RaceResult, class_name: str | None = None,
              top_n: int = 8, figsize=(11, 6)):
    """Class position over race time - the story of who was winning, when."""
    pos = result.positions()
    classification = result.classification()
    if class_name:
        pos = pos[pos["class"] == class_name]
        classification = classification[classification["class"] == class_name]

    leaders = classification.head(top_n)["car_id"].tolist()

    fig, ax = plt.subplots(figsize=figsize)
    _caution_bands(ax, result)

    for car_id in leaders:
        sub = pos[pos["car_id"] == car_id]
        ax.plot(sub["t"] / 3600, sub["class_position"], linewidth=1.4, label=car_id)

    ax.invert_yaxis()
    ax.set_xlabel("race time (hours)")
    ax.set_ylabel("position in class")
    ax.set_title(f"{result.config.name} - {class_name or 'all classes'}")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(alpha=0.6)
    fig.tight_layout()
    return fig


@_styled
def plot_gaps(result: RaceResult, class_name: str, top_n: int = 6, figsize=(11, 6)):
    """Gap to the class leader, in seconds - where races are actually won."""
    gaps = result.gap_to_leader(class_name)
    classification = result.classification()
    order = classification[classification["class"] == class_name]["car_id"].tolist()
    keep = [c for c in order[:top_n] if c in gaps.columns]

    fig, ax = plt.subplots(figsize=figsize)
    for car_id in keep:
        ax.plot(gaps.index, gaps[car_id], linewidth=1.4, label=car_id)

    ax.set_xlabel("lap")
    ax.set_ylabel("gap to leader (s)")
    ax.set_title(f"Gap to class leader - {class_name}")
    ax.invert_yaxis()
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.6)
    fig.tight_layout()
    return fig


@_styled
def plot_gap_normalisation(gaps, scales: dict[str, float],
                           figsize=(11, 4.5)):
    """Where the gap observations actually land, under each candidate scale.

    `gaps` is one row per decision point with a `seconds` column and a `row`
    column naming which observation it is, which is the frame 03a's notebook
    builds by walking `run_stream` and asking `RaceState.gap_ahead_s` and
    `gap_behind_s`. `scales` maps a label to the seconds that normalisation
    treats as 1.0 - the blueprint's flat 120 against `pit_time_mean_s`.

    Left panel is the distribution itself, in seconds, with each candidate
    scale drawn as a rule. Right panel is the consequence: the span from the
    first to the ninety-ninth percentile once divided and clipped, which is
    how much of the unit interval the row ever occupies. A row whose
    observations all sit in the bottom few hundredths of its range is a
    near-constant input, and a policy network learns little from one.

    The quantiles are computed here, as `plot_paired_deltas` computes its
    own - a quantile of the frame handed in is a view of that frame rather
    than a second calculation of it. Nothing is read from the engine.

    Rows where nobody was within a lap either side carry no gap at all.
    Those are dropped and their share reported in the title, because "there
    was nobody there" is a different statement from a wide gap even though
    both normalise to 1.0.
    """
    if "seconds" not in gaps.columns or "row" not in gaps.columns:
        raise ValueError("gaps needs `row` and `seconds` columns")

    present = gaps.dropna(subset=["seconds"])
    missing = 1.0 - len(present) / max(len(gaps), 1)
    names = sorted(present["row"].unique())
    if not names:
        return _placeholder("no gaps were recorded - was a focal car set?",
                            figsize)

    fig, (ax_secs, ax_range) = plt.subplots(1, 2, figsize=figsize)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # -- left: the distribution, in the unit it was measured in ----------
    for colour, name in zip(colours, names):
        seconds = present.loc[present["row"] == name, "seconds"].sort_values()
        ax_secs.plot(seconds.values,
                     [i / len(seconds) for i in range(len(seconds))],
                     color=colour, linewidth=1.6, label=name)
    for style, (label, scale) in zip(("--", ":", "-."), sorted(scales.items())):
        ax_secs.axvline(scale, color=SILVER_MARK, linewidth=1.0, linestyle=style)
        ax_secs.text(scale, 0.02, f" {label}", fontsize=8, color=SILVER_TEXT,
                     rotation=90, va="bottom")
    ax_secs.set_xlabel("gap to the nearest class rival (s)")
    ax_secs.set_ylabel("share of decision points")
    ax_secs.set_title("Where the gaps actually are")
    ax_secs.legend(fontsize=8, frameon=False)

    # -- right: how much of the row each scale uses ----------------------
    labels, y = [], 0
    for label, scale in sorted(scales.items()):
        for colour, name in zip(colours, names):
            seconds = present.loc[present["row"] == name, "seconds"]
            scaled = (seconds / max(scale, 1e-9)).clip(0.0, 1.0)
            lo, med, hi = (float(scaled.quantile(0.01)),
                           float(scaled.median()),
                           float(scaled.quantile(0.99)))
            ax_range.plot([lo, hi], [y, y], color=colour, linewidth=4,
                          solid_capstyle="butt", alpha=0.75)
            ax_range.plot([med], [y], marker="|", markersize=12, color=INK)
            labels.append(f"{name} / {label}")
            y += 1
        y += 0.5

    ax_range.set_yticks([i + (i // len(names)) * 0.5 for i in range(len(labels))],
                        labels, fontsize=8)
    ax_range.set_xlim(0, 1)
    ax_range.set_xlabel("normalised value (p1 to p99, median marked)")
    ax_range.set_title("How much of the row is used")
    ax_range.grid(alpha=0.6, axis="x")

    fig.suptitle(f"Gap observations - {len(present)} decision points, "
                 f"{missing:.0%} with nobody within a lap", fontsize=11,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


@_styled
def plot_stint_pace(result: RaceResult, class_name: str, figsize=(9, 5)):
    """Lap time against tyre age - the degradation dial, as the race saw it.

    This is the same view as notebook 00's stint-pace plot, but produced by
    the simulator rather than the data, which makes the two directly
    comparable.
    """
    laps = result.laps
    laps = laps[(laps["class"] == class_name)
                & (~laps["under_caution"]) & (~laps["pitted"])]
    by_age = laps.groupby("tyre_age")["lap_time"].mean()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(by_age.index, by_age.values, marker="o", markersize=3,
            linewidth=1.2, color=ACCENT)
    ax.set_xlabel("tyre age (laps into stint)")
    ax.set_ylabel("mean green lap time (s)")
    ax.set_title(f"Simulated stint pace - {class_name}")
    ax.grid(alpha=0.6)
    fig.tight_layout()
    return fig


@_styled
def plot_lever_sweep(sweep: pd.DataFrame, x: str, y: str,
                     label: str | None = None, figsize=(9, 5)):
    """What happens to an outcome as one dial is twisted."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(sweep[x], sweep[y], marker="o", color=ACCENT)
    ax.set_xlabel(label or x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} against {label or x}")
    ax.grid(alpha=0.6)
    fig.tight_layout()
    return fig


@_styled
def plot_strategy_comparison(results: dict[str, RaceResult], class_name: str,
                             figsize=(9, 5)):
    """Laps completed by each strategy, run on identical races.

    Because the caution timeline is drawn before the race and does not
    depend on what anyone does, the same seed gives every strategy exactly
    the same race. The spread here is therefore strategy, not luck.
    """
    rows = []
    for name, res in results.items():
        c = res.classification()
        c = c[c["class"] == class_name]
        rows.append({"strategy": name,
                     "mean_laps": c["laps"].mean(),
                     "mean_stops": c["stops"].mean(),
                     "pit_time_s": c["pit_time_s"].mean(),
                     "traffic_time_s": c["traffic_time_s"].mean()})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].bar(df["strategy"], df["mean_laps"] - df["mean_laps"].min(),
                color=ACCENT)
    axes[0].set_ylabel(f"laps gained vs worst ({df['mean_laps'].min():.1f})")
    axes[0].set_title("Laps completed")
    axes[0].tick_params(axis="x", rotation=20)

    # Teal rather than the orange that means "lost a place" elsewhere in
    # this module. Time in the pits is a cost and not an outcome, and one
    # colour cannot mean both without the reader learning it twice.
    axes[1].bar(df["strategy"], df["pit_time_s"], color="#3E8A8F")
    axes[1].set_ylabel("mean time in the pits (s)")
    axes[1].set_title("Pit time")
    axes[1].tick_params(axis="x", rotation=20)

    for ax in axes:
        ax.grid(alpha=0.6, axis="y")
    fig.tight_layout()
    return fig, df


# ----------------------------------------------------------------------
# 02c: the comparison
# ----------------------------------------------------------------------
def _one_series(rows: pd.DataFrame) -> str:
    """Refuse a frame carrying more than one series.

    Enforced here as well as in `harness.summarise` because a figure is the
    thing that ends up in the write-up, and a pooled bar is a claim nobody
    checks once it has been drawn.
    """
    series = sorted(rows["series"].unique())
    if len(series) != 1:
        raise ValueError(
            f"one series per figure; got {series}. Decision 10's budget is "
            "per strategy per series and the two are not pooled.")
    return series[0]


@_styled
def _placeholder(message: str, figsize):
    """A figure that says why it is empty, rather than an exception.

    Used where an artefact the notebook does not control may not exist yet.
    An empty axis with no explanation is worse than no plot at all.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True,
            fontsize=10, color=SILVER_TEXT)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


@_styled
def plot_paired_deltas(rows: pd.DataFrame, null_name: str = "fuel_window",
                       figsize=(11, 5)):
    """Decision 10's headline: how often a strategy gains, and by how much.

    Left panel is the sentence the decision record asks for - "gains a place
    in 40% of races, loses one in 12%" - drawn rather than written. Right
    panel is the spread behind it, as a median with a tenth-to-ninetieth
    percentile range, because two strategies can gain equally often and
    differ entirely in what happens when they do not.

    The null is drawn with the rest rather than dropped. Its row is
    identically zero by construction, so it is the verification gate sitting
    in the middle of the figure: if that bar is not wholly grey, the paired
    comparison is not paired and nothing else in the chart means anything.
    """
    series = _one_series(rows)
    order = (rows.groupby("strategy")["d_class_pos"]
                 .apply(lambda d: (d > 0).mean() - (d < 0).mean())
                 .sort_values().index.tolist())

    fig, (ax_share, ax_spread) = plt.subplots(1, 2, figsize=figsize)
    y = range(len(order))

    lost, level, gained, med, lo, hi = [], [], [], [], [], []
    for name in order:
        d = rows.loc[rows["strategy"] == name, "d_class_pos"]
        lost.append(float((d < 0).mean()))
        level.append(float((d == 0).mean()))
        gained.append(float((d > 0).mean()))
        med.append(float(d.median()))
        lo.append(float(d.quantile(0.10)))
        hi.append(float(d.quantile(0.90)))

    ax_share.barh(y, lost, color=_LOST, label="lost a place or more")
    ax_share.barh(y, level, left=lost, color=_LEVEL, label="level")
    ax_share.barh(y, gained, left=[a + b for a, b in zip(lost, level)],
                  color=_GAINED, label="gained a place or more")
    ax_share.set_yticks(list(y), order)
    ax_share.set_xlim(0, 1)
    ax_share.set_xlabel("share of races")
    ax_share.set_title(f"Paired outcome against {null_name}")

    for i, name in enumerate(order):
        ax_spread.plot([lo[i], hi[i]], [i, i], color=SILVER_MARK, linewidth=1.5,
                       solid_capstyle="butt")
        ax_spread.plot([med[i]], [i], marker="o", markersize=6,
                       color=_GAINED if med[i] > 0 else
                       (_LOST if med[i] < 0 else SILVER_TEXT))
    ax_spread.axvline(0, color=INK, linewidth=0.8)
    ax_spread.set_yticks(list(y), ["" for _ in order])
    ax_spread.set_xlabel("class positions gained (median, p10-p90)")
    ax_spread.set_title("Spread behind the shares")

    for ax in (ax_share, ax_spread):
        ax.grid(alpha=0.6, axis="x")
    n = int(rows.groupby("strategy").size().max())
    fig.suptitle(f"{series.upper()} - {n} paired races", fontsize=11, color=INK)
    # In the footer rather than inside the axis: the bars always span the
    # full width, so any in-axis placement covers a strategy - and the one
    # it covers is whichever sorted to the bottom, which is the strategy the
    # reader most needs to see.
    fig.legend(*ax_share.get_legend_handles_labels(), loc="lower center",
               ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


@_styled
def plot_benchmark_gap(rows: pd.DataFrame, figsize=(9, 5)):
    """How far each strategy finished behind the per-race reference.

    Takes the frame after `harness.attach_benchmark`. Zero means the
    strategy matched what the benchmark achieved on that race; positive is
    positions behind it, so shorter is better and the axis runs the
    intuitive way.

    Read with 02b's third finding in mind. That stage found its top twenty
    plans spanning six seconds of predicted time against thirty seconds of
    model error, and concluded that a ranking whose spread is smaller than
    its error is noise. The same question applies here: if the strategies'
    medians sit inside one another's p10-p90 ranges, this figure shows they
    are not separated, and the honest reading is that the roster is not
    ordered rather than that it is ordered narrowly.
    """
    if "gap_to_benchmark_pos" not in rows.columns:
        return _placeholder(
            "No benchmark cache attached.\n\n"
            "02b's per-seed plans cost 50-130 s a seed, so they are produced "
            "by a script and read from disk.\nRun it, pass the cache through "
            "harness.attach_benchmark, and this figure fills in.",
            figsize)

    series = _one_series(rows)
    order = (rows.groupby("strategy")["gap_to_benchmark_pos"]
                 .median().sort_values().index.tolist())

    fig, ax = plt.subplots(figsize=figsize)
    for i, name in enumerate(order):
        g = rows.loc[rows["strategy"] == name, "gap_to_benchmark_pos"].dropna()
        if g.empty:
            continue
        ax.plot([g.quantile(0.10), g.quantile(0.90)], [i, i],
                color=SILVER_MARK, linewidth=1.5, solid_capstyle="butt")
        ax.plot([g.median()], [i], marker="o", markersize=6, color=ACCENT)

    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("class positions behind the benchmark (median, p10-p90)")
    ax.set_title(f"{series.upper()} - strategy against the per-race reference")
    ax.grid(alpha=0.6, axis="x")
    fig.tight_layout()
    return fig


@_styled
def plot_sweep_response(sweep: pd.DataFrame, statistic: str = "gained",
                        figsize=(9, 5)):
    """Decision 11's one-at-a-time sweep: does the claim move with the dial?

    Takes the frame from `harness.sweep_dial`. One line per strategy against
    the multiplier, with the default marked, because the question decision
    11 asks is not "what is the value here" but "is this claim invariant or
    dependent" - and that is a question about the shape of a line, not about
    a point on it.

    `gained` by default rather than the median delta. On fifty seeds a
    median position delta moves in whole places and steps visibly when
    nothing has really happened; a share moves smoothly and is what the
    headline claim is stated in anyway.
    """
    series = _one_series(sweep)
    dial = sorted(sweep["dial"].unique())
    fig, ax = plt.subplots(figsize=figsize)

    # A sweep by value carries no multiplier - `sweep_dial` leaves the column
    # empty rather than back-computing one off a zero default - so the x axis
    # is chosen from the frame instead of assumed. The default marker moves
    # with it: on a multiplier axis the calibrated point is 1.0, and on a
    # value axis it is whatever the dial actually is.
    by_value = ("multiplier" not in sweep.columns
                or sweep["multiplier"].isna().all())
    x = "value" if by_value else "multiplier"
    if x not in sweep.columns:
        raise ValueError(f"this sweep frame has neither a 'multiplier' nor a "
                         f"'value' column: {list(sweep.columns)}")

    for name, g in sweep.groupby("strategy", sort=False):
        g = g.sort_values(x)
        ax.plot(g[x], g[statistic], marker="o", markersize=4,
                linewidth=1.4, label=name)

    if not by_value:
        ax.axvline(1.0, color=SILVER_MARK, linewidth=0.8, linestyle="--")
        ax.annotate("calibrated", xy=(1.0, ax.get_ylim()[1]), xytext=(3, -10),
                    textcoords="offset points", fontsize=8, color=SILVER_TEXT)
    ax.set_xlabel(f"{'value of' if by_value else 'multiplier on'} "
                  f"{', '.join(dial)}")
    ax.set_ylabel(statistic.replace("_", " "))
    ax.set_title(f"{series.upper()} - sweep response")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.6)
    fig.tight_layout()
    return fig


@_styled
def plot_sweep_grid(grid: pd.DataFrame, strategy: str,
                    dial_a: str, dial_b: str, statistic: str = "gained",
                    figsize=(6.5, 5)):
    """Decision 11's 2-D grid, for one strategy at a time.

    Beyond the three plots the decision record names, because decision 11
    asks for the grid as well and a pair of one-at-a-time lines cannot show
    what it shows: `pit_caution_discount` and the caution rate interact by
    construction - how often a caution arrives and how much a caution stop
    saves - and the interaction is the whole caution story.

    One strategy per figure. A grid is already two dimensions and a third
    drawn on top of it is a table pretending to be a picture.
    """
    series = _one_series(grid)
    sub = grid[grid["strategy"] == strategy]
    if sub.empty:
        return _placeholder(f"no rows for strategy {strategy!r}", figsize)

    table = sub.pivot(index=dial_b, columns=dial_a, values=statistic)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(table.values, origin="lower", aspect="auto", cmap=_SURFACE)
    ax.set_xticks(range(len(table.columns)), [f"{v:g}" for v in table.columns])
    ax.set_yticks(range(len(table.index)), [f"{v:g}" for v in table.index])
    ax.set_xlabel(f"multiplier on {dial_a}")
    ax.set_ylabel(f"multiplier on {dial_b}")
    ax.set_title(f"{series.upper()} - {strategy}, {statistic.replace('_', ' ')}")

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color=_label_colour(im.cmap(im.norm(value))))

    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, label=statistic.replace("_", " "))
    fig.tight_layout()
    return fig
