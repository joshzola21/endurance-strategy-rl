"""Two separate leaks in the caution model, and a draw that has neither.

The units check showed the engine producing well under the caution load it
was calibrated to. Some of that is the lap/time units. The rest is the merge
in `CautionTimeline.draw`: overlapping episodes are collapsed, so the total
caution time realised is less than the total drawn. This measures each
leak on its own, then tests a replacement draw.

The replacement is an alternating on/off process: green gaps exponential
with mean G, caution episodes exponential with mean D, a caution only able
to start when none is running. It cannot overlap by construction, so there
is nothing to merge and nothing to reject. Its long-run caution share is
D / (D + G) exactly, so G is solved for rather than tuned.
"""

import sys

sys.path.insert(0, "/home/claude/src")

import numpy as np

from endurance.engine import CautionTimeline


DURATION = 24 * 3600.0


def drawn_vs_realised(caution_rate: float, mean_dur_s: float, n_seeds=400):
    """How much caution time the current draw loses to the merge."""
    drawn, realised, n_ep_drawn, n_ep_kept = [], [], [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        # Replicate CautionTimeline.draw's own draws so the pre-merge total
        # is visible.
        expected = (caution_rate * DURATION) / mean_dur_s
        n = rng.poisson(max(expected, 0.0))
        if n == 0:
            continue
        starts = np.sort(rng.uniform(0, DURATION, size=n))
        lengths = rng.exponential(mean_dur_s, size=n)

        merged = []
        for s, ln in zip(starts, lengths):
            e = min(s + ln, DURATION)
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        drawn.append(sum(min(s + ln, DURATION) - s for s, ln in zip(starts, lengths)))
        realised.append(sum(b - a for a, b in merged))
        n_ep_drawn.append(n)
        n_ep_kept.append(len(merged))

    return (np.mean(drawn) / DURATION, np.mean(realised) / DURATION,
            np.mean(n_ep_drawn), np.mean(n_ep_kept))


def draw_alternating(duration_s: float, caution_rate: float, mean_dur_s: float,
                     rng: np.random.Generator) -> CautionTimeline:
    """Alternating on/off cautions: exponential green gaps, exponential episodes.

    Non-overlapping by construction, so the marginal episode length stays
    exactly exponential - which is what makes the residual duration of an
    ongoing caution memoryless, and the causal benchmark exactly computable.
    """
    if caution_rate <= 0 or mean_dur_s <= 0:
        return CautionTimeline([])
    if caution_rate >= 1:
        return CautionTimeline([(0.0, duration_s)])

    mean_gap_s = mean_dur_s * (1.0 - caution_rate) / caution_rate

    periods, t = [], 0.0
    while True:
        t += rng.exponential(mean_gap_s)
        if t >= duration_s:
            break
        end = min(t + rng.exponential(mean_dur_s), duration_s)
        periods.append((t, end))
        t = end
    return CautionTimeline(periods)


def measure(draw_fn, caution_rate, mean_dur_s, n_seeds=400):
    shares, counts, lens = [], [], []
    for seed in range(n_seeds):
        tl = draw_fn(DURATION, caution_rate, mean_dur_s,
                     np.random.default_rng(seed))
        shares.append(tl.total_caution_s() / DURATION)
        counts.append(len(tl.periods))
        lens.extend([e - s for s, e in tl.periods])
    return np.mean(shares), np.mean(counts), np.mean(lens), np.array(lens)


if __name__ == "__main__":
    for label, rate, dur in [("as fed today", 0.30, 800.0),
                             ("units fixed", 0.407, 1280.0)]:
        d, r, nd, nk = drawn_vs_realised(rate, dur)
        print(f"{label}: target {rate:.3f}")
        print(f"   drawn before merge   {d:.3f}")
        print(f"   realised after merge {r:.3f}   "
              f"({100 * (1 - r / d):.1f}% lost to the merge)")
        print(f"   episodes {nd:.1f} drawn -> {nk:.1f} kept")
        print()

    rate, dur = 0.407, 1280.0
    print(f"Two draws, same target share {rate:.3f}, same mean episode {dur:.0f} s")
    for label, fn in [("current (merge)", CautionTimeline.draw),
                      ("alternating on/off", draw_alternating)]:
        share, count, mean_len, lens = measure(fn, rate, dur)
        # An exponential has cv == 1. The merge inflates it; alternating
        # should leave it alone.
        cv = lens.std() / lens.mean()
        print(f"   {label:20s} share {share:.3f}  episodes {count:5.1f}  "
              f"mean {mean_len:6.0f} s  cv {cv:.3f}")

    print()
    print("   cv of an exponential is 1.000 by definition; departures from it")
    print("   are the merge distorting the episode-length distribution.")
