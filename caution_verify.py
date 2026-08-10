"""
02a verification harness -- the caution process, from first principles.

Self-contained: no dependency on src/endurance. This verifies the *model*
described in decision 02a, so the engine's implementation has something
independent to be checked against rather than only itself.

Checks:
  A. alternating on/off process -- realised time share is unbiased
  B. episode lengths are marginally exponential (cv = 1) and never touch
  C. a green (non-stationary) start loses about a point of share
  D. the legacy merging draw loses coverage, and merged episodes are not
     exponential
  E. rejection sampling (decision 17) reweights lengths away from exponential
  F. arithmetic reconstruction of the units chain: 0.30 lap share -> 0.19
"""

from __future__ import annotations
import numpy as np

DURATION_S = 24 * 3600.0
MEAN_DUR_S = 0.30 * DURATION_S / 8.0     # ~8 episodes in 24h at share 0.30
N_SEEDS = 3000


# ---------------------------------------------------------------- processes
def draw_alternating(rng, rate, mean_dur_s, duration_s=DURATION_S,
                     stationary_start=True):
    """Exponential green gaps, exponential episodes, a caution only able to
    start when none is running.  Non-overlapping by construction.

    For an alternating renewal process the stationary occupancy is
        rate = mean_dur / (mean_dur + mean_green)
    so the green mean is pinned by the other two.

    Returns (episodes, drawn_lengths).  Drawn lengths are pre-clipping, so the
    cv check is not contaminated by censoring at the chequered flag.
    """
    mean_green_s = mean_dur_s * (1.0 - rate) / rate
    episodes, drawn = [], []
    t = 0.0

    if stationary_start and rng.random() < rate:
        # memorylessness: the residual of an in-progress episode is itself
        # Exp(mean_dur), so opening under caution needs no special case
        length = rng.exponential(mean_dur_s)
        drawn.append(length)
        episodes.append((0.0, min(length, duration_s)))
        t = length

    while t < duration_s:
        t += rng.exponential(mean_green_s)
        if t >= duration_s:
            break
        length = rng.exponential(mean_dur_s)
        drawn.append(length)
        episodes.append((t, min(t + length, duration_s)))
        t += length

    return episodes, drawn


def draw_legacy_merged(rng, rate, mean_dur_s, duration_s=DURATION_S):
    """Pre-02a: n episodes at uniform starts, overlaps collapsed."""
    n = int(round(rate * duration_s / mean_dur_s))
    starts = rng.random(n) * duration_s
    lengths = rng.exponential(mean_dur_s, size=n)
    raw = sorted(zip(starts, np.minimum(starts + lengths, duration_s)))
    merged = []
    for s, e in raw:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def draw_rejection(rng, rate, mean_dur_s, duration_s=DURATION_S,
                   max_attempts=100000):
    """Decision 17: redraw the whole configuration until nothing overlaps."""
    n = int(round(rate * duration_s / mean_dur_s))
    for attempt in range(1, max_attempts + 1):
        starts = rng.random(n) * duration_s
        lengths = rng.exponential(mean_dur_s, size=n)
        ep = sorted(zip(starts, starts + lengths))
        ok = ep[-1][1] <= duration_s and all(
            ep[i][1] <= ep[i + 1][0] for i in range(n - 1))
        if ok:
            return ep, attempt
    return None, max_attempts


# ----------------------------------------------------------------- helpers
def share(episodes, duration_s=DURATION_S):
    return sum(e - s for s, e in episodes) / duration_s

def lengths(episodes):
    return np.array([e - s for s, e in episodes])

def cv(x):
    x = np.asarray(x, dtype=float)
    return x.std(ddof=1) / x.mean()

def touching(episodes, tol=1e-9):
    return any(episodes[i][1] >= episodes[i + 1][0] - tol
               for i in range(len(episodes) - 1))

def interior(episodes, duration_s=DURATION_S, tol=1e-6):
    return [(s, e) for s, e in episodes if s > tol and e < duration_s - tol]

def rule(title):
    print("\n" + "=" * 74); print(title); print("=" * 74)


def main():
    master = np.random.default_rng(20260804)
    seeds = lambda n: master.integers(2**63, size=n)

    rule("A + B  alternating process: share, cv, non-overlap")
    print(f"{'target':>7} {'realised':>9} {'sd':>7} {'bias':>9} {'se':>7} "
          f"{'cv(drawn)':>10} {'episodes':>9} {'touching':>9}")
    for target in (0.05, 0.18, 0.30):
        shares, pooled, counts, overlaps = [], [], [], 0
        for s in seeds(N_SEEDS):
            ep, drawn = draw_alternating(np.random.default_rng(s),
                                         target, MEAN_DUR_S)
            shares.append(share(ep)); pooled.extend(drawn)
            counts.append(len(ep)); overlaps += touching(ep)
        shares = np.array(shares)
        se = shares.std() / np.sqrt(N_SEEDS)
        print(f"{target:7.2f} {shares.mean():9.4f} {shares.std():7.4f} "
              f"{shares.mean()-target:+9.4f} {se:7.4f} {cv(pooled):10.3f} "
              f"{np.mean(counts):9.1f} {overlaps:9d}")
    print(f"\n  cv pooled over all episodes from {N_SEEDS} seeds; per-race cv "
          f"on ~8\n  episodes is biased low by ~10% whatever the model, so it "
          f"cannot\n  distinguish these processes and is not used.")

    rule("C  stationary vs green start")
    n_c = 20000
    for target in (0.18, 0.30):
        out = {}
        for stat in (True, False):
            sh = [share(draw_alternating(np.random.default_rng(s), target,
                                         MEAN_DUR_S, stationary_start=stat)[0])
                  for s in seeds(n_c)]
            out[stat] = (np.mean(sh), np.std(sh) / np.sqrt(n_c))
        tau = 1.0 / (1.0 / MEAN_DUR_S + target / (MEAN_DUR_S * (1 - target)))
        print(f"  target {target:.2f}   stationary {out[True][0]:.4f} "
              f"(+/-{out[True][1]:.4f})   green start {out[False][0]:.4f} "
              f"(+/-{out[False][1]:.4f})")
        print(f"{'':16} green-start deficit {out[True][0]-out[False][0]:.4f}, "
              f"predicted {target*tau/DURATION_S:.4f}")

    rule("D  the legacy merging draw")
    print(f"{'target':>7} {'realised':>9} {'shortfall':>10} {'1-exp(-r)':>10} "
          f"{'cv(merged)':>11}")
    for target in (0.30, 0.407):
        shares, pooled = [], []
        for s in seeds(N_SEEDS):
            ep = draw_legacy_merged(np.random.default_rng(s), target, MEAN_DUR_S)
            shares.append(share(ep)); pooled.extend(lengths(interior(ep)))
        realised = np.mean(shares)
        print(f"{target:7.3f} {realised:9.4f} "
              f"{(realised-target)/target*100:9.1f}% {1-np.exp(-target):10.4f} "
              f"{cv(pooled):11.3f}")
    print("\n  1-exp(-r) is the no-edge-effect coverage; the simulated figure "
          "sits\n  below it because episodes are also clipped at the "
          "chequered flag.")

    rule("E  rejection sampling (decision 17)")
    target = 0.30
    pooled, shares, tries = [], [], []
    for s in seeds(N_SEEDS):
        ep, att = draw_rejection(np.random.default_rng(s), target, MEAN_DUR_S)
        if ep is None:
            continue
        tries.append(att); shares.append(share(ep)); pooled.extend(lengths(ep))
    print(f"  acceptance rate    : 1 in {np.mean(tries):.1f} draws")
    print(f"  cv of lengths      : {cv(pooled):.3f}   (exponential = 1.000)")
    print(f"  realised share     : {np.mean(shares):.4f}   (target {target:.2f})")
    print(f"  mean episode       : {np.mean(pooled):.0f}s   "
          f"(drawn mean {MEAN_DUR_S:.0f}s)")

    rule("F  the units chain: a 0.30 lap share arriving at 0.19")
    lap_to_time = lambda f, m: f * m / (f * m + (1 - f))
    time_to_lap = lambda t, m: (t / m) / (t / m + (1 - t))
    legacy_time = 1 - np.exp(-0.30)
    print(f"{'mult':>6} {'true time share':>16} {'legacy time':>12} "
          f"{'reported lap':>13}")
    for m in (1.4, 1.5, 1.6, 1.7, 2.0):
        print(f"{m:6.2f} {lap_to_time(0.30, m):16.4f} {legacy_time:12.4f} "
              f"{time_to_lap(legacy_time, m):13.4f}")


if __name__ == "__main__":
    main()
