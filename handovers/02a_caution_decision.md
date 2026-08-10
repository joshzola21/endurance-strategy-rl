# 02a — the caution units question, resolved

Taken in the same spirit as the 02 decisions. Not to be silently reversed.

---

## The problem

`calibrate_cautions` measured cautions in **laps** and `CautionTimeline.draw`
consumed the result as **seconds of race time**. Two errors followed, and they
compounded rather than cancelling.

**Units.** `caution_rate` was `is_caution.mean()` over the reference car's lap
records — a share of laps. The engine treats it as a share of race time. A
caution lap occupies `caution_pace_multiplier` times the wall clock of a green
one, so the two are not the same number. `caution_mean_dur_s` had the same
fault in the other direction: caution laps multiplied by *green* pace, so
episodes were drawn about a multiplier too short, and because episode count is
`rate x duration / mean_dur`, correspondingly too many.

**The merge.** Overlapping episodes were collapsed. The overlap was time the
race never ran under caution, so the realised share came in short of the drawn
one — 14% short at the old settings, 18% at corrected ones. The merge also
destroyed exactly the property 02b's benchmark rests on: the union of
overlapping exponentials is not exponential, measured as a coefficient of
variation of 1.09 where an exponential gives 1.00.

Together, on plausible Daytona figures, a calibrated 0.30 caution lap share
came out of the engine at **0.19**.

## The fix

**Calibrate in seconds, not laps.** `calibrate_cautions` now sums observed
`lap_time` over caution laps and over episodes. No conversion is performed,
so the assumed `caution_pace_multiplier` no longer sits in the middle of a
measured quantity. The observed multiplier is returned alongside as
`observed_caution_multiplier` — a measured check on an assumed dial, reported
rather than substituted, because promoting a dial out of `ASSUMED_FIELDS` is a
decision to take deliberately.

Red-flag laps are excluded: the race clock runs while the cars sit still, so a
single lap record can absorb hours. The count and total are reported rather
than dropped quietly.

**Alternating on/off caution process.** Exponential green gaps, exponential
episodes, a caution only able to start when none is running. Non-overlapping
by construction, so nothing is merged and nothing is rejected.

This supersedes decision 17's redraw-until-non-overlapping. Rejection sampling
conditions the whole configuration on non-overlap, which reweights towards
shorter length vectors — so lengths stop being marginally exponential and the
residual duration of an ongoing caution stops being memoryless, which is the
property the decision existed to protect. The alternating process gets
non-overlap for free and keeps memorylessness exactly. Same intent, no
conditioning artefact.

The process starts in its stationary state — with probability `caution_rate`
the race opens under caution. Without that the timeline always begins green
and the realised share comes in about a point light.

Verified: realised share unbiased at 0.05, 0.18 and 0.30 over 3000 seeds;
episode-length cv 1.00; no two episodes touch.

## Switches

- `legacy_cautions=True` restores the merging draw.
- `split_streams=False` restores 01's single shared generator.
- Both together reproduce 01 bit for bit, and there is a test asserting it.

The two are separate on purpose. Under one shared generator the caution draw
and the field build consume from the same stream, so changing the caution
model reassigns every car's base pace as a side effect and the before/after
comparison measures both at once. Split streams make the caution change the
only thing that moves.

## Consequences

**Notebook 01 Part 6 compares a simulated caution *lap* share against a real
caution *lap* share.** Both sides should become time shares. That check is one
of the few places the units error was visible, so it is worth reporting what
it said before as well as after.

**Caution load varies a lot seed to seed** — a standard deviation of roughly
0.07 on the realised share, with about eight episodes in a 24-hour race. That
is honest to endurance racing, but it means caution-sensitive claims carry
real seed noise, and the 50-seed sweep budget in decision 11 will be noticeably
noisier than the 200-seed headline. Worth checking before reading a sweep
curve as a trend.

**Stops fall.** Longer, better-calibrated cautions mean more laps run at
caution fuel burn, so fewer stops and less pit time. Every number in 01's
lever table moves.

## Still open

- `flags != 'GF'` lumps full-course cautions, safety cars and WEC slow zones
  together. One multiplier averages over regimes that are not alike. A
  per-flag breakdown is cheap and would say whether it matters.
- Ordering: the per-car noise streams should land **before** the caution
  before/after is reported. Lap noise and pit cost still come off the shared
  generator during the race, so a caution change still perturbs them. The
  caution comparison is attributable in pace but not yet in noise.
