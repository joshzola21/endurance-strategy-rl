"""What a caution does to the field, other than slow it down.

Notebook 01's cautions slowed every car in proportion to its own pace, so
the field kept its shape and even spread out a little: a fast car gained on
a slow one behind the safety car, which is the opposite of what a safety
car is for. A caution therefore cost nothing positionally and gained
nothing, and the whole strategic value of stopping under yellow had to be
carried by one assumed dial, `pit_caution_discount`.

Two things fix that, and both are written here as adjustments to lap times
rather than as adjustments to the running order. That is deliberate and it
is the property most worth protecting in this stage: position is derived
from accumulated race time everywhere else in the engine, and if compression
reached in and reordered cars directly, it would stop being derived.

Compression
-----------
Behind the safety car everyone runs the safety car's lap, so the first
change is that a caution lap is the same length for every car in the field
rather than a multiple of each car's own pace. On top of that, a car with
a gap to the car ahead runs a slightly shorter lap until that gap closes to
the spacing of a queue. Because the adjustment is applied to the car
*behind*, and only ever closes a positive gap towards a positive target,
the order cannot inverting - overtaking under yellow is prohibited in both
rulebooks, and here it is prohibited by the arithmetic.

Wave-arounds
------------
Both rulebooks use the same eligibility rule: a car whose class leader is
behind it in the order circulating behind the safety car (IMSA art. 46.2.2
and 46.4.1, WEC art. 14.6.4). Only a lapped car can satisfy that, which is
the point. IMSA runs it twice, once as the field forms up and once before
the restart; WEC runs it once.

A wave-by is a timing-system credit rather than a piece of physics: the car
passes the safety car, completes a lap without overtaking and rejoins at
the back, and the timing loop puts it on the leader's lap. Expressed as a
lap time, that is one short caution lap - short because the car is being
credited a crossing it would otherwise have had to wait a full safety-car
lap for. It is flagged `wave_by` in the lap record so it can be excluded
from anything that reads caution lap times as pace, because it is not pace.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CautionRules:
    """How many wave-arounds a series runs, and when.

    `unknown` is what a config gets when its `series_code` matches no
    rulebook. It runs none, because inventing one would be asserting a
    regulation nobody wrote.
    """

    series_code: str
    n_waves: int = 0
    # Caution laps before the restart at which the second wave is run. IMSA's
    # Final Wave-By happens before the field is released (art. 46.4.1).
    final_wave_lead_laps: float = 1.0

    @classmethod
    def for_series(cls, series_code: str) -> "CautionRules":
        code = (series_code or "").strip().lower()
        if code == "imsa":
            return cls("imsa", n_waves=2)      # Pass-Around and Final Wave-By
        if code == "wec":
            return cls("wec", n_waves=1)       # Pass-Around only, art. 14.6.4
        return cls("unknown", n_waves=0)

    def wave_times(self, start: float, end: float, caution_lap_s: float,
                   open_delay_laps: float) -> list[float]:
        """When each wave is announced, in race time.

        The first goes with the field forming up, which is the same moment
        the pits are declared open; the last goes shortly before the green.
        Both are dropped if the caution is too short to hold them in order.
        """
        if self.n_waves <= 0:
            return []
        times = [start + open_delay_laps * caution_lap_s]
        if self.n_waves >= 2:
            times.append(end - self.final_wave_lead_laps * caution_lap_s)
        return [t for t in times if start <= t < end]


def wave_eligible(cars: dict, progress: dict[str, float]) -> set[str]:
    """Cars a lap or more down that are ahead of their class leader on the road.

    The same rule in both rulebooks. `progress` is laps completed plus the
    fraction of the current lap done, which is the only measure that
    survives the obvious trap: a bare lap *count* differs by one between two
    cars merely spread around a lap, so counting crossings would call half
    a strung-out field lapped.

    Being ahead on the road is then a comparison of the fractional parts,
    and being genuinely lapped is a whole lap of progress. Both are needed:
    the first alone catches anyone further round the lap than the leader,
    the second alone catches lapped cars sitting behind it.

    Eligibility is frozen by the caller at the moment of announcement and
    does not change afterwards, even if the class leader then stops - which
    is what the rulebooks say.
    """
    by_class: dict[str, list[str]] = {}
    for car_id in progress:
        by_class.setdefault(cars[car_id].class_name, []).append(car_id)

    eligible: set[str] = set()
    for group in by_class.values():
        leader = max(group, key=lambda cid: progress[cid])
        lead_p = progress[leader]
        lead_frac = lead_p % 1.0
        for car_id in group:
            if car_id == leader:
                continue
            p = progress[car_id]
            if lead_p - p >= 1.0 and (p % 1.0) > lead_frac:
                eligible.add(car_id)
    return eligible


def compressed_lap_time(sc_lap_s: float, gap_s: float, queue_gap_s: float,
                        close_frac: float, floor_s: float) -> float:
    """The caution lap a car runs while closing on the car ahead.

    The lap is anchored to the safety car's lap and shortened by a share of
    the excess gap. Anchoring matters: pricing it off the lap the car ahead
    is running instead looks equivalent and is not, because each car then
    shaves a little more than the one in front and the whole queue winds
    itself down to the floor - the tail of an eighteen-car train ends up
    lapping at racing pace for the entire caution.

    With this form the next gap works out at
    `(1 - close) * gap + close * gap_ahead`, a weighted average of two
    positive numbers, so no car can compress its way past the one in front.
    Gaps settle on the queue spacing from the front backwards, which is how
    a field forms up behind a safety car.

    `floor_s` stops the arithmetic asking for a lap quicker than the car can
    physically go; when it binds the gap simply closes more slowly.
    """
    if gap_s <= 0.0:
        return sc_lap_s
    return max(sc_lap_s - (gap_s - queue_gap_s) * close_frac, floor_s)


def wave_lap_time(t: float, back_of_queue_s: float, queue_gap_s: float,
                  floor_s: float) -> float:
    """The lap that puts a waved car on the leader's lap, at the back.

    Its length is the time until the last car in the queue has passed, plus
    a car's spacing, so the waved car rejoins behind the field rather than
    in front of it - and crosses the line once more than the leader does in
    the same stretch, which is the lap it is owed.
    """
    return max(back_of_queue_s + queue_gap_s - t, floor_s)
