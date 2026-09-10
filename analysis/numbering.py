"""
analysis/numbering.py
What is a bar, and what is merely part of the score.

Two numberings coexist in this project. `measure_index` is ordinal position in
sounding order and counts everything. `measure_number` is the printed bar a
performer reads, and by convention several things that are unquestionably part
of the score are not bars and carry no number of their own:

  - the opening anacrusis;
  - the upbeat written *after* a repeat barline, which supplies the pickup on
    the way round -- Op. 2 No. 2/i writes bar 228 short at 1.5 of 2.0 beats and
    puts the missing 0.5 after the barline, so it can lead back into the
    exposition or on into the recapitulation;
  - a deliberately unbarred passage, such as Op. 2 No. 3/i's cadenza.

music21 is right to decline to number these; a performer would not call them
bars either. The mistake this module exists to correct is storing that refusal
as the integer `0`, which then reads as a bar number and gets printed as one --
the pipeline located Op. 2 No. 2/i's recapitulation correctly and reported it
as "mm. 0-247" because the range opened on that upbeat.

So a non-bar gets `measure_number = None`, which cannot be printed by accident,
plus a `role` saying which kind it is and the bar it `belongs_to` -- a pickup
belongs to the bar it leads into, everything else to the bar it follows. Range
reporting resolves through that, which is how a musician says it anyway: "from
the upbeat to m. 229", never "from m. 0".
"""

from __future__ import annotations
import collections
from dataclasses import dataclass

NUMBERING_VERSION = "1.0"

BAR = "bar"                 # a numbered measure
ANACRUSIS = "anacrusis"     # incomplete measure opening the movement
UPBEAT = "upbeat"           # pickup written after a repeat or ending barline
UNBARRED = "unbarred"       # passage the score deliberately leaves unbarred
EMPTY = "empty"             # no music at all: an artefact of a final barline

NON_BAR_ROLES = (ANACRUSIS, UPBEAT, UNBARRED, EMPTY)


@dataclass
class MeasureNumbering:
    measure_index: int
    measure_number: int | None   # None for anything that is not a bar
    role: str
    belongs_to: int | None       # printed bar this attaches to when reported


def _prevailing_bar_duration(rows: list[tuple[int, int | None, float, float | None]]) -> float:
    """Fallback bar length when the caller supplies no per-measure one: the
    commonest duration among measures the score does number."""
    durations = collections.Counter(
        duration for _, number, duration, _ in rows if number and duration > 0
    )
    return durations.most_common(1)[0][0] if durations else 0.0


def assign_roles(
    rows: list[tuple[int, int | None, float, float | None]]
) -> list[MeasureNumbering]:
    """Classify a movement's measures.

    Each row is `(measure_index, printed_number, total_duration,
    bar_duration)`, in score order. `printed_number` is what the importer
    produced, where 0 or None means "not numbered"; `bar_duration` is the
    meter's bar length at that point and may be None, in which case the
    commonest numbered duration stands in for it.

    An upbeat is recognised arithmetically rather than from the barline symbol:
    a partial measure that, together with the short measure before it, makes
    exactly one bar. That is what a pickup written after a barline *is*, and it
    holds whether the barline was a repeat, a double bar, or an ending.
    """
    fallback = _prevailing_bar_duration(rows)
    result: list[MeasureNumbering] = []

    for position, (index, number, duration, bar_duration) in enumerate(rows):
        if number:
            result.append(MeasureNumbering(index, number, BAR, number))
            continue

        full = bar_duration or fallback
        if duration <= 0:
            role = EMPTY
        elif position == 0:
            role = ANACRUSIS
        else:
            # Look back past empty artefacts for the measure this might complete.
            previous = next(
                (rows[j] for j in range(position - 1, -1, -1) if rows[j][2] > 0), None
            )
            previous_duration = previous[2] if previous else 0.0
            completes = full > 0 and abs(previous_duration + duration - full) < 1e-6
            role = UPBEAT if completes and duration < full else UNBARRED
        result.append(MeasureNumbering(index, None, role, None))

    _attach(result)
    return result


def _attach(rows: list[MeasureNumbering]) -> None:
    """Give every non-bar the bar it is reported against.

    A pickup leads *into* the bar that follows it; everything else continues
    from the bar before. A movement with no numbered bars at all leaves these
    None rather than inventing one.
    """
    for position, row in enumerate(rows):
        if row.role == BAR:
            continue
        if row.role in (ANACRUSIS, UPBEAT):
            following = next((r for r in rows[position + 1:] if r.role == BAR), None)
            row.belongs_to = following.measure_number if following else None
        else:
            preceding = next((r for r in reversed(rows[:position]) if r.role == BAR), None)
            row.belongs_to = preceding.measure_number if preceding else None


def resolve_range(
    rows: list[MeasureNumbering], start_index: int, end_index: int
) -> dict:
    """Report an index range in printed bars.

    Returns the bars a reader should be given, and whether the range opens on a
    pickup -- worth saying, since "from the upbeat to m. 229" is both what the
    score shows and what distinguishes this range from one starting on the
    downbeat.
    """
    by_index = {row.measure_index: row for row in rows}
    start, end = by_index.get(start_index), by_index.get(end_index)
    return {
        "measure_start": (start.measure_number if start and start.role == BAR
                          else start.belongs_to if start else None),
        "measure_end": (end.measure_number if end and end.role == BAR
                        else end.belongs_to if end else None),
        "starts_with_upbeat": bool(start and start.role in (ANACRUSIS, UPBEAT)),
    }
