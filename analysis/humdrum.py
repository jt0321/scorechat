"""
analysis/humdrum.py
Reading the Humdrum source directly, spine by spine.

`music21` is the project's parser, but it is not a faithful reader of
everything a `.krn` encodes. It discards the expansion records `sections.py`
recovers; it misreads the `**dynam` spine, cramming hairpin tokens into
`Dynamic` objects so a crescendo *span* is destroyed rather than captured; and
it estimates a home key statistically for scores that state theirs outright.
This module reads those facts from the text, so the score speaks for itself.

The reason it cannot be done by column index -- which is how a first attempt at
the dynamics went wrong -- is that Humdrum spines split and rejoin. `*^` turns
one spine into two and every column to its right shifts; a run of `*v` merges
them back. All 103 movements in this corpus do it, 2,396 splits and 4,772
joins in total, so the third column is `**dynam` only until the first left-hand
arpeggio divides a staff. `spine_layout()` tracks that properly and every
reader here goes through it.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterator

HUMDRUM_SOURCE_VERSION = "1.0"

_DECLARED_KEY = re.compile(r"^\*([a-gA-G])([#-]?):\s*$")
_BARLINE = re.compile(r"^=+(\d+)")
_METER = re.compile(r"^\*M(\d+)/(\d+)$")

# Humdrum spells a key with a letter case that carries the mode -- lower case
# is minor -- and "-" for a flat, which is how the corpus writes D-flat major
# as "*D-:". The database stores "D- major", so the two meet here.
_ACCIDENTAL = {"#": "#", "-": "-", "": ""}


@dataclass
class Token:
    """One non-null token, with the spine it belongs to and the bar it is in."""
    line: int
    measure: int | None      # printed bar number in force, None before bar 1
    spine: int               # position in the layout at this line
    exclusive: str           # "**kern", "**dynam"
    text: str


def spine_layout(source_text: str) -> Iterator[tuple[int, str, list[str], list[str]]]:
    """Walk the file, yielding `(line_number, line, spine_types, tokens)`.

    `spine_types` is the exclusive interpretation of each column *on that
    line*, after applying any split, join or termination the line itself
    performs. Comment lines yield the layout unchanged.
    """
    types: list[str] = []
    for number, line in enumerate(source_text.splitlines()):
        if line.startswith("!!"):
            yield number, line, list(types), []
            continue
        tokens = line.split("\t")
        if line.startswith("**"):
            types = [token for token in tokens]
            yield number, line, list(types), tokens
            continue
        yield number, line, list(types), tokens
        if line.startswith("*"):
            types = _apply(types, tokens)


def _apply(types: list[str], tokens: list[str]) -> list[str]:
    """The spine layout after an interpretation line.

    `*^` replaces one spine with two of the same type. A *run* of adjacent `*v`
    merges those spines into one -- the count matters, which is why this is a
    scan rather than a map. `*-` terminates. Anything else leaves the layout
    alone.
    """
    if len(tokens) != len(types):
        return types  # malformed line: keep the layout rather than corrupt it
    updated: list[str] = []
    index = 0
    while index < len(tokens):
        token, kind = tokens[index], types[index]
        if token == "*^":
            updated.extend([kind, kind])
            index += 1
        elif token == "*v":
            run = index
            while run < len(tokens) and tokens[run] == "*v":
                run += 1
            updated.append(kind)      # the merged spine keeps the type
            index = run
        elif token == "*-":
            index += 1                # terminated: contributes no spine
        else:
            updated.append(kind)
            index += 1
    return updated


def iter_tokens(source_text: str, exclusive: str = "**kern") -> Iterator[Token]:
    """Every data token belonging to spines of one exclusive interpretation.

    Null tokens (".") are skipped -- they mean "the previous token is still
    sounding", not an event.
    """
    measure: int | None = None
    for number, line, types, tokens in spine_layout(source_text):
        if line.startswith(("!", "*")):
            continue
        if line.startswith("="):
            match = _BARLINE.match(tokens[0] if tokens else "")
            measure = int(match.group(1)) if match else measure
            continue
        for index, token in enumerate(tokens):
            if index >= len(types) or types[index] != exclusive:
                continue
            if token in (".", ""):
                continue
            yield Token(number, measure, index, exclusive, token)


def declared_key(source_text: str) -> str | None:
    """The key the score states, as `*f:` or `*D-:`, in the database's spelling.

    Every movement in this corpus declares one, and it is notated rather than
    inferred -- so it settles a question the Krumhansl estimate gets wrong for
    six movements, most often by reporting the relative major.
    """
    for _, line, _, tokens in spine_layout(source_text):
        if not line.startswith("*") or line.startswith("**"):
            continue
        for token in tokens:
            match = _DECLARED_KEY.match(token)
            if match:
                letter, accidental = match.group(1), _ACCIDENTAL[match.group(2)]
                tonic = letter.upper() + accidental
                return f"{tonic.lower()} minor" if letter.islower() else f"{tonic} major"
    return None


def meter_changes(source_text: str) -> dict[int, str]:
    """Printed bar number -> the time signature that starts there.

    music21 loses these for the movements that need them most. Op. 111's
    Arietta is written 9/16, 6/16, 12/32, 9/16 as the variations subdivide the
    beat; only the opening 9/16 survived the parse, so every later bar was
    measured against a bar length half again too long and read as incomplete.
    Op. 110/iii and Op. 53/iii lose changes the same way -- 3 of 103 movements,
    all of them places where the meter is doing something worth reading.

    A bar length is what tells an upbeat from a whole bar in a faster meter
    (`analysis/numbering.py`), so a stale meter does not merely mislabel the
    metre: it invents anacruses.
    """
    meters: dict[int, str] = {}
    measure = 0
    pending: str | None = None
    for _, line, types, tokens in spine_layout(source_text):
        if line.startswith("!"):
            continue
        if line.startswith("=") and tokens:
            match = _BARLINE.match(tokens[0])
            if match:
                measure = int(match.group(1))
                if pending is not None:
                    meters[measure] = pending
                    pending = None
            continue
        if line.startswith("*") and not line.startswith("**"):
            for token in tokens:
                match = _METER.match(token)
                if match:
                    signature = f"{match.group(1)}/{match.group(2)}"
                    # A meter written before the first barline belongs to the
                    # opening measure; one written after a barline has already
                    # been passed applies from that bar.
                    if measure:
                        meters[measure] = signature
                    else:
                        pending = signature
                        meters[0] = signature
                    break
    return meters


def bar_duration(time_signature: str | None) -> float | None:
    """Quarter-note length of one bar, e.g. "6/16" -> 1.5."""
    if not time_signature:
        return None
    try:
        beats, unit = time_signature.split("/")
        return int(beats) * 4.0 / int(unit)
    except (ValueError, ZeroDivisionError):
        return None
