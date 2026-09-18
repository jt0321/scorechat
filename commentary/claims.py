"""
commentary/claims.py
Claims read out of commentary passages, and what the score says about them.

Three kinds of claim are extracted, and only one is checkable:

    key       "an allegretto, D flat major, follows" -- checked
    bar_ref   "at measure 19" -- the author's own edition; a location hint only
    form      "the second theme", "the development", "the coda" -- interpretive

A key claim is checked in two steps, and the order matters. First against the
*engraved* key -- the one each movement declares (`*f:`), stored as
`global_key`. Agreement there is `supported`. Disagreement is not a
contradiction: "E-flat major" in a passage about a C minor movement may mean it
modulates there. So a key that is not the home key is compared with the
*estimated* key regions instead, and the outcome says exactly that --
`agrees_with_estimate` or `disagrees_with_estimate`. The estimator smooths
away brief tonicisations and is known to be sticky (22 of 103 movements report
a single region), so a disagreement is as likely to be the estimator's fault as
the author's; `relative_of` records when the claimed key is the relative of an
estimated one, the estimator's most common confusion.

Meters are not extracted: the OCR renders Elterlein's printed fractions as
"f time", "a time", and nothing honest can be recovered from that. Tempo
markings are not claims either -- they are what movements are *found* by
(`anchoring.find_cues`), and checking a passage against the cue that placed it
would confirm it by construction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from analysis.harmony import parse_key_name

_KEY = re.compile(
    r"\b(?P<tonic>(?-i:[A-G]))(?:\s*[-‐]?\s*(?P<acc>sharp|flat|♯|♭|#))?\s*[-‐]?\s*(?P<mode>major|minor)\b",
    re.IGNORECASE)
_NAMING = re.compile(r"^\W{0,3}(?:\w+\W+){0,2}?Sonata\b", re.IGNORECASE)
_BAR = re.compile(r"\b(?P<unit>bars?|measures?)\s+(?P<first>\d{1,3})"
                  r"(?:\s*(?:-|–|to|and)\s*(?P<last>\d{1,3}))?", re.IGNORECASE)
_FORM = [
    ("first theme", r"\b(?:first|principal|main|chief)\s+(?:theme|subject)\b"),
    ("second theme", r"\b(?:second|subsidiary|secondary)\s+(?:theme|subject)\b"),
    ("development", r"\bdevelopment\b|\bworking[- ]out\b"),
    ("recapitulation", r"\brecapitulation\b"),
    ("coda", r"\bcoda\b"),
    ("episode", r"\bepisode\b"),
    ("introduction", r"\bintroduction\b"),
    ("trio", r"\btrio\b"),
    ("variations", r"\bvariations?\b"),
    ("fugue", r"\bfugue\b|\bfuga\b"),
]
_SENTENCE = re.compile(r"[^.!?]*(?:[.!?]|$)")


@dataclass(frozen=True)
class Claim:
    claim_type: str
    value: str
    quote: str
    start: int


def key_label(tonic: str, accidental: str | None, mode: str) -> str:
    """"D", "flat", "major" -> "D-flat major": how a reader would write it."""
    accidental = {"#": "sharp", "♯": "sharp", "♭": "flat"}.get(accidental or "", (accidental or "").lower())
    return f"{tonic.upper()}{'-' + accidental if accidental else ''} {mode.lower()}"


def key_value(label: str) -> tuple[int, str] | None:
    """A reader's key name as (pitch class, mode), comparable with the stored
    music21 spellings: "D-flat major" and "D- major" are the same key."""
    name, mode = label.rsplit(" ", 1)
    spelled = name.replace("-sharp", "#").replace("-flat", "-")
    return parse_key_name(f"{spelled} {mode}")


def relative(key: tuple[int, str]) -> tuple[int, str]:
    tonic, mode = key
    return ((tonic + 3) % 12, "major") if mode == "minor" else ((tonic + 9) % 12, "minor")


def _sentence(text: str, position: int) -> str:
    for found in _SENTENCE.finditer(text):
        if found.start() <= position < max(found.end(), found.start() + 1):
            return found.group(0).strip()[:300]
    return text[max(0, position - 120):position + 180].strip()


def extract_claims(text: str) -> list[Claim]:
    claims: list[Claim] = []
    for found in _KEY.finditer(text):
        # "the F minor Sonata" names a work; it asserts nothing about harmony.
        if _NAMING.match(text[found.end():]) or re.search(r"Sonata\s+in\s*$", text[:found.start()]):
            continue
        label = key_label(found.group("tonic"), found.group("acc"), found.group("mode"))
        claims.append(Claim("key", label, _sentence(text, found.start()), found.start()))
    for found in _BAR.finditer(text):
        unit = "measure" if found.group("unit").lower().startswith("measure") else "bar"
        span = found.group("first") + (f"-{found.group('last')}" if found.group("last") else "")
        claims.append(Claim("bar_ref", f"{unit} {span}", _sentence(text, found.start()), found.start()))
    for value, pattern in _FORM:
        for found in re.finditer(pattern, text, re.IGNORECASE):
            claims.append(Claim("form", value, _sentence(text, found.start()), found.start()))

    seen, unique = set(), []
    for claim in sorted(claims, key=lambda c: c.start):
        if (claim.claim_type, claim.value) not in seen:
            seen.add((claim.claim_type, claim.value))
            unique.append(claim)
    return unique


# --- checking ------------------------------------------------------------------------

@dataclass(frozen=True)
class MovementKeys:
    """What the score says about one movement's keys."""
    work_id: int
    number: int
    declared: str | None                              # global_key, engraved
    regions: tuple[dict, ...] = field(default=())     # {"value", "measure_start", "measure_end"}


@dataclass
class Check:
    status: str
    evidence: dict


def movements_for(claim: Claim, passage: dict, sonata_movements: list[MovementKeys],
                  sonata=None) -> tuple[list[MovementKeys], str]:
    """The movements a claim is about, as narrowly as the text allows.

    The claim's own sentence first -- "In the last movement, presto agitato,
    C sharp minor" is about the finale even in a passage that spans all three
    movements -- then the passage's movements, then the whole sonata.
    """
    if sonata is not None:
        from commentary.anchoring import find_cues
        named = {c.movement.work_id for c in find_cues(claim.quote, sonata)}
        in_sentence = [m for m in sonata_movements if m.work_id in named]
        if in_sentence:
            return in_sentence, "sentence"
    anchored = [m for m in sonata_movements if m.work_id in (passage.get("work_ids") or [])]
    if anchored:
        return anchored, "passage"
    return sonata_movements, "sonata"


def check_claim(claim: Claim, passage: dict, movements: list[MovementKeys]) -> Check:
    """Check one claim from `passage` against the movements it is anchored to.

    `movements` are the passage's movements, or every movement of its sonata
    when it is anchored only to the sonata.
    """
    if claim.claim_type == "bar_ref":
        return Check("not_checkable", {"reason": "bar numbers are the author's edition's, not ours"})
    if claim.claim_type == "form":
        return Check("not_checkable", {"reason": "interpretive: the score marks sections, not their names"})
    if passage["anchor_status"] == "conflict":
        return Check("not_checkable", {"reason": "the passage also names other sonatas; the key may be theirs"})
    if passage["sonata"] is None or not movements:
        return Check("not_checkable", {"reason": "the passage is not anchored to a sonata"})

    claimed = key_value(claim.value)
    if claimed is None:
        return Check("not_checkable", {"reason": f"cannot read {claim.value!r} as a key"})

    for movement in movements:
        if key_value_stored(movement.declared) == claimed:
            return Check("supported", {"movement": movement.number, "declared_key": movement.declared,
                                       "basis": "engraved"})

    matching, relatives = [], []
    for movement in movements:
        for region in movement.regions:
            estimated = key_value_stored(region["value"])
            place = {"movement": movement.number, "key": region["value"],
                     "bars": [region["measure_start"], region["measure_end"]]}
            if estimated == claimed:
                matching.append(place)
            elif estimated is not None and relative(estimated) == claimed:
                relatives.append(place)
    if matching:
        return Check("agrees_with_estimate", {"regions": matching[:5], "basis": "estimated"})
    evidence = {"basis": "estimated",
                "declared_keys": {m.number: m.declared for m in movements},
                "estimated_keys": sorted({r["value"] for m in movements for r in m.regions})}
    if relatives:
        evidence["relative_of"] = relatives[:5]
    return Check("disagrees_with_estimate", evidence)


def key_value_stored(name: str | None) -> tuple[int, str] | None:
    return parse_key_name(name) if name else None
