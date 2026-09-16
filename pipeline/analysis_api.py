"""
pipeline/analysis_api.py
The questions ScoreChat can answer, as plain functions.

This is the layer the chat model calls. It exists separately from the tool
bindings in `pipeline/tools.py` so the answers can be tested without a model in
the loop, and so the rule the whole project rests on stays checkable: every
value returned here comes from `score_measures`, `measure_analyses`,
`span_analyses` or `span_relations`. Nothing is inferred at call time and
nothing is phrased -- the model does the phrasing, from these facts.

Two conventions hold throughout:

*Printed bar numbers in, printed bar numbers out.* Callers speak the numbering
a performer reads. Internally everything keys on `measure_index`, and a range
resolves through `measure_belongs_to`, so asking for mm. 229-247 includes the
upbeat into 229 and still reports itself as 229. See `analysis/numbering.py`.

*Absence is an answer.* A movement with no notated sections, a range with no
recurrences, a key estimate below confidence -- each returns an empty result
with a `note` saying so, never a guess. The model is instructed to relay that.
"""

from __future__ import annotations
import re
import unicodedata
from typing import Any

from analysis.humdrum import reference_edition
from analysis.span_relations import (
    WorkFeatures, check_repeats, check_varies, transposition_interval,
)
from db.store import (
    get_measure_evidence, get_notated_sections, get_source_text,
    get_span_relations, list_works, load_work_features,
)

MAX_EVIDENCE_MEASURES = 24
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8}
WORD_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
# "last" cannot be a fixed number -- these sonatas have two, three or four
# movements -- so it is resolved against the candidates instead.
LAST_MOVEMENT = re.compile(r"\b(last|final|finale)\b")


# --- finding a movement -----------------------------------------------------

def _movement_tokens(text: str) -> tuple[str, int | None]:
    """Split a movement number off a free-text request.

    Users say "moonlight 3rd movement", "op 27 no 2 iii", "waldstein mvt 1".
    The number is matched separately from the work because it is the part most
    likely to be right and least likely to match any title text.
    """
    lowered = _fold(text)
    movement = None
    digits = re.search(r"\b(\d+)(?:st|nd|rd|th)\s*(?:mvt|movement)\b", lowered)
    words = re.search(r"\b(" + "|".join(WORD_ORDINALS) + r")\s*(?:mvt|movement)\b", lowered)
    after = re.search(r"\b(?:mvt|movement)\.?\s*(\d+)\b", lowered)
    roman = re.search(r"(?:^|[\s,/])(i{1,3}|iv|v|vi{1,3})(?:$|[\s,.])", lowered)
    if digits:
        movement = int(digits.group(1))
    elif words:
        movement = WORD_ORDINALS[words.group(1)]
    elif after:
        movement = int(after.group(1))
    elif roman:
        movement = ROMAN.get(roman.group(1))
    cleaned = re.sub(
        r"\b(\d+(?:st|nd|rd|th)|" + "|".join(WORD_ORDINALS) + r")?\s*(mvt|movement)\.?\s*\d*\b",
        " ", lowered)
    return cleaned, movement


_OPUS = re.compile(r"\bop(?:us|\.)?\s*(\d+)\s*(?:(?:no|nr|number)\.?\s*(\d+))?", re.I)


def _opus_designation(text: str) -> str | None:
    """Normalise "op 27 no 2", "Op.27 No.2", "opus 27 number 2" to the stored
    "Op. 27 No. 2".

    Matched as a unit rather than as loose tokens, because "27" and "2" on
    their own match Op. 27 No. 1 exactly as well as Op. 27 No. 2 -- which left
    every opus-and-number request ambiguous.
    """
    match = _OPUS.search(text)
    if not match:
        return None
    opus, number = match.group(1), match.group(2)
    return f"Op. {opus} No. {number}" if number else f"Op. {opus}"


def _fold(text: str) -> str:
    """Lowercase and strip accents, so "pathetique" finds "Pathetique" -- users
    type nicknames without the diacritics the catalogue carries."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _score_work(work: dict, needle: str) -> int:
    """How well a work matches free text. Higher is better, 0 is no match."""
    haystacks = [_fold(str(work.get(field) or ""))
                 for field in ("title", "nickname", "opus", "composer")]
    score = 0
    for term in re.findall(r"[a-z0-9#\-]+", needle):
        if term in ("no", "op", "opus", "sonata", "piano", "in", "major", "minor", "the"):
            continue
        for weight, haystack in zip((4, 5, 5, 1), haystacks):
            if term and term in haystack:
                score += weight
                break
    return score


def resolve_work(query: str, limit: int = 5) -> dict:
    """Find the movement a free-text request means.

    Returns every plausible match rather than one, with a `resolved` work only
    when the best is unambiguous. Guessing between "Op. 27 No. 1" and "Op. 27
    No. 2" would put every later answer in the wrong piece, so ambiguity is
    handed back for the model to ask about.
    """
    needle, movement = _movement_tokens(query)
    works = list_works()
    if movement is not None:
        works = [w for w in works if w["movement_number"] == movement] or works

    # An explicit opus is decisive: it names one work, so nothing else needs
    # scoring and a nickname elsewhere in the query cannot pull the match away.
    designation = _opus_designation(query)
    if designation:
        exact = [w for w in works if (w.get("opus") or "") == designation]
        if exact:
            works = exact

    # A nickname names one sonata just as an opus does, so it narrows the same
    # way -- "the finale of the Moonlight" needs the field down to one work
    # before "finale" can mean anything.
    folded = _fold(query)
    named = [w for w in works if w.get("nickname") and _fold(w["nickname"]) in folded]
    if named:
        works = named

    if movement is None and LAST_MOVEMENT.search(folded):
        opus_numbers = {w.get("opus") for w in works}
        if len(opus_numbers) == 1:
            # Answerable only once the field is one sonata: these have two,
            # three or four movements, so "last" has no fixed number.
            highest = max(w["movement_number"] or 0 for w in works)
            works = [w for w in works if w["movement_number"] == highest]

    if len(works) == 1:
        return {"resolved": _work_summary(works[0]),
                "matches": [_work_summary(works[0])], "note": None}

    scored = sorted(
        ((_score_work(w, needle), w) for w in works),
        key=lambda pair: (-pair[0], pair[1]["id"]),
    )
    matches = [w for score, w in scored if score > 0][:limit]
    if not matches:
        return {"resolved": None, "matches": [],
                "note": f"No ingested movement matches {query!r}."}

    best, runner_up = scored[0][0], scored[1][0] if len(scored) > 1 else 0
    unambiguous = best > runner_up
    return {
        "resolved": _work_summary(matches[0]) if unambiguous else None,
        "matches": [_work_summary(w) for w in matches],
        "note": None if unambiguous else
                "Several movements match equally well; ask which is meant.",
    }


def _work_summary(work: dict) -> dict:
    # The edition is part of the answer, not trivia: bar numbers belong to an
    # edition, and this corpus is transcribed from a performing edition whose
    # numbering parts company with an urtext wherever a repeat has first and
    # second endings.
    edition = reference_edition(get_source_text(work["id"]) or "")
    return {
        "work_id": work["id"],
        "reference_edition": edition,
        "composer": work["composer"],
        "title": work["title"],
        "opus": work.get("opus"),
        "nickname": work.get("nickname"),
        "movement_number": work.get("movement_number"),
    }


# --- reading a range --------------------------------------------------------

def _bars(work_id: int) -> list[dict]:
    return load_work_features(work_id)


def _index_range(measures: list[dict], start: int, end: int) -> tuple[int | None, int | None]:
    """Printed bar range -> index range, resolving through `measure_belongs_to`
    so a range starting at a bar picks up the upbeat into it."""
    inside = [m for m in measures
              if m["measure_belongs_to"] is not None
              and start <= m["measure_belongs_to"] <= end]
    if not inside:
        return None, None
    return inside[0]["measure_index"], inside[-1]["measure_index"]


def describe_span(work_id: int, measure_start: int, measure_end: int) -> dict:
    """What happens in a range of bars, on its own terms.

    The only non-relational question ScoreChat answers: everything else about
    "describe mm. x-y in context" is a lookup in the relations graph. Chords,
    keys and directions come back as stored, with their confidences, so the
    model can report an unlabelled passage as unlabelled.
    """
    measures = _bars(work_id)
    if not measures:
        return {"error": f"Work {work_id} has no stored measures."}
    start_index, end_index = _index_range(measures, measure_start, measure_end)
    if start_index is None:
        return {"error": f"Work {work_id} has no bars in mm. {measure_start}-{measure_end}."}

    evidence = get_measure_evidence(work_id, measure_start, measure_end)
    truncated = len(evidence) > MAX_EVIDENCE_MEASURES
    evidence = evidence[:MAX_EVIDENCE_MEASURES]

    keys, chords, directions = [], [], []
    for measure in evidence:
        analysis = measure.get("analysis") or {}
        bar = measure["measure_belongs_to"]
        if analysis.get("local_key"):
            keys.append((bar, analysis["local_key"]))
        for chord in analysis.get("chords") or []:
            chords.append({"measure": bar, "figure": chord.get("figure"),
                           "confidence": chord.get("confidence")})
        for direction in analysis.get("directions") or []:
            directions.append({"measure": bar, "type": direction.get("type"),
                               "value": direction.get("value")})

    opening = evidence[0]
    return {
        "work_id": work_id,
        "measure_start": measure_start,
        "measure_end": measure_end,
        "starts_with_upbeat": opening["measure_role"] in ("anacrusis", "upbeat"),
        "keys": _runs(keys),
        "chords": chords,
        "directions": directions,
        "section": locate_in_form(work_id, measure_start).get("section"),
        "measures_returned": len(evidence),
        "note": (f"Only the first {MAX_EVIDENCE_MEASURES} bars of the range are "
                 "summarised; ask about a narrower range for the rest."
                 if truncated else None),
    }


def _runs(pairs: list[tuple[int, str]]) -> list[dict]:
    """Collapse consecutive equal values into ranges: a key plan is a list of
    regions, not one row per bar."""
    runs: list[dict] = []
    for bar, value in pairs:
        if runs and runs[-1]["value"] == value:
            runs[-1]["measure_end"] = bar
        else:
            runs.append({"value": value, "measure_start": bar, "measure_end": bar})
    return runs


def get_key_plan(work_id: int) -> dict:
    """The movement's key regions, as estimated.

    Reported as regions rather than per bar, and with the estimator's known
    weakness stated: the trajectory is smoothed, so a brief tonicisation is
    deliberately not a region, and a movement reported as one key throughout
    may simply not have had its modulations survive smoothing.
    """
    measures = _bars(work_id)
    if not measures:
        return {"error": f"Work {work_id} has no stored measures."}
    regions = _runs([(m["measure_belongs_to"], m["local_key"])
                     for m in measures if m["local_key"] and m["measure_belongs_to"]])
    return {
        "work_id": work_id,
        "regions": regions,
        "note": ("The whole movement is estimated in one key; this estimator "
                 "smooths its trajectory, so treat an absence of modulation as "
                 "unconfirmed rather than established."
                 if len(regions) <= 1 else None),
    }


# --- the relations graph ----------------------------------------------------

def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


def _relation_summary(relation: dict, direction: str) -> dict:
    evidence = relation.get("evidence") or {}
    return {
        "relation_type": relation["relation_type"],
        "confidence": round(relation["confidence"], 3),
        "direction": direction,
        "source_measures": [relation["source_start"], relation["source_end"]],
        "target_measures": [relation["target_start"], relation["target_end"]],
        "returns_in_same_key": evidence.get("returns_in_same_key"),
        "transposed_semitones": evidence.get("transposed_semitones"),
        "transposition_consistency": evidence.get("transposition_consistency"),
        "opens_on_pickup": evidence.get(
            "target_opens_on_pickup" if direction == "returns_at" else "source_opens_on_pickup"),
        "status": relation.get("status"),
    }


def find_recurrences(work_id: int, measure_start: int, measure_end: int) -> dict:
    """Where else in the movement this material appears.

    Both directions are reported and labelled. `returns_at` is a later passage
    that restates the queried range; `restates` means the queried range is
    itself the return of something earlier. Conflating them would let a
    recapitulation and its exposition read identically, which is the one
    distinction the whole graph exists to preserve.

    `transposed_semitones` is recorded separately from the confidence because
    the matcher is transposition-invariant by design: 0 is a theme returning at
    pitch, 5 a second group brought home a fourth away. A low
    `transposition_consistency` means no single interval explains the pair, so
    the interval must not be reported as one.
    """
    relations = get_span_relations(work_id)
    found = []
    for relation in relations:
        if _overlaps(measure_start, measure_end, relation["source_start"], relation["source_end"]):
            found.append(_relation_summary(relation, "returns_at"))
        elif _overlaps(measure_start, measure_end, relation["target_start"], relation["target_end"]):
            found.append(_relation_summary(relation, "restates"))
    found.sort(key=lambda r: -r["confidence"])
    return {
        "work_id": work_id,
        "measure_start": measure_start,
        "measure_end": measure_end,
        "recurrences": found,
        "note": (None if found else
                 "No recurrence of this material was proposed. The pass only "
                 "reports matches above its confidence threshold, so this means "
                 "none was found, not that the material is unique."),
    }


def compare_spans(
    work_id: int, a_start: int, a_end: int, b_start: int, b_end: int
) -> dict:
    """Compare two ranges directly, whether or not a relation was stored.

    The stored graph keeps only matches above threshold, so a passage a reader
    asks about specifically -- "is the coda built from the second subject?" --
    may have no relation to look up. This runs the same comparison on demand
    and reports what it scores, including a low score.
    """
    measures = _bars(work_id)
    if not measures:
        return {"error": f"Work {work_id} has no stored measures."}
    features = WorkFeatures(work_id, measures)
    a_first, a_last = _index_range(measures, a_start, a_end)
    b_first, b_last = _index_range(measures, b_start, b_end)
    if a_first is None or b_first is None:
        return {"error": "One of the ranges has no bars in this movement."}

    span_a = {"measure_start_index": a_first, "measure_end_index": a_last}
    span_b = {"measure_start_index": b_first, "measure_end_index": b_last}
    repeats, _ = check_repeats(span_a, span_b, features)
    varies, _ = check_varies(span_a, span_b, features)
    interval, consistency = transposition_interval(
        features.pitch_classes(a_first, a_last), features.pitch_classes(b_first, b_last)
    )
    return {
        "work_id": work_id,
        "a_measures": [a_start, a_end],
        "b_measures": [b_start, b_end],
        "repeats_confidence": round(repeats, 3),
        "varies_confidence": round(varies, 3),
        "transposed_semitones": interval,
        "transposition_consistency": round(consistency, 3),
        "a_key": features.local_key(a_first),
        "b_key": features.local_key(b_first),
        "note": ("Neither score is high enough to call this a restatement; "
                 "report the numbers rather than a relationship."
                 if max(repeats, varies) < 0.75 else None),
    }


# --- where a bar sits in the movement ---------------------------------------

def locate_in_form(work_id: int, measure: int) -> dict:
    """Which notated section a bar falls in, and what the repeat scheme is.

    Sections here are *engraved*, read from the score's own expansion records,
    so they carry no formal names -- the score marks that a stretch repeats, not
    that it is an exposition. The repeat scheme is reported for the model to
    reason from: a repeat covering much of a movement is the clearest indicator
    of sonata form, and which section repeats characterises it. Naming the
    section would be a claim this evidence does not make.
    """
    sections = get_notated_sections(work_id)
    if not sections:
        return {
            "work_id": work_id, "measure": measure, "section": None,
            "repeat_scheme": None,
            "note": ("This movement notates no section structure -- 42 of the "
                     "103 in the corpus do not. That absence is itself "
                     "evidence: no repeat scheme was engraved."),
        }
    containing = next((s for s in sections
                       if s["measure_start"] <= measure <= s["measure_end"]), None)
    scheme = sections[0]["evidence"].get("repeat_scheme")
    return {
        "work_id": work_id,
        "measure": measure,
        "section": None if containing is None else {
            "label": containing["label"],
            "measure_start": containing["measure_start"],
            "measure_end": containing["measure_end"],
            "play_count": containing["evidence"].get("play_count"),
            "is_alternate_ending": containing["evidence"].get("is_alternate_ending"),
        },
        "repeat_scheme": scheme,
        "sections": [{"label": s["label"], "measure_start": s["measure_start"],
                      "measure_end": s["measure_end"],
                      "play_count": s["evidence"].get("play_count")}
                     for s in sections],
        "note": None if containing else f"No notated section contains m. {measure}.",
    }
