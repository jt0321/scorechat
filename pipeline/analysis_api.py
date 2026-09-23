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

The last two functions are the exception to "every value comes from the
score": they return published *commentary* -- what Elterlein, Marx and
Shedlock wrote -- and say so in every result. Commentary is attributed opinion,
anchored to a work so it can be found and quoted beside the score; it is never
evidence of a musical fact, and where it makes a checkable claim the result
carries what the score says about it.
"""

from __future__ import annotations
import re
import unicodedata
from typing import Any

from analysis.humdrum import meter_changes, reference_edition
from analysis.span_relations import (
    WorkFeatures, check_repeats, check_varies, transposition_interval,
)
from db.store import (
    get_measure_evidence, get_notated_sections, get_source_text,
    get_movement_keys, get_span_relations, list_works, load_work_features,
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


# The separator between opus and number is whatever the user typed: "Op. 31
# No. 3", "op 31, no 3", "op31/no3" and "op31no3" all name one sonata.
_OPUS = re.compile(
    r"\bop(?:us|\.)?\s*(\d+)[\s/,\-]*(?:(?:no|nr|number)\.?\s*(\d+))?", re.I)


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


# How a movement is named when it is not numbered: by the heading it opens with
# ("the Scherzo", "the Adagio sostenuto", "the fugue" for Op. 106/iv, whose full
# heading is "Introduzione: Largo---Fuga: Allegro risoluto"). Words that qualify
# a tempo rather than name one identify nothing on their own.
_MARKING_FILLER = {
    "e", "ed", "con", "ma", "non", "troppo", "molto", "poco", "assai", "piu",
    "quasi", "un", "una", "di", "il", "la", "le", "l", "alla", "in", "tempo",
    "mit", "und", "nicht", "zu", "sehr", "quarter", "half", "eighth", "dot",
}
# English names for the Italian headings the scores carry.
_MARKING_SYNONYMS = {
    "fugue": "fuga", "minuet": "menuetto minuetto", "menuet": "menuetto minuetto",
    "introduction": "introduzione", "intro": "introduzione", "march": "marcia",
}


def _marking_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", _fold(text)) if w not in _MARKING_FILLER}


def _movement_by_marking(works: list[dict], folded_query: str) -> list[dict]:
    """The movements of one sonata whose heading the query names, best first.

    Returns only the uniquely best match; a tie (Op. 106's first movement and
    its fugue are both "Allegro") returns nothing, since guessing between
    movements would put the answer in the wrong music.
    """
    asked = set()
    for word in re.findall(r"[a-z]+", folded_query):
        asked.update(_MARKING_SYNONYMS.get(word, word).split())
    asked -= _MARKING_FILLER
    scored = sorted(((len(asked & _marking_words(w.get("tempo_indication") or "")), w)
                     for w in works), key=lambda pair: -pair[0])
    if not scored or scored[0][0] == 0:
        return []
    if len(scored) > 1 and scored[1][0] == scored[0][0]:
        return []
    return [scored[0][1]]


def _readable_key(key: str | None) -> str | None:
    """music21's "E- major" / "c# minor" as a reader writes it: "E-flat major",
    "C-sharp minor". The spelling reaches the model verbatim, and "E- major"
    is not something to repeat to a musician."""
    if not key or " " not in key:
        return key
    name, mode = key.rsplit(" ", 1)
    tonic, accidentals = name[0].upper(), name[1:]
    spelled = {"-": "-flat", "--": "-double-flat", "#": "-sharp", "##": "-double-sharp"}
    return f"{tonic}{spelled.get(accidentals, accidentals)} {mode}"


def _sonata_summary(opus: str) -> dict | None:
    """The whole sonata one movement belongs to.

    The *work* a musician names is the sonata; the corpus stores each movement
    as its own work_id, the way a recording has one track per movement. This is
    what lets an answer say "Op. 31 No. 3 has four movements" and give each one's
    heading and key, rather than seeing only the movement it was asked about.
    """
    movements = [w for w in list_works() if (w.get("opus") or "") == opus]
    if not movements:
        return None
    keys = get_movement_keys([w["id"] for w in movements])
    first = movements[0]
    return {
        "opus": opus,
        "nickname": first.get("nickname") or None,
        "sonata_number": first.get("work_number"),
        "movement_count": len(movements),
        "movements": [{
            "work_id": w["id"],
            "movement_number": w.get("movement_number"),
            "heading": w.get("tempo_indication"),
            # Engraved (`*f:`), not estimated: every movement declares one.
            "key": _readable_key(keys.get(w["id"])),
        } for w in movements],
    }


def _nickname_in(nickname: str, folded_query: str) -> bool:
    """Whether the query names a sonata by its nickname. A leading article is
    optional -- "the Hunt" and "hunt" are the same request."""
    name = _fold(nickname)
    return name in folded_query or re.sub(r"^the\s+", "", name) in folded_query


def resolve_work(query: str, limit: int = 5) -> dict:
    """Find the movement a free-text request means, and the sonata it is in.

    Every result that narrows to one sonata carries `sonata`: its opus,
    nickname and every movement with its heading and engraved key.
    """
    result = _resolve_movement(query, limit)
    opera = {m["opus"] for m in result["matches"]}
    result["sonata"] = _sonata_summary(opera.pop()) if len(opera) == 1 else None
    if result["resolved"] is None and result["sonata"]:
        sonata = result["sonata"]
        name = f" ({sonata['nickname']})" if sonata["nickname"] else ""
        result["note"] = (
            f"The request names the sonata {sonata['opus']}{name}, which has "
            f"{sonata['movement_count']} movements (listed in `sonata`). For a "
            "question about the sonata as a whole, call outline_sonata_tool with "
            "any movement's work_id; ask which movement is meant only when the "
            "question needs one.")
    elif result["resolved"] is None and len(opera) > 1:
        # "Op. 31" is a set of three sonatas published together, not one work.
        base = {re.sub(r"\s+No\.\s*\d+$", "", o) for o in opera if o}
        if len(base) == 1:
            opus = base.pop()
            numbers = sorted({w["opus"] for w in list_works()
                              if (w.get("opus") or "").startswith(opus + " No.")})
            result["note"] = (
                f"{opus} is a set of {len(numbers)} sonatas published "
                f"together ({', '.join(numbers)}); ask which is meant.")
    return result


def _resolve_movement(query: str, limit: int = 5) -> dict:
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
        # A bare "Op. 31" names the set, so it narrows to the set's sonatas.
        in_set = [w for w in works
                  if (w.get("opus") or "").startswith(designation + " No.")]
        if exact or in_set:
            works = exact or in_set

    # A nickname names one sonata just as an opus does, so it narrows the same
    # way -- "the finale of the Moonlight" needs the field down to one work
    # before "finale" can mean anything.
    folded = _fold(query)
    named = [w for w in works if w.get("nickname") and _nickname_in(w["nickname"], folded)]
    if named:
        works = named

    if movement is None and len({w.get("opus") for w in works}) == 1:
        works = _movement_by_marking(works, folded) or works

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
    if len({w.get("opus") for w in works}) == 1:
        # Narrowed to one sonata by opus or nickname, with no single movement
        # named: the movements are the answer, not a ranking of them by
        # whatever words are left in the query.
        return {"resolved": None, "matches": [_work_summary(w) for w in works],
                "note": None}

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
        "heading": work.get("tempo_indication"),
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


# --- the sonata as a whole --------------------------------------------------

# A synopsis names the keys a movement passes through, not every region the
# estimator found: the Hunt's finale has fifteen, and a list that long is an
# analysis rather than an overview.
MAX_OUTLINE_KEY_REGIONS = 8


def _meter_sequence(work_id: int) -> list[str]:
    """The meters a movement is written in, in order, from the source's own
    `*M` records (music21 loses changes in three movements)."""
    changes = meter_changes(get_source_text(work_id) or "")
    sequence: list[str] = []
    for bar in sorted(changes):
        if not sequence or sequence[-1] != changes[bar]:
            sequence.append(changes[bar])
    return sequence


def _outline_movement(movement: dict) -> dict:
    work_id = movement["work_id"]
    measures = _bars(work_id)
    regions = _runs([(m["measure_belongs_to"], m["local_key"])
                     for m in measures if m["local_key"] and m["measure_belongs_to"]])
    sections = [s for s in get_notated_sections(work_id)
                if not s["evidence"].get("is_alternate_ending")]
    # A section overlapped by its successor is only the pickup into it (the Hunt
    # finale's "I", bar 1 of an A that also starts at bar 1), not a part of the form.
    sections = [s for s, after in zip(sections, sections[1:] + [None])
                if after is None or after["measure_start"] > s["measure_end"]]
    return {
        **movement,
        "bars": sum(1 for m in measures if m["measure_role"] == "bar"),
        "meters": _meter_sequence(work_id),
        # Engraved: which stretches the score marks to be played twice.
        "sections": [{"label": s["label"],
                      "measures": f"{s['measure_start']}-{s['measure_end']}",
                      "repeated": (s["evidence"].get("play_count") or 1) > 1}
                     for s in sections] or None,
        # Estimated, and smoothed: a key the harmony only touches is not here.
        "estimated_keys": [{"key": _readable_key(r["value"]),
                            "measures": f"{r['measure_start']}-{r['measure_end']}"}
                           for r in regions[:MAX_OUTLINE_KEY_REGIONS]],
        "estimated_key_regions": len(regions),
    }


def outline_sonata(work_id: int) -> dict:
    """Every movement of the sonata `work_id` belongs to, in order, with what a
    synopsis can rest on: heading, engraved key, length, meter, the repeats the
    score marks, and the keys the movement is estimated to pass through.

    For a question about a sonata as a whole. It is an overview by design --
    nothing here is bar-by-bar -- so an answer built on it stays one.
    """
    work = next((w for w in list_works() if w["id"] == work_id), None)
    if work is None:
        return {"error": f"No work with id {work_id}."}
    sonata = _sonata_summary(work.get("opus") or "")
    if sonata is None:
        return {"error": f"Work {work_id} is not a movement of a catalogued sonata."}
    return {
        **{k: v for k, v in sonata.items() if k != "movements"},
        "movements": [_outline_movement(m) for m in sonata["movements"]],
        "note": ("`sections` are engraved; a movement with none notates no repeats. "
                 "`estimated_keys` are smoothed estimates, capped at "
                 f"{MAX_OUTLINE_KEY_REGIONS} regions (`estimated_key_regions` gives the "
                 "full count); a single region means modulations are unconfirmed, not absent."),
    }


# --- what the commentators say ----------------------------------------------

COMMENTARY_NOTE = ("Published commentary, quoted as the author's opinion. It is not evidence "
                   "of a musical fact; bar numbers in it follow the author's edition, not ours.")


def _movements_of(work_id: int) -> tuple[int | None, dict[int, int]]:
    """A movement's sonata number, and that sonata's work ids -> movement numbers."""
    works = list_works()
    work = next((w for w in works if w["id"] == work_id), None)
    if work is None or work.get("work_number") is None:
        return None, {}
    numbers = {w["id"]: w.get("movement_number") for w in works
               if w.get("work_number") == work["work_number"]}
    return work["work_number"], numbers


def search_commentary(query: str, work_id: int | None = None, whole_sonata: bool = False,
                      limit: int = 4) -> dict:
    """Passages of published commentary answering `query`.

    With `work_id`, passages about that movement come back along with remarks
    on its sonata as a whole, each marked by `scope`; `whole_sonata` widens to
    every movement. Without it the whole of the commentary is searched.
    """
    from commentary.search import search_passages
    sonata, movements = (None, {})
    if work_id is not None:
        sonata, movements = _movements_of(work_id)
        if sonata is None:
            return {"error": f"Work {work_id} is not a movement of a catalogued sonata."}
    result = search_passages(query, sonata=sonata if whole_sonata else None,
                             work_id=None if whole_sonata else work_id, limit=limit)
    passages = [{
        "author": p["author"], "title": p["title"], "year": p["year"],
        "page": p["page"], "line": p["line"], "url": p["url"],
        "about_sonata": p["sonata"],
        "about_movements": sorted(movements.get(w, w) for w in p["work_ids"]) if movements else p["work_ids"],
        "scope": p["scope"], "anchor": p["anchor_status"], "found_by": p["found_by"],
        "text": p["content"],
    } for p in result["passages"]]
    note = result["note"] or COMMENTARY_NOTE
    if not passages and work_id is not None:
        note = ("None of the ingested commentaries discusses this in words the search can match. "
                "Marx covers only twenty of the sonatas, and Shedlock names works rather than "
                "discussing them; absence here is not evidence about the music.")
    return {"query": query, "retrievers": result["retrievers"], "model": result["model"],
            "passages": passages, "note": note}


def commentary_claims(work_id: int, limit: int = 40) -> dict:
    """What the commentators assert about a movement, and what the score says.

    Key claims carry their check: `supported` by the engraved key,
    `agrees_with_estimate` / `disagrees_with_estimate` against the estimated
    key regions -- where a disagreement may be the estimator's fault as easily
    as the author's -- or `not_checkable` with the reason. Formal terms are
    interpretive; bar references are the author's edition.
    """
    from sqlalchemy import text
    from db.session import session_scope
    sonata, movements = _movements_of(work_id)
    if sonata is None:
        return {"error": f"Work {work_id} is not a movement of a catalogued sonata."}
    with session_scope() as session:
        rows = session.execute(text("""
            SELECT d.author, d.year, p.page, p.line, p.work_ids, c.claim_type, c.value,
                   c.check_status, c.check_evidence, c.quote
            FROM passage_claims c JOIN text_passages p ON p.id = c.passage_id
            JOIN text_documents d ON d.id = p.document_id
            WHERE :work = ANY(p.work_ids) OR (cardinality(p.work_ids) = 0 AND p.sonata = :sonata)
            ORDER BY CASE c.claim_type WHEN 'key' THEN 0 WHEN 'form' THEN 1 ELSE 2 END, p.id
        """), {"work": work_id, "sonata": sonata}).mappings().all()
    movement = movements.get(work_id)
    # A claim whose own sentence named its movement belongs to that movement,
    # even inside a passage that spans several.
    rows = [r for r in rows if _placed_movements(r["check_evidence"]) in (None, set())
            or movement in _placed_movements(r["check_evidence"])]
    claims = [{
        "author": r["author"], "year": r["year"], "page": r["page"], "line": r["line"],
        "type": r["claim_type"], "value": r["value"], "check": r["check_status"],
        "evidence": r["check_evidence"], "quote": r["quote"],
        "scope": "movement" if work_id in (r["work_ids"] or []) else "sonata",
    } for r in rows[:limit]]
    counts: dict[str, int] = {}
    for r in rows:
        if r["claim_type"] == "key":
            counts[r["check_status"]] = counts.get(r["check_status"], 0) + 1
    return {
        "work_id": work_id, "movement": movement, "claims": claims,
        "key_checks": counts, "truncated": len(rows) > limit,
        "note": (COMMENTARY_NOTE + " A key the estimate does not bear out may be the estimator's "
                 "error: it smooths away brief modulations." if rows else
                 "No commentator's claim is anchored to this movement."),
    }


def _placed_movements(evidence: dict) -> set[int] | None:
    """The movements a key claim was placed at by its own sentence, or None
    when it was placed only by its passage or sonata."""
    if not evidence or evidence.get("placed_by") != "sentence":
        return None
    if "movement" in evidence:
        return {evidence["movement"]}
    places = evidence.get("regions") or evidence.get("relative_of") or []
    found = {p["movement"] for p in places}
    return found or {int(n) for n in (evidence.get("declared_keys") or {})}
