"""
analysis/span_relations.py
Deterministic comparison of two spans (measure_index ranges) to score how
well they support a `repeats` or `varies` span_relations candidate.

v1 scope: primary part only (part_index=0), equal measure count only. A
span pair with unequal measure counts is not attempted (confidence 0.0) —
alignment across unequal-length spans is deferred to v2.
"""

from __future__ import annotations
from typing import Any

from analysis.sections import parse_notated_sections
from db.store import (
    extract_ordered_pitch_classes, extract_ordered_rhythm, get_measure_total_durations,
    get_max_measure_index, get_global_key, get_local_key, get_span_candidates,
    get_measure_index, get_source_text, get_tempo_markings, load_work_features,
)

RHYTHM_MATCH_TOLERANCE = 0.05  # quarter-length slack for measure-total duration comparisons


class WorkFeatures:
    """One work's per-measure comparison features, held in memory.

    Relation search scores a reference span against every same-length window in
    the movement. Fetching each window from the database would issue tens of
    thousands of queries per movement; loading once and slicing makes the same
    search a matter of seconds. The comparison functions below take an optional
    `features` argument so a batch pass can supply this while a one-off call
    still works straight off the database.
    """

    def __init__(self, work_id: int, measures: list[dict]):
        self.work_id = work_id
        self._by_index = {measure["measure_index"]: measure for measure in measures}
        self.max_index = max(self._by_index) if self._by_index else None

    @classmethod
    def load(cls, work_id: int, part_index: int = 0) -> "WorkFeatures":
        return cls(work_id, load_work_features(work_id, part_index))

    def _range(self, start: int, end: int) -> list[dict]:
        return [self._by_index[i] for i in range(start, end + 1) if i in self._by_index]

    def pitch_classes(self, start: int, end: int) -> list[int]:
        return [pc for m in self._range(start, end) for pc in m["pitch_classes"]]

    def rhythm(self, start: int, end: int) -> list[float]:
        return [value for m in self._range(start, end) for value in m["rhythm"]]

    def measure_durations(self, start: int, end: int) -> list[float]:
        return [m["total_duration"] for m in self._range(start, end)]

    def local_key(self, measure_index: int) -> str | None:
        measure = self._by_index.get(measure_index)
        return measure["local_key"] if measure else None

    def measure_number(self, measure_index: int) -> int | None:
        measure = self._by_index.get(measure_index)
        return measure["measure_number"] if measure else None


def _pitch_classes(span: dict, features: WorkFeatures | None) -> list[int]:
    if features is not None:
        return features.pitch_classes(span["measure_start_index"], span["measure_end_index"])
    return extract_ordered_pitch_classes(
        span["work_id"], span["measure_start_index"], span["measure_end_index"])


def _rhythm(span: dict, features: WorkFeatures | None) -> list[float]:
    if features is not None:
        return features.rhythm(span["measure_start_index"], span["measure_end_index"])
    return extract_ordered_rhythm(
        span["work_id"], span["measure_start_index"], span["measure_end_index"])


def _measure_durations(span: dict, features: WorkFeatures | None) -> list[float]:
    if features is not None:
        return features.measure_durations(span["measure_start_index"], span["measure_end_index"])
    return get_measure_total_durations(
        span["work_id"], span["measure_start_index"], span["measure_end_index"])


def _span_length(span: dict) -> int:
    return span["measure_end_index"] - span["measure_start_index"] + 1


def _sequence_match_ratio(a: list, b: list) -> float:
    """Fraction of positions that match, using the longer sequence's length
    as the denominator so extra/missing elements also count against it."""
    length = max(len(a), len(b))
    if length == 0:
        return 1.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / length


def _measure_duration_similarity(a: list[float], b: list[float], tolerance: float = RHYTHM_MATCH_TOLERANCE) -> float:
    length = max(len(a), len(b))
    if length == 0:
        return 1.0
    matches = sum(1 for x, y in zip(a, b) if abs(x - y) <= tolerance)
    return matches / length


def transposition_interval(
    pitches_a: list[int], pitches_b: list[int]
) -> tuple[int | None, float]:
    """Semitones separating two pitch-class sequences, and how consistently.

    `check_varies` compares interval sequences, which is transposition
    invariant by design -- it recognises a transposed return but says nothing
    about the transposition. That distinction is the difference between a
    recapitulation restating material at pitch and one bringing a second group
    home from another key, so the offset is measured here and recorded as
    evidence: 0 for a return at pitch, 5 for one a perfect fourth higher.

    Returns the most common offset and the fraction of positions holding it.
    A low consistency means no single transposition explains the pair, so the
    interval should not be read as one.
    """
    length = min(len(pitches_a), len(pitches_b))
    if length == 0:
        return None, 0.0
    offsets: dict[int, int] = {}
    for a, b in zip(pitches_a[:length], pitches_b[:length]):
        offset = (b - a) % 12
        offsets[offset] = offsets.get(offset, 0) + 1
    interval = max(offsets, key=offsets.__getitem__)
    return interval, offsets[interval] / length


def _interval_sequence(pitch_classes: list[int]) -> list[int]:
    return [(b - a) % 12 for a, b in zip(pitch_classes, pitch_classes[1:])]


def _worth_comparing(
    span_a: dict, span_b: dict, features: WorkFeatures, min_confidence: float
) -> bool:
    """Cheap exact bound: can this window possibly reach min_confidence?

    Both scorers divide matches by the *longer* sequence, so two spans whose
    event counts are far apart cannot score well however their contents line
    up. `varies` averages the interval ratio with a rhythm ratio that can reach
    1.0, making (ratio + 1) / 2 the highest either scorer could return -- so a
    window failing that bound is skipped without comparing a single pitch.
    """
    count_a = len(features.pitch_classes(span_a["measure_start_index"], span_a["measure_end_index"]))
    count_b = len(features.pitch_classes(span_b["measure_start_index"], span_b["measure_end_index"]))
    if count_a == 0 or count_b == 0:
        return count_a == count_b
    ratio = min(count_a, count_b) / max(count_a, count_b)
    return (ratio + 1) / 2 >= min_confidence


def check_repeats(
    span_a: dict, span_b: dict, features: WorkFeatures | None = None
) -> tuple[float, dict[str, Any]]:
    """Does span_b look like a literal repeat of span_a: near-identical
    ordered pitch-class sequence and rhythm sequence, on the primary part.

    span_a/span_b: dicts with work_id, measure_start_index, measure_end_index.
    """
    if _span_length(span_a) != _span_length(span_b):
        return 0.0, {
            "reason": "unequal_measure_count",
            "span_a_measures": _span_length(span_a),
            "span_b_measures": _span_length(span_b),
        }

    pitches_a = _pitch_classes(span_a, features)
    pitches_b = _pitch_classes(span_b, features)
    rhythm_a = _rhythm(span_a, features)
    rhythm_b = _rhythm(span_b, features)

    pitch_ratio = _sequence_match_ratio(pitches_a, pitches_b)
    rhythm_ratio = _sequence_match_ratio(rhythm_a, rhythm_b)
    confidence = (pitch_ratio + rhythm_ratio) / 2

    evidence = {
        "comparison": "repeats",
        "part_index": 0,
        "pitch_classes_a": pitches_a,
        "pitch_classes_b": pitches_b,
        "pitch_match_ratio": pitch_ratio,
        "rhythm_a": rhythm_a,
        "rhythm_b": rhythm_b,
        "rhythm_match_ratio": rhythm_ratio,
    }
    return confidence, evidence


def check_varies(
    span_a: dict, span_b: dict, features: WorkFeatures | None = None
) -> tuple[float, dict[str, Any]]:
    """Does span_b look like a varied restatement of span_a: similar melodic
    contour (interval sequence mod 12) and similar rhythmic weight per
    measure, without requiring pitch identity or event-for-event rhythm.

    span_a/span_b: dicts with work_id, measure_start_index, measure_end_index.
    """
    if _span_length(span_a) != _span_length(span_b):
        return 0.0, {
            "reason": "unequal_measure_count",
            "span_a_measures": _span_length(span_a),
            "span_b_measures": _span_length(span_b),
        }

    pitches_a = _pitch_classes(span_a, features)
    pitches_b = _pitch_classes(span_b, features)
    intervals_a = _interval_sequence(pitches_a)
    intervals_b = _interval_sequence(pitches_b)
    interval_ratio = _sequence_match_ratio(intervals_a, intervals_b)

    durations_a = _measure_durations(span_a, features)
    durations_b = _measure_durations(span_b, features)
    rhythm_ratio = _measure_duration_similarity(durations_a, durations_b)

    confidence = (interval_ratio + rhythm_ratio) / 2

    what_varied = []
    if pitches_a != pitches_b:
        what_varied.append("pitch")
    if durations_a != durations_b:
        what_varied.append("rhythm")

    evidence = {
        "comparison": "varies",
        "part_index": 0,
        "interval_sequence_a": intervals_a,
        "interval_sequence_b": intervals_b,
        "interval_match_ratio": interval_ratio,
        "measure_durations_a": durations_a,
        "measure_durations_b": durations_b,
        "rhythm_similarity_ratio": rhythm_ratio,
        "what_varied": what_varied,
    }
    return confidence, evidence


def search_for_matches(
    reference_span: dict,
    search_start_index: int | None = None,
    search_end_index: int | None = None,
    min_confidence: float = 0.5,
    features: WorkFeatures | None = None,
) -> list[dict]:
    """Slide a window the same length as reference_span across
    [search_start_index, search_end_index] of the same work (default: the
    whole work) and score every non-overlapping position with check_repeats
    and check_varies. This finds thematic returns (e.g. a transposed
    recapitulation) without needing pre-identified section boundaries or
    repeat signs — the reference span is the only thing that must be known
    up front.

    Returns candidates sorted by best(repeats, varies) confidence
    descending, each entry: {measure_start_index, measure_end_index,
    repeats_confidence, repeats_evidence, varies_confidence, varies_evidence}.
    """
    work_id = reference_span["work_id"]
    length = _span_length(reference_span)

    if search_start_index is None:
        search_start_index = 0
    if search_end_index is None:
        max_index = features.max_index if features is not None else get_max_measure_index(work_id)
        if max_index is None:
            return []
        search_end_index = max_index

    ref_start = reference_span["measure_start_index"]
    ref_end = reference_span["measure_end_index"]

    results = []
    for start in range(search_start_index, search_end_index - length + 2):
        end = start + length - 1
        if start <= ref_end and end >= ref_start:
            continue  # skip windows overlapping the reference span itself

        candidate = {"work_id": work_id, "measure_start_index": start, "measure_end_index": end}
        if features is not None and not _worth_comparing(
            reference_span, candidate, features, min_confidence
        ):
            continue
        repeats_confidence, repeats_evidence = check_repeats(reference_span, candidate, features)
        varies_confidence, varies_evidence = check_varies(reference_span, candidate, features)

        if max(repeats_confidence, varies_confidence) >= min_confidence:
            results.append({
                "measure_start_index": start,
                "measure_end_index": end,
                "repeats_confidence": repeats_confidence,
                "repeats_evidence": repeats_evidence,
                "varies_confidence": varies_confidence,
                "varies_evidence": varies_evidence,
            })

    results.sort(key=lambda r: max(r["repeats_confidence"], r["varies_confidence"]), reverse=True)
    return results


# An introduction stands outside the form, so it is short: the longest in this
# corpus is Op. 111's 18-bar Maestoso, an eighth of the movement. Op. 57's
# finale also opens with an unrepeated section followed by a repeated one, but
# that section is a third of the movement and is its exposition -- the
# proportion is what separates the two cases.
MAX_INTRO_SHARE = 0.20


def detect_intro_end_index(work_id: int) -> int:
    """Measure_index where the movement's main theme begins, excluding a slow
    introduction, if one exists.

    Primary signal: the notated section structure. A first section the
    expansion record plays once, followed by one it plays twice, is an
    introduction standing outside a repeated exposition -- provided it is short
    (`MAX_INTRO_SHARE`), which is what distinguishes Op. 111's 18-bar Maestoso
    from Op. 57's finale, where the unrepeated opening section *is* the
    exposition. This is engraved rather than inferred, so it is exact: it puts
    the Pathetique's Allegro at bar 11 and the Waldstein's at bar 3.

    Fallback, for the 42 movements notating no sections: a change in tempo
    marking from the movement's opening one. Less precise -- it can be a
    measure or two off -- but it catches introductions with no repeat scheme.

    Returns 0 when neither signal fires, meaning no detectable introduction.
    """
    source = get_source_text(work_id)
    if source:
        expansion, sections = parse_notated_sections(source)
        body = [s for s in sections if not s.is_alternate_ending]
        if expansion and len(body) >= 2:
            first, second = body[0], body[1]
            total = max(s.measure_end for s in body) - first.measure_start + 1
            length = first.measure_end - first.measure_start + 1
            if (first.play_count <= 1 and second.play_count > 1
                    and total > 0 and length / total <= MAX_INTRO_SHARE):
                index = get_measure_index(work_id, second.measure_start)
                if index is not None:
                    return index
            # The engraving describes the movement's shape and does not
            # describe an introduction, so it is evidence there is none --
            # stronger than the tempo guess below, which would otherwise read
            # the Presto of Op. 57's coda as the end of a 315-bar introduction.
            return 0

    markings = get_tempo_markings(work_id)
    if not markings:
        return 0

    opening_tempo = next((tempo for _, tempo in markings if tempo is not None), None)
    if opening_tempo is None:
        return 0

    last_index = max(index for index, _ in markings)
    for measure_index, tempo in markings:
        if tempo is not None and tempo != opening_tempo:
            # A tempo change late in a movement is a coda or a trio, not an
            # introduction; only one near the opening can end one.
            if last_index and measure_index / last_index > MAX_INTRO_SHARE:
                return 0
            return measure_index
    return 0


def corroborate_key_match(
    work_id: int, reference_span: dict, candidate: dict,
    features: WorkFeatures | None = None,
) -> bool:
    """Does the candidate span open in the same key as the reference span?

    An independent check on a repeats/varies match: a genuine primary-theme
    recapitulation returns in the tonic, whereas the same material quoted in
    the development or restated in the second group does not.

    This compares the *local* key from the windowed trajectory. It formerly
    compared `global_key`, which is now one value for a whole movement, so the
    check would have corroborated every match unconditionally.

    Returns False (not corroborated) when either key is unknown, since an
    absent key cannot confirm anything.
    """
    if features is not None:
        reference_key = features.local_key(reference_span["measure_start_index"])
        candidate_key = features.local_key(candidate["measure_start_index"])
    else:
        reference_key = get_local_key(reference_span["work_id"], reference_span["measure_start_index"])
        candidate_key = get_local_key(work_id, candidate["measure_start_index"])
    if reference_key is None or candidate_key is None:
        return False
    return reference_key == candidate_key


# --- The relations pass ---------------------------------------------------

RELATION_ANALYSIS_VERSION = "1.0"

# A relation must clear a higher bar than a bare search hit. At the search
# default of 0.5, four-bar windows of ordinary accompaniment figuration match
# each other constantly; these thresholds keep only matches strong enough to
# be worth a musician's attention.
MIN_RELATION_CONFIDENCE = 0.75
# One- and two-measure spans recur by coincidence in tonal music, so a
# reference span shorter than this says nothing about thematic identity.
MIN_RELATION_MEASURES = 3
# ... as does a span with too few notes to have a shape, however many bars it
# spans (a held chord, a bar of rests).
MIN_RELATION_EVENTS = 8
# Matches per reference span. A theme returning three or four times is normal;
# a reference producing dozens of hits is matching filler, not a theme.
MAX_MATCHES_PER_SPAN = 4

# Reference spans are tiled at these lengths rather than taken solely from
# `span_analyses`. Boundary candidates segment a movement but do not index its
# themes: they break at every notated direction, so two thirds of them are one
# or two measures long, while a movement with no internal directions collapses
# into a single 226-measure span. Neither extreme can act as a thematic
# reference, which left a quarter of the corpus with no relations at all.
# Four and eight measures are the phrase lengths this repertoire is built from.
REFERENCE_WINDOW_LENGTHS = (4, 8)


def _reference_spans(
    features: WorkFeatures, spans: list[dict], min_measures: int, min_events: int
) -> list[dict]:
    """Spans to search for recurrences of: musically-bounded candidates that
    are long enough, plus a uniform tiling so coverage does not depend on how
    finely the boundary analyser happened to cut the movement."""
    references: dict[tuple[int, int], dict] = {}

    def offer(start: int, end: int) -> None:
        if (start, end) in references or features.measure_number(start) is None:
            return
        if end - start + 1 < min_measures or features.measure_number(end) is None:
            return
        if len(features.pitch_classes(start, end)) < min_events:
            return
        references[(start, end)] = {
            "measure_start_index": start, "measure_end_index": end,
            "measure_start": features.measure_number(start),
            "measure_end": features.measure_number(end),
        }

    for span in spans:
        offer(span["measure_start_index"], span["measure_end_index"])
    if features.max_index is not None:
        for length in REFERENCE_WINDOW_LENGTHS:
            stride = max(1, length // 2)
            for start in range(0, features.max_index - length + 2, stride):
                offer(start, start + length - 1)
    return [references[key] for key in sorted(references)]


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


def _select_distinct_matches(matches: list[dict], limit: int) -> list[dict]:
    """Keep the strongest matches that do not overlap each other.

    A sliding window scores a thematic return at several adjacent offsets, so
    the raw hits cluster around each real recurrence. Taking the best of each
    cluster turns that into one relation per return.
    """
    selected: list[dict] = []
    for match in matches:  # already sorted by confidence descending
        if any(_overlaps(match["measure_start_index"], match["measure_end_index"],
                         chosen["measure_start_index"], chosen["measure_end_index"])
               for chosen in selected):
            continue
        selected.append(match)
        if len(selected) >= limit:
            break
    return selected


def _merge_by_offset(relations: list[dict]) -> list[dict]:
    """Collapse relations that describe one correspondence into one relation.

    References are tiled with overlap, so a single thematic return is reported
    by every window covering it. Such reports share an *offset* (target start
    minus source start) and have touching source ranges; merging those keeps
    the musical fact and drops the duplication. Ranges at the same offset that
    do not touch stay separate -- the gap between them was never compared, and
    joining them would claim a longer correspondence than was verified.

    A merged relation takes the *lowest* confidence of its parts: the claim
    spans all of them, so it is only as good as its weakest verified section.
    """
    by_offset: dict[int, list[dict]] = {}
    for relation in relations:
        offset = relation["target_start_index"] - relation["source_start_index"]
        by_offset.setdefault(offset, []).append(relation)

    merged: list[dict] = []
    for offset, group in by_offset.items():
        group.sort(key=lambda r: r["source_start_index"])
        current = dict(group[0])
        for relation in group[1:]:
            if relation["source_start_index"] <= current["source_end_index"] + 1:
                if relation["source_end_index"] > current["source_end_index"]:
                    current["source_end_index"] = relation["source_end_index"]
                    current["source_end"] = relation["source_end"]
                    current["target_end_index"] = relation["target_end_index"]
                    current["target_end"] = relation["target_end"]
                current["confidence"] = min(current["confidence"], relation["confidence"])
                if relation["relation_type"] == "varies":
                    current["relation_type"] = "varies"
                current["evidence"] = dict(current["evidence"])
                current["evidence"]["merged_windows"] = current["evidence"].get("merged_windows", 1) + 1
                current["evidence"]["returns_in_same_key"] = (
                    current["evidence"]["returns_in_same_key"]
                    and relation["evidence"]["returns_in_same_key"]
                )
                # Parts disagreeing on the transposition describe a passage no
                # single interval explains; report none rather than one part's.
                if (current["evidence"].get("transposed_semitones")
                        != relation["evidence"].get("transposed_semitones")):
                    current["evidence"]["transposed_semitones"] = None
                current["evidence"]["transposition_consistency"] = min(
                    current["evidence"].get("transposition_consistency", 0.0),
                    relation["evidence"].get("transposition_consistency", 0.0),
                )
            else:
                merged.append(current)
                current = dict(relation)
        merged.append(current)
    merged.sort(key=lambda r: (r["source_start_index"], r["target_start_index"]))
    return merged


def build_span_relations(
    work_id: int,
    min_confidence: float = MIN_RELATION_CONFIDENCE,
    min_measures: int = MIN_RELATION_MEASURES,
    min_events: int = MIN_RELATION_EVENTS,
    max_matches: int = MAX_MATCHES_PER_SPAN,
    features: WorkFeatures | None = None,
) -> list[dict]:
    """Propose `repeats`/`varies` relations between spans of one work.

    For each candidate span long enough to carry an identity, slide it across
    the movement and keep the strongest non-overlapping matches. Relations are
    emitted forward in time only -- a return points back to its first statement
    rather than each recurrence relating to every other -- which keeps the
    graph a statement-and-return structure instead of a clique.

    Returns dicts ready for `store_span_relations`; nothing is written here, so
    thresholds can be explored without touching the database.
    """
    if features is None:
        features = WorkFeatures.load(work_id)
    references = _reference_spans(
        features, get_span_candidates(work_id), min_measures, min_events
    )

    relations: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()
    for span in references:
        matches = search_for_matches(
            {"work_id": work_id,
             "measure_start_index": span["measure_start_index"],
             "measure_end_index": span["measure_end_index"]},
            min_confidence=min_confidence,
            features=features,
        )
        for match in _select_distinct_matches(matches, max_matches):
            # Forward only: the earlier span is the statement, the later one
            # the return. Without this every pair appears twice, mirrored.
            if match["measure_start_index"] <= span["measure_start_index"]:
                continue
            key = (span["measure_start_index"], span["measure_end_index"],
                   match["measure_start_index"], match["measure_end_index"])
            if key in seen:
                continue
            seen.add(key)

            repeats, varies = match["repeats_confidence"], match["varies_confidence"]
            is_repeat = repeats >= varies
            candidate = {"measure_start_index": match["measure_start_index"]}
            same_key = corroborate_key_match(work_id, span, candidate, features)
            interval, consistency = transposition_interval(
                features.pitch_classes(span["measure_start_index"], span["measure_end_index"]),
                features.pitch_classes(match["measure_start_index"], match["measure_end_index"]),
            )
            relations.append({
                "relation_type": "repeats" if is_repeat else "varies",
                "confidence": round(max(repeats, varies), 4),
                "source_start_index": span["measure_start_index"],
                "source_end_index": span["measure_end_index"],
                "source_start": span["measure_start"],
                "source_end": span["measure_end"],
                "target_start_index": match["measure_start_index"],
                "target_end_index": match["measure_end_index"],
                "target_start": features.measure_number(match["measure_start_index"]),
                "target_end": features.measure_number(match["measure_end_index"]),
                "evidence": {
                    "analysis_version": RELATION_ANALYSIS_VERSION,
                    "repeats_confidence": round(repeats, 4),
                    "varies_confidence": round(varies, 4),
                    "comparison": (match["repeats_evidence"] if is_repeat
                                   else match["varies_evidence"]),
                    # An independent signal, recorded rather than folded into
                    # the confidence: material returning in the key it was
                    # stated in is the mark of a recapitulation, while the same
                    # material transposed is a sequence or a second-group
                    # restatement. Downstream form analysis needs to tell those
                    # apart, so the raw fact is kept separate from the score.
                    "returns_in_same_key": same_key,
                    "transposed_semitones": interval,
                    "transposition_consistency": round(consistency, 4),
                    "source_local_key": features.local_key(span["measure_start_index"]),
                    "target_local_key": features.local_key(match["measure_start_index"]),
                },
            })
    return _merge_by_offset(relations)
