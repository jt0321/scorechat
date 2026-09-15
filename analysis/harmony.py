"""
analysis/harmony.py
Harmonic analysis over the canonical symbolic layer.

Operates on stored `score_measures.symbolic_data` dicts rather than a music21
score, so a harmonic pass is reproducible from the database alone and can be
re-run (and re-versioned) without re-parsing the source .krn.

Two stages, deliberately separated because they fail differently:
  1. `estimate_key_trajectory` — which key are we in, per measure.  Windowed
     evidence + Viterbi smoothing; a key change must be paid for, so brief
     tonicisations don't register as modulations.
  2. chord segmentation / labelling (see below) — what chord is sounding,
     given that key.

Stage 1 is reliable enough to publish.  Stage 2 carries a confidence and is
explicitly a candidate: see MEMORY analysis-tiers.
"""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

HARMONY_ANALYSIS_VERSION = "1.0"

# Krumhansl-Kessler key profiles.
KK_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
KK_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

PITCH_NAMES = ("C", "C#", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B")

# A key estimate for one measure is drawn from a window this many measures
# wide, centred on it.  Single-measure evidence is far too sparse to correlate
# against a 12-dimensional profile -- that is the defect this replaces.
KEY_WINDOW_MEASURES = 13

# Bonus, in correlation units, for a key consistent with the notated key
# signature.  The signature is engraved evidence, not inference, so it
# outranks a marginal profile correlation -- without it a cantabile melody
# sitting on the fifth pulls the estimate to the dominant (the Pathetique
# Adagio, notated in 4 flats, reads as E- major on raw correlation alone).
# Deliberately smaller than KEY_CHANGE_PENALTY: it biases, never constrains,
# so a genuine modulation away from the notated signature can still win.
SIGNATURE_BONUS = 0.25

# Cost, in correlation units, of the key changing between adjacent measures.
# Tonal music stays put; a modulation must be worth more than this to be
# reported.  Tuned so Op. 2 No. 1/i holds f minor through its first group.
KEY_CHANGE_PENALTY = 0.9


@dataclass
class KeyEstimate:
    """The key in force at one measure, with the evidence behind it."""
    measure_index: int
    key: str                 # e.g. "f minor", "A- major"
    tonic: int               # pitch class 0-11
    mode: str                # "major" | "minor"
    correlation: float       # windowed profile fit alone, -1..1; excludes the
                             # signature/home-key bonuses used to pick the path
    margin: float            # lead over the runner-up key in the same window
    in_key_signature: bool   # is this key the one the staff signature notates


# Major-key tonic for each signature, from 7 flats to 7 sharps.
_MAJOR_TONIC_BY_SHARPS = {
    -7: 11, -6: 6, -5: 1, -4: 8, -3: 3, -2: 10, -1: 5,
    0: 0, 1: 7, 2: 2, 3: 9, 4: 4, 5: 11, 6: 6, 7: 1,
}


def signature_keys(sharps: int | None) -> set[tuple[int, str]]:
    """The two keys a staff signature notates: a major key and its relative
    minor.  An unknown signature constrains nothing and returns empty."""
    if sharps is None or sharps not in _MAJOR_TONIC_BY_SHARPS:
        return set()
    major_tonic = _MAJOR_TONIC_BY_SHARPS[sharps]
    return {(major_tonic, "major"), ((major_tonic + 9) % 12, "minor")}


def measure_key_signatures(measures: list[tuple[int, dict]]) -> list[int | None]:
    """Signature in force at each measure, forward-filled: a signature is
    notated only where it changes, so an absent value inherits the last one."""
    filled: list[int | None] = []
    current: int | None = None
    for _, data in measures:
        for part in data.get("parts", []):
            sharps = part.get("key_signature_sharps")
            if sharps is not None:
                current = sharps
                break
        filled.append(current)
    return filled


def _quarter_length(value: Any) -> float:
    """Canonical durations are stored as strings, and music21 emits exact
    fractions ('1/3') for tuplets, so plain float() is not enough."""
    if value is None:
        return 0.0
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return 0.0


def measure_pitch_class_weights(symbolic_data: dict) -> list[float]:
    """Duration-weighted pitch-class histogram for one measure, all parts.

    Weighting by sounding duration is what separates structural tones from
    passing figuration: a half-note tonic outweighs four sixteenth passing
    notes, which is precisely the discrimination the per-measure estimate
    it replaces did not have.
    """
    weights = [0.0] * 12
    for part in symbolic_data.get("parts", []):
        for event in part.get("events", []):
            kind = event.get("kind")
            if kind == "rest":
                continue
            duration = _quarter_length(event.get("duration", {}).get("quarter_length"))
            if duration <= 0:
                continue
            if kind == "note":
                pitch = event.get("pitch")
                if pitch is not None:
                    weights[pitch["pitch_class"] % 12] += duration
            elif kind == "chord":
                for pitch in event.get("pitches", []):
                    weights[pitch["pitch_class"] % 12] += duration
    return weights


def _correlation(observed: list[float], profile: tuple[float, ...]) -> float:
    """Pearson correlation between a pitch-class histogram and a key profile."""
    n = 12
    mean_o = sum(observed) / n
    mean_p = sum(profile) / n
    dev_o = [value - mean_o for value in observed]
    dev_p = [value - mean_p for value in profile]
    numerator = sum(a * b for a, b in zip(dev_o, dev_p))
    denominator = (sum(a * a for a in dev_o) * sum(b * b for b in dev_p)) ** 0.5
    return numerator / denominator if denominator else 0.0


# Enharmonic spellings the corpus writes that PITCH_NAMES does not use. A
# score declaring "*D-:" means the same pitch class as our "C#", and the
# declaration is the better spelling: D-flat major is what the page says.
_SPELLINGS = {
    "C": 0, "B#": 0, "C#": 1, "D-": 1, "D": 2, "D#": 3, "E-": 3, "E": 4,
    "F-": 4, "E#": 5, "F": 5, "F#": 6, "G-": 6, "G": 7, "G#": 8, "A-": 8,
    "A": 9, "A#": 10, "B-": 10, "B": 11, "C-": 11,
}


def parse_key_name(name: str | None) -> tuple[int, str] | None:
    """"f minor" or "D- major" -> (pitch class, mode). None if unparsable."""
    if not name:
        return None
    parts = name.split()
    if len(parts) != 2 or parts[1] not in ("major", "minor"):
        return None
    tonic = _SPELLINGS.get(parts[0][0].upper() + parts[0][1:])
    return None if tonic is None else (tonic, parts[1])


def _key_name(tonic: int, mode: str) -> str:
    name = PITCH_NAMES[tonic % 12]
    return f"{name} major" if mode == "major" else f"{name.lower()} minor"


def score_all_keys(weights: list[float]) -> list[tuple[float, int, str]]:
    """Correlate a pitch-class histogram against all 24 keys, best first."""
    if not any(weights):
        return []
    scored = []
    for tonic in range(12):
        rotated = weights[tonic:] + weights[:tonic]
        scored.append((_correlation(rotated, KK_MAJOR), tonic, "major"))
        scored.append((_correlation(rotated, KK_MINOR), tonic, "minor"))
    scored.sort(reverse=True)
    return scored



# Bonus for the movement's home key, applied at every measure.  Small: it
# exists to break the relative major/minor tie that the signature bonus
# necessarily splits evenly, not to suppress real modulation.
HOME_KEY_BONUS = 0.15


def _lowest_pitch_class(symbolic_data: dict) -> int | None:
    """Pitch class of the lowest sounding note in a measure, or None if silent."""
    lowest = None
    for part in symbolic_data.get("parts", []):
        for event in part.get("events", []):
            kind = event.get("kind")
            if kind == "note":
                candidate = event.get("pitch")
            elif kind == "chord":
                pitches = event.get("pitches") or []
                candidate = min(pitches, key=lambda p: p["midi"]) if pitches else None
            else:
                continue
            if candidate and (lowest is None or candidate["midi"] < lowest["midi"]):
                lowest = candidate
    return lowest["pitch_class"] % 12 if lowest else None


def detect_home_key(
    measures: list[tuple[int, dict]], signatures: list[int | None]
) -> tuple[int, str] | None:
    """The movement's tonic, from the bass of its final sounding measure.

    Tonal movements close on the tonic in the bass, so this is notated
    evidence rather than inference.  It is what separates a key from its
    relative -- the pair a key signature cannot distinguish between.
    Returns None when the final bass is not one of the keys the signature
    admits (a movement ending on a half cadence or in a foreign key), in
    which case no anchor is asserted.
    """
    for index in range(len(measures) - 1, -1, -1):
        final_bass = _lowest_pitch_class(measures[index][1])
        if final_bass is None:
            continue  # empty trailing measure carries no cadence
        admitted = signature_keys(signatures[index])
        matching = [state for state in admitted if state[0] == final_bass]
        return matching[0] if len(matching) == 1 else None
    return None


def _window_weights(per_measure: list[list[float]], index: int, width: int) -> list[float]:
    half = width // 2
    start = max(0, index - half)
    end = min(len(per_measure), index + half + 1)
    summed = [0.0] * 12
    for measure_weights in per_measure[start:end]:
        for pitch_class in range(12):
            summed[pitch_class] += measure_weights[pitch_class]
    return summed


def estimate_key_trajectory(
    measures: Iterable[Any],
    window: int = KEY_WINDOW_MEASURES,
    change_penalty: float = KEY_CHANGE_PENALTY,
    signature_bonus: float = SIGNATURE_BONUS,
    home_key_bonus: float = HOME_KEY_BONUS,
    declared: tuple[int, str] | None = None,
) -> list[KeyEstimate]:
    """Per-measure key estimates, smoothed so a key change must earn its cost.

    `measures` is any iterable of objects or dicts carrying `measure_index`
    and `symbolic_data` (i.e. CanonicalMeasure or a row from score_measures).

    Each measure is scored from a window of surrounding measures, then a
    Viterbi pass over the 24 key states finds the trajectory maximising total
    fit minus a penalty per key change.  This is what stops a secondary
    dominant or a brief tonicisation from being reported as a modulation --
    the failure mode of correlating a single measure in isolation.
    """
    items = [
        (
            item["measure_index"] if isinstance(item, dict) else item.measure_index,
            item["symbolic_data"] if isinstance(item, dict) else item.symbolic_data,
        )
        for item in measures
    ]
    if not items:
        return []

    per_measure = [measure_pitch_class_weights(data) for _, data in items]
    signatures = measure_key_signatures(items)
    # A declared key is notated evidence and outranks the cadential guess:
    # every movement in this corpus states its key, and inferring it from the
    # final bass misreads six of them, usually as the relative major.
    home_key = declared if declared is not None else detect_home_key(items, signatures)
    states = [(tonic, mode) for tonic in range(12) for mode in ("major", "minor")]

    # Emission scores: correlation of each measure's window against each key,
    # biased toward the keys the notated signature admits.
    emissions: list[dict[tuple[int, str], float]] = []
    # Raw correlations are kept apart from the bonused scores the Viterbi pass
    # optimises, so a reported confidence is the profile evidence alone and
    # never a bonus-inflated number greater than 1.
    raw_correlations: list[dict[tuple[int, str], float]] = []
    for index in range(len(items)):
        admitted = signature_keys(signatures[index])
        bonuses = {state: signature_bonus for state in admitted}
        if home_key is not None:
            bonuses[home_key] = bonuses.get(home_key, 0.0) + home_key_bonus
        weights = _window_weights(per_measure, index, window)
        scored = score_all_keys(weights)
        if not scored:
            # A measure of rests carries no key evidence; stay neutral so the
            # transition penalty alone decides, i.e. hold the current key.
            emissions.append({state: bonuses.get(state, 0.0) for state in states})
            raw_correlations.append({state: 0.0 for state in states})
            continue
        emissions.append({
            (tonic, mode): value + bonuses.get((tonic, mode), 0.0)
            for value, tonic, mode in scored
        })
        raw_correlations.append({(tonic, mode): value for value, tonic, mode in scored})

    # Viterbi forward pass.
    best: dict[tuple[int, str], float] = dict(emissions[0])
    backpointers: list[dict[tuple[int, str], tuple[int, str]]] = []
    for index in range(1, len(items)):
        previous, best = best, {}
        pointers: dict[tuple[int, str], tuple[int, str]] = {}
        # Only the single best predecessor can win a stay-or-switch race, so
        # the usual O(24^2) inner loop collapses to a max plus a lookup.
        champion = max(previous, key=previous.__getitem__)
        champion_score = previous[champion] - change_penalty
        for state in states:
            stay_score = previous[state]
            if stay_score >= champion_score:
                best[state] = stay_score + emissions[index][state]
                pointers[state] = state
            else:
                best[state] = champion_score + emissions[index][state]
                pointers[state] = champion
        backpointers.append(pointers)

    # Backward pass.
    state = max(best, key=best.__getitem__)
    path = [state]
    for pointers in reversed(backpointers):
        state = pointers[state]
        path.append(state)
    path.reverse()

    estimates = []
    for index, ((measure_index, _), (tonic, mode)) in enumerate(zip(items, path)):
        ranked = sorted(raw_correlations[index].items(), key=lambda kv: kv[1], reverse=True)
        runner_up = next((value for key, value in ranked if key != (tonic, mode)), 0.0)
        estimates.append(KeyEstimate(
            measure_index=measure_index,
            key=_key_name(tonic, mode),
            tonic=tonic,
            mode=mode,
            correlation=raw_correlations[index][(tonic, mode)],
            margin=raw_correlations[index][(tonic, mode)] - runner_up,
            in_key_signature=(tonic, mode) in signature_keys(signatures[index]),
        ))
    return estimates



# --- Chord segmentation and labelling -------------------------------------
#
# The defect this replaces: chordifying a score creates a new vertical
# sonority at every note onset in any part, so a measure of Alberti bass
# yields six "chords" -- note density, not harmonic rhythm -- and every
# passing tone becomes a chord member, producing figures like "III+6543".
#
# Instead: aggregate sounding duration into beat-sized windows, fit a chord
# template to each, and treat pitches the template does not explain as
# non-chord tones.  Adjacent windows agreeing on a chord are merged, so the
# result is harmonic rhythm rather than onset rhythm.

CHORD_TEMPLATES: dict[str, tuple[int, ...]] = {
    "major":          (0, 4, 7),
    "minor":          (0, 3, 7),
    "diminished":     (0, 3, 6),
    "augmented":      (0, 4, 8),
    "dominant-7":     (0, 4, 7, 10),
    "minor-7":        (0, 3, 7, 10),
    "major-7":        (0, 4, 7, 11),
    "half-diminished-7": (0, 3, 6, 10),
    "diminished-7":   (0, 3, 6, 9),
}

# A template is rewarded for the sounding duration it explains and penalised
# for its own members that never sound, so a seventh chord cannot win merely
# by covering more pitch classes than a triad.
MISSING_TONE_PENALTY = 0.35

# Extra weight given to the bass pitch class when fitting.  The bass is the
# strongest cue to a chord's root and inversion.
BASS_WEIGHT = 0.6

# Share of a beat a pitch must sound for to count as that beat's bass.  The
# harmonic bass is the lowest *sustained* pitch, not the lowest to occur:
# a broken-chord figure momentarily dipping below the bass would otherwise
# invert the chord, while requiring the bass to sound exactly on the beat
# would miss an accompaniment that enters after the melody.
BASS_MIN_DURATION_SHARE = 0.25

# A triad needs three distinct pitch classes to be determined.  A window with
# fewer cannot discriminate between the templates containing them -- an
# arpeggiated measure sounding one note per beat would otherwise have each
# note fitted separately, and the tie broken arbitrarily (a lone F fitted as
# "C# major").  Such a window borrows pitch content from its neighbours --
# the *narrowest* context that determines a triad, so a fast harmonic rhythm
# is not blurred by aggregating a whole measure that holds two harmonies.
MIN_DISTINCT_PITCH_CLASSES = 3

# Below this fit, the window is reported as an unlabelled candidate rather
# than a chord: dense chromatic or purely melodic passages have no single
# defensible label, and silence is a better answer than "III+6543".
MIN_CHORD_CONFIDENCE = 0.55


@dataclass
class ChordSpan:
    """One harmony, spanning a contiguous run of beats."""
    measure_index: int
    beat_start: float          # in quarter lengths from the measure start
    beat_end: float
    root: int                  # pitch class
    quality: str               # a CHORD_TEMPLATES key
    bass: int | None           # pitch class of the lowest sounding note
    confidence: float          # 0..1 duration-weighted template fit
    non_chord_tones: list[int]
    figure: str | None = None  # Roman numeral, filled in against the local key
    key: str | None = None


def _beat_length(time_signature: str | None) -> float:
    """Quarter lengths per beat.  Compound meters (x/8 where x is divisible
    by 3) are felt in dotted beats, so 6/8 gives 1.5, not 0.5."""
    if not time_signature or "/" not in time_signature:
        return 1.0
    try:
        numerator, denominator = (int(part) for part in time_signature.split("/"))
    except ValueError:
        return 1.0
    beat = 4.0 / denominator
    if denominator >= 8 and numerator % 3 == 0 and numerator > 3:
        beat *= 3
    return beat


def _events_with_span(symbolic_data: dict):
    """Yield (offset, duration, [pitch dicts], is_lowest_candidate) per event."""
    for part in symbolic_data.get("parts", []):
        for event in part.get("events", []):
            kind = event.get("kind")
            if kind == "rest":
                continue
            duration = _quarter_length(event.get("duration", {}).get("quarter_length"))
            offset = _quarter_length(event.get("offset"))
            if duration <= 0:
                continue
            if kind == "note":
                pitches = [event["pitch"]] if event.get("pitch") else []
            else:
                pitches = event.get("pitches") or []
            if pitches:
                yield offset, duration, pitches


def beat_windows(symbolic_data: dict, time_signature: str | None) -> list[dict]:
    """Split one measure into beat-sized windows of sounding pitch content.

    An event's duration is apportioned to every window it overlaps, so a bass
    note held under a running figure supports the harmony for its whole
    length instead of only the beat it was struck on.
    """
    beat = _beat_length(time_signature)
    events = list(_events_with_span(symbolic_data))
    if not events or beat <= 0:
        return []
    measure_end = max(offset + duration for offset, duration, _ in events)
    window_count = max(1, int(-(-measure_end // beat)))  # ceil

    windows = []
    for index in range(window_count):
        start, end = index * beat, (index + 1) * beat
        weights = [0.0] * 12
        sounding: dict[int, float] = {}  # midi -> duration sounded in this window
        for offset, duration, pitches in events:
            overlap = min(offset + duration, end) - max(offset, start)
            if overlap <= 0:
                continue
            for pitch in pitches:
                weights[pitch["pitch_class"] % 12] += overlap
                sounding[pitch["midi"]] = sounding.get(pitch["midi"], 0.0) + overlap
        bass = None
        if sounding:
            threshold = (end - start) * BASS_MIN_DURATION_SHARE
            sustained = [midi for midi, held in sounding.items() if held >= threshold]
            bass = min(sustained) if sustained else min(sounding)
        if any(weights):
            windows.append({
                "beat_start": start, "beat_end": end,
                "weights": weights,
                "bass": bass % 12 if bass is not None else None,
            })
    return windows


def fit_chord(weights: list[float], bass: int | None) -> tuple[int, str, float, list[int]]:
    """Best (root, quality, confidence, non_chord_tones) for a pitch-class
    histogram.  Confidence is the share of sounding duration the winning
    template explains, less a penalty for template members that never sound.
    """
    total = sum(weights)
    if total <= 0:
        return 0, "major", 0.0, []
    boosted = list(weights)
    if bass is not None:
        boosted[bass] += total * BASS_WEIGHT
    boosted_total = sum(boosted)

    best = (0, "major", -1.0)
    for root in range(12):
        for quality, intervals in CHORD_TEMPLATES.items():
            members = {(root + interval) % 12 for interval in intervals}
            explained = sum(boosted[pc] for pc in members) / boosted_total
            missing = sum(1 for pc in members if weights[pc] <= 0) / len(members)
            score = explained - MISSING_TONE_PENALTY * missing
            if score > best[2]:
                best = (root, quality, score)

    root, quality, _ = best
    members = {(root + interval) % 12 for interval in CHORD_TEMPLATES[quality]}
    # Report the fit against the *unboosted* histogram: the bass weighting
    # exists to pick the right root and inversion, but folding it into the
    # reported confidence would inflate every span (the bass is nearly always
    # a chord tone of the winning template) and leave the gate below inert.
    explained = sum(weights[pc] for pc in members) / total
    missing = sum(1 for pc in members if weights[pc] <= 0) / len(members)
    confidence = explained - MISSING_TONE_PENALTY * missing
    non_chord_tones = sorted(pc for pc in range(12) if weights[pc] > 0 and pc not in members)
    return root, quality, max(0.0, min(1.0, confidence)), non_chord_tones


def _distinct_pitch_classes(weights: list[float]) -> int:
    return sum(1 for weight in weights if weight > 0)


def _determining_weights(windows: list[dict], index: int) -> list[float] | None:
    """Pitch content for fitting window `index`, widened only as far as needed.

    Grows outward one neighbouring beat at a time -- forward first, since a
    harmony is more often completed by what follows than by what preceded it
    -- and stops as soon as three distinct pitch classes are in view.
    Returns None when the whole measure never reaches three.
    """
    weights = list(windows[index]["weights"])
    if _distinct_pitch_classes(weights) >= MIN_DISTINCT_PITCH_CLASSES:
        return weights
    low = high = index
    while low > 0 or high < len(windows) - 1:
        if high < len(windows) - 1:
            high += 1
            source = windows[high]["weights"]
        else:
            low -= 1
            source = windows[low]["weights"]
        for pitch_class in range(12):
            weights[pitch_class] += source[pitch_class]
        if _distinct_pitch_classes(weights) >= MIN_DISTINCT_PITCH_CLASSES:
            return weights
    return None


def segment_measure_chords(
    measure_index: int, symbolic_data: dict, time_signature: str | None
) -> list[ChordSpan]:
    """Beat-level chord spans for one measure, adjacent duplicates merged.

    A beat too thin to determine a triad on its own borrows its measure's
    pitch content, so an arpeggiated texture is read as the harmony it
    outlines rather than as one spurious chord per note.
    """
    windows = beat_windows(symbolic_data, time_signature)
    if not windows:
        return []

    spans: list[ChordSpan] = []
    for index, window in enumerate(windows):
        weights = _determining_weights(windows, index)
        if weights is None:
            # No context within the measure spells a triad (a bare octave, a
            # solo anacrusis): no chord is determinable, so none is asserted.
            spans.append(ChordSpan(
                measure_index=measure_index,
                beat_start=window["beat_start"], beat_end=window["beat_end"],
                root=0, quality="major", bass=window["bass"],
                confidence=0.0, non_chord_tones=[],
            ))
            continue
        root, quality, confidence, non_chord_tones = fit_chord(weights, window["bass"])
        if spans and spans[-1].root == root and spans[-1].quality == quality:
            # Same harmony continuing: extend it rather than emitting a second
            # chord, so the output tracks harmonic rhythm, not note onsets.
            previous = spans[-1]
            previous.beat_end = window["beat_end"]
            previous.confidence = max(previous.confidence, confidence)
            previous.non_chord_tones = sorted(
                set(previous.non_chord_tones) & set(non_chord_tones)
            )
            continue
        spans.append(ChordSpan(
            measure_index=measure_index,
            beat_start=window["beat_start"], beat_end=window["beat_end"],
            root=root, quality=quality, bass=window["bass"],
            confidence=confidence, non_chord_tones=non_chord_tones,
        ))
    return spans


# Semitones above the tonic -> Roman numeral degree.  Minor is referenced to
# the *harmonic* minor scale, which is why 11 is a plain VII (the leading-tone
# chord, viio) and 10 is bVII (the subtonic) -- the standard convention.
_DEGREES_MAJOR = {
    0: ("I", ""), 1: ("II", "b"), 2: ("II", ""), 3: ("III", "b"), 4: ("III", ""),
    5: ("IV", ""), 6: ("IV", "#"), 7: ("V", ""), 8: ("VI", "b"), 9: ("VI", ""),
    10: ("VII", "b"), 11: ("VII", ""),
}
_DEGREES_MINOR = {
    0: ("I", ""), 1: ("II", "b"), 2: ("II", ""), 3: ("III", ""), 4: ("III", "#"),
    5: ("IV", ""), 6: ("IV", "#"), 7: ("V", ""), 8: ("VI", ""), 9: ("VI", "#"),
    10: ("VII", "b"), 11: ("VII", ""),
}

# Qualities written with a lowercase numeral, and the symbol each appends.
_LOWERCASE_QUALITIES = {"minor", "diminished", "minor-7", "half-diminished-7", "diminished-7"}
_QUALITY_SYMBOL = {
    "diminished": "o", "diminished-7": "o", "half-diminished-7": "ø", "augmented": "+",
}
_TRIAD_INVERSIONS = {0: "", 1: "6", 2: "64"}
_SEVENTH_INVERSIONS = {0: "7", 1: "65", 2: "43", 3: "42"}


def roman_figure(root: int, quality: str, bass: int | None, tonic: int, mode: str) -> str:
    """Roman numeral for a chord in a key, from scale degree and inversion.

    Computed arithmetically rather than by handing a chord back to music21 to
    re-derive: a chord built from bare pitch classes carries no spelling, so
    in A- major music21 reads the tonic triad as G#-B#-D# and reports "#VII".
    Working in scale degrees sidesteps enharmonic spelling entirely.
    """
    degrees = _DEGREES_MAJOR if mode == "major" else _DEGREES_MINOR
    numeral, accidental = degrees[(root - tonic) % 12]
    if quality in _LOWERCASE_QUALITIES:
        numeral = numeral.lower()

    intervals = CHORD_TEMPLATES[quality]
    inversion = 0
    if bass is not None:
        offset = (bass - root) % 12
        if offset in intervals:
            inversion = intervals.index(offset)
    table = _SEVENTH_INVERSIONS if len(intervals) == 4 else _TRIAD_INVERSIONS
    return accidental + numeral + _QUALITY_SYMBOL.get(quality, "") + table.get(inversion, "")


def analyze_harmony(
    measures: Iterable[Any],
    analyses: Iterable[Any],
    min_confidence: float = MIN_CHORD_CONFIDENCE,
    declared_key: str | None = None,
) -> tuple[list[KeyEstimate], list[ChordSpan]]:
    """Full harmonic pass: key trajectory, then chords labelled in local key.

    Chord spans whose template fit falls below `min_confidence` keep their
    root and quality but are left without a figure -- an unlabelled span is
    a truthful answer for a chromatic or purely melodic passage, where the
    approach this replaces would emit a confident nonsense figure.
    """
    measures = list(measures)
    declared = parse_key_name(declared_key)
    trajectory = estimate_key_trajectory(measures, declared=declared)
    if declared is not None and declared_key:
        # Report the spelling the score uses. PITCH_NAMES has one name per
        # pitch class, so an estimate agreeing with a score that writes
        # "D- major" would otherwise be reported as "C# major".
        for estimate in trajectory:
            if (estimate.tonic, estimate.mode) == declared:
                estimate.key = declared_key
    key_by_index = {estimate.measure_index: estimate for estimate in trajectory}
    time_signature_by_index = {
        (item["measure_index"] if isinstance(item, dict) else item.measure_index):
        (item["analysis_data"] if isinstance(item, dict) else item.analysis_data
         ).get("time_signature")
        for item in analyses
    }

    spans: list[ChordSpan] = []
    current_time_signature = None
    for item in measures:
        measure_index = item["measure_index"] if isinstance(item, dict) else item.measure_index
        symbolic_data = item["symbolic_data"] if isinstance(item, dict) else item.symbolic_data
        # Meter is notated only where it changes, so carry the last one forward.
        current_time_signature = (
            time_signature_by_index.get(measure_index) or current_time_signature
        )
        estimate = key_by_index.get(measure_index)
        for span in segment_measure_chords(measure_index, symbolic_data, current_time_signature):
            if estimate is not None:
                span.key = estimate.key
                if span.confidence >= min_confidence:
                    span.figure = roman_figure(
                        span.root, span.quality, span.bass, estimate.tonic, estimate.mode
                    )
            spans.append(span)
    return trajectory, spans
