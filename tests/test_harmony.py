"""
tests/test_harmony.py
Ground-truth tests for key estimation.

The expected keys here are uncontroversial published facts about these
movements, not this implementation's own output -- they are what stops the
tuning constants in analysis/harmony.py from being fitted to one movement.
"""
import pytest
from pathlib import Path

from analysis.analyzer import build_symbolic_layers
from analysis.harmony import (
    detect_home_key, estimate_key_trajectory, measure_key_signatures,
    measure_pitch_class_weights, score_all_keys, signature_keys,
)
from analysis.corpus import KERN_DIR


def _load(name: str):
    path = KERN_DIR / name
    if not path.exists():
        pytest.skip(f"{name} is not available")
    return build_symbolic_layers(str(path))[0]


def _runs(trajectory):
    """Collapse a per-measure trajectory into [(key, start_index, end_index)]."""
    runs = []
    for estimate in trajectory:
        if runs and runs[-1][0] == estimate.key:
            runs[-1][2] = estimate.measure_index
        else:
            runs.append([estimate.key, estimate.measure_index, estimate.measure_index])
    return [tuple(run) for run in runs]


def test_synthetic_c_major_scores_c_major_first():
    weights = [0.0] * 12
    for pitch_class, duration in [(0, 4), (4, 3), (7, 3), (2, 1), (5, 1), (9, 1), (11, 1)]:
        weights[pitch_class] = duration
    correlation, tonic, mode = score_all_keys(weights)[0]
    assert (tonic, mode) == (0, "major")
    assert correlation > 0.9


def test_signature_keys_pairs_relative_major_and_minor():
    assert signature_keys(-4) == {(8, "major"), (5, "minor")}   # A- major / f minor
    assert signature_keys(4) == {(4, "major"), (1, "minor")}    # E major / c# minor
    assert signature_keys(0) == {(0, "major"), (9, "minor")}    # C major / a minor
    assert signature_keys(None) == set()


def test_silent_measure_contributes_no_key_weight():
    assert measure_pitch_class_weights({"parts": [{"events": [
        {"kind": "rest", "duration": {"quarter_length": "2.0"}}
    ]}]}) == [0.0] * 12


def test_tuplet_durations_do_not_break_weighting():
    """music21 emits exact fractions for tuplets; plain float() would raise."""
    weights = measure_pitch_class_weights({"parts": [{"events": [
        {"kind": "note", "duration": {"quarter_length": "1/3"},
         "pitch": {"name": "C4", "midi": 60, "pitch_class": 0}}
    ]}]})
    assert weights[0] == pytest.approx(1 / 3)


def test_key_signature_forward_fills_across_measures():
    measures = [
        (0, {"parts": [{"key_signature_sharps": -4}]}),
        (1, {"parts": [{"key_signature_sharps": None}]}),
        (2, {"parts": [{"key_signature_sharps": 3}]}),
        (3, {"parts": [{"key_signature_sharps": None}]}),
    ]
    assert measure_key_signatures(measures) == [-4, -4, 3, 3]


def test_pathetique_adagio_is_a_flat_major_not_its_dominant():
    """Op. 13/ii is in A- major. Raw Krumhansl correlation reports E- major
    here -- the cantabile melody dwells on the fifth -- so this asserts the
    notated key signature actually overrides the profile."""
    trajectory = estimate_key_trajectory(_load("sonata08-2.krn"))
    assert trajectory[0].key == "A- major"
    assert trajectory[-1].key == "A- major"
    keys = {estimate.key for estimate in trajectory}
    assert "E- major" not in keys


def test_pathetique_adagio_finds_its_minor_middle_section():
    trajectory = estimate_key_trajectory(_load("sonata08-2.krn"))
    assert "a- minor" in {estimate.key for estimate in trajectory}
    assert len(_runs(trajectory)) <= 4  # A- / a- / A-, not a churn of estimates


def test_moonlight_opens_in_c_sharp_minor_not_its_relative_major():
    """Op. 27 No. 2/i. E major and c# minor share a signature, so only the
    home-key anchor separates them."""
    trajectory = estimate_key_trajectory(_load("sonata14-1.krn"))
    assert trajectory[0].key == "c# minor"
    assert trajectory[-1].key == "c# minor"


def test_op2no1_first_movement_moves_to_the_relative_major():
    """Op. 2 No. 1/i: first group f minor, second group A- major, recap f minor."""
    runs = _runs(estimate_key_trajectory(_load("sonata01-1.krn")))
    assert runs[0][0] == "f minor"
    assert runs[-1][0] == "f minor"
    assert any(key == "A- major" for key, _, _ in runs)
    assert len(runs) <= 5


def test_home_key_anchor_uses_last_sounding_measure():
    """Op. 2 No. 1/i ends with a measure carrying no notes; the anchor must
    fall back past it rather than giving up."""
    measures = _load("sonata01-1.krn")
    items = [(m.measure_index, m.symbolic_data) for m in measures]
    assert detect_home_key(items, measure_key_signatures(items)) == (5, "minor")


def test_empty_input_returns_empty_trajectory():
    assert estimate_key_trajectory([]) == []


def test_reported_correlation_excludes_the_bonuses():
    """A confidence handed to the LLM as evidence must be the profile fit
    alone -- the signature and home-key bonuses can push the Viterbi score
    above 1.0, which would be meaningless as a correlation."""
    for estimate in estimate_key_trajectory(_load("sonata08-2.krn")):
        assert -1.0 <= estimate.correlation <= 1.0


def test_roman_figures_use_scale_degrees_not_enharmonic_spelling():
    """A- major's tonic triad must read I, not the #VII that a pitch-class
    chord round-tripped through music21 spelling produces."""
    from analysis.harmony import roman_figure
    assert roman_figure(8, "major", 8, 8, "major") == "I"           # A- in A- major
    assert roman_figure(3, "dominant-7", 3, 8, "major") == "V7"     # E-7 in A- major
    assert roman_figure(3, "dominant-7", 7, 8, "major") == "V65"    # G in the bass
    assert roman_figure(0, "minor", 0, 8, "major") == "iii"
    assert roman_figure(11, "diminished", 11, 0, "minor") == "viio"  # leading tone in c minor
    assert roman_figure(10, "major", 10, 0, "minor") == "bVII"       # subtonic in c minor


def test_beat_windows_apportion_a_sustained_note_across_beats():
    """A bass note held under a running figure must support the harmony for
    its whole length, not only the beat it was struck on."""
    from analysis.harmony import beat_windows
    windows = beat_windows({"parts": [{"events": [
        {"kind": "note", "offset": "0.0", "duration": {"quarter_length": "4.0"},
         "pitch": {"name": "C2", "midi": 36, "pitch_class": 0}},
    ]}]}, "4/4")
    assert len(windows) == 4
    assert all(window["weights"][0] == pytest.approx(1.0) for window in windows)
    assert all(window["bass"] == 0 for window in windows)


def test_compound_meter_is_felt_in_dotted_beats():
    from analysis.harmony import _beat_length
    assert _beat_length("6/8") == 1.5
    assert _beat_length("4/4") == 1.0
    assert _beat_length("2/2") == 2.0
    assert _beat_length("3/4") == 1.0
    assert _beat_length(None) == 1.0


def test_alberti_bass_yields_one_chord_not_one_per_onset():
    """The defect being replaced: chordify emitted a vertical per onset, so a
    measure of broken-chord accompaniment read as six chord changes."""
    from analysis.harmony import segment_measure_chords
    events = []
    for index, (name, midi, pc) in enumerate(
        [("C3", 48, 0), ("G3", 55, 7), ("E3", 52, 4), ("G3", 55, 7)] * 2
    ):
        events.append({"kind": "note", "offset": str(index * 0.5),
                       "duration": {"quarter_length": "0.5"},
                       "pitch": {"name": name, "midi": midi, "pitch_class": pc}})
    spans = segment_measure_chords(0, {"parts": [{"events": events}]}, "4/4")
    assert len(spans) == 1
    assert (spans[0].root, spans[0].quality) == (0, "major")


def test_low_confidence_chord_is_left_unlabelled_rather_than_guessed():
    from analysis.harmony import analyze_harmony, ChordSpan
    chromatic = [
        {"kind": "note", "offset": str(i * 0.25), "duration": {"quarter_length": "0.25"},
         "pitch": {"name": f"p{i}", "midi": 60 + i, "pitch_class": i % 12}}
        for i in range(16)
    ]
    measures = [{"measure_index": 0, "symbolic_data": {"parts": [{"events": chromatic}]}}]
    analyses = [{"measure_index": 0, "analysis_data": {"time_signature": "4/4"}}]
    _, spans = analyze_harmony(measures, analyses, min_confidence=0.9)
    assert spans and all(span.figure is None for span in spans)


def test_arpeggiated_measure_reads_as_one_chord_not_one_per_note():
    """Op. 2 No. 1/i opens with a monophonic F minor arpeggio, one note per
    beat. Fitted beat-by-beat, each lone note matches many templates equally
    and the tie breaks arbitrarily; the measure as a whole spells i."""
    measures = _load("sonata01-1.krn")
    from analysis.analyzer import build_symbolic_layers
    from analysis.harmony import analyze_harmony
    _, analyses, _ = build_symbolic_layers(str(KERN_DIR / "sonata01-1.krn"))
    _, spans = analyze_harmony(measures, analyses)
    by_measure = {}
    for span in spans:
        by_measure.setdefault(span.measure_index, []).append(span)
    assert [span.figure for span in by_measure[1]] == ["i"]
    assert [span.figure for span in by_measure[2]] == ["i"]


def test_solo_anacrusis_is_left_unlabelled():
    """Op. 2 No. 1/i begins with a single unaccompanied C. One pitch class
    determines no chord, and asserting one would be a fabrication."""
    from analysis.analyzer import build_symbolic_layers
    from analysis.harmony import analyze_harmony
    measures, analyses, _ = build_symbolic_layers(str(KERN_DIR / "sonata01-1.krn"))
    _, spans = analyze_harmony(measures, analyses)
    opening = [span for span in spans if span.measure_index == 0]
    assert opening and all(span.figure is None for span in opening)


def test_third_inversion_dominant_seventh_is_read_from_the_notated_bass():
    """Op. 13/ii m1-2 is I - V42 - I6: the left hand holds D-3 under beat 2,
    and a third-inversion V7 resolving to I6 is the textbook progression.
    Guards against 'simplifying' the bass rule back to a root-position bias."""
    from analysis.analyzer import build_symbolic_layers
    from analysis.harmony import analyze_harmony
    measures, analyses, _ = build_symbolic_layers(str(KERN_DIR / "sonata08-2.krn"))
    _, spans = analyze_harmony(measures, analyses)
    figures = [span.figure for span in spans if span.measure_index in (0, 1)]
    assert figures == ["I", "V42", "I6", "V65"]


def test_bass_is_the_lowest_sustained_pitch_not_the_lowest_to_occur():
    """A figure dipping briefly below the held bass must not invert the chord."""
    from analysis.harmony import beat_windows
    windows = beat_windows({"parts": [{"events": [
        {"kind": "note", "offset": "0.0", "duration": {"quarter_length": "1.0"},
         "pitch": {"name": "C3", "midi": 48, "pitch_class": 0}},
        {"kind": "note", "offset": "0.75", "duration": {"quarter_length": "0.125"},
         "pitch": {"name": "G2", "midi": 43, "pitch_class": 7}},
    ]}]}, "4/4")
    assert windows[0]["bass"] == 0  # the fleeting G2 is not the bass


def test_bass_entering_after_the_beat_still_counts():
    """A left hand entering under a held melody note is the bass, even though
    nothing low sounds at the beat's onset."""
    from analysis.harmony import beat_windows
    windows = beat_windows({"parts": [{"events": [
        {"kind": "note", "offset": "0.0", "duration": {"quarter_length": "2.0"},
         "pitch": {"name": "A-5", "midi": 80, "pitch_class": 8}},
        {"kind": "note", "offset": "1.0", "duration": {"quarter_length": "1.0"},
         "pitch": {"name": "F3", "midi": 53, "pitch_class": 5}},
    ]}]}, "2/2")
    assert windows[0]["bass"] == 5


# --- score loading -------------------------------------------------------

def test_humdrum_signatures_read_from_source():
    """Key signature and meter are notated facts in the .krn; they must not
    depend on which importer parsed the score."""
    from analysis.analyzer import humdrum_signatures
    assert humdrum_signatures("*k[b-e-a-d-]\n*M2/4\n") == (-4, "2/4")
    assert humdrum_signatures("*k[f#c#g#d#]\n*M3/4\n") == (4, "3/4")
    assert humdrum_signatures("*k[]\n") == (0, None)
    assert humdrum_signatures("no signatures here\n") == (None, None)


def test_humdrum_measure_count_counts_distinct_barlines():
    from analysis.analyzer import humdrum_measure_count
    assert humdrum_measure_count("=1\tx\n=2\tx\n=2\tx\n=3\tx\n") == 3
    assert humdrum_measure_count("no barlines\n") == 0


@pytest.mark.parametrize("name,expected", [
    ("sonata09-1.krn", 162),   # Op. 14 No. 1/i  -- music21's Humdrum reader gave 12
    ("sonata18-1.krn", 253),   # Op. 31 No. 3/i  -- gave 136
    ("sonata04-1.krn", 362),   # Op. 7/i         -- gave 264
])
def test_scores_with_nested_spine_splits_parse_complete(name, expected):
    """music21's Humdrum reader silently truncates these at their first nested
    spine split; load_score must detect the shortfall and fall back to Verovio."""
    measures = _load(name)
    assert len(measures) >= expected - 2


def test_recovered_score_still_carries_its_key_signature():
    """The Verovio fallback's MEI drops signatures music21 would have read, so
    the source-derived fallback must supply them -- without it Op. 31 No. 3
    loses the key-signature anchor entirely."""
    measures = _load("sonata18-1.krn")
    signatures = measure_key_signatures(
        [(m.measure_index, m.symbolic_data) for m in measures]
    )
    assert signatures[0] == -3  # E- major
    assert estimate_key_trajectory(measures)[0].key == "E- major"


def test_op31no3_reaches_its_tonic_ending():
    """The truncated parse ended on an F minor sonority mid-movement; the
    movement actually closes in E- major."""
    trajectory = estimate_key_trajectory(_load("sonata18-1.krn"))
    assert trajectory[-1].key == "E- major"
