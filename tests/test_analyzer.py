"""
tests/test_analyzer.py
Smoke tests for the music21 analysis module.
"""
from pathlib import Path

import pytest

from analysis.analyzer import build_span_candidates, build_symbolic_layers, detect_texture


def test_detect_texture_with_music21():
    from music21 import stream, note
    s = stream.Stream()
    for pitch in ["C4", "D4", "E4", "F4", "G4", "A4"]:
        s.append(note.Note(pitch, quarterLength=1))
    tag = detect_texture(s)
    assert tag in {"stepwise_melody", "cantabile", "chordal", "octaves_or_leaps"}


def test_build_symbolic_layers_preserves_events_and_measure_analysis():
    score_path = Path("data/sonata32-2.krn")
    if not score_path.exists():
        pytest.skip("Op. 111 source is not available")

    measures, analyses, global_key = build_symbolic_layers(str(score_path))

    assert measures
    assert len(measures) == len(analyses)
    assert global_key != "unknown"
    first = measures[0]
    assert first.symbolic_data["encoding_version"] == "1.0"
    assert first.symbolic_data["parts"]
    assert any(part["events"] for part in first.symbolic_data["parts"])
    assert "pitch_classes" in analyses[0].analysis_data
    assert analyses[0].analysis_data["analysis_version"] == "2.0"


def test_span_candidates_cover_score_with_score_derived_boundaries():
    score_path = Path("data/sonata32-2.krn")
    if not score_path.exists():
        pytest.skip("Op. 111 source is not available")

    measures, analyses, _ = build_symbolic_layers(str(score_path))
    candidates = build_span_candidates(measures, analyses)

    assert candidates
    assert candidates[0].measure_start == measures[0].measure_number
    assert candidates[-1].measure_end == measures[-1].measure_number
    assert all(candidate.measure_start_index <= candidate.measure_end_index for candidate in candidates)
    assert all("start_boundary" in candidate.evidence for candidate in candidates)
