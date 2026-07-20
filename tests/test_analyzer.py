"""
tests/test_analyzer.py
Smoke tests for the music21 analysis module.
Requires music21 installed and a sample MusicXML file.
"""
import pytest
from pathlib import Path
from analysis.analyzer import analyze_score, detect_texture


SAMPLE_XML = Path(__file__).parent / "fixtures" / "sample.musicxml"


@pytest.mark.skipif(not SAMPLE_XML.exists(), reason="No fixture MusicXML")
def test_analyze_produces_chunks():
    chunks, global_key = analyze_score(str(SAMPLE_XML), window=4)
    assert len(chunks) > 0
    assert global_key is not None
    for c in chunks:
        assert c.measure_start <= c.measure_end
        assert c.summary_text is not None
        assert len(c.summary_text) > 10


def test_detect_texture_with_music21():
    from music21 import stream, note
    s = stream.Stream()
    for pitch in ["C4", "D4", "E4", "F4", "G4", "A4"]:
        s.append(note.Note(pitch, quarterLength=1))
    tag = detect_texture(s)
    assert tag in {"stepwise_melody", "cantabile", "chordal", "octaves_or_leaps"}
