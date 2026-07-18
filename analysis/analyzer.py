"""
analysis/analyzer.py
Parses MusicXML via music21 and produces per-measure chunk dicts
suitable for embedding and storage in score_segments.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import music21
from music21 import converter, analysis, stream, harmony, roman


@dataclass
class MeasureChunk:
    measure_start:   int
    measure_end:     int
    part:            str = "grand_staff"
    local_key:       Optional[str] = None
    roman_numerals:  Optional[str] = None
    harmonic_rhythm: Optional[str] = None
    texture_tag:     Optional[str] = None
    formal_function: Optional[str] = None
    motif_tags:      list[str] = field(default_factory=list)
    summary_text:    Optional[str] = None
    musicxml_slice:  Optional[str] = None


def detect_texture(measures: stream.Stream) -> str:
    """Heuristic texture detection based on note/chord density and intervals."""
    note_count = len(measures.flatten().notes)
    chord_count = len(measures.flatten().getElementsByClass("Chord"))
    if chord_count > note_count * 0.6:
        return "chordal"
    pitches = [n.pitch for n in measures.flatten().notes if hasattr(n, "pitch")]
    if len(pitches) > 1:
        intervals = [abs(pitches[i+1].midi - pitches[i].midi) for i in range(len(pitches)-1)]
        avg_interval = sum(intervals) / len(intervals) if intervals else 0
        if avg_interval <= 2:
            return "stepwise_melody"
        if avg_interval >= 10:
            return "octaves_or_leaps"
    return "cantabile"


def analyze_score(score_path: str, window: int = 4) -> tuple[list[MeasureChunk], str]:
    """
    Parse a score file (Humdrum, MusicXML, etc.) and return a list of MeasureChunk objects,
    windowed by `window` measures (analogous to paragraph-level chunking).
    """
    score = converter.parse(score_path)
    parts = score.parts

    # Key analysis over full score
    key_analyzer = analysis.discrete.KrumhanslSchmuckler()
    key_obj = key_analyzer.getSolution(score)
    global_key = str(key_obj) if key_obj else "unknown"

    # Chordify for Roman numeral analysis
    chordified = score.chordify()

    all_measures = list(score.parts[0].getElementsByClass("Measure"))
    total = len(all_measures)
    chunks: list[MeasureChunk] = []

    for start_idx in range(0, total, window):
        end_idx = min(start_idx + window - 1, total - 1)
        m_start = all_measures[start_idx].number
        m_end   = all_measures[end_idx].number

        # Slice of chordified score for this window
        window_chords = chordified.measures(m_start, m_end)

        # Local key via window analysis
        try:
            local_key_obj = key_analyzer.getSolution(window_chords)
            local_key = str(local_key_obj)
        except Exception:
            local_key = global_key

        # Roman numerals (up to 8 per window to keep summary concise)
        rn_labels = []
        for c in window_chords.flatten().getElementsByClass("Chord"):
            try:
                rn = roman.romanNumeralFromChord(c, music21.key.Key(local_key.split()[0]))
                rn_labels.append(rn.figure)
            except Exception:
                pass
        rn_str = " ".join(rn_labels[:8]) if rn_labels else None

        # Harmonic rhythm: rough count of chord changes per measure
        changes_per_measure = len(rn_labels) / max(window, 1)
        if changes_per_measure <= 1:
            harmonic_rhythm = "slow"
        elif changes_per_measure <= 3:
            harmonic_rhythm = "moderate"
        else:
            harmonic_rhythm = "fast"

        # Texture from soprano (right hand) part
        rh_window = parts[0].measures(m_start, m_end)
        texture = detect_texture(rh_window)

        # Build summary text (this is what gets embedded)
        summary = (
            f"Measures {m_start}–{m_end} of the score. "
            f"Local key: {local_key}. "
            f"Harmonic progression: {rn_str or 'undetermined'}. "
            f"Harmonic rhythm: {harmonic_rhythm}. "
            f"Texture: {texture}."
        )

        # Extract raw MusicXML slice
        try:
            xml_slice = rh_window.write("musicxml").read_text()
        except Exception:
            xml_slice = None

        chunks.append(MeasureChunk(
            measure_start   = m_start,
            measure_end     = m_end,
            part            = "grand_staff",
            local_key       = local_key,
            roman_numerals  = rn_str,
            harmonic_rhythm = harmonic_rhythm,
            texture_tag     = texture,
            summary_text    = summary,
            musicxml_slice  = xml_slice,
        ))

    return chunks, global_key
