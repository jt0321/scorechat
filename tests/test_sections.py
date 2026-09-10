"""
tests/test_sections.py
Notated section structure read from Humdrum expansion records.

The expected structures here are engraved facts about specific movements --
what Beethoven marked to be repeated -- not this parser's own output.
"""
import os
import pytest
from pathlib import Path

from analysis.sections import (
    parse_notated_sections, repeated_sections, section_evidence,
)


def _load(name: str) -> str:
    path = Path("data") / name
    if not path.exists():
        pytest.skip(f"{name} is not available")
    return path.read_text(encoding="utf-8")


def _sections(name: str):
    return parse_notated_sections(_load(name))


def _by_label(sections):
    return {s.label: s for s in sections}


# --- parsing ---------------------------------------------------------------

def test_expansion_and_ranges_are_read_from_the_first_column():
    source = "*>[A,A,B]\tx\n*>A\tx\n=1\tx\n=2\tx\n=3\tx\n*>B\tx\n=4\tx\n"
    expansion, sections = parse_notated_sections(source)
    assert expansion == ["A", "A", "B"]
    assert [(s.label, s.measure_start, s.measure_end, s.play_count) for s in sections] == [
        ("A", 1, 2, 2), ("B", 3, 4, 1),
    ]


def test_a_label_starts_at_the_barline_it_follows():
    """`*>B` is written immediately after the barline that opens section B, so
    that barline is B's first measure and not A's last.

    Op. 111 marks `=19!|:` then `*>B` and its Allegro is bar 19; reading the
    barline as the end of the introduction would date every section in the
    corpus one bar late.
    """
    source = "*>[A,B,B]\tx\n*>A\tx\n=1\tx\n4c\tx\n=19!|:\tx\n*>B\tx\n4c\tx\n=20\tx\n"
    _, sections = parse_notated_sections(source)
    ranges = _by_label(sections)
    assert ranges["A"].measure_start == 1 and ranges["A"].measure_end == 18
    assert ranges["B"].measure_start == 19


def test_a_label_after_an_unnumbered_barline_takes_the_next_number():
    """A plain repeat sign carries no measure number, so the section it opens
    starts at the next numbered barline -- how 46 of the corpus's in-score
    labels are introduced."""
    source = "*>[A,A,B,B]\tx\n*>A\tx\n=1\tx\n4c\tx\n=:|!|:\tx\n*>B\tx\n4c\tx\n=49\tx\n"
    _, sections = parse_notated_sections(source)
    assert _by_label(sections)["B"].measure_start == 49


def test_a_comment_does_not_separate_a_label_from_its_barline():
    """Humdrum layout comments (`!!LO:...`) sit between the barline and the
    section label throughout this corpus."""
    source = "*>[A,B]\tx\n*>A\tx\n=1\tx\n4c\tx\n=9\tx\n!!LO:LB:g=original\n*>B\tx\n4c\tx\n=10\tx\n"
    _, sections = parse_notated_sections(source)
    assert _by_label(sections)["B"].measure_start == 9


def test_norep_expansion_is_ignored():
    """*>norep[...] is the performance option that omits repeats; the written
    structure is the plain expansion."""
    source = "*>[A,A,B]\tx\n*>norep[A,B]\tx\n*>A\tx\n=1\tx\n*>B\tx\n=2\tx\n"
    expansion, _ = parse_notated_sections(source)
    assert expansion == ["A", "A", "B"]


def test_alternate_endings_are_identified_by_their_parent_section():
    source = "*>[A,A1,A,A2,B]\tx\n*>A\tx\n=1\tx\n*>A1\tx\n=2\tx\n*>A2\tx\n=3\tx\n*>B\tx\n=4\tx\n"
    _, sections = parse_notated_sections(source)
    by_label = _by_label(sections)
    assert by_label["A1"].is_alternate_ending and by_label["A1"].parent_label == "A"
    assert not by_label["A"].is_alternate_ending
    assert not by_label["B"].is_alternate_ending


def test_a_trailing_digit_without_a_parent_section_is_not_an_ending():
    source = "*>[X1,Y]\tx\n*>X1\tx\n=1\tx\n*>Y\tx\n=2\tx\n"
    _, sections = parse_notated_sections(source)
    assert not _by_label(sections)["X1"].is_alternate_ending


def test_a_score_with_no_expansion_record_yields_nothing():
    assert parse_notated_sections("**kern\n=1\tx\n=2\tx\n") == ([], [])


def test_repeated_sections_excludes_alternate_endings():
    source = "*>[A,A1,A,A2,B]\tx\n*>A\tx\n=1\tx\n*>A1\tx\n=2\tx\n*>A2\tx\n=3\tx\n*>B\tx\n=4\tx\n"
    _, sections = parse_notated_sections(source)
    assert [s.label for s in repeated_sections(sections)] == ["A"]


# --- engraved structure of specific movements -----------------------------

def test_op57_finale_repeats_its_second_section_not_its_first():
    """Op. 57/iii inverts the Classical scheme: the exposition (A) is played
    once and the development-and-recapitulation (B) is what repeats."""
    expansion, sections = _sections("sonata23-3.krn")
    by_label = _by_label(sections)
    assert by_label["A"].play_count == 1
    assert by_label["B"].play_count == 2
    assert by_label["B"].measure_start > by_label["A"].measure_start
    # and B is the substantial one: the repeat covers most of the movement
    assert by_label["B"].measure_end - by_label["B"].measure_start > 100


def test_op2no1_first_movement_repeats_both_halves():
    """The Classical norm, against which Op. 57/iii is the departure."""
    _, sections = _sections("sonata01-1.krn")
    assert all(s.play_count == 2 for s in repeated_sections(sections))
    assert [s.label for s in repeated_sections(sections)] == ["A", "B"]


def test_op111_slow_introduction_stands_outside_the_repeat():
    """Op. 111/i opens with a Maestoso introduction: a short unrepeated
    section before the repeated exposition."""
    _, sections = _sections("sonata32-1.krn")
    by_label = _by_label(sections)
    assert by_label["A"].play_count == 1
    assert by_label["A"].measure_end < 25          # short
    assert by_label["B"].play_count == 2           # the exposition repeats
    assert by_label["B"].measure_start > by_label["A"].measure_end


def test_appassionata_first_movement_notates_no_repeat_at_all():
    """Op. 57/i famously omits the exposition repeat; the absence of an
    expansion record is itself the evidence."""
    expansion, sections = _sections("sonata23-1.krn")
    assert expansion == []
    assert sections == []


def test_evidence_records_the_scheme_without_naming_form_parts():
    expansion, sections = _sections("sonata23-3.krn")
    evidence = section_evidence(expansion, sections, _by_label(sections)["B"])
    assert evidence["is_repeated"] is True
    assert evidence["repeat_scheme"].startswith("A,B,B1")
    assert "B" in evidence["repeated_labels"]
    # No form vocabulary is asserted anywhere in the evidence.
    assert not {"exposition", "development", "recapitulation"} & set(
        str(v).lower() for v in evidence.values()
    )


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="no DATABASE_URL configured")
def test_sections_are_stored_as_notated_spans():
    from db.store import get_source_text, list_works
    from db.session import session_scope
    from sqlalchemy import text
    work = next((w for w in list_works()
                 if "No. 23" in w["title"] and w.get("movement_number") == 3), None)
    if work is None or get_source_text(work["id"]) is None:
        pytest.skip("Op. 57/iii is not ingested")
    with session_scope() as session:
        rows = session.execute(text(
            "SELECT label, confidence FROM span_analyses "
            "WHERE work_id=:w AND span_type='section' ORDER BY measure_start_index"
        ), {"w": work["id"]}).all()
    assert rows, "notated sections were not stored (run build_sections.py)"
    assert all(confidence == 1.0 for _, confidence in rows)  # notated, not estimated
    assert "A" in [label for label, _ in rows]
