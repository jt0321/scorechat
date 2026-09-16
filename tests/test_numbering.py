"""
tests/test_numbering.py
What counts as a bar.

`measure_index` counts every measure in sounding order. `measure_number` is
the printed bar a performer reads, and several things that are unquestionably
part of the score are not bars: the opening anacrusis, the upbeat written
after a repeat barline, an unbarred cadenza. These pin the rule that separates
them, and the resolution that keeps a range citable anyway.
"""

import pytest

from analysis.numbering import (
    ANACRUSIS, BAR, EMPTY, UNBARRED, UPBEAT, assign_roles, resolve_range,
)


def rows(*specs):
    """(number, duration, bar_duration) triples -> indexed rows."""
    return [(index, *spec) for index, spec in enumerate(specs)]


# --- classification ---------------------------------------------------------

def test_a_numbered_measure_is_a_bar_and_keeps_its_number():
    result = assign_roles(rows((1, 4.0, 4.0), (2, 4.0, 4.0)))
    assert [r.role for r in result] == [BAR, BAR]
    assert [r.measure_number for r in result] == [1, 2]
    assert [r.belongs_to for r in result] == [1, 2]


def test_an_opening_partial_measure_is_an_anacrusis():
    result = assign_roles(rows((None, 1.0, 4.0), (1, 4.0, 4.0)))
    assert result[0].role == ANACRUSIS
    assert result[0].measure_number is None


def test_a_partial_measure_completing_a_short_bar_is_an_upbeat():
    """Op. 2 No. 2/i is in 2/2. Bar 228 is written short at 1.5 beats and the
    missing 0.5 is written after the barline, so it can lead back into the
    exposition or on into the recapitulation. That fragment is a pickup, not a
    bar, and not debris."""
    result = assign_roles(rows((228, 1.5, 2.0), (None, 0.5, 2.0), (229, 2.0, 2.0)))
    assert result[1].role == UPBEAT
    assert result[1].measure_number is None


def test_an_upbeat_belongs_to_the_bar_it_leads_into():
    """Which is why the recapitulation reads as m. 229 and not as m. 228: the
    pickup is the start of the new material, not the end of the old."""
    result = assign_roles(rows((228, 1.5, 2.0), (None, 0.5, 2.0), (229, 2.0, 2.0)))
    assert result[1].belongs_to == 229


def test_an_unnumbered_measure_that_completes_nothing_is_unbarred():
    result = assign_roles(rows((1, 4.0, 4.0), (None, 6.0, 4.0), (2, 4.0, 4.0)))
    assert result[1].role == UNBARRED
    assert result[1].belongs_to == 1   # it continues from the bar before it


def test_a_measure_with_no_music_is_an_artefact():
    result = assign_roles(rows((1, 4.0, 4.0), (None, 0.0, 4.0)))
    assert result[1].role == EMPTY


def test_the_meter_decides_what_counts_as_short():
    """Without the bar length, two 2.0 measures in 2/2 look like a bar split in
    half. They are two whole bars, and only the meter says so."""
    in_cut_time = assign_roles(rows((1, 2.0, 4.0), (None, 2.0, 4.0), (2, 4.0, 4.0)))
    in_two_four = assign_roles(rows((1, 2.0, 2.0), (None, 2.0, 2.0), (2, 2.0, 2.0)))
    assert in_cut_time[1].role == UPBEAT      # 2.0 + 2.0 makes one 4.0 bar
    assert in_two_four[1].role == UNBARRED    # already a whole bar on its own


def test_an_empty_artefact_does_not_hide_the_bar_an_upbeat_completes():
    """A structural barline can produce an empty measure between the short bar
    and its pickup; the pair must still be recognised."""
    result = assign_roles(rows((14, 2.0, 3.0), (None, 0.0, 3.0), (None, 1.0, 3.0), (15, 3.0, 3.0)))
    assert [r.role for r in result] == [BAR, EMPTY, UPBEAT, BAR]


def test_bar_length_falls_back_to_the_commonest_numbered_measure():
    result = assign_roles([(0, 1, 4.0, None), (1, 2, 4.0, None),
                           (2, None, 1.0, None), (3, 3, 4.0, None)])
    assert result[2].role == UPBEAT or result[2].role == UNBARRED
    assert result[2].measure_number is None


# --- reporting --------------------------------------------------------------

def test_a_range_opening_on_a_pickup_is_reported_as_the_bar_it_leads_into():
    result = assign_roles(rows((228, 1.5, 2.0), (None, 0.5, 2.0), (229, 2.0, 2.0), (230, 2.0, 2.0)))
    assert resolve_range(result, 1, 3) == {
        "measure_start": 229, "measure_end": 230, "starts_with_upbeat": True,
    }


def test_a_range_on_ordinary_bars_says_nothing_about_a_pickup():
    result = assign_roles(rows((1, 4.0, 4.0), (2, 4.0, 4.0), (3, 4.0, 4.0)))
    assert resolve_range(result, 0, 2) == {
        "measure_start": 1, "measure_end": 3, "starts_with_upbeat": False,
    }


def test_no_measure_ever_resolves_to_zero():
    """0 was the sentinel for "not numbered". It also read as a bar number and
    got printed as one, which is the whole reason this module exists."""
    result = assign_roles(rows((None, 1.0, 4.0), (1, 4.0, 4.0), (2, 4.0, 4.0)))
    assert all(r.measure_number != 0 for r in result)
    assert all(r.belongs_to != 0 for r in result)
    assert resolve_range(result, 0, 2)["measure_start"] == 1


# --- how long a measure is --------------------------------------------------

def test_a_measure_is_as_long_as_its_span_not_the_sum_of_its_events():
    """Two voices sounding together do not make the measure twice as long.
    Summing durations made a divided staff in Op. 111's Arietta total 3.0 in a
    bar 1.5 long, so two complete bars were classified as unbarred."""
    from db.store import measure_span
    divided = {"parts": [{"events": [
        {"offset": "0.0", "duration": {"quarter_length": "1.5"}},          # upper voice
        {"offset": "0.0", "duration": {"quarter_length": "0.75"}},         # lower voice
        {"offset": "0.75", "duration": {"quarter_length": "0.75"}},
    ]}]}
    assert measure_span(divided) == 1.5


def test_span_spans_every_part():
    from db.store import measure_span
    data = {"parts": [
        {"events": [{"offset": "0.0", "duration": {"quarter_length": "1.0"}}]},
        {"events": [{"offset": "0.0", "duration": {"quarter_length": "4.0"}}]},
    ]}
    assert measure_span(data) == 4.0


def test_a_whole_bar_in_a_faster_meter_is_not_an_upbeat():
    """The rule needs the meter that is actually in force. A 1.5-beat measure
    is a whole bar in 6/16 and an incomplete one in 9/16, and reading a stale
    9/16 across Op. 111/ii invented anacruses out of complete bars."""
    in_six_sixteen = assign_roles(rows((36, 1.5, 1.5), (None, 1.5, 1.5), (37, 1.5, 1.5)))
    assert in_six_sixteen[1].role == UNBARRED     # a whole bar, merely unnumbered
    in_nine_sixteen = assign_roles(rows((36, 0.75, 2.25), (None, 1.5, 2.25), (37, 2.25, 2.25)))
    assert in_nine_sixteen[1].role == UPBEAT      # 0.75 + 1.5 makes one 9/16 bar
