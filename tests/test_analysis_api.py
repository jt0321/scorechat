"""
tests/test_analysis_api.py
The functions the chat model calls.

These run against the ingested corpus and skip without it. What they pin is
not the musical content -- `evaluate_form.py` scores that -- but the API's
contract: printed bar numbers in and out, ranges resolved through the bar a
measure belongs to, and an empty result where the pipeline found nothing
rather than a value the model might mistake for a finding.
"""

import pytest

from pipeline.analysis_api import (
    compare_spans, describe_span, find_recurrences, get_key_plan,
    locate_in_form, outline_sonata, resolve_work,
)


def work(query):
    resolved = resolve_work(query)["resolved"]
    if resolved is None:
        pytest.skip(f"{query!r} is not ingested")
    return resolved["work_id"]


# --- resolving a movement ---------------------------------------------------

@pytest.mark.parametrize("query,expected_movement", [
    ("moonlight 3rd movement", 3),
    ("Moonlight sonata third movement", 3),
    ("op 27 no 2 iii", 3),
    ("waldstein mvt 1", 1),
    ("Pathetique 1st mvt", 1),          # typed without the diacritic
    ("the last movement of op 57", 3),  # Op. 57 has three, so "last" is 3
    ("finale of the moonlight", 3),
])
def test_a_movement_is_found_however_it_is_named(query, expected_movement):
    resolved = resolve_work(query)["resolved"]
    if resolved is None:
        pytest.skip("corpus not ingested")
    assert resolved["movement_number"] == expected_movement


def test_an_opus_and_number_are_matched_as_a_unit():
    """"27" and "2" as loose tokens match Op. 27 No. 1 exactly as well as
    Op. 27 No. 2, which left every request of this shape ambiguous."""
    resolved = resolve_work("op 27 no 1 ii")["resolved"]
    if resolved is None:
        pytest.skip("corpus not ingested")
    assert resolved["opus"] == "Op. 27 No. 1"


@pytest.mark.parametrize("query", [
    "op31/no3", "Op.31/No.3", "op 31, no 3", "op31no3", "Op. 31 No. 3", "the Hunt",
])
def test_a_sonata_is_found_however_its_opus_is_punctuated(query):
    """Users drop the spaces and periods; "op31/no3" is still one sonata, not
    the whole of Op. 31."""
    result = resolve_work(query)
    if not result["matches"]:
        pytest.skip("corpus not ingested")
    assert result["sonata"]["opus"] == "Op. 31 No. 3"
    assert result["resolved"] is None          # no movement was named
    assert result["note"]


def test_a_sonata_lists_its_movements_with_headings_and_engraved_keys():
    """The work is the sonata; its movements are what a recording splits into
    tracks. Op. 31 No. 3's Scherzo is the one movement not in E-flat."""
    sonata = resolve_work("op31/no3")["sonata"]
    if sonata is None:
        pytest.skip("corpus not ingested")
    assert sonata["nickname"] == "The Hunt"
    assert [(m["movement_number"], m["key"]) for m in sonata["movements"]] == [
        (1, "E-flat major"), (2, "A-flat major"), (3, "E-flat major"), (4, "E-flat major")]
    assert sonata["movements"][1]["heading"].startswith("Scherzo")


def test_a_sonata_is_outlined_movement_by_movement_in_one_call():
    """A question about the whole sonata wants every movement, in order, with
    what a short synopsis rests on -- and no more than that."""
    sonata = resolve_work("op31/no3")["sonata"]
    if sonata is None:
        pytest.skip("corpus not ingested")
    outline = outline_sonata(sonata["movements"][2]["work_id"])  # any movement will do
    movements = outline["movements"]
    assert [m["movement_number"] for m in movements] == [1, 2, 3, 4]
    finale = movements[3]
    assert finale["meters"] == ["6/8"] and finale["bars"] > 300
    # The pickup marked "I" is not reported as a section of its own.
    assert [(s["label"], s["repeated"]) for s in finale["sections"]] == [("A", True), ("B", False)]
    # Capped: the finale has more estimated regions than an overview lists.
    assert len(finale["estimated_keys"]) <= finale["estimated_key_regions"]
    assert finale["estimated_keys"][0]["key"] == "E-flat major"


def test_outlining_an_unknown_work_is_an_error_not_an_empty_sonata():
    assert "error" in outline_sonata(-1)


@pytest.mark.parametrize("query,opus,expected_movement", [
    ("the fugue of op 106", "Op. 106", 4),              # "Introduzione: Largo---Fuga: ..."
    ("Hammerklavier Fuga: Allegro risoluto", "Op. 106", 4),
    ("the scherzo of the hunt", "Op. 31 No. 3", 2),
    ("op31/no3 minuet", "Op. 31 No. 3", 3),             # heading reads "Menuetto"
])
def test_a_movement_is_found_by_its_heading(query, opus, expected_movement):
    resolved = resolve_work(query)["resolved"]
    if resolved is None:
        pytest.skip("corpus not ingested")
    assert (resolved["opus"], resolved["movement_number"]) == (opus, expected_movement)


def test_a_heading_shared_by_two_movements_resolves_to_nothing():
    """Op. 106's first movement and its fugue are both marked Allegro."""
    result = resolve_work("hammerklavier allegro")
    if not result["matches"]:
        pytest.skip("corpus not ingested")
    assert result["resolved"] is None


def test_a_bare_opus_names_the_set_not_one_sonata():
    result = resolve_work("op 31")
    if not result["matches"]:
        pytest.skip("corpus not ingested")
    assert result["resolved"] is None and result["sonata"] is None
    assert "set of 3 sonatas" in result["note"]


def test_an_ambiguous_request_resolves_to_nothing_and_says_why():
    """Guessing between movements would put every later answer in the wrong
    music, so ambiguity is handed back rather than resolved."""
    result = resolve_work("moonlight")
    if not result["matches"]:
        pytest.skip("corpus not ingested")
    assert result["resolved"] is None
    assert result["note"]


def test_an_unknown_request_returns_no_matches_rather_than_the_nearest_thing():
    result = resolve_work("Rachmaninoff third concerto")
    assert result["resolved"] is None and result["matches"] == []


# --- describing and locating ------------------------------------------------

def test_describe_span_reports_printed_bars_and_its_section():
    work_id = work("op 111 i")
    described = describe_span(work_id, 20, 30)
    assert described["measure_start"] == 20
    assert described["section"]["label"] == "B"       # the repeated exposition
    assert described["section"]["play_count"] == 2


def test_a_range_starting_on_a_pickup_says_so():
    """Op. 2 No. 2/i's recapitulation is preceded by an upbeat written after
    the barline. The range still reports itself as m. 229."""
    work_id = work("op 2 no 2 i")
    described = describe_span(work_id, 229, 236)
    assert described["measure_start"] == 229
    assert described["starts_with_upbeat"] is True


def test_locate_in_form_gives_the_scheme_without_naming_the_form():
    work_id = work("op 111 i")
    located = locate_in_form(work_id, 55)
    assert located["repeat_scheme"] == "A,B,B1,B,B2,C"
    text = str(located).lower()
    assert "exposition" not in text and "recapitulation" not in text


def test_a_movement_notating_no_sections_says_that_rather_than_returning_nothing():
    work_id = work("op 57 i")   # the famous omitted exposition repeat
    located = locate_in_form(work_id, 1)
    assert located["section"] is None
    assert located["note"] and "no section structure" in located["note"]


# --- recurrences ------------------------------------------------------------

def test_the_recapitulation_is_found_and_cited_by_printed_bar():
    """Op. 2 No. 2/i: the return was always found; before measure roles it was
    reported as mm. 0-247 because it opens on the upbeat into m. 229."""
    work_id = work("op 2 no 2 i")
    found = find_recurrences(work_id, 1, 19)["recurrences"]
    returns = [r for r in found if r["direction"] == "returns_at"
               and r["target_measures"][0] == 229]
    assert returns, "the recapitulation was not proposed"
    assert returns[0]["transposed_semitones"] == 0     # home, not transposed


def test_direction_distinguishes_a_return_from_its_statement():
    """Asking about the recapitulation must not read like asking about the
    exposition; the direction is what keeps them apart."""
    work_id = work("op 2 no 2 i")
    at_recap = find_recurrences(work_id, 229, 247)["recurrences"]
    assert any(r["direction"] == "restates" for r in at_recap)


def test_no_recurrence_is_reported_as_none_found_not_as_no_answer():
    work_id = work("op 57 i")
    result = find_recurrences(work_id, 1, 8)
    if result["recurrences"]:
        pytest.skip("this movement now has proposals here")
    assert result["note"] and "not that the material is unique" in result["note"]


# --- comparison and key plan ------------------------------------------------

def test_compare_spans_measures_a_pair_with_no_stored_relation():
    """The Waldstein's recapitulation is recomposed for four bars, so mm. 3ff
    resume verbatim only at m. 160."""
    work_id = work("waldstein mvt 1")
    compared = compare_spans(work_id, 3, 10, 160, 167)
    assert compared["repeats_confidence"] > 0.9
    assert compared["transposed_semitones"] == 0


def test_a_weak_comparison_is_reported_with_a_warning_not_as_a_relationship():
    work_id = work("waldstein mvt 1")
    compared = compare_spans(work_id, 3, 10, 90, 97)
    if max(compared["repeats_confidence"], compared["varies_confidence"]) >= 0.75:
        pytest.skip("these ranges now match")
    assert compared["note"] and "rather than a relationship" in compared["note"]


def test_a_single_key_region_is_flagged_as_unconfirmed():
    """The estimator smooths its trajectory and reports the Moonlight finale as
    c# minor throughout, missing the g# minor second group. The tool must not
    let that read as an established absence of modulation."""
    work_id = work("moonlight 3rd movement")
    plan = get_key_plan(work_id)
    if len(plan["regions"]) > 1:
        pytest.skip("the estimator now finds modulations here")
    assert plan["note"] and "unconfirmed" in plan["note"]
