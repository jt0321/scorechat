"""
tests/test_claims.py
Claims read from commentary, and how each is checked against the score.

Synthetic movements throughout, except the last test, which pins one real
finding against the ingested books: Elterlein's "an allegretto, D flat major
... follows as the second movement" is supported by the Moonlight Allegretto's
engraved key. That single line exercises extraction, the sentence placing the
claim, the spelling bridge between "D flat" and music21's "D-", and the
engraved check.
"""

import pytest
from sqlalchemy import text

from commentary.anchoring import Movement, Sonata
from commentary.claims import (
    Claim, MovementKeys, check_claim, extract_claims, key_value, movements_for,
)
from analysis.harmony import parse_key_name
from db.session import session_scope


def claims(text_, kind=None):
    return [(c.claim_type, c.value) for c in extract_claims(text_) if kind in (None, c.claim_type)]


def test_a_key_in_the_prose_is_a_claim_and_a_key_naming_a_sonata_is_not():
    assert claims("It modulates to E flat major.", "key") == [("key", "E-flat major")]
    assert claims("the F minor Sonata, Op. 2", "key") == []
    assert claims("the Sonata in C sharp minor", "key") == []


def test_bar_references_are_kept_as_the_author_wrote_them():
    assert claims("In measures 28 to 31 the first tone is doubled.", "bar_ref") == [("bar_ref", "measure 28-31")]
    assert claims("which appears first at bar 21", "bar_ref") == [("bar_ref", "bar 21")]


def test_formal_terms_are_normalised_across_the_books_vocabularies():
    """Elterlein's "working-out" is Marx's "development"; "subsidiary subject"
    is the second theme."""
    assert claims("the working-out of the subsidiary subject", "form") == [
        ("form", "development"), ("form", "second theme")]


def test_a_claim_is_recorded_once_per_passage():
    assert claims("in E major ... and again in E major", "key") == [("key", "E major")]


def test_a_readers_key_and_a_stored_key_meet_as_pitch_class_and_mode():
    assert key_value("D-flat major") == parse_key_name("D- major")
    assert key_value("C-sharp minor") == parse_key_name("c# minor")
    assert key_value("C-sharp minor") != key_value("C-sharp major")


FINALE = MovementKeys(3, 3, "c# minor", ({"value": "c# minor", "measure_start": 1, "measure_end": 20},
                                         {"value": "E major", "measure_start": 21, "measure_end": 40}))
PASSAGE = {"sonata": 14, "work_ids": [3], "anchor_status": "declared"}


def check(value, movements=(FINALE,), passage=PASSAGE, kind="key"):
    return check_claim(Claim(kind, value, "…", 0), passage, list(movements))


def test_the_engraved_key_supports_a_claim():
    result = check("C-sharp minor")
    assert result.status == "supported" and result.evidence["basis"] == "engraved"


def test_another_key_is_never_contradicted_only_compared_with_the_estimate():
    """"E major" in a C-sharp minor movement may be where it modulates to, so
    the engraved key cannot contradict it; the estimated regions decide."""
    assert check("E major").status == "agrees_with_estimate"
    assert check("G-sharp minor").status == "disagrees_with_estimate"
    assert "contradicted" not in {check(k).status for k in ("E major", "G-sharp minor", "F major")}


def test_a_disagreement_records_when_the_claim_is_the_relative_of_an_estimate():
    """The estimator's commonest confusion is a key for its relative, so a
    disagreement that is only that says so."""
    result = check("A minor", movements=(MovementKeys(1, 1, "C major",
                   ({"value": "C major", "measure_start": 1, "measure_end": 9},)),))
    assert result.status == "disagrees_with_estimate" and result.evidence["relative_of"]


def test_claims_that_cannot_be_checked_say_why():
    assert "edition" in check("bar 21", kind="bar_ref").evidence["reason"]
    assert "interpretive" in check("coda", kind="form").evidence["reason"]
    compared = dict(PASSAGE, anchor_status="conflict")
    assert "other sonatas" in check("C-sharp minor", passage=compared).evidence["reason"]
    general = dict(PASSAGE, sonata=None, work_ids=[])
    assert check("C-sharp minor", movements=(), passage=general).status == "not_checkable"


def test_the_claims_own_sentence_places_it_before_the_passage_does():
    moonlight = Sonata(14, 27, 2, "Op. 27 No. 2", "C-sharp minor", "Moonlight",
                       (Movement(1, 1, "Adagio sostenuto"), Movement(2, 2, "Allegretto"),
                        Movement(3, 3, "Presto agitato")))
    movements = [MovementKeys(n, n, k) for n, k in ((1, "c# minor"), (2, "D- major"), (3, "c# minor"))]
    claim = Claim("key", "C-sharp minor", "In the last movement, presto agitato, C sharp minor", 0)
    placed, by = movements_for(claim, {"work_ids": [1, 2, 3]}, movements, moonlight)
    assert [m.number for m in placed] == [3] and by == "sentence"


def test_elterlein_on_the_moonlight_allegretto_is_borne_out_by_the_engraved_key():
    with session_scope() as session:
        row = session.execute(text("""
            SELECT c.check_status, c.check_evidence FROM passage_claims c
            JOIN text_passages p ON p.id = c.passage_id JOIN text_documents d ON d.id = p.document_id
            WHERE d.source_key = 'elterlein-1879' AND p.sonata = 14 AND c.value = 'D-flat major'
        """)).first()
    if row is None:
        pytest.skip("commentary claims not built")
    status, evidence = row
    assert status == "supported"
    assert evidence["movement"] == 2 and evidence["placed_by"] == "sentence"
