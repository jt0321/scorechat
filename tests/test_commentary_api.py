"""
tests/test_commentary_api.py
The commentary as the chat model reaches it: two API functions and their tools.

Against the ingested commentary, skipping without it. What must hold is the
separation the project rests on -- every commentary result says it is opinion,
names its author and page, and never passes itself off as score evidence --
and that a movement's claims are the ones actually about that movement.
"""

import pytest
from sqlalchemy import text

from db.session import session_scope
from pipeline import analysis_api, tools


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Full-text search only: with an embedded corpus and a key in .env, the
    default would embed each test query through a provider."""
    monkeypatch.setattr("commentary.search.default_model", lambda: None)


def moonlight(movement: int) -> int:
    with session_scope() as session:
        work = session.execute(text(
            "SELECT id FROM works WHERE work_number = 14 AND movement_number = :m"), {"m": movement}).scalar()
        ingested = session.execute(text("SELECT count(*) FROM passage_claims")).scalar()
    if work is None or not ingested:
        pytest.skip("corpus or commentary not ingested")
    return work


def test_every_passage_is_attributed_and_located():
    result = analysis_api.search_commentary("stormy passionate finale", work_id=moonlight(3))
    assert result["passages"]
    assert "opinion" in result["note"]
    for passage in result["passages"]:
        assert passage["author"] and passage["year"] and passage["url"].startswith("https://archive.org/")
        assert passage["page"] is not None or passage["line"] is not None
        assert passage["scope"] in ("movement", "sonata")


def test_a_work_that_is_not_a_sonata_movement_is_an_error_not_a_search():
    assert "error" in analysis_api.search_commentary("anything", work_id=-1)
    assert "error" in analysis_api.commentary_claims(-1)


def test_the_allegretto_claim_is_supported_by_the_engraved_key():
    claims = analysis_api.commentary_claims(moonlight(2))["claims"]
    supported = [c for c in claims if c["value"] == "D-flat major" and c["check"] == "supported"]
    assert supported and supported[0]["author"].endswith("Elterlein")


def test_a_claim_placed_in_the_finale_by_its_own_sentence_stays_with_the_finale():
    """"In the last movement, presto agitato, C sharp minor" sits in a passage
    spanning all three movements; it belongs to the finale alone."""
    def quotes(work):
        return {c["quote"] for c in analysis_api.commentary_claims(work)["claims"]}
    finale_line = next(q for q in quotes(moonlight(3)) if q.startswith("In the last movement"))
    assert finale_line not in quotes(moonlight(1))


def test_the_commentary_tools_are_bound_and_the_model_is_told_they_are_opinion():
    names = {t.name for t in tools.TOOLS}
    assert {"search_commentary_tool", "commentary_claims_tool"} <= names
    assert "never present it as fact" in tools.SYSTEM_PROMPT
    assert "author's edition" in tools.search_commentary_tool.description
