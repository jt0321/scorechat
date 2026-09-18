"""
tests/test_commentary_search.py
Searching the commentary, and the embedding providers behind it.

No network. Provider wiring is checked by constructing clients, not calling
them. The search tests run against the ingested passages (and skip without
them); the vector path is exercised by handing the search a stored passage's
own vector as the "query", which must bring that passage back first -- a
deterministic stand-in for a provider call.
"""

import pytest
from sqlalchemy import text

from commentary.search import _or_query, search_passages
from db.session import session_scope
from pipeline import providers


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-test")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")


def test_a_model_is_stored_under_one_name_however_it_is_asked_for():
    """"gemini" and "gemini:gemini-embedding-2" are the same vectors; storing
    them under two names would embed the corpus twice and split searches."""
    assert providers.embedding_model_id("gemini") == providers.embedding_model_id("gemini:gemini-embedding-2")
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        providers.embedding_model_id("anthropic")          # has no embeddings API


def test_openai_compatible_embeddings_send_text_not_token_ids():
    """langchain-openai tokenises locally and sends token ids unless told not
    to; only OpenAI accepts those, so the free tiers must get plain text."""
    for spec, host in (("openrouter", "openrouter.ai"), ("cloudflare", "accounts/acct-123/ai/v1")):
        client = providers.get_embeddings(spec)
        assert client.check_embedding_ctx_length is False and host in str(client.openai_api_base)


def test_a_question_becomes_any_of_its_words_not_all_of_them():
    assert _or_query("what does Marx say about the stormy finale?") == "marx | stormy | finale"
    assert _or_query("the and of") == ""


@pytest.fixture(scope="module")
def passage():
    with session_scope() as session:
        row = session.execute(text("""
            SELECT p.id, p.sonata, e.model, e.embedding::text AS vector
            FROM text_passages p JOIN passage_embeddings e ON e.passage_id = p.id
            WHERE p.sonata = 14 ORDER BY p.id LIMIT 1""")).mappings().first()
    if not row:
        pytest.skip("commentary not ingested and embedded")
    return dict(row, vector=[float(x) for x in row["vector"].strip("[]").split(",")])


def test_the_vector_path_finds_the_passage_its_vector_came_from(passage):
    result = search_passages("zzzz", model=passage["model"], embed=lambda _q: passage["vector"])
    assert result["retrievers"] == ["fulltext", "vector"]
    assert result["passages"][0]["id"] == passage["id"]
    assert result["passages"][0]["found_by"] == {"vector": 1}


def test_a_model_the_corpus_was_not_embedded_with_falls_back_to_full_text(passage):
    """A query embedded by one model cannot be compared with passages embedded
    by another, so the search says so and uses full text, never mixes them."""
    result = search_passages("Moonlight finale", model="ollama:not-embedded")
    assert result["retrievers"] == ["fulltext"] and "not been embedded" in result["note"]


def test_the_sonata_filter_keeps_to_the_sonata(passage):
    result = search_passages("finale storm passion", sonata=14, model="ollama:not-embedded", limit=10)
    assert result["passages"]
    assert all(p["sonata"] == 14 or p["sonata"] is None for p in result["passages"])


def test_a_movement_search_marks_general_remarks_as_about_the_sonata(passage):
    """Asking about the finale also returns remarks on the Moonlight as a
    whole, but they must say so rather than pass as remarks on the finale."""
    with session_scope() as session:
        finale = session.execute(text(
            "SELECT id FROM works WHERE work_number = 14 AND movement_number = 3")).scalar_one()
    result = search_passages("sonata storm passion lament", work_id=finale,
                             model="ollama:not-embedded", limit=20)
    scopes = {p["scope"] for p in result["passages"]}
    assert scopes <= {"movement", "sonata"} and "movement" in scopes
    for p in result["passages"]:
        assert (p["scope"] == "movement") == (finale in p["work_ids"])


def test_no_match_is_said_rather_than_left_empty(passage):
    result = search_passages("xylophone saxophone", model="ollama:not-embedded")
    assert result["passages"] == [] and result["note"]
