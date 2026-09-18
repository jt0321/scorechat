"""
embed_commentary.py
Embed the commentary passages with a chosen model, for vector search.

Only prose is embedded -- never score content. The model is a property of the
corpus: a query can only be compared with passages embedded by the same model,
so embeddings are stored per model and several can coexist. Running this again
embeds only what is missing (new passages after a re-ingest, or a new model).

Usage:
    python embed_commentary.py --model gemini                    # provider default
    python embed_commentary.py --model cloudflare:@cf/baai/bge-base-en-v1.5
    python embed_commentary.py --list                            # what the corpus has
"""

import time

import click
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text

from commentary.search import embedded_models
from db.session import session_scope
from pipeline.providers import embedding_model_id, embedding_provider_ready, get_embeddings

BATCH = 64
RETRIES = 4


@click.command()
@click.option("--model", "spec", help='"provider" or "provider:model", e.g. gemini')
@click.option("--list", "list_models", is_flag=True, help="Show which models the corpus is embedded with")
def main(spec, list_models):
    if list_models or not spec:
        models = embedded_models()
        with session_scope() as session:
            total = session.execute(text("SELECT count(*) FROM text_passages")).scalar_one()
        click.echo(f"{total} passages." + ("" if models else " Not embedded with any model."))
        for m in models:
            click.echo(f"  {m['model']:55} {m['dimensions']:5} dims  {m['passages']}/{total}")
        return

    model = embedding_model_id(spec)
    if not embedding_provider_ready(model):
        raise click.ClickException(f"No API key for {model.split(':')[0]}; see .env.example.")
    with session_scope() as session:
        todo = session.execute(text("""
            SELECT p.id, p.content FROM text_passages p
            WHERE NOT EXISTS (SELECT 1 FROM passage_embeddings e
                              WHERE e.passage_id = p.id AND e.model = :model)
            ORDER BY p.id"""), {"model": model}).all()
    click.echo(f"{model}: {len(todo)} passages to embed")
    embedder = get_embeddings(model)

    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        vectors = _with_retries(lambda: embedder.embed_documents([row.content for row in batch]))
        with session_scope() as session:
            for row, vector in zip(batch, vectors):
                session.execute(text("""
                    INSERT INTO passage_embeddings (passage_id, model, dimensions, embedding)
                    VALUES (:id, :model, :dims, CAST(:vector AS vector))
                    ON CONFLICT (passage_id, model) DO NOTHING"""),
                    {"id": row.id, "model": model, "dims": len(vector),
                     "vector": "[" + ",".join(map(str, vector)) + "]"})
            session.commit()
        click.echo(f"  {min(start + BATCH, len(todo))}/{len(todo)}")


def _with_retries(call):
    """Free tiers rate-limit; wait and try again rather than lose the batch."""
    for attempt in range(RETRIES):
        try:
            return call()
        except Exception as error:
            if attempt == RETRIES - 1:
                raise
            wait = 10 * 2 ** attempt
            click.echo(f"  {type(error).__name__}: {str(error)[:100]} — retrying in {wait}s")
            time.sleep(wait)


if __name__ == "__main__":
    main()
