"""
ingest_commentary.py
Read the fetched commentary texts into passages anchored to the corpus.

Each passage is placed at the sonata the manifest declares for its section and
checked against what the passage itself names; movements come from the cues
the commentators use (tempo markings, "the Scherzo", "the finale"). Nothing is
embedded here -- see embed_commentary.py -- and nothing reaches the symbolic
layers: commentary is attributed opinion, stored beside the score.

Usage:
    python fetch_sources.py && python ingest_commentary.py
    python ingest_commentary.py --source marx-1895 --dry-run
    python ingest_commentary.py --review          # list every conflict to check
"""

from collections import Counter

import click
from dotenv import load_dotenv

load_dotenv()

from commentary.anchoring import AnchorState, anchor
from commentary.manifest import load_manifest
from commentary.store import load_catalogue, replace_document
from commentary.texts import read_passages


def build_rows(source, catalogue) -> list[dict]:
    rows, state, section = [], AnchorState(), None
    for passage in read_passages(source):
        if passage.section != section:
            # Carry-forward never crosses a section: a new heading is a new subject.
            state, section = AnchorState(), passage.section
        declared = source.sections[passage.section].sonatas
        result = anchor(passage.text, declared, catalogue, state)
        rows.append({
            "ordinal": passage.ordinal, "heading": source.sections[passage.section].heading,
            "declared": list(declared), "sonata": result.sonata, "work_ids": list(result.work_ids),
            "status": result.status, "evidence": result.evidence, "leaf": passage.leaf,
            "page": passage.page, "line": passage.line, "content": passage.text,
        })
    return rows


@click.command()
@click.option("--source", "source_keys", multiple=True, help="Only ingest this source (repeatable)")
@click.option("--dry-run", is_flag=True, help="Anchor and report without writing")
@click.option("--review", is_flag=True, help="List every conflicting passage for review")
def main(source_keys, dry_run, review):
    sources = [s for s in load_manifest() if not source_keys or s.key in source_keys]
    catalogue = load_catalogue()
    for source in sources:
        missing = [f.name for f in source.files if not (source.directory() / f.name).exists()]
        if missing:
            click.echo(f"✗ {source.key}: not fetched ({', '.join(missing)}) — run fetch_sources.py")
            continue
        rows = build_rows(source, catalogue)
        statuses = Counter(r["status"] for r in rows)
        sonatas = {r["sonata"] for r in rows if r["sonata"]}
        with_movement = sum(1 for r in rows if r["work_ids"])
        click.echo(f"▶  {source.key}: {len(rows)} passages on {len(sonatas)} sonatas; "
                   f"{with_movement} placed at a movement")
        click.echo("   " + ", ".join(f"{k} {v}" for k, v in statuses.most_common()))

        if review:
            for row in rows:
                if row["status"] == "conflict":
                    where = f"p. {row['page']}" if row["page"] else f"line {row['line']}"
                    click.echo(f"   ? {where} under {row['heading']!r} also names sonatas "
                               f"{row['evidence'].get('names_instead')}: {row['content'][:110]}…")

        if not dry_run:
            _, dropped = replace_document(source, rows)
            click.echo(f"   ✓ stored" + (f" — {dropped} embeddings dropped with the old passages; "
                                         f"re-run embed_commentary.py" if dropped else ""))
        click.echo()


if __name__ == "__main__":
    main()
