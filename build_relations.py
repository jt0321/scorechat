"""
build_relations.py
Propose `span_relations` for ingested works by symbolic comparison.

A separate pass from ingestion because it depends only on what is already in
the database -- canonical measures and their analyses -- and never on the
source .krn. It can therefore be re-run and re-tuned without re-ingesting.
"""

from __future__ import annotations
import click

# These CLIs read DATABASE_URL and the provider keys straight from the
# environment, so the .env a developer already has must be loaded before any
# db.store import builds an engine from it.
from dotenv import load_dotenv

load_dotenv()

from analysis.span_relations import (
    MAX_MATCHES_PER_SPAN, MIN_RELATION_CONFIDENCE, MIN_RELATION_EVENTS,
    MIN_RELATION_MEASURES, REFERENCE_WINDOW_LENGTHS, RELATION_ANALYSIS_VERSION,
    build_span_relations,
)
from db.store import clear_work_span_relations, list_works, store_span_relations


@click.command()
@click.option("--work-id", type=int, default=None, help="Only this work (default: all).")
@click.option("--min-confidence", type=float, default=MIN_RELATION_CONFIDENCE, show_default=True)
@click.option("--min-measures", type=int, default=MIN_RELATION_MEASURES, show_default=True)
@click.option("--dry-run", is_flag=True, help="Report what would be proposed; write nothing.")
def main(work_id: int | None, min_confidence: float, min_measures: int, dry_run: bool):
    works = [w for w in list_works() if work_id is None or w["id"] == work_id]
    if not works:
        click.echo("No matching works. Has ingest_scores.py run?")
        return

    configuration = {
        "min_confidence": min_confidence,
        "min_measures": min_measures,
        "min_events": MIN_RELATION_EVENTS,
        "max_matches_per_span": MAX_MATCHES_PER_SPAN,
        "reference_window_lengths": list(REFERENCE_WINDOW_LENGTHS),
    }

    total = 0
    for work in works:
        label = f"{work['composer']} — {work['title']}"
        relations = build_span_relations(
            work["id"], min_confidence=min_confidence, min_measures=min_measures
        )
        total += len(relations)
        same_key = sum(1 for r in relations if r["evidence"]["returns_in_same_key"])
        click.echo(f"  {label[:62]:<62} {len(relations):>4} relations ({same_key} in key)")
        if dry_run:
            continue
        # Previous runs are dropped rather than added to: relations are
        # proposals derived from current thresholds, not an accumulating log.
        clear_work_span_relations(work["id"])
        if relations:
            store_span_relations(
                work["id"], relations, RELATION_ANALYSIS_VERSION, configuration
            )

    verb = "would propose" if dry_run else "stored"
    click.echo(f"\n{verb} {total} relations across {len(works)} works (status: proposed)")


if __name__ == "__main__":
    main()
