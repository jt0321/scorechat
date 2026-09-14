"""
renumber_measures.py
Re-derive which stored measures are bars, and which are merely part of the score.

A separate pass because it depends only on what is already in the database --
each measure's sounding duration and the meter in force -- and never on the
source .krn, so an existing corpus is repaired without re-ingesting it. Fresh
ingests get the same treatment inside `build_symbolic_layers`.
"""

from __future__ import annotations
import collections

# These CLIs read DATABASE_URL and the provider keys straight from the
# environment, so the .env a developer already has must be loaded before any
# db.store import builds an engine from it.
from dotenv import load_dotenv

load_dotenv()

import click

from analysis.numbering import BAR, NUMBERING_VERSION, assign_roles
from db.store import get_measure_shapes, list_works, store_measure_roles


@click.command()
@click.option("--work-id", type=int, default=None, help="Only this work (default: all).")
@click.option("--dry-run", is_flag=True, help="Report what would change; write nothing.")
def main(work_id: int | None, dry_run: bool):
    works = [w for w in list_works() if work_id is None or w["id"] == work_id]
    if not works:
        click.echo("No matching works. Has ingest_scores.py run?")
        return

    totals = collections.Counter()
    for work in works:
        shapes = get_measure_shapes(work["id"])
        if not shapes:
            continue
        numbering = assign_roles(shapes)
        non_bars = [row for row in numbering if row.role != BAR]
        for row in numbering:
            totals[row.role] += 1
        if non_bars:
            summary = ", ".join(
                f"{count}×{role}" for role, count in
                collections.Counter(row.role for row in non_bars).most_common()
            )
            label = f"{work['opus']}/{work['movement_number']}"
            click.echo(f"  {label:<22} {len(numbering):>4} measures, {len(non_bars)} not bars ({summary})")
        if not dry_run:
            store_measure_roles(work["id"], numbering)

    click.echo()
    for role, count in totals.most_common():
        click.echo(f"  {role:<12} {count}")
    verb = "would renumber" if dry_run else "renumbered"
    click.echo(f"\n{verb} {sum(totals.values())} measures across {len(works)} works "
               f"(numbering version {NUMBERING_VERSION})")


if __name__ == "__main__":
    main()
