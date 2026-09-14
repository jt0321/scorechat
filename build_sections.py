"""
build_sections.py
Record the section structure notated in each work's Humdrum source.

Reads `score_sources.raw_content`, so like build_relations.py it depends only
on the database and can be re-run without re-ingesting. Run it before
build_relations.py: notated sections are the most reliable structural spans a
score offers, being engraved rather than inferred.
"""

from __future__ import annotations
import click

# These CLIs read DATABASE_URL and the provider keys straight from the
# environment, so the .env a developer already has must be loaded before any
# db.store import builds an engine from it.
from dotenv import load_dotenv

load_dotenv()

from analysis.sections import parse_notated_sections, repeated_sections
from db.store import get_source_text, list_works, store_notated_sections


@click.command()
@click.option("--work-id", type=int, default=None, help="Only this work (default: all).")
@click.option("--dry-run", is_flag=True, help="Report what was found; write nothing.")
def main(work_id: int | None, dry_run: bool):
    works = [w for w in list_works() if work_id is None or w["id"] == work_id]
    if not works:
        click.echo("No matching works. Has ingest_scores.py run?")
        return

    total_sections = with_repeats = unmarked = 0
    for work in works:
        source = get_source_text(work["id"])
        if source is None:
            click.echo(f"  {work['title'][:56]:<56} no stored source")
            continue
        expansion, sections = parse_notated_sections(source)
        repeated = repeated_sections(sections)
        if not sections:
            unmarked += 1
        else:
            total_sections += len(sections)
            with_repeats += bool(repeated)

        summary = ",".join(expansion) if expansion else "(no repeat scheme)"
        click.echo(f"  {work['title'][:52]:<52} {summary[:38]:<38} "
                   f"{len(sections):>2} sections")
        for section in repeated:
            click.echo(f"       repeated: {section.label} mm.{section.measure_start}"
                       f"–{section.measure_end} "
                       f"({section.measure_end - section.measure_start + 1} bars) "
                       f"×{section.play_count}")
        if not dry_run and sections:
            store_notated_sections(work["id"], expansion, sections)

    verb = "would store" if dry_run else "stored"
    click.echo(f"\n{verb} {total_sections} notated sections across "
               f"{len(works) - unmarked} works; {with_repeats} have a repeated section, "
               f"{unmarked} notate none")


if __name__ == "__main__":
    main()
