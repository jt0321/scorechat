"""
build_harmony.py
Re-run the harmonic pass over stored scores.

`analysis/harmony.py` is a pure function of `score_measures.symbolic_data`, so
a harmonic pass is reproducible from the database alone. That seam existed from
the start but had no way to use it: the only route to a new harmonic analysis
was a full re-ingest through music21. This is the missing half -- re-key and
re-chord the corpus from what is already stored, in seconds rather than an hour.

It also reads the key each score *declares* (`*f:`) from the stored source and
anchors the trajectory to it, which no amount of re-estimating would fix.
"""

from __future__ import annotations
import click

from dotenv import load_dotenv

load_dotenv()

from analysis.harmony import HARMONY_ANALYSIS_VERSION, analyze_harmony
from analysis.humdrum import declared_key
from db.store import (
    get_source_text, get_stored_measures, list_works, store_harmony,
)


@click.command()
@click.option("--work-id", type=int, default=None, help="Only this work (default: all).")
@click.option("--dry-run", is_flag=True, help="Report what would change; write nothing.")
def main(work_id: int | None, dry_run: bool):
    works = [w for w in list_works() if work_id is None or w["id"] == work_id]
    if not works:
        click.echo("No matching works. Has ingest_scores.py run?")
        return

    changed_keys, total_measures, labelled, undeclared = 0, 0, 0, []
    for work in works:
        measures = get_stored_measures(work["id"])
        if not measures:
            continue
        source_key = declared_key(get_source_text(work["id"]) or "")
        if source_key is None:
            undeclared.append(f"{work['opus']}/{work['movement_number']}")

        previous = measures[0]["analysis_data"].get("global_key")
        trajectory, chords = analyze_harmony(
            measures, measures, declared_key=source_key
        )
        if source_key and source_key != previous:
            changed_keys += 1
            click.echo(f"  {work['opus']}/{work['movement_number']:<3} "
                       f"{previous!r} → {source_key!r}")

        by_index = {estimate.measure_index: estimate for estimate in trajectory}
        chords_by_index: dict[int, list[dict]] = {}
        for span in chords:
            chords_by_index.setdefault(span.measure_index, []).append({
                "figure": span.figure,
                "root_pitch_class": span.root,
                "quality": span.quality,
                "bass_pitch_class": span.bass,
                "beat_start": span.beat_start,
                "beat_end": span.beat_end,
                "confidence": round(span.confidence, 3),
                "non_chord_tones": span.non_chord_tones,
            })

        updates = {}
        for measure in measures:
            index = measure["measure_index"]
            estimate = by_index.get(index)
            updates[index] = {
                "harmony_version": HARMONY_ANALYSIS_VERSION,
                "global_key": source_key or (trajectory[0].key if trajectory else None),
                "local_key": estimate.key if estimate else None,
                "local_key_scope": "windowed_viterbi",
                "local_key_correlation": round(estimate.correlation, 3) if estimate else None,
                "local_key_in_signature": estimate.in_key_signature if estimate else None,
                "chords": chords_by_index.get(index, []),
            }
        total_measures += len(updates)
        labelled += sum(1 for span in chords if span.figure)
        if not dry_run:
            store_harmony(work["id"], updates)

    verb = "would update" if dry_run else "updated"
    click.echo(f"\n{verb} {total_measures} measures across {len(works)} works")
    click.echo(f"  global keys corrected: {changed_keys}")
    click.echo(f"  chord spans labelled:  {labelled}")
    if undeclared:
        click.echo(f"  no declared key: {', '.join(undeclared)}")


if __name__ == "__main__":
    main()
