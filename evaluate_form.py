"""
evaluate_form.py
Score the analysis pipeline against hand-checked ground truth.

The eval set exists because every other check in this repo asks whether a
function does what it was written to do. This one asks the different question
of whether the pipeline finds the musical facts a reader would ask about --
where the opening material returns, what key the movement is in, where the
main theme begins -- on movements whose answers are known independently.

It reads only the database, so it scores what the pipeline has actually stored
rather than re-deriving anything, and re-running it after a threshold change
says whether that change helped or merely moved the failures around.
"""

from __future__ import annotations
import json
from pathlib import Path

import click

from analysis.span_relations import detect_intro_end_index
from db.store import (
    get_global_key, get_span_relations, list_works, load_work_features,
)
from evaluation.scoring import (
    Outcome, best_return_bar, qualifying_returns, score_citability,
    score_exposition_start, score_home_key, score_return, score_return_recall,
    summarise,
)

GROUND_TRUTH = Path(__file__).parent / "evaluation" / "ground_truth.json"


def _resolve(works: list[dict], opus: str, movement: int) -> dict | None:
    """Ground truth names movements by opus and movement number; work ids are
    assigned at ingest and change whenever the corpus is rebuilt."""
    return next((w for w in works
                 if w["opus"] == opus and w["movement_number"] == movement), None)


def evaluate_movement(work: dict, entry: dict, tolerance: int) -> list[Outcome]:
    measures = load_work_features(work["id"])
    numbers = [m["measure_belongs_to"] for m in measures]
    indices = {m["measure_index"]: m["measure_belongs_to"] for m in measures}
    non_bars = [m for m in measures if m["measure_role"] != "bar"]
    unnumbered = [m for m in non_bars if m["measure_belongs_to"] is None]

    intro_end = detect_intro_end_index(work["id"])
    observed_start = indices.get(intro_end)

    relations = get_span_relations(work["id"])
    last_bar = max((n for n in numbers if n is not None), default=0)
    candidates = qualifying_returns(
        relations, entry["exposition_start"], last_bar, tolerance
    )

    return [
        score_home_key(entry["home_key"], get_global_key(work["id"], 0)),
        score_exposition_start(entry["exposition_start"], observed_start, tolerance),
        score_return_recall(entry["literal_return"], candidates, tolerance),
        score_return(entry["literal_return"], best_return_bar(candidates), tolerance),
        score_citability(len(unnumbered), len(non_bars)),
    ]


def _mark(outcome: Outcome) -> str:
    return "PASS" if outcome.passed else "FAIL"


@click.command()
@click.option("--tolerance", type=int, default=None,
              help="Bars of slack allowed on a located bar (default: from the ground truth file).")
@click.option("--verbose", is_flag=True, help="Show every metric, not only failures.")
def main(tolerance: int | None, verbose: bool):
    truth = json.loads(GROUND_TRUTH.read_text())
    tolerance = tolerance if tolerance is not None else truth["tolerance_measures"]
    works = list_works()

    all_outcomes: list[Outcome] = []
    missing: list[str] = []

    for entry in truth["movements"]:
        label = f"{entry['opus']} mvt {entry['movement']}"
        work = _resolve(works, entry["opus"], entry["movement"])
        if work is None:
            missing.append(label)
            continue

        outcomes = evaluate_movement(work, entry, tolerance)
        all_outcomes.extend(outcomes)

        failed = [o for o in outcomes if not o.passed]
        header = f"{label:<22} {len(outcomes) - len(failed)}/{len(outcomes)}"
        click.echo(header if not failed else click.style(header, fg="red"))
        for outcome in outcomes:
            if verbose or not outcome.passed:
                detail = f"  ({outcome.detail})" if outcome.detail else ""
                click.echo(f"     {_mark(outcome):<4} {outcome.metric:<17} "
                           f"expected {outcome.expected!r}, got {outcome.observed!r}{detail}")

    click.echo()
    for metric, (passed, total) in summarise(all_outcomes).items():
        click.echo(f"  {metric:<17} {passed}/{total}")
    total_passed = sum(1 for o in all_outcomes if o.passed)
    click.echo(f"\n  overall           {total_passed}/{len(all_outcomes)}"
               f"  (tolerance {tolerance} bars)")
    if missing:
        click.echo(f"\n  not ingested: {', '.join(missing)}")


if __name__ == "__main__":
    main()
