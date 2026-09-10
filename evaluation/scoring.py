"""
evaluation/scoring.py
Scoring rules for the form-finding evaluation.

Kept apart from the database glue in `evaluate_form.py` so the rules can be
tested on hand-written observations. Everything here speaks in *printed* bar
numbers, the numbering a performer reads and the LLM must cite.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable

DEFAULT_TOLERANCE = 4


@dataclass
class Outcome:
    """One movement's result for one metric."""
    metric: str
    passed: bool
    expected: Any
    observed: Any
    detail: str = ""


def score_home_key(expected: str | None, observed: str | None) -> Outcome:
    """Key names compare case-sensitively: 'C' and 'c' are different keys, and
    a major/minor confusion is the failure mode key estimation actually has."""
    return Outcome("home_key", expected == observed, expected, observed)


def score_exposition_start(
    expected: int | None, observed: int | None, tolerance: int = DEFAULT_TOLERANCE
) -> Outcome:
    """Where the main theme begins, i.e. after any slow introduction."""
    passed = expected is not None and observed is not None and abs(expected - observed) <= tolerance
    detail = "" if expected is None or observed is None else f"off by {observed - expected:+d}"
    return Outcome("exposition_start", passed, expected, observed, detail)


def qualifying_returns(
    relations: Iterable[dict], exposition_start: int, last_bar: int,
    tolerance: int = DEFAULT_TOLERANCE,
) -> list[dict]:
    """Relations that could be a recapitulation of the opening material.

    Two filters, both musical rather than statistical. The *source* must be the
    opening material -- it starts within `tolerance` of `exposition_start` --
    because a movement is full of relations between inner passages and any of
    them would otherwise read as a recapitulation. And the *target* must lie in
    the movement's later half: an earlier return of the opening is a repeat
    inside the exposition, not a recapitulation. Every recapitulation in the
    ground truth falls between 51% and 67% of the way through its movement.
    """
    midpoint = last_bar / 2
    return [
        r for r in relations
        if r.get("source_start") is not None and r.get("target_start") is not None
        and abs(r["source_start"] - exposition_start) <= tolerance
        and r["target_start"] > r["source_start"]
        and r["target_start"] >= midpoint
    ]


def best_return_bar(candidates: list[dict]) -> dict | None:
    """The candidate the pipeline ranks first: most confident, and among
    equally confident ones the latest, since the recapitulation is the last
    full return rather than a restatement within the exposition."""
    if not candidates:
        return None
    return max(candidates, key=lambda r: (round(r.get("confidence", 0.0), 3), r["target_start"]))


def score_return_recall(
    expected: int | None, candidates: list[dict], tolerance: int = DEFAULT_TOLERANCE
) -> Outcome:
    """Is the true return anywhere among the proposals?

    Separated from `score_return` because finding the passage and ranking it
    first are different jobs -- the analysis pass does the first, and nothing
    yet does the second. Scoring only the top-ranked answer would hide a
    recapitulation the pipeline located but listed third.
    """
    hits = [] if expected is None else [
        r for r in candidates if abs(r["target_start"] - expected) <= tolerance
    ]
    if expected is None:
        return Outcome("return_recall", not candidates, None,
                       [r["target_start"] for r in candidates] or None,
                       "" if not candidates else "false positives")
    return Outcome("return_recall", bool(hits), expected,
                   [r["target_start"] for r in hits] or [r["target_start"] for r in candidates] or None,
                   "" if hits else f"{len(candidates)} proposals, none within {tolerance}")


def score_return(
    expected: int | None, match: dict | None, tolerance: int = DEFAULT_TOLERANCE
) -> Outcome:
    """Did the pipeline find the literal return, and only when there is one?

    `expected` of None is a movement whose opening does not literally return
    (Op. 57's first movement brings the theme back over a pedal the exposition
    never had). Finding nothing there is the correct answer, so it passes; a
    confident claim there is a false positive and fails. This is what keeps the
    metric from rewarding a matcher that simply lowers its threshold.
    """
    observed = None if match is None else match["target_start"]
    if expected is None:
        return Outcome("return", observed is None, None, observed,
                       "" if observed is None else "false positive")
    if observed is None:
        return Outcome("return", False, expected, None, "not found")
    return Outcome("return", abs(observed - expected) <= tolerance, expected, observed,
                   f"off by {observed - expected:+d}")


def score_citability(unnumbered: int, non_bars: int) -> Outcome:
    """Can every measure the pipeline may cite be named?

    A measure with no printed number is fine — an anacrusis, the upbeat after a
    repeat barline and an unbarred cadenza are all part of the score without
    being bars, and a performer does not name them either. What is not fine is
    such a measure having no bar to be *reported against*, because then a range
    opening on it cannot be cited at all. That is what this counts.
    """
    return Outcome("citable", unnumbered == 0, 0, unnumbered,
                   "" if unnumbered == 0 else
                   f"{unnumbered} of {non_bars} non-bars resolve to no printed bar")


def summarise(outcomes: Iterable[Outcome]) -> dict[str, tuple[int, int]]:
    """Passes and totals per metric, in the order the metrics first appear."""
    tally: dict[str, list[int]] = {}
    for outcome in outcomes:
        entry = tally.setdefault(outcome.metric, [0, 0])
        entry[0] += 1 if outcome.passed else 0
        entry[1] += 1
    return {metric: (passed, total) for metric, (passed, total) in tally.items()}
