"""Scoring rules for the form-finding evaluation.

These test the harness, not the corpus: the corpus result is what
`evaluate_form.py` reports, and it is expected to change as the pipeline
improves. What must not change silently is what counts as a pass.
"""

import json
from pathlib import Path

from evaluation.scoring import (
    best_return_bar, qualifying_returns, score_citability, score_exposition_start,
    score_home_key, score_return, score_return_recall, summarise,
)


def relation(source_start, target_start, confidence=0.9, **extra):
    return {"source_start": source_start, "target_start": target_start,
            "confidence": confidence, **extra}


# --- ground truth file ------------------------------------------------------

def test_ground_truth_is_complete_and_well_formed():
    truth = json.loads((Path(__file__).parent.parent / "evaluation" / "ground_truth.json").read_text())
    assert truth["tolerance_measures"] > 0
    for entry in truth["movements"]:
        for field in ("opus", "movement", "home_key", "exposition_start",
                      "recapitulation", "literal_return", "verified", "notes"):
            assert field in entry, f"{entry.get('opus')} is missing {field}"
        assert entry["notes"], "every entry states how its answer was established"
        # A literal return must come after the exposition it returns to.
        if entry["literal_return"] is not None:
            assert entry["literal_return"] > entry["exposition_start"]


def test_every_recorded_literal_return_was_actually_verified():
    """`verified` is the match confidence measured between the exposition and
    the claimed return. A recorded return that did not score as one would mean
    the ground truth was remembered rather than checked."""
    truth = json.loads((Path(__file__).parent.parent / "evaluation" / "ground_truth.json").read_text())
    for entry in truth["movements"]:
        if entry["literal_return"] is not None:
            assert entry["verified"] >= 0.9, entry["opus"]
        else:
            assert entry["verified"] < 0.9, entry["opus"]


# --- keys and bars ----------------------------------------------------------

def test_home_key_distinguishes_a_key_from_its_parallel():
    assert score_home_key("c minor", "c minor").passed
    assert not score_home_key("c minor", "C major").passed


def test_exposition_start_allows_slack_but_not_a_different_section():
    assert score_exposition_start(11, 10, tolerance=4).passed
    assert not score_exposition_start(11, 195, tolerance=4).passed


# --- return candidates ------------------------------------------------------

def test_a_return_must_come_from_the_opening_material():
    """Relations between inner passages are not recapitulation candidates
    however confident they are."""
    relations = [relation(60, 200, 1.0), relation(1, 200, 0.8)]
    candidates = qualifying_returns(relations, exposition_start=1, last_bar=300)
    assert [r["source_start"] for r in candidates] == [1]


def test_a_return_in_the_first_half_is_a_repeat_not_a_recapitulation():
    relations = [relation(1, 40, 1.0), relation(1, 200, 0.8)]
    candidates = qualifying_returns(relations, exposition_start=1, last_bar=300)
    assert [r["target_start"] for r in candidates] == [200]


def test_the_top_candidate_is_the_most_confident_then_the_latest():
    candidates = [relation(1, 200, 0.9), relation(1, 260, 0.9), relation(1, 240, 0.8)]
    assert best_return_bar(candidates)["target_start"] == 260
    assert best_return_bar([]) is None


def test_recall_passes_when_the_return_is_proposed_but_not_ranked_first():
    """Locating the passage and ranking it first are different jobs; scoring
    only the top answer would hide a recapitulation listed third."""
    candidates = [relation(1, 301, 0.95), relation(1, 197, 0.8)]
    assert score_return_recall(197, candidates, tolerance=4).passed
    assert not score_return(197, best_return_bar(candidates), tolerance=4).passed


def test_finding_nothing_is_correct_when_nothing_returns():
    """Op. 57's first movement brings its theme back over a pedal the
    exposition never had. A confident claim there is a false positive, which is
    what stops the metric rewarding a lowered threshold."""
    assert score_return(None, None).passed
    assert score_return_recall(None, []).passed
    assert not score_return(None, relation(1, 200)).passed
    assert not score_return_recall(None, [relation(1, 200)]).passed


# --- citability -------------------------------------------------------------

def test_a_non_bar_is_citable_when_it_resolves_to_a_bar():
    """Having no number of its own is normal for an anacrusis or an upbeat.
    Having no bar to be reported against is what makes a range uncitable."""
    assert score_citability(unnumbered=0, non_bars=3).passed
    assert not score_citability(unnumbered=1, non_bars=3).passed


def test_summary_counts_passes_per_metric():
    outcomes = [score_home_key("c minor", "c minor"), score_home_key("C major", "c minor")]
    assert summarise(outcomes) == {"home_key": (1, 2)}
