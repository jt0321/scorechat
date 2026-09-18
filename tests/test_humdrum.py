"""
tests/test_humdrum.py
Reading the Humdrum source directly.

The spine tracker is what everything here rests on. Humdrum spines split and
rejoin — `*^` turns one spine into two and shifts every column to its right,
a run of `*v` merges them back — and all 103 movements in this corpus do it.
Reading `**dynam` by column index is therefore wrong everywhere after the
first divided staff, which is how the stored dynamics came to be full of kern
data. These pin the tracking rules.
"""

import pytest

from analysis.humdrum import (
    _apply, bar_duration, declared_key, iter_tokens, meter_changes, spine_layout,
)
from analysis.corpus import KERN_DIR


def krn(*lines):
    return "\n".join(lines) + "\n"


# --- spine bookkeeping ------------------------------------------------------

def test_a_split_replaces_one_spine_with_two_of_its_type():
    assert _apply(["**kern", "**dynam"], ["*^", "*"]) == ["**kern", "**kern", "**dynam"]


def test_a_run_of_joins_merges_exactly_that_many_spines():
    """Two `*v` tokens merge two spines into one; the length of the run is the
    count, which is why this cannot be a per-token mapping."""
    assert _apply(["**kern", "**kern", "**dynam"], ["*v", "*v", "*"]) == ["**kern", "**dynam"]


def test_two_separate_joins_on_one_line_stay_separate():
    types = ["**kern", "**kern", "**kern", "**kern"]
    assert _apply(types, ["*v", "*v", "*v", "*v"]) == ["**kern"]
    assert _apply(types, ["*v", "*v", "*", "*"]) == ["**kern", "**kern", "**kern"]


def test_a_terminated_spine_contributes_nothing():
    assert _apply(["**kern", "**dynam"], ["*", "*-"]) == ["**kern"]


def test_a_malformed_line_leaves_the_layout_alone():
    """Better to keep reading with the layout we have than to corrupt it."""
    types = ["**kern", "**dynam"]
    assert _apply(types, ["*^"]) == types


# --- reading through the layout ---------------------------------------------

def test_dynam_is_followed_across_a_kern_split():
    """The whole point: after `*^` the dynam spine is column 3, not column 2,
    and a column-index read would return note tokens as dynamics."""
    source = krn(
        "**kern\t**dynam",
        "=1\t=1",
        "4c\tp",
        "*^\t*",
        "4d\t4e\t<",
        "*v\t*v\t*",
        "4f\tff",
    )
    assert [t.text for t in iter_tokens(source, "**dynam")] == ["p", "<", "ff"]
    assert [t.text for t in iter_tokens(source, "**kern")] == ["4c", "4d", "4e", "4f"]


def test_tokens_carry_the_bar_they_sound_in():
    source = krn("**kern\t**dynam", "=1\t=1", "4c\tp", "=2\t=2", "4d\tf")
    assert [(t.measure, t.text) for t in iter_tokens(source, "**dynam")] == [(1, "p"), (2, "f")]


def test_null_tokens_are_not_events():
    """"." means the previous token is still sounding, not a new dynamic."""
    source = krn("**kern\t**dynam", "=1\t=1", "4c\tp", "4d\t.", "4e\tf")
    assert [t.text for t in iter_tokens(source, "**dynam")] == ["p", "f"]


def test_layout_and_token_counts_agree_on_every_line_of_a_real_score(): 
    """The invariant that makes the tracker trustworthy; it holds across
    175,279 lines of the corpus."""
    from pathlib import Path
    path = KERN_DIR / "sonata08-1.krn"
    if not path.exists():
        pytest.skip("corpus not downloaded")
    for _, line, types, tokens in spine_layout(path.read_text(encoding="utf-8")):
        if line.startswith("!") or line.startswith("**") or not line.strip():
            continue
        assert len(types) == len(tokens), line[:60]


# --- the declared key -------------------------------------------------------

def test_a_lower_case_tonic_is_minor():
    assert declared_key(krn("**kern\t**dynam", "*f:\t*", "=1\t=1")) == "f minor"


def test_an_upper_case_tonic_is_major():
    assert declared_key(krn("**kern", "*E-:", "=1")) == "E- major"


def test_the_score_keeps_its_own_spelling():
    """A score declaring D-flat means the same pitch class our PITCH_NAMES
    calls C#, and the page's spelling is the better one to report."""
    assert declared_key(krn("**kern", "*D-:", "=1")) == "D- major"


def test_a_key_change_later_does_not_replace_the_declaration():
    source = krn("**kern", "*c:", "=1", "4c", "*E-:", "=2", "4e-")
    assert declared_key(source) == "c minor"


def test_a_score_declaring_nothing_returns_nothing():
    assert declared_key(krn("**kern", "=1", "4c")) is None


def test_every_movement_in_the_corpus_declares_its_key():
    """Which is what makes the estimator unnecessary for the home key."""
    from pathlib import Path
    files = sorted(KERN_DIR.glob("sonata*.krn"))
    if not files:
        pytest.skip("corpus not downloaded")
    missing = [f.name for f in files if declared_key(f.read_text(encoding="utf-8")) is None]
    assert missing == []


# --- meter ------------------------------------------------------------------

def test_a_meter_before_the_first_barline_belongs_to_the_opening_measure():
    source = krn("**kern", "*M9/16", "=1", "4c", "=2", "4d")
    assert meter_changes(source)[1] == "9/16"


def test_a_meter_after_a_barline_applies_from_that_bar():
    source = krn("**kern", "*M9/16", "=1", "4c", "=36", "*M6/16", "4d")
    assert meter_changes(source) == {0: "9/16", 1: "9/16", 36: "6/16"}


def test_every_change_is_kept_not_only_the_first():
    """Op. 111's Arietta is written 9/16, 6/16, 12/32, 9/16 as the variations
    subdivide the beat. music21 keeps only the opening 9/16, which measures
    every later bar against a bar half again too long."""
    from pathlib import Path
    path = KERN_DIR / "sonata32-2.krn"
    if not path.exists():
        pytest.skip("corpus not downloaded")
    assert list(meter_changes(path.read_text(encoding="utf-8")).values()) == [
        "9/16", "9/16", "6/16", "12/32", "9/16",
    ]


def test_bar_duration_is_in_quarter_notes():
    assert bar_duration("4/4") == 4.0
    assert bar_duration("9/16") == 2.25
    assert bar_duration("6/16") == 1.5
    assert bar_duration("12/32") == 1.5      # the same length, subdivided finer
    assert bar_duration(None) is None
    assert bar_duration("nonsense") is None
