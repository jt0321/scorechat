"""
tests/test_measure_numbering.py
The printed-vs-internal measure numbering contract.

Printed (engraved) numbering is what a performer reads, what the user types,
and what the LLM cites: bar 1 is the first *complete* measure, and an
anacrusis is not counted. Verovio agrees -- it emits an unnumbered pickup --
but its `measureRange` selection counts ordinal positions from 1, in which the
pickup *is* position 1. These tests pin that one-measure offset, which is
silent when wrong: the viewer simply shows the neighbouring bars.
"""
import pytest
from pathlib import Path

from pipeline.mei_converter import measure_ordinals, pickup_ordinals
from analysis.corpus import KERN_DIR, MEI_DIR


ANACRUSIS_MEI = MEI_DIR / "sonata01-1.mei"   # Op. 2 No. 1/i, upbeat
PLAIN_MEI = MEI_DIR / "sonata08-2.mei"       # Op. 13/ii, no upbeat


def _read(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"{path} has not been generated")
    return path.read_text(encoding="utf-8")


def test_anacrusis_shifts_printed_numbers_by_one():
    ordinals = measure_ordinals(_read(ANACRUSIS_MEI))
    assert 0 not in ordinals  # the pickup has no number, so no entry of its own
    assert ordinals[1] == 2   # printed bar 1 is the *second* physical measure
    assert ordinals[5] == 6


def test_the_pickup_is_reached_through_the_bar_it_belongs_to():
    """An unnumbered measure cannot be addressed by number because it has
    none. It is reachable as the pickup of the bar it leads into, which is how
    a musician refers to it and how the database now stores it."""
    pickups = pickup_ordinals(_read(ANACRUSIS_MEI))
    assert pickups[1] == 1    # the anacrusis sits at ordinal 1, before bar 1
    assert 5 not in pickups   # bar 5 has no pickup of its own


def test_score_without_anacrusis_is_unshifted():
    ordinals = measure_ordinals(_read(PLAIN_MEI))
    assert 0 not in ordinals
    assert ordinals[1] == 1
    assert ordinals[5] == 5


def test_an_unnumbered_measure_gets_no_entry_of_its_own():
    """0 used to denote "unnumbered" here and in score_measures. It also read
    as a bar number and was printed as one, so a range opening on a pickup was
    cited as "mm. 0-247". Nothing is addressable as 0 any more."""
    mei = '<measure><x/></measure><measure n="1"></measure>'
    assert measure_ordinals(mei) == {1: 2}
    assert pickup_ordinals(mei) == {1: 1}


def test_a_pickup_run_is_reported_from_its_first_measure():
    mei = '<measure n="1"></measure><measure><x/></measure><measure><x/></measure><measure n="2"></measure>'
    assert pickup_ordinals(mei) == {2: 2}


def test_ordinals_ignore_unparsable_numbers_without_shifting_the_rest():
    mei = '<measure n="1"></measure><measure n="1a"></measure><measure n="2"></measure>'
    assert measure_ordinals(mei) == {1: 1, 2: 3}


def test_first_occurrence_wins_for_repeated_numbers():
    """A repeated bar number (common around repeats and editorial numbering)
    must not silently retarget an earlier citation to a later measure."""
    mei = '<measure n="1"></measure><measure n="2"></measure><measure n="2"></measure>'
    assert measure_ordinals(mei)[2] == 2


@pytest.mark.parametrize("printed,expected_source_line", [(1, 25), (5, 53), (8, 80)])
def test_rendered_excerpt_matches_the_printed_bar_in_the_source(printed, expected_source_line):
    """Verovio tags each SVG measure with the MEI xml:id, which encodes the
    Humdrum line it came from -- so this checks the rendered bar really is the
    one the source numbers, not its neighbour."""
    import re
    from pipeline.mei_converter import mei_to_svg
    if not ANACRUSIS_MEI.exists():
        pytest.skip("MEI has not been generated")
    svg = mei_to_svg(str(ANACRUSIS_MEI), printed, printed)
    assert f"measure-L{expected_source_line}" in svg
    # Bar 1 is preceded by the anacrusis, and a request for bar 1 renders it
    # too: that is where a performer starts playing, and it is what
    # get_measure_evidence returns for the same range.
    expected = 2 if printed == 1 else 1
    assert len(re.findall(r'id="measure-L\d+"', svg)) == expected


def test_excerpt_renders_only_the_requested_measures():
    """Regression: the old `select` *option* is unsupported in Verovio 6, so a
    request for four bars silently rendered the entire movement."""
    import re
    from pipeline.mei_converter import mei_to_svg
    if not ANACRUSIS_MEI.exists():
        pytest.skip("MEI has not been generated")
    svg = mei_to_svg(str(ANACRUSIS_MEI), 5, 8)
    assert len(re.findall(r'id="measure-L\d+"', svg)) == 4


def test_a_multi_rest_carries_every_bar_it_collapses():
    """Two bars of rest are engraved as one measure holding <multiRest num="2">.
    Op. 110/ii does this twice, so its MEI has no measure numbered 38 and a
    request for bar 38 silently rendered its neighbour."""
    mei = ('<measure n="37"><multiRest num="2" /></measure>'
           '<measure n="39"><note/></measure>')
    ordinals = measure_ordinals(mei)
    assert ordinals[37] == 1
    assert ordinals[38] == 1      # inside the same engraved measure
    assert ordinals[39] == 2


def test_every_bar_of_the_corpus_is_addressable_in_its_rendering():
    """The viewer and the chat must cite the same bar. Two bars in two
    movements are not addressable -- degenerate measures at a tempo change --
    and this pins that number so it cannot quietly grow."""
    import re
    from pathlib import Path
    from analysis.humdrum import spine_layout
    sources = sorted(KERN_DIR.glob("sonata*.krn"))
    if not sources:
        pytest.skip("corpus not downloaded")
    unaddressable = 0
    for source in sources:
        rendering = MEI_DIR / f"{source.stem}.mei"
        if not rendering.exists():
            continue
        bars = [int(m.group(1)) for _, line, _, toks in spine_layout(source.read_text(encoding="utf-8"))
                if line.startswith("=") and toks and (m := re.match(r"^=+(\d+)", toks[0]))]
        ordinals = measure_ordinals(rendering.read_text(encoding="utf-8"))
        unaddressable += sum(1 for b in bars if b not in ordinals)
    assert unaddressable <= 2, f"{unaddressable} bars cannot be rendered by number"
