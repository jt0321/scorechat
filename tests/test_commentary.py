"""
tests/test_commentary.py
Reading commentary into passages, and anchoring each to the right work.

Most of this runs on a small hand-built catalogue and synthetic text, so it
needs neither the database nor the fetched books. The last group reads the
real fetched texts and skips without them: those tests pin the property the
whole layer exists for -- that a passage is filed under the sonata it is
actually about -- against the books as they really are, OCR errors and all.
"""

import pytest

from commentary.anchoring import (
    AnchorState, Catalogue, Movement, Sonata, anchor, find_cues, find_mentions, fold, tempo_phrase,
)
from commentary.manifest import load_manifest
from commentary.texts import Mark, Paragraph, clean, to_passages, _is_running_head


def sonata(number, opus, sub, key, nickname=None, tempos=()):
    return Sonata(number, opus, sub, f"Op. {opus}" + (f" No. {sub}" if sub else ""), key, nickname,
                  tuple(Movement(number * 10 + i, i, t) for i, t in enumerate(tempos, start=1)))


CATALOGUE = Catalogue([
    sonata(1, 2, 1, "F minor", tempos=("Allegro", "Adagio", "MINUETTO Allegretto", "Prestissimo")),
    sonata(5, 10, 1, "C minor"),
    sonata(6, 10, 2, "F major"),
    sonata(7, 10, 3, "D major"),
    sonata(13, 27, 1, "E-flat major", "Quasi una fantasia"),
    sonata(14, 27, 2, "C-sharp minor", "Moonlight", ("Adagio sostenuto", "Allegretto", "Presto agitato")),
    sonata(16, 31, 1, "G major", tempos=("Allegro vivace", "Adagio grazioso", "Rondo: Allegretto")),
    sonata(22, 54, None, "F major"),
    sonata(23, 57, None, "F minor", "Appassionata"),
    sonata(28, 101, None, "A major"),
    sonata(31, 110, None, "A-flat major"),
    sonata(32, 111, None, "C minor"),
])


def names(text):
    return [sorted(m.candidates) for m in find_mentions(text, CATALOGUE)]


# --- what a passage names ------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("OP. loi, A MAJOR", [[28]]),      # the OCR's "loi" is 101
    ("OP. no, A FLAT", [[31]]),        # and "no" is 110
    ("OP. Ill, C MINOR", [[32]]),      # and "Ill" is 111
    ("Op. 27, No. 2", [[14]]),
])
def test_opus_numbers_are_read_through_ocr_errors(text, expected):
    assert names(text) == expected


def test_an_opus_without_its_number_names_the_whole_opus():
    assert names("Op. 10") == [[5, 6, 7]]


def test_a_key_written_beside_an_opus_narrows_it_to_one_sonata():
    """Marx's heading "F-Major Sonata, Op. 10" omits "No. 2". Neither half
    identifies the sonata -- two sonatas are in F major, three are Op. 10 --
    but together they do."""
    opus = [m for m in find_mentions("F-Major Sonata, Op. 10", CATALOGUE) if m.kind == "opus"]
    assert [sorted(m.candidates) for m in opus] == [[6]]


def test_a_list_of_opus_numbers_is_not_read_as_an_opus_and_its_number():
    """Elterlein writes "Op. 2, 7, 10, 13": that is four opus numbers, not
    Op. 2 No. 7, which does not exist."""
    assert names("Op. 2, 7, 10, 13")[0] == [1]


def test_a_bare_key_names_no_sonata():
    """A key is an identifier only when it is a sonata's name ("the F minor
    Sonata"); "modulates to F minor" is a claim about harmony, not a mention."""
    assert names("it modulates to F minor") == []
    assert names("the F minor Sonata") == [[1, 23]]


def test_the_article_a_is_not_the_key_of_a():
    assert names("a major Sonata of the middle period") == []


def test_a_nickname_counts_only_when_written_as_a_name():
    """Op. 2 No. 2's Largo is marked "appassionato" in Marx's text. The lower
    case word is a marking; the capitalised one is the sonata."""
    assert names('The Largo following is marked "appassionata," not ...') == []
    assert names("unlike the Appassionata") == [[23]]


def test_folding_keeps_every_character_in_place():
    """Matches in folded text are quoted from the original by offset."""
    for text in ("Pathétique", "Countess ^idie — Guicciardi", "à Thérèse"):
        assert len(fold(text)) == len(text)


# --- which movement ---------------------------------------------------------------

MOONLIGHT = CATALOGUE.sonatas[14]


def cue_movements(text, which=MOONLIGHT):
    return [c.movement.number for c in find_cues(text, which)]


def test_a_tempo_marking_picks_out_its_movement():
    assert cue_movements("then the Presto agitato bursts in") == [3]


def test_commentators_short_forms_count_when_unambiguous():
    """Elterlein writes "the adagio" for the Adagio sostenuto."""
    assert cue_movements("the spell of the adagio") == [1]


def test_ordinals_and_the_finale():
    assert cue_movements("the second movement") == [2]
    assert cue_movements("in the finale") == [3]


def test_a_marking_two_movements_open_with_picks_out_neither():
    """Op. 31 No. 1 has an Allegro vivace and a Rondo: Allegretto; "allegro"
    alone does not say which, and "adagio" is only the Adagio grazioso."""
    op31 = CATALOGUE.sonatas[16]
    assert cue_movements("the adagio", op31) == [2]
    assert tempo_phrase("Rondo: Allegretto") == "allegretto"
    assert tempo_phrase("SCHERZO Allegretto") == "allegretto"
    assert tempo_phrase("Allegro. [quarter] = 138") == "allegro"


# --- the anchor ------------------------------------------------------------------

def test_declared_and_named_agree_is_confirmed():
    result = anchor("This, the C sharp minor Sonata, Op. 27, No. 2, ...", (14,), CATALOGUE, AnchorState())
    assert result.status == "confirmed" and result.sonata == 14


def test_a_passage_naming_nothing_inherits_the_declaration():
    result = anchor("The musical colouring is bewitching.", (14,), CATALOGUE, AnchorState())
    assert result.status == "declared" and result.sonata == 14


def test_a_passage_naming_only_another_sonata_is_flagged_not_moved():
    """Usually a comparison ("like the Appassionata ..."), so the passage stays
    with the sonata its section discusses -- but a reviewer should look."""
    result = anchor("Its finale recalls the Appassionata.", (14,), CATALOGUE, AnchorState())
    assert result.status == "conflict" and result.sonata == 14
    assert result.evidence["names_instead"] == [23]


def test_undeclared_text_is_anchored_only_by_what_it_names():
    assert anchor("Beethoven's Op. 111 closes the series.", (), CATALOGUE, AnchorState()).sonata == 32
    both = anchor("Compare Op. 57 with Op. 111.", (), CATALOGUE, AnchorState())
    assert both.status == "unanchored" and both.sonata is None


def test_a_movement_carries_forward_and_a_passage_across_a_change_holds_both():
    state = AnchorState()
    first = anchor("The adagio is a lament.", (14,), CATALOGUE, state)
    carried = anchor("Its colouring is dim and strange.", (14,), CATALOGUE, state)
    across = anchor("Then comes the second movement, an allegretto.", (14,), CATALOGUE, state)
    works = {m.number: m.work_id for m in MOONLIGHT.movements}
    assert first.work_ids == (works[1],)
    assert carried.work_ids == (works[1],) and carried.evidence["carried_from_previous"] == 1
    # "Then comes ..." opens with the new movement, so it does not also hold the old one.
    assert across.work_ids == (works[2],)


def test_a_passage_that_moves_on_mid_way_is_recorded_against_both():
    state = AnchorState()
    anchor("The adagio is a lament.", (14,), CATALOGUE, state)
    result = anchor("It fades away. The last movement, Presto agitato, storms.", (14,), CATALOGUE, state)
    works = {m.number: m.work_id for m in MOONLIGHT.movements}
    assert result.work_ids == (works[1], works[3])


# --- passages ---------------------------------------------------------------------

def test_line_end_hyphenation_and_ocr_spacing_are_undone():
    assert clean("the  instru- mental   music") == "the instrumental music"


def test_running_heads_are_recognised_and_prose_is_not():
    assert _is_running_head("54 BEETHOVEN'S SONATAS EXPLAINED.")
    assert _is_running_head("THE INDIVIDUAL WORKS. 97")
    assert not _is_running_head("This Sonata is undoubtedly one of the greatest")


def words(n):
    """n words of prose, a sentence every twelve: "W0 W1 ... W11. W12 ..." """
    return " ".join(f"W{i}." if i % 12 == 11 else f"W{i}" for i in range(n))


def test_a_short_paragraph_joins_the_discussion_it_introduces():
    heading = Paragraph("Presto agitato.", 0, (Mark(0, leaf=5, page=61),))
    body = Paragraph(words(60), 0, (Mark(0, leaf=5, page=61),))
    passages = to_passages([heading, body])
    assert len(passages) == 1 and passages[0].text.startswith("Presto agitato.")


def test_a_passage_never_crosses_a_section():
    passages = to_passages([Paragraph(words(60), 0, (Mark(0, page=1),)),
                            Paragraph(words(60), 1, (Mark(0, page=2),))])
    assert [p.section for p in passages] == [0, 1]


def test_a_split_passage_cites_the_page_its_first_word_is_on():
    """A paragraph running from p. 59 to p. 61 is cut into several passages;
    the later ones must not all claim p. 59."""
    text = words(500)
    turn = text.index("W300")
    paragraph = Paragraph(text, 0, (Mark(0, leaf=65, page=59), Mark(turn, leaf=66, page=60)))
    pages = [p.page for p in to_passages([paragraph])]
    assert pages[0] == 59 and pages[-1] == 60 and len(pages) > 2


# --- the real books ---------------------------------------------------------------

def _fetched(source):
    return all((source.directory() / f.name).exists() for f in source.files)


@pytest.fixture(scope="module")
def rows_by_source():
    from commentary.store import load_catalogue
    from ingest_commentary import build_rows
    sources = [s for s in load_manifest() if _fetched(s)]
    if not sources:
        pytest.skip("commentary sources not fetched")
    catalogue = load_catalogue()
    if not catalogue.sonatas:
        pytest.skip("corpus not ingested")
    return {s.key: build_rows(s, catalogue) for s in sources}


def test_every_declared_section_opens_by_naming_its_own_sonata(rows_by_source):
    """The right-work check, end to end: the manifest declares which sonata a
    section discusses, and the first passage of every declared section
    independently names that sonata. A shifted section table, a heading found
    on the wrong leaf, or an opus the OCR reader cannot parse all break this."""
    for key, rows in rows_by_source.items():
        seen = set()
        for row in rows:
            if row["declared"] and row["heading"] not in seen:
                seen.add(row["heading"])
                assert row["status"] == "confirmed", (key, row["heading"], row["content"][:80])


def test_elterlein_walks_the_moonlight_movement_by_movement(rows_by_source):
    from commentary.store import load_catalogue
    rows = [r for r in rows_by_source.get("elterlein-1879", []) if r["sonata"] == 14]
    if not rows:
        pytest.skip("Elterlein not fetched")
    number = {m.work_id: m.number for m in load_catalogue().sonatas[14].movements}
    reached = [max(number[w] for w in r["work_ids"]) for r in rows if r["work_ids"]]
    assert reached == sorted(reached) and reached[-1] == 3
