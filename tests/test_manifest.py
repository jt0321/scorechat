"""
tests/test_manifest.py
The commentary manifest, and the refusal to use any file it does not pin.

No network. The manifest is checked for the facts every later stage leans on --
that each book's sections are in reading order, and that the sonatas a book is
said to cover are declared exactly once -- and the fetcher is driven by a stub
download so a checksum mismatch can be shown to stop the file cold.
"""

from pathlib import Path

import pytest

from commentary.fetch import fetch_file, sha256_of
from commentary.manifest import ManifestError, get_source, load_manifest


def declared(source):
    return [n for section in source.sections if section.include for n in section.sonatas]


def test_elterlein_declares_every_sonata_exactly_once():
    sonatas = declared(get_source("elterlein-1879"))
    assert sorted(sonatas) == list(range(1, 33))


def test_marx_declares_twenty_sonatas_without_repeats():
    sonatas = declared(get_source("marx-1895"))
    assert len(sonatas) == len(set(sonatas)) == 20


def test_a_section_outside_the_corpus_is_excluded_not_anchored():
    """Elterlein discusses Op. 6, the four-hand sonata. If its pages were
    included they would sit between Op. 2 No. 3 and Op. 7 and inherit nothing --
    but the danger is the reverse, text about a work we do not have being
    anchored to one we do. Excluding the section is what prevents it."""
    op6 = next(s for s in get_source("elterlein-1879").sections if s.heading.startswith("OP. 6"))
    assert op6.include is False and op6.sonatas == ()


def test_every_file_is_pinned():
    for source in load_manifest():
        assert all(len(f.sha256) == 64 for f in source.files), source.key


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "manifest.toml"
    path.write_text(body)
    return path


SOURCE = '''
[[source]]
key = "x"
title = "T"
author = "A"
year = 1900
format = "archive-hocr"
archive_id = "id"
rights = "r"
[[source.file]]
name = "f"
sha256 = "{sha}"
'''


def section(leaf, sonatas):
    return f'[[source.section]]\nleaf = {leaf}\nheading = "h{leaf}"\nsonatas = {sonatas}\n'


def test_sections_out_of_reading_order_are_refused(tmp_path):
    """A section ends where the next begins, so an out-of-order entry would
    hand one sonata's pages to another with no error at all."""
    body = SOURCE.format(sha="0" * 64) + section(40, [2]) + section(30, [1])
    with pytest.raises(ManifestError, match="reading order"):
        load_manifest(write(tmp_path, body))


def test_a_sonata_outside_the_corpus_is_refused(tmp_path):
    body = SOURCE.format(sha="0" * 64) + section(10, [33])
    with pytest.raises(ManifestError, match="outside the corpus"):
        load_manifest(write(tmp_path, body))


def test_an_unpinned_file_is_refused(tmp_path):
    body = SOURCE.format(sha="not-a-checksum") + section(10, [1])
    with pytest.raises(ManifestError, match="sha256"):
        load_manifest(write(tmp_path, body))


def fake_download(content: bytes):
    def download(_url, dest: Path):
        dest.write_bytes(content)
    return download


def test_a_file_that_does_not_match_its_pin_is_rejected_not_kept(tmp_path):
    dest = tmp_path / "book.txt"
    wanted = tmp_path / "wanted"
    wanted.write_bytes(b"the pinned edition")
    pin = sha256_of(wanted)

    result = fetch_file("url", dest, pin, download=fake_download(b"a re-OCR'd edition"))

    assert result.status == "rejected" and not result.ok
    assert not dest.exists()
    assert (tmp_path / "book.txt.rejected").read_bytes() == b"a re-OCR'd edition"


def test_a_matching_file_is_kept_and_then_served_from_cache(tmp_path):
    dest = tmp_path / "book.txt"
    wanted = tmp_path / "wanted"
    wanted.write_bytes(b"the pinned edition")
    pin = sha256_of(wanted)

    assert fetch_file("url", dest, pin, download=fake_download(b"the pinned edition")).status == "downloaded"

    def must_not_download(_url, _dest):
        raise AssertionError("a matching cached file was downloaded again")
    assert fetch_file("url", dest, pin, download=must_not_download).status == "cached"


def test_a_cached_file_that_no_longer_matches_is_fetched_again(tmp_path):
    dest = tmp_path / "book.txt"
    dest.write_bytes(b"edited by hand")
    wanted = tmp_path / "wanted"
    wanted.write_bytes(b"the pinned edition")

    result = fetch_file("url", dest, sha256_of(wanted), download=fake_download(b"the pinned edition"))
    assert result.status == "downloaded" and dest.read_bytes() == b"the pinned edition"
