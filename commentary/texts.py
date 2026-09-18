"""
commentary/texts.py
A fetched source, read into paragraphs that know where they came from.

Two layouts are supported. archive.org's hOCR bundle gives the OCR text with
a per-scan-page character index, so every paragraph carries the leaf a link
can open and the page number printed on it. A Gutenberg plain-text file has
no pages, only lines, so paragraphs carry the line they start on.

Reading is where the manifest gets checked against the text itself: a declared
heading that is not on its declared leaf, or a page_offset that the running
heads contradict, is an error here rather than a quietly misfiled passage later.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from commentary.manifest import SOURCES_DIR, ManifestError, Source

# Passage size, in words. A printed page of these books runs 250-300 words;
# a passage is shorter so that one quote card carries one point.
MIN_PASSAGE_WORDS = 40
MAX_PASSAGE_WORDS = 220

# How far the running heads may disagree with page_offset before the offset is
# declared wrong. OCR mangles some page numbers ("6o" for 60), so not all agree.
MIN_PAGE_AGREEMENT = 0.8

_TERMINAL = re.compile(r"[.!?:;\"'”’)\]]\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“(])")


@dataclass(frozen=True)
class Mark:
    """Where in the source the text from `offset` onward comes from."""
    offset: int
    leaf: int | None = None
    page: int | None = None
    line: int | None = None


@dataclass(frozen=True)
class Paragraph:
    text: str
    section: int              # index into source.sections
    marks: tuple[Mark, ...]   # one per page turn (or source line) inside the text

    def at(self, offset: int) -> Mark:
        """The source position of a character in `text`."""
        current = self.marks[0]
        for mark in self.marks:
            if mark.offset > offset:
                break
            current = mark
        return current


@dataclass(frozen=True)
class Passage:
    text: str
    section: int
    ordinal: int
    leaf: int | None = None
    page: int | None = None
    line: int | None = None

    @property
    def words(self) -> int:
        return len(self.text.split())


def read_paragraphs(source: Source, root: Path = SOURCES_DIR) -> list[Paragraph]:
    if source.format == "archive-hocr":
        return _hocr_paragraphs(source, source.directory(root))
    if source.format == "gutenberg-text":
        return _gutenberg_paragraphs(source, source.directory(root))
    raise ManifestError(f"{source.key}: no reader for format {source.format!r}")


def read_passages(source: Source, root: Path = SOURCES_DIR) -> list[Passage]:
    return to_passages(read_paragraphs(source, root))


# --- archive.org hOCR ---------------------------------------------------------

def _hocr_paragraphs(source: Source, directory: Path) -> list[Paragraph]:
    base = directory / source.archive_id
    text = gzip.open(f"{base}_hocr_searchtext.txt.gz", "rt", encoding="utf-8").read()
    index = json.load(gzip.open(f"{base}_hocr_pageindex.json.gz", "rt"))
    detected = {p["leafNum"]: p["pageNumber"]
                for p in json.load(open(f"{base}_page_numbers.json"))["pages"]}
    pages = printed_pages(source, text, index, detected)
    bounds = [(entry[0], entry[1]) for entry in index]

    starts = _section_starts(source, text, bounds)
    paragraphs: list[Paragraph] = []
    for number, (section, start) in enumerate(zip(source.sections, starts)):
        if not section.include:
            continue
        end = starts[number + 1] if number + 1 < len(starts) else len(text)
        paragraphs.extend(_hocr_section(text, bounds, pages, number, start, end))
    return paragraphs


def printed_pages(source: Source, text: str, index: list, detected: dict) -> dict[int, int]:
    """Printed page number for every leaf.

    archive.org's detection is shifted by `page_offset` and leaves some leaves
    blank; the blanks are filled from the dominant leaf-to-page gap. The
    running heads are the independent witness: if too few of them agree with
    the result, the declared offset is wrong and nothing is read.
    """
    known = {leaf: int(n) + source.page_offset for leaf, n in detected.items() if str(n).isdigit()}
    gap = Counter(leaf - page for leaf, page in known.items()).most_common(1)[0][0]
    pages = {leaf: known.get(leaf, leaf - gap) for leaf in range(len(index))}

    agree = checked = 0
    for leaf, (start, end, *_) in enumerate(index):
        head = text[start:end].split("\n", 1)[0]
        found = re.search(r"(?:^|\s)(\d{1,3})(?:\s|$)", head)
        if found and leaf in known:
            checked += 1
            agree += int(found.group(1)) == known[leaf]
    if checked and agree / checked < MIN_PAGE_AGREEMENT:
        raise ManifestError(f"{source.key}: page_offset {source.page_offset} agrees with only "
                            f"{agree} of {checked} running heads")
    return pages


def _heading_pattern(heading: str) -> re.Pattern:
    return re.compile(r"\s+".join(re.escape(token) for token in heading.split()))


def _section_starts(source: Source, text: str, bounds: list) -> list[int]:
    starts = []
    for section in source.sections:
        start, end = bounds[section.leaf]
        found = _heading_pattern(section.heading).search(text, start, end)
        if not found:
            raise ManifestError(f"{source.key}: heading {section.heading!r} is not on leaf "
                                f"{section.leaf} (page {section.page})")
        starts.append(found.start())
    if starts != sorted(starts):
        raise ManifestError(f"{source.key}: two sections on one leaf are listed out of order")
    return starts


def _is_running_head(line: str) -> bool:
    words = line.split()
    if not words or len(words) > 8:
        return False
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return True                      # a bare page number
    return sum(c.isupper() for c in letters) / len(letters) > 0.8


def _hocr_section(text, bounds, pages, number, start, end) -> list[Paragraph]:
    built: list[tuple[str, list[Mark]]] = []
    for leaf, (page_start, page_end) in enumerate(bounds):
        lo, hi = max(page_start, start), min(page_end, end)
        if lo >= hi:
            continue
        lines = [clean(line) for line in text[lo:hi].split("\n")]
        if lo == page_start:
            # Running heads sit at the top of a page, one or two lines of them.
            for _ in range(2):
                if lines and _is_running_head(lines[0]):
                    lines.pop(0)
        for position, line in enumerate(line for line in lines if line):
            here = Mark(0, leaf=leaf, page=pages.get(leaf))
            if position == 0 and built and not _TERMINAL.search(built[-1][0]):
                # A paragraph interrupted by the page turn continues here, and
                # the passage cut from this point on cites the new page.
                body, marks = built[-1]
                glue = re.search(r"\w-$", body) and line[:1].islower()
                body = body[:-1] if glue else body + " "
                marks.append(Mark(len(body), leaf=leaf, page=pages.get(leaf)))
                built[-1] = (body + line, marks)
            else:
                built.append((line, [here]))
    return [Paragraph(body, number, tuple(marks)) for body, marks in built]


# --- Project Gutenberg plain text ----------------------------------------------

def _gutenberg_paragraphs(source: Source, directory: Path) -> list[Paragraph]:
    lines = (directory / source.files[0].name).read_text(encoding=source.encoding).splitlines()
    for section in source.sections:
        if not 0 < section.line <= len(lines) or \
                not _heading_pattern(section.heading).match(lines[section.line - 1].strip()):
            raise ManifestError(f"{source.key}: heading {section.heading!r} is not at "
                                f"line {section.line}")
    paragraphs: list[Paragraph] = []
    for number, section in enumerate(source.sections):
        if not section.include:
            continue
        end = source.sections[number + 1].line - 1 if number + 1 < len(source.sections) else len(lines)
        block: list[str] = []
        block_line = section.line
        for line_number in range(section.line, end + 1):
            line = lines[line_number - 1].strip()
            if line:
                if not block:
                    block_line = line_number
                block.append(line)
                continue
            _flush(block, paragraphs, number, block_line)
            block = []
        _flush(block, paragraphs, number, block_line)
    return paragraphs


def _flush(block, paragraphs, section, line):
    if not block:
        return
    body, marks = "", []
    for number, text in enumerate(block):
        text = clean(text)
        if body:
            body += " "
        marks.append(Mark(len(body), line=line + number))
        body += text
    # Chapter headings ("CHAPTER VII", "LUDWIG VAN BEETHOVEN") are not commentary.
    if body and not _is_running_head(body):
        paragraphs.append(Paragraph(body, section, tuple(marks)))


# --- shared ---------------------------------------------------------------------

def clean(text: str) -> str:
    """Undo what the page layout did to one line of prose: words split across
    a line end ("instru- mental"), and the OCR's doubled spaces. Applied per
    line, before lines are joined, so the offsets that locate a page turn stay
    exact."""
    text = re.sub(r"(\w)- (\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def to_passages(paragraphs: list[Paragraph]) -> list[Passage]:
    """Paragraphs into passages of a quotable size, never crossing a section.

    A short paragraph -- often a movement heading on its own line, "Presto
    agitato." -- is joined to the one after it, which is the discussion it
    introduces; a short one with nothing after it in its section joins the one
    before. A long one is split at sentence ends, and each piece cites the page
    its first word is on, not the page its paragraph began on.
    """
    merged: list[Paragraph] = []
    carry: Paragraph | None = None
    for paragraph in paragraphs:
        if carry and carry.section == paragraph.section:
            paragraph = _join(carry, paragraph)
        elif carry:
            _append_short(merged, carry)
        carry = None
        if len(paragraph.text.split()) < MIN_PASSAGE_WORDS:
            carry = paragraph
        else:
            merged.append(paragraph)
    if carry:
        _append_short(merged, carry)

    passages: list[Passage] = []
    for paragraph in merged:
        cursor = 0
        for chunk in _split(paragraph.text):
            offset = paragraph.text.index(chunk, cursor)
            cursor = offset + len(chunk)
            where = paragraph.at(offset)
            passages.append(Passage(chunk, paragraph.section, len(passages),
                                    leaf=where.leaf, page=where.page, line=where.line))
    return passages


def _join(first: Paragraph, second: Paragraph) -> Paragraph:
    shift = len(first.text) + 1
    marks = first.marks + tuple(Mark(m.offset + shift, m.leaf, m.page, m.line) for m in second.marks)
    return Paragraph(first.text + " " + second.text, second.section, marks)


def _append_short(merged: list[Paragraph], short: Paragraph) -> None:
    if merged and merged[-1].section == short.section:
        merged[-1] = _join(merged[-1], short)
    else:
        merged.append(short)


def _split(text: str) -> list[str]:
    if len(text.split()) <= MAX_PASSAGE_WORDS:
        return [text]
    chunks, current = [], []
    for sentence in _SENTENCE_END.split(text):
        if current and len(" ".join(current + [sentence]).split()) > MAX_PASSAGE_WORDS:
            chunks.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        tail = " ".join(current)
        if chunks and len(tail.split()) < MIN_PASSAGE_WORDS:
            chunks[-1] += " " + tail
        else:
            chunks.append(tail)
    return chunks
