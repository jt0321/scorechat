"""
commentary/manifest.py
Reading and validating sources/manifest.toml.

The manifest is the only committed record of the commentary texts: where each
one is downloaded from, the sha256 that pins it, and which pages discuss which
sonata. Everything downstream -- fetching, segmenting, anchoring -- trusts it,
so it is validated on load rather than when something later goes wrong: a
section list out of reading order would silently assign one sonata's pages to
another.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "sources" / "manifest.toml"
SOURCES_DIR = ROOT / "data" / "sources"

FORMATS = ("archive-hocr", "gutenberg-text")
CORPUS_SONATAS = range(1, 33)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """The manifest contradicts itself or the corpus."""


@dataclass(frozen=True)
class SourceFile:
    name: str
    sha256: str


@dataclass(frozen=True)
class Section:
    heading: str
    sonatas: tuple[int, ...]
    include: bool = True
    leaf: int | None = None     # archive-hocr: scan page index
    page: int | None = None     # archive-hocr: printed page number
    line: int | None = None     # gutenberg-text: 1-based line number
    note: str | None = None

    @property
    def declared(self) -> bool:
        """Whether the section is declared to discuss particular sonatas, as
        opposed to being anchored only by what its passages name."""
        return bool(self.sonatas)


@dataclass(frozen=True)
class Source:
    key: str
    title: str
    author: str
    year: int
    format: str
    archive_id: str
    rights: str
    files: tuple[SourceFile, ...]
    sections: tuple[Section, ...]
    translator: str | None = None
    published: str | None = None
    language: str = "en"
    page_offset: int = 0
    encoding: str = "utf-8"
    coverage: str | None = None

    def url(self, file: SourceFile) -> str:
        return f"https://archive.org/download/{self.archive_id}/{file.name}"

    def directory(self, root: Path = SOURCES_DIR) -> Path:
        return root / self.key

    def citation(self) -> str:
        translated = f", tr. {self.translator}" if self.translator else ""
        return f"{self.author}, {self.title}{translated} ({self.year})"


def load_manifest(path: Path = MANIFEST_PATH) -> list[Source]:
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    sources = [_source(entry) for entry in raw.get("source", [])]
    keys = [s.key for s in sources]
    duplicates = {k for k in keys if keys.count(k) > 1}
    if duplicates:
        raise ManifestError(f"duplicate source keys: {sorted(duplicates)}")
    return sources


def get_source(key: str, path: Path = MANIFEST_PATH) -> Source:
    for source in load_manifest(path):
        if source.key == key:
            return source
    raise ManifestError(f"no source {key!r} in {path.name}")


def _source(entry: dict) -> Source:
    key = entry.get("key") or "<missing key>"
    fmt = entry.get("format")
    if fmt not in FORMATS:
        raise ManifestError(f"{key}: format {fmt!r} is not one of {FORMATS}")

    files = tuple(SourceFile(f["name"], f["sha256"]) for f in entry.get("file", []))
    if not files:
        raise ManifestError(f"{key}: no files listed")
    for f in files:
        if not _SHA256.match(f.sha256):
            raise ManifestError(f"{key}: {f.name} has no valid sha256 pin")

    sections = tuple(_section(key, fmt, s) for s in entry.get("section", []))
    if not sections:
        raise ManifestError(f"{key}: no sections -- nothing would be ingested")
    positions = [(s.leaf if fmt == "archive-hocr" else s.line) for s in sections]
    if positions != sorted(positions):
        # Sections end where the next begins, so an out-of-order entry would
        # hand one sonata's pages to another without any error.
        raise ManifestError(f"{key}: sections are not in reading order")

    fields = {name: entry[name] for name in
              ("translator", "published", "language", "page_offset", "encoding", "coverage")
              if name in entry}
    return Source(key=key, title=entry["title"], author=entry["author"], year=entry["year"],
                  format=fmt, archive_id=entry["archive_id"], rights=entry["rights"],
                  files=files, sections=sections, **fields)


def _section(key: str, fmt: str, entry: dict) -> Section:
    sonatas = tuple(entry.get("sonatas", ()))
    outside = [n for n in sonatas if n not in CORPUS_SONATAS]
    if outside:
        raise ManifestError(f"{key}: section {entry.get('heading')!r} names sonatas "
                            f"outside the corpus: {outside}")
    locator = "leaf" if fmt == "archive-hocr" else "line"
    if locator not in entry:
        raise ManifestError(f"{key}: section {entry.get('heading')!r} needs a {locator}")
    return Section(heading=entry["heading"], sonatas=sonatas,
                   include=entry.get("include", True), leaf=entry.get("leaf"),
                   page=entry.get("page"), line=entry.get("line"), note=entry.get("note"))
