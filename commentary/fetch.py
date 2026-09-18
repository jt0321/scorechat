"""
commentary/fetch.py
Downloading a pinned file, and refusing it if it is not the file we pinned.

Everything downstream counts on the exact text: a section starts at a given
leaf or line, a quote is located by offset. A re-OCR'd or re-issued file would
still download cleanly and look fine, and every anchor in it would be wrong.
So a checksum mismatch is not a warning -- the file is set aside as
`<name>.rejected` for inspection and nothing is ingested from it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

USER_AGENT = "ScoreChat/1.0 (educational use; github.com/jt0321/scorechat)"


@dataclass(frozen=True)
class FetchResult:
    name: str
    status: str        # "cached" | "downloaded" | "rejected"
    sha256: str
    expected: str

    @property
    def ok(self) -> bool:
        return self.status != "rejected"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def http_download(url: str, dest: Path) -> None:
    import requests
    with requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in response.iter_content(1 << 16):
                handle.write(chunk)


def fetch_file(url: str, dest: Path, sha256: str, *, force: bool = False,
               download: Callable[[str, Path], None] = http_download) -> FetchResult:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        actual = sha256_of(dest)
        if actual == sha256:
            return FetchResult(dest.name, "cached", actual, sha256)
        # A cached file that no longer matches is re-fetched rather than trusted.

    partial = dest.with_name(dest.name + ".part")
    try:
        download(url, partial)
        actual = sha256_of(partial)
        if actual != sha256:
            partial.replace(dest.with_name(dest.name + ".rejected"))
            if dest.exists():
                dest.unlink()
            return FetchResult(dest.name, "rejected", actual, sha256)
        partial.replace(dest)
        return FetchResult(dest.name, "downloaded", actual, sha256)
    finally:
        if partial.exists():
            partial.unlink()
