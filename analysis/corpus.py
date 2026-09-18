"""
analysis/corpus.py
Where the corpus lives on disk.

The Humdrum sources are Craig Sapp's beethoven-piano-sonatas repository,
included as a git submodule pinned to one commit rather than copied in: the
provenance is the upstream history, and a checkout reproduces exactly the text
the stored analysis was derived from. The MEI renderings are *derived* -- ingest
regenerates them -- so they are written outside the submodule and not tracked.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERN_DIR = ROOT / "data" / "beethoven-piano-sonatas" / "kern"
MEI_DIR = ROOT / "data" / "mei"
