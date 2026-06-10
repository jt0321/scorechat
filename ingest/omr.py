"""
ingest/omr.py
Converts a PDF or image file to MusicXML via oemer (default) or Audiveris.
Returns the path to the generated MusicXML file.
"""

import subprocess
import shutil
import tempfile
from pathlib import Path


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[Path]:
    """Convert each PDF page to a PNG image using pdf2image."""
    from pdf2image import convert_from_path
    pdf_path = Path(pdf_path)
    out_dir = pdf_path.parent / f"{pdf_path.stem}_pages"
    out_dir.mkdir(exist_ok=True)
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    paths = []
    for i, page in enumerate(pages):
        p = out_dir / f"page_{i+1:03d}.png"
        page.save(str(p), "PNG")
        paths.append(p)
    return paths


def run_oemer(image_path: str, output_dir: str | None = None) -> Path:
    """
    Run oemer on a single score image and return the MusicXML path.
    oemer CLI: `oemer <image_path> --output-dir <dir>`
    """
    image_path = Path(image_path)
    if output_dir is None:
        output_dir = image_path.parent / "musicxml"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["oemer", str(image_path), "--output-dir", str(output_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"oemer failed:\n{result.stderr}")

    # oemer writes <stem>.musicxml
    out = Path(output_dir) / f"{image_path.stem}.musicxml"
    if not out.exists():
        # fallback: find any .musicxml in output_dir
        candidates = list(Path(output_dir).glob("*.musicxml"))
        if not candidates:
            raise FileNotFoundError(f"oemer produced no MusicXML in {output_dir}")
        out = candidates[0]
    return out


def run_audiveris(image_path: str, output_dir: str | None = None) -> Path:
    """
    Run Audiveris CLI on a single score image and return the MusicXML path.
    Requires Audiveris installed and on PATH as `audiveris`.
    """
    image_path = Path(image_path)
    if output_dir is None:
        output_dir = image_path.parent / "musicxml"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "audiveris", "-batch", "-export",
            "-output", str(output_dir),
            str(image_path)
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Audiveris failed:\n{result.stderr}")

    candidates = list(Path(output_dir).glob("**/*.xml"))
    if not candidates:
        raise FileNotFoundError(f"Audiveris produced no XML in {output_dir}")
    return candidates[0]


def ingest_score(
    source_path: str,
    tool: str = "oemer",
    output_dir: str | None = None
) -> list[Path]:
    """
    Main entry point. Accepts a PDF or image file.
    Returns list of MusicXML paths (one per page if PDF).
    tool: "oemer" | "audiveris"
    """
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    if source.suffix.lower() == ".pdf":
        images = pdf_to_images(str(source))
    else:
        images = [source]

    runner = run_oemer if tool == "oemer" else run_audiveris
    musicxml_paths = []
    for img in images:
        xml = runner(str(img), output_dir)
        musicxml_paths.append(xml)

    return musicxml_paths
