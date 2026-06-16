"""
ingest_scores.py
----------------
Batch-ingests all downloaded PDFs in ./data/ through the full pipeline:
  PDF → OMR (oemer/Audiveris) → MusicXML → music21 analysis → pgvector

Metadata comes from the SCORES manifest in download_scores.py.
PDFs in ./data/ that are not in the manifest are ingested with best-effort
metadata derived from their filename.

Usage:
    python ingest_scores.py [--tool oemer|audiveris] [--window 4] [--mei]

Prerequisites:
    python download_scores.py   # fetch PDFs
    psql $DATABASE_URL -f db/schema.sql  # create tables (first time only)
"""

import click
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from download_scores import SCORES, DATA_DIR
from db.store import (
    upsert_work, store_asset, store_segments, store_text_chunks,
    clear_work_segments_and_assets
)
from ingest.omr import ingest_score
from analysis.analyzer import analyze_musicxml
from pipeline.mei_converter import musicxml_to_mei

# Build a filename → metadata lookup from the manifest.
_MANIFEST: dict[str, dict] = {s["filename"]: s for s in SCORES}


def _metadata_from_filename(filename: str) -> dict:
    """Best-effort metadata for PDFs not in the manifest."""
    stem = Path(filename).stem.replace("_", " ")
    parts = stem.split(" ", 1)
    return {
        "composer": parts[0] if parts else "Unknown",
        "title":    parts[1] if len(parts) > 1 else stem,
        "opus":     None,
        "key":      None,
        "year":     None,
        "imslp":    None,
    }


@click.command()
@click.option("--tool",   default="oemer", type=click.Choice(["oemer", "audiveris"]),
              show_default=True, help="OMR engine")
@click.option("--window", default=4, type=int, show_default=True,
              help="Measures per analysis chunk")
@click.option("--mei",    is_flag=True, default=False,
              help="Also produce MEI files via Verovio (for front-end rendering)")
@click.option("--force",  is_flag=True, default=False,
              help="Re-ingest scores that are already in the database")
def main(tool: str, window: int, mei: bool, force: bool):
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        click.echo(f"No PDFs found in ./{DATA_DIR}/  —  run download_scores.py first.")
        return

    click.echo(f"Found {len(pdfs)} PDF(s) in ./{DATA_DIR}/\n")

    for pdf in pdfs:
        meta = _MANIFEST.get(pdf.name) or _metadata_from_filename(pdf.name)

        click.echo(f"▶  {meta['composer']} — {meta['title']}")

        work_meta = dict(
            composer     = meta["composer"],
            title        = meta["title"],
            opus         = meta.get("opus"),
            key_signature= meta.get("key"),
            year_composed= meta.get("year"),
            imslp_url    = meta.get("imslp"),
        )
        work_id = upsert_work(work_meta)
        click.echo(f"   Work ID: {work_id}")

        # Clear existing assets/segments to avoid duplicates
        clear_work_segments_and_assets(work_id)

        store_asset(work_id, "pdf", str(pdf))

        # Check if pre-existing symbolic score (.mscx, .musicxml, or .krn) exists in data/
        mscx_path = pdf.with_suffix(".mscx")
        xml_path = pdf.with_suffix(".musicxml")
        krn_path = pdf.with_suffix(".krn")
        xml_paths = []
        omr_used = False

        if xml_path.exists():
            click.echo(f"   ✓ Pre-existing MusicXML score found ({xml_path.name}). Bypassing OMR.")
            xml_paths = [xml_path]
        elif krn_path.exists():
            click.echo(f"   ✓ Pre-existing Humdrum score found ({krn_path.name}). Bypassing OMR.")
            try:
                from music21 import converter
                click.echo(f"     Converting {krn_path.name} to MusicXML...")
                score = converter.parse(krn_path)
                score.write("musicxml", fp=xml_path)
                click.echo(f"     ✓ Converted and saved: {xml_path.name}")
                xml_paths = [xml_path]
            except Exception as e:
                click.echo(f"     ✗ Conversion to MusicXML failed: {e}")
                continue
        elif mscx_path.exists():
            click.echo(f"   ✓ Pre-existing MuseScore score found ({mscx_path.name}). Bypassing OMR.")
            try:
                from music21 import converter
                click.echo(f"     Converting {mscx_path.name} to MusicXML...")
                score = converter.parse(mscx_path)
                score.write("musicxml", fp=xml_path)
                click.echo(f"     ✓ Converted and saved: {xml_path.name}")
                xml_paths = [xml_path]
            except Exception as e:
                click.echo(f"     ✗ Conversion to MusicXML failed: {e}")
                continue
        else:
            click.echo(f"   Running OMR ({tool})...")
            try:
                xml_paths = ingest_score(str(pdf), tool=tool)
                omr_used = True
            except Exception as e:
                click.echo(f"   ✗ OMR failed: {e}")
                continue

        click.echo(f"   → {len(xml_paths)} score file(s)")

        for x_path in xml_paths:
            store_asset(work_id, "musicxml", str(x_path), omr_tool=tool if omr_used else "omr_bypass")

            if mei:
                mei_path = musicxml_to_mei(str(x_path))
                if mei_path:
                    store_asset(work_id, "mei", str(mei_path))
                    click.echo(f"   MEI → {mei_path}")

            try:
                chunks, global_key = analyze_musicxml(str(x_path), window=window)
            except Exception as e:
                click.echo(f"   ✗ Analysis failed: {e}")
                continue

            click.echo(f"   {len(chunks)} chunks  (global key: {global_key})")
            store_segments(work_id, chunks)

        # Store IMSLP URL as a text source for hybrid retrieval.
        if meta.get("imslp"):
            store_text_chunks(work_id, [{
                "source_type": "imslp",
                "content": (
                    f"IMSLP entry for {meta['title']} by {meta['composer']}. "
                    f"URL: {meta['imslp']}"
                ),
                "url": meta["imslp"],
            }])

        click.echo(f"   ✓ Done\n")

    click.echo("All scores ingested. Run `streamlit run scorechat_app.py` to start the app.")


if __name__ == "__main__":
    main()
