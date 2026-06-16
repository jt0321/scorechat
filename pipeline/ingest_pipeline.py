"""
pipeline/ingest_pipeline.py

Full pipeline:
  PDF/image → OMR → MusicXML → (optional MEI) → music21 analysis
  → chunk summaries → embeddings → Postgres (pgvector)

Usage:
  python -m pipeline.ingest_pipeline \
    --source /path/to/score.pdf \
    --composer "Frédéric Chopin" \
    --title "Nocturne in E-flat major" \
    --opus "Op. 9 No. 2" \
    --imslp "https://imslp.org/wiki/..." \
    --wikipedia "https://en.wikipedia.org/wiki/..." \
    [--tool oemer|audiveris] \
    [--window 4]
"""

import click
import os
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

load_dotenv()

from ingest.omr import ingest_score
from analysis.analyzer import analyze_musicxml
from db.store import (
    upsert_work, store_asset, store_segments, store_text_chunks,
    clear_work_segments_and_assets
)
from pipeline.mei_converter import musicxml_to_mei  # optional


@click.command()
@click.option("--source",    required=True,  help="Path to PDF or image score")
@click.option("--composer",  required=True)
@click.option("--title",     required=True)
@click.option("--opus",      default=None)
@click.option("--catalog",   default=None,   help="Catalog number e.g. K.331")
@click.option("--key",       default=None,   help="e.g. 'E-flat major'")
@click.option("--year",      default=None,   type=int)
@click.option("--imslp",     default=None)
@click.option("--wikipedia", default=None)
@click.option("--tool",      default="oemer", type=click.Choice(["oemer","audiveris"]))
@click.option("--window",    default=4,       type=int, help="Measures per chunk")
@click.option("--convert-mei", is_flag=True, default=False, help="Also produce MEI via Verovio")
def run(source, composer, title, opus, catalog, key, year,
        imslp, wikipedia, tool, window, convert_mei):

    click.echo(f"▶ Ingesting: {composer} — {title}")

    # 1. Register work in DB
    work_metadata = dict(
        composer=composer, title=title, opus=opus,
        catalog_no=catalog, key_signature=key,
        year_composed=year, imslp_url=imslp, wikipedia_url=wikipedia,
    )
    work_id = upsert_work(work_metadata)
    click.echo(f"  Work ID: {work_id}")

    # Clear existing assets/segments to avoid duplicates
    clear_work_segments_and_assets(work_id)

    # 2. Store original source asset
    is_pdf = source.lower().endswith(".pdf")
    store_asset(work_id, asset_type="pdf" if is_pdf else "page_image", file_path=source)

    # Check for pre-existing symbolic score or if source is already symbolic
    source_path = Path(source)
    mscx_path = source_path.with_suffix(".mscx")
    xml_path = source_path.with_suffix(".musicxml")
    krn_path = source_path.with_suffix(".krn")
    
    xml_paths = []
    omr_used = False

    if not is_pdf:
        # If source itself is a symbolic score
        if source_path.suffix.lower() == ".musicxml":
            xml_paths = [source_path]
        elif source_path.suffix.lower() in [".krn", ".mscx"]:
            target_xml = source_path.with_suffix(".musicxml")
            try:
                from music21 import converter
                click.echo(f"  Converting {source_path.name} to MusicXML...")
                score = converter.parse(source_path)
                score.write("musicxml", fp=target_xml)
                click.echo(f"  ✓ Converted and saved: {target_xml.name}")
                xml_paths = [target_xml]
            except Exception as e:
                click.echo(f"  ✗ Conversion to MusicXML failed: {e}")
                return
        else:
            # Assume it's a page image to run OMR on
            click.echo(f"  Running OMR ({tool})...")
            xml_paths = ingest_score(source, tool=tool)
            omr_used = True
    else:
        # If source is PDF, check for pre-existing symbolic files in the same directory
        if xml_path.exists():
            click.echo(f"  ✓ Pre-existing MusicXML score found ({xml_path.name}). Bypassing OMR.")
            xml_paths = [xml_path]
        elif krn_path.exists():
            click.echo(f"  ✓ Pre-existing Humdrum score found ({krn_path.name}). Bypassing OMR.")
            try:
                from music21 import converter
                click.echo(f"  Converting {krn_path.name} to MusicXML...")
                score = converter.parse(krn_path)
                score.write("musicxml", fp=xml_path)
                click.echo(f"  ✓ Converted and saved: {xml_path.name}")
                xml_paths = [xml_path]
            except Exception as e:
                click.echo(f"  ✗ Conversion to MusicXML failed: {e}")
                return
        elif mscx_path.exists():
            click.echo(f"  ✓ Pre-existing MuseScore score found ({mscx_path.name}). Bypassing OMR.")
            try:
                from music21 import converter
                click.echo(f"  Converting {mscx_path.name} to MusicXML...")
                score = converter.parse(mscx_path)
                score.write("musicxml", fp=xml_path)
                click.echo(f"  ✓ Converted and saved: {xml_path.name}")
                xml_paths = [xml_path]
            except Exception as e:
                click.echo(f"  ✗ Conversion to MusicXML failed: {e}")
                return
        else:
            click.echo(f"  Running OMR ({tool})...")
            try:
                xml_paths = ingest_score(source, tool=tool)
                omr_used = True
            except Exception as e:
                click.echo(f"  ✗ OMR failed: {e}")
                return

    click.echo(f"  → {len(xml_paths)} score file(s)")

    for x_path in tqdm(xml_paths, desc="  Analyzing pages"):
        store_asset(work_id, "musicxml", str(x_path), omr_tool=tool if omr_used else "omr_bypass")

        # 4. Optional MEI conversion via Verovio CLI
        if convert_mei:
            mei_path = musicxml_to_mei(str(x_path))
            if mei_path:
                store_asset(work_id, "mei", str(mei_path))
                click.echo(f"    MEI → {mei_path}")

        # 5. music21 analysis → measure chunks
        chunks, global_key = analyze_musicxml(str(x_path), window=window)
        click.echo(f"    {len(chunks)} measure chunks extracted (global key: {global_key})")

        # 6. Embed summaries + store in score_segments
        store_segments(work_id, chunks)

    # 7. Store linked text sources if URLs provided
    text_chunks = []
    if wikipedia:
        text_chunks.append({
            "source_type": "wikipedia",
            "content": f"Wikipedia article for {title} by {composer}. URL: {wikipedia}",
            "url": wikipedia,
        })
    if imslp:
        text_chunks.append({
            "source_type": "imslp",
            "content": f"IMSLP entry for {title} by {composer}. URL: {imslp}",
            "url": imslp,
        })
    if text_chunks:
        store_text_chunks(work_id, text_chunks)

    click.echo(f"✓ Done. Work ID {work_id} fully ingested.")


if __name__ == "__main__":
    run()
