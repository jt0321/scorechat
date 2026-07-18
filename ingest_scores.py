"""
ingest_scores.py
----------------
Batch-ingests all Humdrum (.krn) files in ./data/ through the symbolic RAG pipeline:
  Humdrum (.krn) → MusicXML → MEI (via Verovio) → music21 analysis → pgvector (PostgreSQL)

Prerequisites:
    python download_beethoven_piano_sonatas.py  # fetch .krn files
    psql $DATABASE_URL -f db/schema.sql  # create tables (first time only)
"""

import re
import click
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from download_beethoven_piano_sonatas import DATA_DIR
from db.store import (
    upsert_work, store_asset, store_segments,
    clear_work_segments_and_assets
)
from analysis.analyzer import analyze_musicxml
from pipeline.mei_converter import musicxml_to_mei

def parse_krn_metadata(krn_path: Path) -> dict:
    """Parse standard Humdrum metadata headers for title, composer, opus, movement, etc."""
    meta = {
        "composer": "Ludwig van Beethoven",
        "title": "Piano Sonata",
        "opus": None,
        "movement": "",
        "key": None,
        "year": None,
    }
    
    # Try parsing filename first to guess sonata number and movement
    # e.g., sonata32-1.krn
    match = re.search(r"sonata(\d+)-(\d+)", krn_path.name)
    sonata_num = None
    mvt_num = None
    if match:
        sonata_num = int(match.group(1))
        mvt_num = int(match.group(2))
        meta["movement"] = f"Mvt {mvt_num}"
        meta["title"] = f"Piano Sonata No. {sonata_num}"
        
    try:
        content = krn_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            if not line.startswith("!!!"):
                continue
            
            # Composer
            if line.startswith("!!!COM:"):
                val = line.split(":", 1)[1].strip()
                # Reformat "Beethoven, Ludwig van" -> "Ludwig van Beethoven"
                if "," in val:
                    parts = [p.strip() for p in val.split(",", 1)]
                    meta["composer"] = f"{parts[1]} {parts[0]}"
                else:
                    meta["composer"] = val
            
            # Original Title (fallback if filename parsing didn't work)
            elif line.startswith("!!!OTL:"):
                val = line.split(":", 1)[1].strip()
                # Clean up title formatting if it has sonata info
                if not sonata_num:
                    meta["title"] = val
            
            # Opus
            elif line.startswith("!!!OPS:"):
                val = line.split(":", 1)[1].strip()
                if val.isdigit():
                    meta["opus"] = f"Op. {val}"
                elif not val.lower().startswith("op"):
                    meta["opus"] = f"Op. {val}"
                else:
                    meta["opus"] = val
                    
            # Composition Date / Year
            elif line.startswith("!!!ODT:"):
                val = line.split(":", 1)[1].strip()
                year_match = re.search(r"\b(1[789]\d{2})\b", val)
                if year_match:
                    meta["year"] = int(year_match.group(1))
    except Exception as e:
        click.echo(f"   ⚠ Metadata parsing error for {krn_path.name}: {e}")
        
    # Standardize titles for late Beethoven sonatas
    if sonata_num == 32:
        meta["title"] = f"Piano Sonata No. 32 in C minor"
        meta["opus"] = "Op. 111"
        meta["key"] = "C minor"
    elif sonata_num == 21:
        meta["title"] = f"Piano Sonata No. 21 in C major (Waldstein)"
        meta["opus"] = "Op. 53"
        meta["key"] = "C major"
    elif sonata_num == 23:
        meta["title"] = f"Piano Sonata No. 23 in F minor (Appassionata)"
        meta["opus"] = "Op. 57"
        meta["key"] = "F minor"

    return meta


@click.command()
@click.option("--window", default=4, type=int, show_default=True,
              help="Measures per analysis chunk")
@click.option("--force",  is_flag=True, default=False,
              help="Force re-conversion and re-analysis")
def main(window: int, force: bool):
    krns = sorted(DATA_DIR.glob("*.krn"))
    if not krns:
        click.echo(f"No Humdrum (.krn) files found in ./{DATA_DIR}/ — run download_beethoven_piano_sonatas.py first.")
        return

    click.echo(f"Found {len(krns)} Humdrum score(s) in ./{DATA_DIR}/\n")

    for krn in krns:
        meta = parse_krn_metadata(krn)
        mvt_suffix = f" ({meta['movement']})" if meta["movement"] else ""
        click.echo(f"▶  {meta['composer']} — {meta['title']}{mvt_suffix}")

        work_meta = dict(
            composer     = meta["composer"],
            title        = f"{meta['title']}{mvt_suffix}",
            opus         = meta["opus"],
            key_signature= meta["key"],
            year_composed= meta["year"],
            imslp_url    = None,
        )
        work_id = upsert_work(work_meta)
        click.echo(f"   Work ID: {work_id}")

        # Clear existing assets/segments to avoid duplicates
        clear_work_segments_and_assets(work_id)

        # Convert Humdrum to MusicXML for analysis & slicing
        xml_path = krn.with_suffix(".musicxml")
        if not xml_path.exists() or force:
            try:
                from music21 import converter
                click.echo(f"   Converting {krn.name} to MusicXML...")
                score = converter.parse(krn)
                # Write to MusicXML
                score.write("musicxml", fp=xml_path)
                click.echo(f"   ✓ Converted and saved: {xml_path.name}")
            except Exception as e:
                click.echo(f"   ✗ Conversion to MusicXML failed: {e}")
                continue
        else:
            click.echo(f"   ✓ Existing MusicXML score found ({xml_path.name}).")

        # Save assets in DB
        store_asset(work_id, "musicxml", str(xml_path))

        # Generate MEI file using Verovio (needed for SVG rendering)
        mei_path = musicxml_to_mei(str(xml_path))
        if mei_path:
            store_asset(work_id, "mei", str(mei_path))
            click.echo(f"   ✓ MEI file generated → {mei_path.name}")
        else:
            click.echo("   ✗ MEI generation failed.")

        # Run music21 analysis and segment score
        click.echo("   Analyzing musical features...")
        try:
            chunks, global_key = analyze_musicxml(str(xml_path), window=window)
            click.echo(f"   ✓ {len(chunks)} chunks extracted (global key: {global_key})")
            store_segments(work_id, chunks)
            click.echo(f"   ✓ Done ingesting\n")
        except Exception as e:
            click.echo(f"   ✗ Analysis failed: {e}\n")
            continue

    click.echo("All scores ingested. Run `python server.py` or `streamlit run scorechat_app.py` to start the app.")


if __name__ == "__main__":
    main()
