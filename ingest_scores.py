"""
ingest_scores.py
----------------
Batch-ingests all Humdrum (.krn) files in ./data/ through the symbolic RAG pipeline:
  Humdrum (.krn) → MEI (via Verovio) → music21 analysis → pgvector (PostgreSQL)

Prerequisites:
    git submodule update --init                 # fetch the .krn sources
    psql $DATABASE_URL -f db/schema.sql  # create tables (first time only)
"""

import re
import click
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from analysis.corpus import KERN_DIR, MEI_DIR
from db.store import (
    upsert_work, store_asset, clear_work_assets, clear_work_symbolic_layers,
    store_symbolic_layers, store_symbolic_source, store_span_candidates,
)
from analysis.analyzer import (
    build_span_candidates, build_symbolic_layers, humdrum_measure_count,
)
from pipeline.mei_converter import score_to_mei

# Sonata number -> (title, opus, key, nickname). Covers all 32 published
# Beethoven piano sonatas as catalogued in craigsapp/beethoven-piano-sonatas.
SONATA_CATALOG = {
    1:  ("Piano Sonata No. 1 in F minor",        "Op. 2 No. 1",  "F minor",  None),
    2:  ("Piano Sonata No. 2 in A major",         "Op. 2 No. 2",  "A major",  None),
    3:  ("Piano Sonata No. 3 in C major",         "Op. 2 No. 3",  "C major",  None),
    4:  ("Piano Sonata No. 4 in E-flat major",    "Op. 7",        "E-flat major", None),
    5:  ("Piano Sonata No. 5 in C minor",         "Op. 10 No. 1", "C minor",  None),
    6:  ("Piano Sonata No. 6 in F major",         "Op. 10 No. 2", "F major",  None),
    7:  ("Piano Sonata No. 7 in D major",         "Op. 10 No. 3", "D major",  None),
    8:  ("Piano Sonata No. 8 in C minor",         "Op. 13",       "C minor",  "Pathétique"),
    9:  ("Piano Sonata No. 9 in E major",         "Op. 14 No. 1", "E major",  None),
    10: ("Piano Sonata No. 10 in G major",        "Op. 14 No. 2", "G major",  None),
    11: ("Piano Sonata No. 11 in B-flat major",   "Op. 22",       "B-flat major", None),
    12: ("Piano Sonata No. 12 in A-flat major",   "Op. 26",       "A-flat major", None),
    13: ("Piano Sonata No. 13 in E-flat major",   "Op. 27 No. 1", "E-flat major", "Quasi una fantasia"),
    14: ("Piano Sonata No. 14 in C-sharp minor",  "Op. 27 No. 2", "C-sharp minor", "Moonlight"),
    15: ("Piano Sonata No. 15 in D major",        "Op. 28",       "D major",  "Pastoral"),
    16: ("Piano Sonata No. 16 in G major",        "Op. 31 No. 1", "G major",  None),
    17: ("Piano Sonata No. 17 in D minor",        "Op. 31 No. 2", "D minor",  "Tempest"),
    18: ("Piano Sonata No. 18 in E-flat major",   "Op. 31 No. 3", "E-flat major", "The Hunt"),
    19: ("Piano Sonata No. 19 in G minor",        "Op. 49 No. 1", "G minor",  None),
    20: ("Piano Sonata No. 20 in G major",        "Op. 49 No. 2", "G major",  None),
    21: ("Piano Sonata No. 21 in C major",        "Op. 53",       "C major",  "Waldstein"),
    22: ("Piano Sonata No. 22 in F major",        "Op. 54",       "F major",  None),
    23: ("Piano Sonata No. 23 in F minor",        "Op. 57",       "F minor",  "Appassionata"),
    24: ("Piano Sonata No. 24 in F-sharp major",  "Op. 78",       "F-sharp major", "à Thérèse"),
    25: ("Piano Sonata No. 25 in G major",        "Op. 79",       "G major",  None),
    26: ("Piano Sonata No. 26 in E-flat major",   "Op. 81a",      "E-flat major", "Les Adieux"),
    27: ("Piano Sonata No. 27 in E minor",        "Op. 90",       "E minor",  None),
    28: ("Piano Sonata No. 28 in A major",        "Op. 101",      "A major",  None),
    29: ("Piano Sonata No. 29 in B-flat major",   "Op. 106",      "B-flat major", "Hammerklavier"),
    30: ("Piano Sonata No. 30 in E major",        "Op. 109",      "E major",  None),
    31: ("Piano Sonata No. 31 in A-flat major",   "Op. 110",      "A-flat major", None),
    32: ("Piano Sonata No. 32 in C minor",        "Op. 111",      "C minor",  None),
}


ROMAN_NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]


def parse_krn_metadata(krn_path: Path) -> dict:
    """Parse standard Humdrum metadata headers for title, composer, opus, movement, etc."""
    meta = {
        "composer": "Ludwig van Beethoven",
        "title": "Piano Sonata",
        "opus": None,
        "nickname": None,
        "work_number": None,
        "movement_number": None,
        "tempo_indication": None,
        "key": None,
        "year": None,
        "source_url": None,
    }

    # Try parsing filename first to guess sonata number and movement
    # e.g., sonata32-1.krn
    match = re.search(r"sonata(\d+)-(\d+)", krn_path.name)
    sonata_num = None
    if match:
        sonata_num = int(match.group(1))
        meta["work_number"] = sonata_num
        meta["movement_number"] = int(match.group(2))
        meta["title"] = f"Piano Sonata No. {sonata_num}"

    try:
        content = krn_path.read_text(encoding="utf-8")
        # This provenance record is commonly emitted at the end of a Humdrum
        # file, after the data spine, whereas work metadata is at the start.
        source_url_match = re.search(r"^!!!URL-github:\s*(.+)$", content, re.MULTILINE)
        if source_url_match:
            meta["source_url"] = source_url_match.group(1).strip() or None
        for line in content.splitlines():
            # Header metadata ends once the data spine (**kern) starts; later
            # !!!OMD lines are mid-movement tempo changes, not the movement's
            # identifying tempo indication.
            if line.startswith("**"):
                break
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

            # Movement number (overrides filename guess when present)
            elif line.startswith("!!!OMV:"):
                val = line.split(":", 1)[1].strip()
                if val.isdigit():
                    meta["movement_number"] = int(val)

            # Movement tempo indication, e.g. "Presto agitato". Humdrum uses
            # a literal "\n" (not an actual newline) for manual line breaks
            # within the field, e.g. "ARIETTA\nAdagio molto..." — normalize
            # to a space for display.
            elif line.startswith("!!!OMD:") and not meta["tempo_indication"]:
                val = line.split(":", 1)[1].strip().replace("\\n", " ")
                if val:
                    meta["tempo_indication"] = val

            # Composition Date / Year
            elif line.startswith("!!!ODT:"):
                val = line.split(":", 1)[1].strip()
                year_match = re.search(r"\b(1[789]\d{2})\b", val)
                if year_match:
                    meta["year"] = int(year_match.group(1))
    except Exception as e:
        click.echo(f"   ⚠ Metadata parsing error for {krn_path.name}: {e}")

    # Standardize title/opus/key from the known catalog when we recognize
    # the sonata number, regardless of what the Humdrum headers say.
    if sonata_num in SONATA_CATALOG:
        title, opus, key, nickname = SONATA_CATALOG[sonata_num]
        meta["title"] = f"{title} ({nickname})" if nickname else title
        meta["opus"] = opus
        meta["key"] = key
        meta["nickname"] = nickname

    return meta


@click.command()
@click.option("--symbolic-only", is_flag=True,
              help="Rebuild the symbolic layers only, keeping the rendered MEI assets")
def main(symbolic_only: bool):
    krns = sorted(KERN_DIR.glob("*.krn"))
    if not krns:
        click.echo(f"No Humdrum (.krn) files found in {KERN_DIR} — run `git submodule update --init` first.")
        return

    click.echo(f"Found {len(krns)} Humdrum score(s) in {KERN_DIR}\n")

    for krn in krns:
        meta = parse_krn_metadata(krn)

        mvt_num = meta["movement_number"]
        mvt_roman = ROMAN_NUMERALS[mvt_num - 1] if mvt_num and mvt_num <= len(ROMAN_NUMERALS) else None
        mvt_suffix = ""
        if mvt_roman and meta["tempo_indication"]:
            mvt_suffix = f". {mvt_roman}. {meta['tempo_indication']}"
        elif mvt_roman:
            mvt_suffix = f". {mvt_roman}"

        click.echo(f"▶  {meta['composer']} — {meta['title']}{mvt_suffix}")

        work_meta = dict(
            composer        = meta["composer"],
            title           = f"{meta['title']}{mvt_suffix}",
            opus            = meta["opus"],
            nickname        = meta["nickname"],
            work_number     = meta["work_number"],
            movement_number = meta["movement_number"],
            tempo_indication= meta["tempo_indication"],
            key_signature   = meta["key"],
            year_composed   = meta["year"],
            imslp_url       = None,
        )
        work_id = upsert_work(work_meta)
        click.echo(f"   Work ID: {work_id}")

        # Symbolic-only rebuilding preserves already-rendered MEI assets; they
        # are independent of the symbolic layers rebuilt below.
        if symbolic_only:
            clear_work_symbolic_layers(work_id)
        else:
            clear_work_assets(work_id)

        # Save Humdrum (.krn) as an asset during a full ingest.  The immutable
        # source row below is stored in both modes.
        if not symbolic_only:
            store_asset(work_id, "krn", str(krn))
        store_symbolic_source(work_id, str(krn), source_url=meta["source_url"])

        if not symbolic_only:
            # Generate MEI file dynamically using Verovio (needed for SVG rendering)
            mei_path = score_to_mei(str(krn), output_dir=MEI_DIR)
            if mei_path:
                # Stored as text as well as a path: data/mei is derived and
                # gitignored, so a deployed instance has no copy to read.
                store_asset(work_id, "mei", str(mei_path),
                            content=mei_path.read_text(encoding="utf-8"))
                click.echo(f"   ✓ MEI file generated → {mei_path.name}")
            else:
                click.echo("   ✗ MEI generation failed.")

        # The canonical layer: every later pass and every answer reads these
        # records, and nothing re-parses the source to serve a question.
        click.echo("   Encoding score and analysing measures...")
        try:
            measures, measure_analyses, _ = build_symbolic_layers(str(krn))
            # A parse that silently drops music is worse than a failed one: the
            # stored analysis looks healthy while covering only part of the
            # movement. Check it against the barlines the source itself notates.
            expected_measures = humdrum_measure_count(krn.read_text(encoding="utf-8"))
            if expected_measures and len(measures) < expected_measures - 2:
                click.echo(
                    f"   ✗ Parsed only {len(measures)} of {expected_measures} notated "
                    f"measures — skipping rather than storing a truncated score."
                )
                continue
            store_symbolic_layers(work_id, measures, measure_analyses)
            click.echo(f"   ✓ {len(measures)} canonical measures and analyses stored")
            span_candidates = build_span_candidates(measures, measure_analyses)
            store_span_candidates(work_id, span_candidates)
            click.echo(f"   ✓ {len(span_candidates)} evidence-backed span candidates stored")
            click.echo("   ✓ Done ingesting\n")
        except Exception as e:
            click.echo(f"   ✗ Analysis failed: {e}\n")
            continue

    click.echo("All scores ingested. Run `python build_sections.py && python build_relations.py`, "
               "then `python server.py` to start the app.")


if __name__ == "__main__":
    main()
