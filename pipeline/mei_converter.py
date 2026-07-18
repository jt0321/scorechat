"""
pipeline/mei_converter.py
Converts a MusicXML file to MEI using the Verovio Python bindings.
MEI is used for front-end rendering (Verovio → SVG) and scholarly addressing.
Install: pip install verovio
"""

from pathlib import Path
from typing import Optional


def score_to_mei(score_path: str, output_dir: Optional[str] = None) -> Optional[Path]:
    """
    Convert a symbolic score file (MusicXML or Humdrum) → MEI using verovio Python bindings.
    Returns the path to the .mei file, or None on failure.
    """
    try:
        import verovio
    except ImportError:
        print("verovio not installed. Skipping MEI conversion. (pip install verovio)")
        return None

    score_path = Path(score_path)
    if output_dir is None:
        output_dir = score_path.parent / "mei"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ext = score_path.suffix.lower()
    input_format = "humdrum" if ext == ".krn" else "musicxml"

    tk = verovio.toolkit()
    tk.setOptions({
        "inputFrom": input_format,
        "outputTo":  "mei",
    })

    with open(score_path, "r", encoding="utf-8") as f:
        score_str = f.read()

    ok = tk.loadData(score_str)
    if not ok:
        print(f"Verovio failed to load {score_path}")
        return None

    mei_str = tk.getMEI()
    mei_path = Path(output_dir) / f"{score_path.stem}.mei"
    mei_path.write_text(mei_str, encoding="utf-8")
    return mei_path


def mei_to_svg(mei_path: str, measure_start: int = 1, measure_end: int = 4) -> str:
    """
    Render a range of measures from an MEI file to SVG.
    Used by the API to return rendered score excerpts for RAG results.
    """
    import verovio
    tk = verovio.toolkit()
    tk.setOptions({
        "inputFrom": "mei",
        "select":    [f"{measure_start}-{measure_end}"],
    })
    with open(mei_path, "r", encoding="utf-8") as f:
        mei_str = f.read()

    tk.loadData(mei_str)
    svg = tk.renderToSVG(1)
    return svg
