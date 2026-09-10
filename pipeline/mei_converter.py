"""
pipeline/mei_converter.py
Converts a MusicXML file to MEI using the Verovio Python bindings.
MEI is used for front-end rendering (Verovio → SVG) and scholarly addressing.
Install: pip install verovio
"""

import re
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


# Printed measure numbers follow engraving convention: an anacrusis is not
# numbered, so bar 1 is the first *complete* measure.  Verovio agrees (it emits
# the pickup with no @n) but its measureRange selection counts ordinal
# positions from 1, in which the pickup *is* position 1.  The two therefore
# differ by one for any movement with an upbeat, and this is the seam where
# that gets reconciled -- see measure_ordinals().
_MEI_MEASURE = re.compile(r"<measure\b([^>]*)>")
_MEI_MEASURE_NUMBER = re.compile(r'\bn="([^"]*)"')


def measure_ordinals(mei_str: str) -> dict[int, int]:
    """Map printed measure number -> Verovio ordinal position for one MEI.

    Built from the document itself rather than assumed, because a score's
    printed numbering need not be contiguous (pickups, repeated bar numbers,
    editorial renumbering) and the MEI is the authority for what Verovio will
    count.

    Measures the engraving does not number carry no `@n` and so appear in no
    entry of their own -- they are unreachable by number because they have
    none. `pickup_ordinals` below is how a range still renders them.
    """
    ordinals: dict[int, int] = {}
    for position, attributes in enumerate(_MEI_MEASURE.findall(mei_str), start=1):
        match = _MEI_MEASURE_NUMBER.search(attributes)
        if match is None:
            continue
        try:
            number = int(match.group(1))
        except ValueError:
            continue
        ordinals.setdefault(number, position)
    return ordinals


def pickup_ordinals(mei_str: str) -> dict[int, int]:
    """Map printed measure number -> ordinal of the unnumbered measure(s)
    immediately before it.

    A bar preceded by an unnumbered measure is a bar with a pickup: the
    movement's anacrusis, or the upbeat written after a repeat barline. Asking
    for mm. 229-247 means the music a performer would start playing, which
    begins on that upbeat -- so rendering starts there, matching how the range
    is cited ("from the upbeat to m. 229") and how `get_measure_evidence`
    selects the same range from the database.
    """
    pickups: dict[int, int] = {}
    run_start: int | None = None
    for position, attributes in enumerate(_MEI_MEASURE.findall(mei_str), start=1):
        match = _MEI_MEASURE_NUMBER.search(attributes)
        if match is None:
            run_start = position if run_start is None else run_start
            continue
        if run_start is not None:
            try:
                pickups.setdefault(int(match.group(1)), run_start)
            except ValueError:
                pass
            run_start = None
    return pickups


def mei_to_svg(mei_path: str, measure_start: int = 1, measure_end: int = 4) -> str:
    """
    Render a range of measures from an MEI file to SVG.
    Used by the API to return rendered score excerpts for RAG results.

    `measure_start`/`measure_end` are *printed* measure numbers, as stored in
    `score_measures.measure_number` and as cited to the user, not internal
    indices.
    """
    import verovio

    with open(mei_path, "r", encoding="utf-8") as f:
        mei_str = f.read()

    ordinals = measure_ordinals(mei_str)
    start = ordinals.get(measure_start)
    end = ordinals.get(measure_end)
    if start is None or end is None:
        # Fall back to the nearest numbers actually present rather than
        # rendering the whole movement, which is what a failed selection does.
        known = sorted(ordinals)
        if not known:
            return ""
        start = start or ordinals[min(known, key=lambda n: abs(n - measure_start))]
        end = end or ordinals[min(known, key=lambda n: abs(n - measure_end))]
    if end < start:
        start, end = end, start

    # Back up over a pickup so the excerpt begins where a performer would.
    start = min(start, pickup_ordinals(mei_str).get(measure_start, start))

    tk = verovio.toolkit()
    tk.setOptions({"inputFrom": "mei", "adjustPageHeight": True,
                   "header": "none", "footer": "none"})
    # select() takes a dict and must be called before loadData; the older
    # "select" *option* is silently unsupported and renders the entire
    # movement instead of the excerpt.
    tk.select({"measureRange": f"{start}-{end}"})
    tk.loadData(mei_str)
    return tk.renderToSVG(1)
