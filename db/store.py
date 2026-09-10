"""
db/store.py
Persists works, score assets, score segments (with embeddings),
and text source chunks to Postgres via SQLAlchemy.
"""

from __future__ import annotations
import hashlib
import re
from fractions import Fraction
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.orm import aliased
from db.models import (
    Work, ScoreAsset, ScoreSegment, TextSource, ScoreSource, ScoreMeasure,
    MeasureAnalysis, AnalysisRun, SpanAnalysis, SpanRelation,
)
from db.session import session_scope
from analysis.analyzer import (
    CanonicalMeasure, MeasureChunk, PerMeasureAnalysis, SpanCandidate,
    SPAN_ANALYSIS_VERSION,
)
from pipeline.embedder import embed_texts


def upsert_work(metadata: dict) -> int:
    """Insert or update a Work record. Returns work.id."""
    with session_scope() as session:
        existing = (
            session.query(Work)
            .filter_by(composer=metadata["composer"], title=metadata["title"])
            .first()
        )
        if existing:
            for k, v in metadata.items():
                setattr(existing, k, v)
            session.commit()
            return existing.id

        work = Work(**metadata)
        session.add(work)
        session.commit()
        return work.id


def store_asset(work_id: int, asset_type: str, file_path: str,
                page_number: int | None = None, omr_tool: str | None = None,
                omr_quality: str = "auto") -> int:
    with session_scope() as session:
        asset = ScoreAsset(
            work_id=work_id, asset_type=asset_type,
            file_path=str(file_path), page_number=page_number,
            omr_tool=omr_tool, omr_quality=omr_quality
        )
        session.add(asset)
        session.commit()
        return asset.id


def store_symbolic_source(work_id: int, file_path: str, source_url: str | None = None) -> int:
    """Persist an immutable Humdrum source copy, checksum, and provenance."""
    path = Path(file_path)
    raw_content = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
    with session_scope() as session:
        source = ScoreSource(
            work_id=work_id,
            format="humdrum-kern",
            file_path=str(path),
            source_url=source_url,
            sha256=digest,
            raw_content=raw_content,
        )
        session.add(source)
        session.commit()
        return source.id


def store_symbolic_layers(
    work_id: int,
    measures: list[CanonicalMeasure],
    analyses: list[PerMeasureAnalysis],
) -> None:
    """Store canonical notation and versioned measure analysis atomically."""
    analyses_by_index = {analysis.measure_index: analysis for analysis in analyses}
    with session_scope() as session:
        for encoded_measure in measures:
            measure = ScoreMeasure(
                work_id=work_id,
                measure_index=encoded_measure.measure_index,
                measure_number=encoded_measure.measure_number,
                measure_role=encoded_measure.measure_role,
                measure_belongs_to=encoded_measure.measure_belongs_to,
                symbolic_data=encoded_measure.symbolic_data,
            )
            session.add(measure)
            session.flush()
            analysis = analyses_by_index.get(encoded_measure.measure_index)
            if analysis is not None:
                session.add(MeasureAnalysis(
                    measure_id=measure.id,
                    analysis_version=analysis.analysis_data["analysis_version"],
                    analysis_data=analysis.analysis_data,
                ))
        session.commit()


def store_span_candidates(work_id: int, candidates: list[SpanCandidate]) -> int:
    """Persist one deterministic candidate-span analysis run for a work."""
    with session_scope() as session:
        source = session.query(ScoreSource).filter_by(work_id=work_id).order_by(ScoreSource.id.desc()).first()
        if source is None:
            raise ValueError(f"Cannot analyse spans without a symbolic source for work {work_id}")
        run = AnalysisRun(
            work_id=work_id,
            analyzer_name="deterministic_boundary_candidates",
            analyzer_version=SPAN_ANALYSIS_VERSION,
            configuration_data={"boundary_signals": ["meter_change", "notated_direction", "structural_barline"]},
            source_sha256=source.sha256,
        )
        session.add(run)
        session.flush()
        for candidate in candidates:
            session.add(SpanAnalysis(
                work_id=work_id,
                analysis_run_id=run.id,
                measure_start_index=candidate.measure_start_index,
                measure_end_index=candidate.measure_end_index,
                measure_start=candidate.measure_start,
                measure_end=candidate.measure_end,
                span_type="candidate",
                confidence=1.0,
                status="proposed",
                evidence_data=candidate.evidence,
                features_data=candidate.features,
            ))
        session.commit()
        return run.id


def get_measure_evidence(work_id: int, measure_start: int, measure_end: int) -> list[dict]:
    """Return canonical notation and current analysis for an inclusive range."""
    with session_scope() as session:
        rows = (
            session.query(ScoreMeasure, MeasureAnalysis)
            .outerjoin(MeasureAnalysis, MeasureAnalysis.measure_id == ScoreMeasure.id)
            .filter(
                ScoreMeasure.work_id == work_id,
                # Selected on measure_belongs_to, not measure_number. Both
                # carry printed numbering -- the numbers a performer reads and
                # the LLM cites -- but a bar's pickup has no number of its own
                # while belonging to that bar, and asking for mm. 229-247 must
                # return the upbeat into 229 along with it. For a bar the two
                # columns agree, so this only ever adds the non-bars.
                ScoreMeasure.measure_belongs_to >= measure_start,
                ScoreMeasure.measure_belongs_to <= measure_end,
            )
            .order_by(
                ScoreMeasure.measure_index,
                MeasureAnalysis.created_at.desc(),
                # PostgreSQL NOW() is transaction-scoped, so versions written
                # in one transaction share created_at. The sequence-backed ID
                # is a deterministic insertion-order tie-breaker.
                MeasureAnalysis.id.desc(),
            )
            .all()
        )

        # A measure may gain a newer analysis version later. Keep the newest
        # inserted one (created_at, then ID) while retaining one score record.
        evidence_by_measure: dict[int, dict] = {}
        for measure, measure_analysis in rows:
            if measure.id not in evidence_by_measure:
                evidence_by_measure[measure.id] = {
                    "measure_index": measure.measure_index,
                    "measure_number": measure.measure_number,
                    "measure_role": measure.measure_role,
                    "measure_belongs_to": measure.measure_belongs_to,
                    "notation": measure.symbolic_data,
                    "analysis": measure_analysis.analysis_data if measure_analysis else None,
                }
        return list(evidence_by_measure.values())


def _parse_quarter_length(value: str) -> float:
    """symbolic_data stores durations/offsets as music21 quarterLength
    strings, which may be plain decimals ("1.5") or fractions ("5/3")."""
    return float(Fraction(value))


def _ordered_events(
    work_id: int, measure_start_index: int, measure_end_index: int, part_index: int = 0
) -> list[dict]:
    """One part's note/chord/rest events across an inclusive measure_index
    range, ordered by measure then offset within the measure."""
    with session_scope() as session:
        measures = (
            session.query(ScoreMeasure)
            .filter(
                ScoreMeasure.work_id == work_id,
                ScoreMeasure.measure_index >= measure_start_index,
                ScoreMeasure.measure_index <= measure_end_index,
            )
            .order_by(ScoreMeasure.measure_index)
            .all()
        )
        events: list[dict] = []
        for measure in measures:
            parts = measure.symbolic_data.get("parts", [])
            if part_index >= len(parts):
                continue
            events.extend(sorted(parts[part_index]["events"], key=lambda e: _parse_quarter_length(e["offset"])))
        return events


def extract_ordered_pitch_classes(
    work_id: int, measure_start_index: int, measure_end_index: int, part_index: int = 0
) -> list[int]:
    """Ordered pitch-class sequence for one part across a measure range.
    Chords contribute every pitch class they contain, in stored order, not
    just one note. Rests contribute nothing (no pitch to compare)."""
    pitch_classes: list[int] = []
    for event in _ordered_events(work_id, measure_start_index, measure_end_index, part_index):
        if event["kind"] == "note":
            pitch_classes.append(event["pitch"]["pitch_class"])
        elif event["kind"] == "chord":
            pitch_classes.extend(p["pitch_class"] for p in event["pitches"])
    return pitch_classes


def extract_ordered_rhythm(
    work_id: int, measure_start_index: int, measure_end_index: int, part_index: int = 0
) -> list[float]:
    """Ordered quarter-length duration per event for one part across a
    measure range. Notes, chords, and rests all count."""
    return [
        _parse_quarter_length(event["duration"]["quarter_length"])
        for event in _ordered_events(work_id, measure_start_index, measure_end_index, part_index)
    ]


def get_measure_total_durations(
    work_id: int, measure_start_index: int, measure_end_index: int, part_index: int = 0
) -> list[float]:
    """Total event duration per measure (one entry per measure_index in
    range), for loose rhythmic comparison that doesn't require identical
    subdivision of each measure."""
    with session_scope() as session:
        measures = (
            session.query(ScoreMeasure)
            .filter(
                ScoreMeasure.work_id == work_id,
                ScoreMeasure.measure_index >= measure_start_index,
                ScoreMeasure.measure_index <= measure_end_index,
            )
            .order_by(ScoreMeasure.measure_index)
            .all()
        )
        totals = []
        for measure in measures:
            parts = measure.symbolic_data.get("parts", [])
            if part_index >= len(parts):
                totals.append(0.0)
                continue
            totals.append(sum(
                _parse_quarter_length(e["duration"]["quarter_length"]) for e in parts[part_index]["events"]
            ))
        return totals


def get_max_measure_index(work_id: int) -> int | None:
    """Highest measure_index stored for a work, or None if it has no measures."""
    with session_scope() as session:
        return session.query(ScoreMeasure.measure_index).filter_by(work_id=work_id).order_by(
            ScoreMeasure.measure_index.desc()
        ).limit(1).scalar()


def get_measure_index(work_id: int, measure_number: int) -> int | None:
    """measure_index of a printed bar number, or None if the score has no such
    bar. Unnumbered measures all carry 0, so the first match wins."""
    with session_scope() as session:
        return (
            session.query(ScoreMeasure.measure_index)
            .filter_by(work_id=work_id, measure_number=measure_number)
            .order_by(ScoreMeasure.measure_index)
            .limit(1)
            .scalar()
        )


def get_global_key(work_id: int, measure_index: int) -> str | None:
    """The stored global_key (e.g. "f minor") analysis for one measure, or
    None if that measure has no analysis or no key was determined."""
    with session_scope() as session:
        row = (
            session.query(MeasureAnalysis.analysis_data)
            .join(ScoreMeasure, MeasureAnalysis.measure_id == ScoreMeasure.id)
            .filter(ScoreMeasure.work_id == work_id, ScoreMeasure.measure_index == measure_index)
            .order_by(MeasureAnalysis.created_at.desc(), MeasureAnalysis.id.desc())
            .first()
        )
        return row[0].get("global_key") if row else None


def get_local_key(work_id: int, measure_index: int) -> str | None:
    """The windowed local_key estimate for one measure, or None if absent."""
    with session_scope() as session:
        row = (
            session.query(MeasureAnalysis.analysis_data)
            .join(ScoreMeasure, MeasureAnalysis.measure_id == ScoreMeasure.id)
            .filter(ScoreMeasure.work_id == work_id, ScoreMeasure.measure_index == measure_index)
            .order_by(MeasureAnalysis.created_at.desc(), MeasureAnalysis.id.desc())
            .first()
        )
        return row[0].get("local_key") if row else None


def get_tempo_markings(work_id: int) -> list[tuple[int, str | None]]:
    """(measure_index, first MetronomeMark value found in that measure's
    stored directions) for every measure of a work, in order. A measure
    with no tempo direction of its own gets None (tempo unchanged)."""
    with session_scope() as session:
        rows = (
            session.query(ScoreMeasure.measure_index, MeasureAnalysis.analysis_data)
            .join(MeasureAnalysis, MeasureAnalysis.measure_id == ScoreMeasure.id)
            .filter(ScoreMeasure.work_id == work_id)
            .order_by(ScoreMeasure.measure_index)
            .all()
        )
        markings = []
        for measure_index, analysis_data in rows:
            tempo = None
            for direction in analysis_data.get("directions", []):
                if direction.get("type") == "MetronomeMark":
                    tempo = direction.get("value")
                    break
            markings.append((measure_index, tempo))
        return markings


_REPEAT_OPEN_BARLINE = re.compile(r"^=(\d+)")


def get_theme_repeat_open_index(work_id: int) -> int | None:
    """measure_index of the first repeat-open barline (Humdrum "|:") in the
    raw source, or None if the movement has none. When a movement opens
    with a slow introduction, the notated theme conventionally begins
    exactly at this barline, making it a more precise theme-start signal
    than tempo-marking changes.
    """
    with session_scope() as session:
        source = (
            session.query(ScoreSource)
            .filter_by(work_id=work_id)
            .order_by(ScoreSource.id.desc())
            .first()
        )
        if source is None:
            return None

        measure_number = None
        for line in source.raw_content.splitlines():
            if not line.startswith("="):
                continue
            first_field = line.split("\t", 1)[0]
            if "|:" not in first_field:
                continue
            match = _REPEAT_OPEN_BARLINE.match(first_field)
            if match:
                measure_number = int(match.group(1))
                break

        if measure_number is None:
            return None

        return (
            session.query(ScoreMeasure.measure_index)
            .filter_by(work_id=work_id, measure_number=measure_number)
            .order_by(ScoreMeasure.measure_index)
            .limit(1)
            .scalar()
        )


def store_segments(work_id: int, chunks: list[MeasureChunk]) -> None:
    """Embed all chunk summaries and bulk-insert into score_segments."""
    texts = [c.summary_text for c in chunks]
    vectors = embed_texts(texts)

    with session_scope() as session:
        for chunk, vec in zip(chunks, vectors):
            seg = ScoreSegment(
                work_id         = work_id,
                part            = chunk.part,
                measure_start   = chunk.measure_start,
                measure_end     = chunk.measure_end,
                local_key       = chunk.local_key,
                roman_numerals  = chunk.roman_numerals,
                harmonic_rhythm = chunk.harmonic_rhythm,
                texture_tag     = chunk.texture_tag,
                formal_function = chunk.formal_function,
                motif_tags      = chunk.motif_tags or [],
                summary_text    = chunk.summary_text,
                musicxml_slice  = chunk.musicxml_slice,
                embedding       = vec,
            )
            session.add(seg)

        session.commit()


def store_text_chunks(work_id: int, chunks: list[dict]) -> None:
    """
    chunks: list of dicts with keys: source_type, content, url (optional)
    Embeds each chunk and inserts into text_sources.
    """
    texts = [c["content"] for c in chunks]
    vectors = embed_texts(texts)

    with session_scope() as session:
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            ts = TextSource(
                work_id     = work_id,
                source_type = chunk["source_type"],
                content     = chunk["content"],
                chunk_index = i,
                embedding   = vec,
                url         = chunk.get("url"),
            )
            session.add(ts)

        session.commit()


def list_works() -> list[dict]:
    """
    List all ingested works (id, composer, title, opus, nickname, work_number,
    movement_number, tempo_indication) for the sidebar/work picker, in natural
    sonata/movement order (No. 5-9 before No. 10, No. 32 after No. 29) rather
    than lexicographic title order.
    """
    with session_scope() as session:
        rows = session.execute(text(r"""
            SELECT id, composer, title, opus, nickname,
                   work_number, movement_number, tempo_indication
            FROM works
            ORDER BY
                composer,
                COALESCE(work_number, 0),
                COALESCE(movement_number, 0),
                title
        """)).mappings().all()
        return [dict(r) for r in rows]


def get_work_mei(work_id: int) -> str | None:
    """Return the full MEI XML content for a work's generated MEI asset, or None."""
    with session_scope() as session:
        asset = (
            session.query(ScoreAsset)
            .filter_by(work_id=work_id, asset_type="mei")
            .order_by(ScoreAsset.id.desc())
            .first()
        )
        if not asset:
            return None
        path = Path(asset.file_path)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")


def clear_work_symbolic_layers(work_id: int) -> None:
    """Clear only reproducible symbolic source derivatives for a work."""
    with session_scope() as session:
        # Span analyses and relations cascade from their analysis run.
        session.query(AnalysisRun).filter_by(work_id=work_id).delete()
        measure_ids = [m.id for m in session.query(ScoreMeasure.id).filter_by(work_id=work_id)]
        if measure_ids:
            session.query(MeasureAnalysis).filter(MeasureAnalysis.measure_id.in_(measure_ids)).delete(
                synchronize_session=False
            )
        session.query(ScoreMeasure).filter_by(work_id=work_id).delete()
        session.query(ScoreSource).filter_by(work_id=work_id).delete()
        session.commit()


def clear_work_segments_and_assets(work_id: int) -> None:
    """Clear all derived records for a work to allow a complete re-ingestion."""
    clear_work_symbolic_layers(work_id)
    with session_scope() as session:
        session.query(ScoreSegment).filter_by(work_id=work_id).delete()
        # Delete text sources (like wikipedia or imslp text chunks)
        session.query(TextSource).filter_by(work_id=work_id).delete()
        # Delete all assets (PDF will be re-added by the ingestion script)
        session.query(ScoreAsset).filter_by(work_id=work_id).delete()
        session.commit()


def load_work_features(work_id: int, part_index: int = 0) -> list[dict]:
    """Every measure's comparison features for one work, in one query.

    Relation search compares a reference span against every same-length window
    in the movement, so a per-comparison query would issue tens of thousands of
    round trips for a single movement. Loading once and slicing in memory is
    what makes the pass tractable.

    Each entry: measure_index, measure_number, pitch_classes (ordered, chords
    contributing every pitch), rhythm (ordered quarter lengths), total_duration,
    and local_key.
    """
    with session_scope() as session:
        rows = (
            session.query(ScoreMeasure, MeasureAnalysis.analysis_data)
            .outerjoin(MeasureAnalysis, MeasureAnalysis.measure_id == ScoreMeasure.id)
            .filter(ScoreMeasure.work_id == work_id)
            .order_by(ScoreMeasure.measure_index, MeasureAnalysis.created_at.desc())
            .all()
        )
        features: list[dict] = []
        seen: set[int] = set()
        for measure, analysis_data in rows:
            if measure.measure_index in seen:
                continue  # keep only the newest analysis per measure
            seen.add(measure.measure_index)
            parts = measure.symbolic_data.get("parts", [])
            events = (
                sorted(parts[part_index]["events"], key=lambda e: _parse_quarter_length(e["offset"]))
                if part_index < len(parts) else []
            )
            pitch_classes: list[int] = []
            rhythm: list[float] = []
            for event in events:
                rhythm.append(_parse_quarter_length(event["duration"]["quarter_length"]))
                if event["kind"] == "note":
                    pitch_classes.append(event["pitch"]["pitch_class"])
                elif event["kind"] == "chord":
                    pitch_classes.extend(p["pitch_class"] for p in event["pitches"])
            features.append({
                "measure_index": measure.measure_index,
                "measure_number": measure.measure_number,
                "measure_role": measure.measure_role,
                "measure_belongs_to": measure.measure_belongs_to,
                "pitch_classes": pitch_classes,
                "rhythm": rhythm,
                "total_duration": sum(rhythm),
                "local_key": (analysis_data or {}).get("local_key"),
            })
        return features


def get_measure_shapes(work_id: int) -> list[tuple[int, int | None, float, float | None]]:
    """`(measure_index, printed number, sounding duration, bar length)` per
    measure -- what `analysis.numbering.assign_roles` needs, read back from the
    database so numbering can be re-derived without re-parsing the .krn.

    The meter is stored only where it changes, so it is forward-filled here the
    way the harmony pass forward-fills key signatures.
    """
    shapes: list[tuple[int, int | None, float, float | None]] = []
    bar_duration: float | None = None
    with session_scope() as session:
        rows = (
            session.query(ScoreMeasure, MeasureAnalysis)
            .outerjoin(MeasureAnalysis, MeasureAnalysis.measure_id == ScoreMeasure.id)
            .filter(ScoreMeasure.work_id == work_id)
            .order_by(ScoreMeasure.measure_index, MeasureAnalysis.id.desc())
            .all()
        )
        seen: set[int] = set()
        for measure, analysis in rows:
            if measure.measure_index in seen:
                continue
            seen.add(measure.measure_index)
            signature = (analysis.analysis_data or {}).get("time_signature") if analysis else None
            if signature:
                bar_duration = _bar_duration(signature) or bar_duration
            duration = 0.0
            for part in measure.symbolic_data.get("parts", []):
                part_duration = sum(
                    _parse_quarter_length(event["duration"]["quarter_length"])
                    for event in part.get("events", [])
                )
                duration = max(duration, part_duration)
            shapes.append((measure.measure_index, measure.measure_number or None,
                           duration, bar_duration))
    return shapes


def _bar_duration(time_signature: str) -> float | None:
    """Quarter-note length of one bar in a "6/8"-style signature."""
    try:
        beats, unit = time_signature.split("/")
        return int(beats) * 4.0 / int(unit)
    except (ValueError, ZeroDivisionError):
        return None


def store_measure_roles(work_id: int, numbering: list) -> int:
    """Apply a numbering pass to a work's stored measures.

    Rewrites `measure_number`, `measure_role` and `measure_belongs_to` only --
    the canonical notation itself is untouched, so this repairs an existing
    corpus without re-ingesting it.
    """
    updated = 0
    with session_scope() as session:
        by_index = {
            measure.measure_index: measure
            for measure in session.query(ScoreMeasure).filter_by(work_id=work_id).all()
        }
        for row in numbering:
            measure = by_index.get(row.measure_index)
            if measure is None:
                continue
            measure.measure_number = row.measure_number
            measure.measure_role = row.role
            measure.measure_belongs_to = row.belongs_to
            measure.symbolic_data = {
                **measure.symbolic_data,
                "measure_number": row.measure_number,
                "measure_role": row.role,
            }
            updated += 1
        session.commit()
    return updated


def get_span_candidates(work_id: int) -> list[dict]:
    """Candidate spans from the work's most recent boundary-analysis run."""
    with session_scope() as session:
        latest_run = (
            session.query(SpanAnalysis.analysis_run_id)
            .filter(SpanAnalysis.work_id == work_id)
            .order_by(SpanAnalysis.analysis_run_id.desc())
            .first()
        )
        if latest_run is None:
            return []
        spans = (
            session.query(SpanAnalysis)
            .filter(SpanAnalysis.work_id == work_id,
                    SpanAnalysis.analysis_run_id == latest_run[0])
            .order_by(SpanAnalysis.measure_start_index)
            .all()
        )
        return [{
            "id": span.id,
            "work_id": span.work_id,
            "measure_start_index": span.measure_start_index,
            "measure_end_index": span.measure_end_index,
            "measure_start": span.measure_start,
            "measure_end": span.measure_end,
            "span_type": span.span_type,
        } for span in spans]


def get_notated_sections(work_id: int) -> list[dict]:
    """The work's engraved section structure, newest run first.

    Stored as spans with `span_type='section'` by `build_sections.py`; read
    back here rather than re-parsing the .krn, so the chat layer sees exactly
    what the pipeline recorded.
    """
    with session_scope() as session:
        spans = (
            session.query(SpanAnalysis)
            .filter(SpanAnalysis.work_id == work_id, SpanAnalysis.span_type == "section")
            .order_by(SpanAnalysis.analysis_run_id.desc(), SpanAnalysis.measure_start_index)
            .all()
        )
        if not spans:
            return []
        latest = spans[0].analysis_run_id
        return [{
            "label": span.label,
            "measure_start": span.measure_start,
            "measure_end": span.measure_end,
            "measure_start_index": span.measure_start_index,
            "measure_end_index": span.measure_end_index,
            "evidence": span.evidence_data or {},
        } for span in spans if span.analysis_run_id == latest]


def get_span_relations(work_id: int) -> list[dict]:
    """Stored relations for one work, in printed bar numbers.

    Relations are keyed internally by measure_index; readers -- the evaluation
    harness, and eventually the chat layer -- need the printed numbers a user
    would type, so both are returned side by side.
    """
    with session_scope() as session:
        source_span = aliased(SpanAnalysis)
        target_span = aliased(SpanAnalysis)
        rows = (
            session.query(SpanRelation, source_span, target_span)
            .join(source_span, SpanRelation.source_span_id == source_span.id)
            .join(target_span, SpanRelation.target_span_id == target_span.id)
            .filter(source_span.work_id == work_id)
            .order_by(SpanRelation.confidence.desc())
            .all()
        )
        return [{
            "relation_type": relation.relation_type,
            "confidence": float(relation.confidence),
            "status": relation.status,
            "source_start_index": source.measure_start_index,
            "source_end_index": source.measure_end_index,
            "source_start": source.measure_start,
            "source_end": source.measure_end,
            "target_start_index": target.measure_start_index,
            "target_end_index": target.measure_end_index,
            "target_start": target.measure_start,
            "target_end": target.measure_end,
            "evidence": relation.evidence_data or {},
        } for relation, source, target in rows]


def store_span_relations(
    work_id: int, relations: list[dict], analyzer_version: str,
    configuration: dict | None = None,
) -> int:
    """Persist one symbolic-comparison run and the relations it proposed.

    A relation's target need not coincide with an existing candidate span --
    a thematic return rarely aligns with a boundary-derived segmentation -- so
    a target range without a span of its own gets one created here, as a
    `candidate`. Naming it a theme or a recapitulation would be a formal claim
    the comparison alone does not support.
    """
    with session_scope() as session:
        source = (session.query(ScoreSource).filter_by(work_id=work_id)
                  .order_by(ScoreSource.id.desc()).first())
        if source is None:
            raise ValueError(f"Cannot relate spans without a symbolic source for work {work_id}")
        run = AnalysisRun(
            work_id=work_id,
            analyzer_name="symbolic_span_comparison",
            analyzer_version=analyzer_version,
            configuration_data=configuration or {},
            source_sha256=source.sha256,
        )
        session.add(run)
        session.flush()

        existing = {
            (span.measure_start_index, span.measure_end_index): span.id
            for span in session.query(SpanAnalysis).filter_by(work_id=work_id).all()
        }

        def span_id_for(relation: dict, prefix: str) -> int:
            key = (relation[f"{prefix}_start_index"], relation[f"{prefix}_end_index"])
            if key not in existing:
                span = SpanAnalysis(
                    work_id=work_id,
                    analysis_run_id=run.id,
                    measure_start_index=key[0],
                    measure_end_index=key[1],
                    measure_start=relation[f"{prefix}_start"],
                    measure_end=relation[f"{prefix}_end"],
                    span_type="candidate",
                    confidence=relation["confidence"],
                    status="proposed",
                    evidence_data={"origin": "symbolic_span_comparison"},
                    features_data={},
                )
                session.add(span)
                session.flush()
                existing[key] = span.id
            return existing[key]

        for relation in relations:
            session.add(SpanRelation(
                analysis_run_id=run.id,
                source_span_id=span_id_for(relation, "source"),
                target_span_id=span_id_for(relation, "target"),
                relation_type=relation["relation_type"],
                confidence=relation["confidence"],
                status="proposed",
                evidence_data=relation["evidence"],
            ))
        session.commit()
        return run.id


def clear_work_span_relations(work_id: int) -> None:
    """Drop previous comparison runs for a work so the pass is re-runnable."""
    with session_scope() as session:
        runs = (session.query(AnalysisRun.id)
                .filter(AnalysisRun.work_id == work_id,
                        AnalysisRun.analyzer_name == "symbolic_span_comparison").all())
        for (run_id,) in runs:
            session.query(SpanRelation).filter_by(analysis_run_id=run_id).delete()
            session.query(SpanAnalysis).filter_by(analysis_run_id=run_id).delete()
            session.query(AnalysisRun).filter_by(id=run_id).delete()
        session.commit()


def get_source_text(work_id: int) -> str | None:
    """The work's raw Humdrum source, as stored at ingest."""
    with session_scope() as session:
        source = (session.query(ScoreSource.raw_content).filter_by(work_id=work_id)
                  .order_by(ScoreSource.id.desc()).first())
        return source[0] if source else None


def store_notated_sections(work_id: int, expansion: list, sections: list) -> int:
    """Persist notated sections as spans, replacing any previous section run.

    Sections are engraved, not inferred, so they are stored with span_type
    'section' and confidence 1.0 to distinguish them from derived candidates.
    """
    from analysis.sections import SECTION_ANALYSIS_VERSION, section_evidence

    with session_scope() as session:
        source = (session.query(ScoreSource).filter_by(work_id=work_id)
                  .order_by(ScoreSource.id.desc()).first())
        if source is None:
            raise ValueError(f"Cannot store sections without a symbolic source for work {work_id}")

        for (run_id,) in session.query(AnalysisRun.id).filter(
            AnalysisRun.work_id == work_id,
            AnalysisRun.analyzer_name == "notated_sections",
        ).all():
            session.query(SpanAnalysis).filter_by(analysis_run_id=run_id).delete()
            session.query(AnalysisRun).filter_by(id=run_id).delete()

        # Printed measure numbers come from the source; spans are keyed by
        # measure_index, so resolve through the stored measures. A number can
        # repeat (unnumbered measures are all 0), so the first wins.
        index_of: dict[int, int] = {}
        for number, index in session.query(
            ScoreMeasure.measure_number, ScoreMeasure.measure_index
        ).filter_by(work_id=work_id).order_by(ScoreMeasure.measure_index).all():
            index_of.setdefault(number, index)
        if not index_of:
            return 0

        run = AnalysisRun(
            work_id=work_id,
            analyzer_name="notated_sections",
            analyzer_version=SECTION_ANALYSIS_VERSION,
            configuration_data={"repeat_scheme": ",".join(expansion)},
            source_sha256=source.sha256,
        )
        session.add(run)
        session.flush()

        stored = 0
        for section in sections:
            start = index_of.get(section.measure_start)
            end = index_of.get(section.measure_end)
            if start is None or end is None:
                continue  # a label whose barlines are not in the parsed score
            session.add(SpanAnalysis(
                work_id=work_id,
                analysis_run_id=run.id,
                measure_start_index=start,
                measure_end_index=end,
                measure_start=section.measure_start,
                measure_end=section.measure_end,
                span_type="section",
                label=section.label,
                confidence=1.0,  # notated, not estimated
                status="proposed",
                evidence_data=section_evidence(expansion, sections, section),
                features_data={},
            ))
            stored += 1
        session.commit()
        return stored
