from sqlalchemy import (
    Column, Integer, Float, Text, ARRAY, TIMESTAMP, ForeignKey, CheckConstraint,
    UniqueConstraint, func
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class Work(Base):
    __tablename__ = "works"

    id              = Column(Integer, primary_key=True)
    composer        = Column(Text, nullable=False)
    title           = Column(Text, nullable=False)
    opus            = Column(Text)
    nickname        = Column(Text)
    catalog_no      = Column(Text)
    work_number     = Column(Integer)
    movement_number = Column(Integer)
    tempo_indication = Column(Text)
    key_signature   = Column(Text)
    time_signature  = Column(Text)
    year_composed   = Column(Integer)
    instrumentation = Column(Text, default="solo piano")
    imslp_url       = Column(Text)
    wikipedia_url   = Column(Text)
    source_license  = Column(Text, default="public domain")
    created_at      = Column(TIMESTAMP(timezone=True), server_default=func.now())

    segments        = relationship("ScoreSegment", back_populates="work", cascade="all, delete")
    assets          = relationship("ScoreAsset",   back_populates="work", cascade="all, delete")
    text_sources    = relationship("TextSource",   back_populates="work", cascade="all, delete")
    sources         = relationship("ScoreSource",  back_populates="work", cascade="all, delete")
    measures        = relationship("ScoreMeasure", back_populates="work", cascade="all, delete")
    analysis_runs   = relationship("AnalysisRun",  back_populates="work", cascade="all, delete")
    spans           = relationship("SpanAnalysis", back_populates="work", cascade="all, delete")


class ScoreAsset(Base):
    __tablename__ = "score_assets"
    __table_args__ = (
        CheckConstraint("asset_type IN ('krn','mei')"),
    )

    id          = Column(Integer, primary_key=True)
    work_id     = Column(Integer, ForeignKey("works.id", ondelete="CASCADE"), nullable=False)
    asset_type  = Column(Text, nullable=False)
    file_path   = Column(Text, nullable=False)
    created_at  = Column(TIMESTAMP(timezone=True), server_default=func.now())

    work = relationship("Work", back_populates="assets")


class ScoreSource(Base):
    """Immutable copy and provenance of the symbolic source used for analysis."""
    __tablename__ = "score_sources"
    __table_args__ = (UniqueConstraint("work_id", "sha256", name="score_sources_work_sha256_key"),)

    id          = Column(Integer, primary_key=True)
    work_id     = Column(Integer, ForeignKey("works.id", ondelete="CASCADE"), nullable=False)
    format      = Column(Text, nullable=False, default="humdrum-kern")
    file_path   = Column(Text, nullable=False)
    source_url  = Column(Text)
    sha256      = Column(Text, nullable=False)
    raw_content = Column(Text, nullable=False)
    created_at  = Column(TIMESTAMP(timezone=True), server_default=func.now())

    work = relationship("Work", back_populates="sources")


class ScoreMeasure(Base):
    """Canonical, JSON-safe notation facts for one measure across all parts."""
    __tablename__ = "score_measures"
    __table_args__ = (UniqueConstraint("work_id", "measure_index", name="score_measures_work_index_key"),)

    id              = Column(Integer, primary_key=True)
    work_id         = Column(Integer, ForeignKey("works.id", ondelete="CASCADE"), nullable=False)
    measure_index   = Column(Integer, nullable=False)
    measure_number  = Column(Integer)          # NULL when the measure is not a bar
    measure_role    = Column(Text, nullable=False, default="bar")
    measure_belongs_to = Column(Integer)       # printed bar a non-bar is reported against
    symbolic_data   = Column(JSONB, nullable=False)
    created_at      = Column(TIMESTAMP(timezone=True), server_default=func.now())

    work = relationship("Work", back_populates="measures")
    analyses = relationship("MeasureAnalysis", back_populates="measure", cascade="all, delete")


class MeasureAnalysis(Base):
    """Versioned, reproducible per-measure analysis derived from score_measures."""
    __tablename__ = "measure_analyses"
    __table_args__ = (
        UniqueConstraint("measure_id", "analysis_version", name="measure_analyses_measure_version_key"),
    )

    id               = Column(Integer, primary_key=True)
    measure_id       = Column(Integer, ForeignKey("score_measures.id", ondelete="CASCADE"), nullable=False)
    analysis_version = Column(Text, nullable=False)
    analysis_data    = Column(JSONB, nullable=False)
    created_at       = Column(TIMESTAMP(timezone=True), server_default=func.now())

    measure = relationship("ScoreMeasure", back_populates="analyses")


class AnalysisRun(Base):
    """Provenance for a reproducible derived-analysis pass over one work."""
    __tablename__ = "analysis_runs"

    id                 = Column(Integer, primary_key=True)
    work_id            = Column(Integer, ForeignKey("works.id", ondelete="CASCADE"), nullable=False)
    analyzer_name      = Column(Text, nullable=False)
    analyzer_version   = Column(Text, nullable=False)
    configuration_data = Column(JSONB, nullable=False, default=dict)
    source_sha256      = Column(Text, nullable=False)
    created_at         = Column(TIMESTAMP(timezone=True), server_default=func.now())

    work = relationship("Work", back_populates="analysis_runs")
    spans = relationship("SpanAnalysis", back_populates="analysis_run", cascade="all, delete")
    relations = relationship("SpanRelation", back_populates="analysis_run", cascade="all, delete")


class SpanAnalysis(Base):
    """A variable-length, evidence-backed candidate or formal analysis span."""
    __tablename__ = "span_analyses"
    __table_args__ = (
        CheckConstraint("measure_start_index <= measure_end_index"),
        CheckConstraint("span_type IN ('candidate','section','phrase','theme','variation','transition')"),
        CheckConstraint("status IN ('proposed','accepted','rejected')"),
    )

    id            = Column(Integer, primary_key=True)
    work_id       = Column(Integer, ForeignKey("works.id", ondelete="CASCADE"), nullable=False)
    analysis_run_id = Column(Integer, ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    measure_start_index = Column(Integer, nullable=False)
    measure_end_index   = Column(Integer, nullable=False)
    measure_start = Column(Integer, nullable=False)
    measure_end   = Column(Integer, nullable=False)
    span_type     = Column(Text, nullable=False, default="candidate")
    label         = Column(Text)
    confidence    = Column(Float)
    status        = Column(Text, nullable=False, default="proposed")
    evidence_data = Column(JSONB, nullable=False)
    features_data = Column(JSONB, nullable=False)
    created_at    = Column(TIMESTAMP(timezone=True), server_default=func.now())

    work = relationship("Work", back_populates="spans")
    analysis_run = relationship("AnalysisRun", back_populates="spans")
    outgoing_relations = relationship("SpanRelation", foreign_keys="SpanRelation.source_span_id", back_populates="source_span")
    incoming_relations = relationship("SpanRelation", foreign_keys="SpanRelation.target_span_id", back_populates="target_span")


class SpanRelation(Base):
    """Evidence-backed relation between two variable-length analysis spans."""
    __tablename__ = "span_relations"
    __table_args__ = (
        CheckConstraint("relation_type IN ('repeats','varies','inverts','diminishes','augments','changes_meter_from','contrasts_with')"),
        CheckConstraint("status IN ('proposed','accepted','rejected')"),
    )

    id              = Column(Integer, primary_key=True)
    analysis_run_id = Column(Integer, ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    source_span_id  = Column(Integer, ForeignKey("span_analyses.id", ondelete="CASCADE"), nullable=False)
    target_span_id  = Column(Integer, ForeignKey("span_analyses.id", ondelete="CASCADE"), nullable=False)
    relation_type   = Column(Text, nullable=False)
    confidence      = Column(Float)
    status          = Column(Text, nullable=False, default="proposed")
    evidence_data   = Column(JSONB, nullable=False)
    created_at      = Column(TIMESTAMP(timezone=True), server_default=func.now())

    analysis_run = relationship("AnalysisRun", back_populates="relations")
    source_span = relationship("SpanAnalysis", foreign_keys=[source_span_id], back_populates="outgoing_relations")
    target_span = relationship("SpanAnalysis", foreign_keys=[target_span_id], back_populates="incoming_relations")


class ScoreSegment(Base):
    __tablename__ = "score_segments"
    __table_args__ = (
        CheckConstraint("difficulty BETWEEN 1 AND 10"),
    )

    id              = Column(Integer, primary_key=True)
    work_id         = Column(Integer, ForeignKey("works.id", ondelete="CASCADE"), nullable=False)
    part            = Column(Text, default="grand_staff")
    measure_start   = Column(Integer, nullable=False)
    measure_end     = Column(Integer, nullable=False)
    local_key       = Column(Text)
    roman_numerals  = Column(Text)
    harmonic_rhythm = Column(Text)
    texture_tag     = Column(Text)
    formal_function = Column(Text)
    motif_tags      = Column(ARRAY(Text))
    difficulty      = Column(Integer)
    summary_text    = Column(Text)
    musicxml_slice  = Column(Text)
    embedding       = Column(Vector(768))
    created_at      = Column(TIMESTAMP(timezone=True), server_default=func.now())

    work = relationship("Work", back_populates="segments")


class TextSource(Base):
    __tablename__ = "text_sources"
    __table_args__ = (
        CheckConstraint("source_type IN ('wikipedia','imslp','program_note','annotation')"),
    )

    id          = Column(Integer, primary_key=True)
    work_id     = Column(Integer, ForeignKey("works.id", ondelete="CASCADE"), nullable=False)
    source_type = Column(Text, nullable=False)
    content     = Column(Text, nullable=False)
    chunk_index = Column(Integer, default=0)
    embedding   = Column(Vector(768))
    url         = Column(Text)
    created_at  = Column(TIMESTAMP(timezone=True), server_default=func.now())

    work = relationship("Work", back_populates="text_sources")
