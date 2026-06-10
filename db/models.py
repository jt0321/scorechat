from sqlalchemy import (
    Column, Integer, Text, ARRAY, TIMESTAMP, ForeignKey, CheckConstraint, func
)
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
    catalog_no      = Column(Text)
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


class ScoreAsset(Base):
    __tablename__ = "score_assets"
    __table_args__ = (
        CheckConstraint("asset_type IN ('pdf','page_image','musicxml','mei','midi')"),
    )

    id          = Column(Integer, primary_key=True)
    work_id     = Column(Integer, ForeignKey("works.id", ondelete="CASCADE"), nullable=False)
    asset_type  = Column(Text, nullable=False)
    file_path   = Column(Text, nullable=False)
    page_number = Column(Integer)
    omr_tool    = Column(Text)
    omr_quality = Column(Text)
    created_at  = Column(TIMESTAMP(timezone=True), server_default=func.now())

    work = relationship("Work", back_populates="assets")


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
    embedding       = Column(Vector(1536))
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
    embedding   = Column(Vector(1536))
    url         = Column(Text)
    created_at  = Column(TIMESTAMP(timezone=True), server_default=func.now())

    work = relationship("Work", back_populates="text_sources")
