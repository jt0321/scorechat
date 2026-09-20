-- 009: keep a generated asset's content in the database, not only its path.
--
-- score_assets stored a file_path, so serving a score meant the process could
-- reach data/mei/ on disk — a directory that is gitignored and only produced by
-- a full re-ingest. That is fine locally and impossible on a stateless host, so
-- the MEI text now lives here and file_path stays as provenance.
--
-- Backfill an existing database with `python backfill_mei.py` (no re-ingest).

ALTER TABLE score_assets ADD COLUMN IF NOT EXISTS content TEXT;
