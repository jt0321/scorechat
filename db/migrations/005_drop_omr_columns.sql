-- 005_drop_omr_columns.sql
--
-- Removes the last of the scanned-score/OMR scope the project dropped when it
-- moved to symbolic Humdrum sources. score_assets only ever holds the .krn
-- source and the MEI rendered from it: no PDF or page image is ingested and no
-- OMR tool runs, so page_number, omr_tool and omr_quality are always NULL and
-- the wider asset_type list describes an ingest path that does not exist.

ALTER TABLE score_assets
    DROP COLUMN IF EXISTS page_number,
    DROP COLUMN IF EXISTS omr_tool,
    DROP COLUMN IF EXISTS omr_quality;

DELETE FROM score_assets WHERE asset_type NOT IN ('krn', 'mei');

ALTER TABLE score_assets DROP CONSTRAINT IF EXISTS score_assets_asset_type_check;
ALTER TABLE score_assets ADD CONSTRAINT score_assets_asset_type_check
    CHECK (asset_type IN ('krn', 'mei'));
