-- A measure that is not a bar.
--
-- `measure_number` was NOT NULL, so the importer's refusal to number a
-- measure was stored as the integer 0 -- for an opening anacrusis, for the
-- upbeat written after a repeat barline, for an unbarred cadenza, and for an
-- empty artefact alike. All four then read as a bar number and were printed as
-- one: a correctly located recapitulation was reported as "mm. 0-247".
--
-- NULL cannot be printed by accident. `measure_role` says which kind of
-- non-bar it is and `measure_belongs_to` the printed bar it is reported
-- against, so a range opening on a pickup still resolves to a citable bar.

ALTER TABLE score_measures ALTER COLUMN measure_number DROP NOT NULL;

ALTER TABLE score_measures
    ADD COLUMN IF NOT EXISTS measure_role       TEXT NOT NULL DEFAULT 'bar',
    ADD COLUMN IF NOT EXISTS measure_belongs_to INT;

ALTER TABLE score_measures DROP CONSTRAINT IF EXISTS score_measures_role_check;
ALTER TABLE score_measures ADD CONSTRAINT score_measures_role_check
    CHECK (measure_role IN ('bar', 'anacrusis', 'upbeat', 'unbarred', 'empty'));

-- A bar has a number; anything else does not. Enforced so the two columns
-- cannot drift apart.
ALTER TABLE score_measures DROP CONSTRAINT IF EXISTS score_measures_number_role_check;
ALTER TABLE score_measures ADD CONSTRAINT score_measures_number_role_check
    CHECK ((measure_role = 'bar') = (measure_number IS NOT NULL));

CREATE INDEX IF NOT EXISTS idx_score_measures_belongs_to
    ON score_measures (work_id, measure_belongs_to);
