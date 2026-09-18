-- 008: claims read out of commentary passages, and what the score says.
--
-- Rebuilt wholesale by build_claims.py from text_passages and the stored
-- analysis; nothing here is written by hand, and nothing here is read by the
-- symbolic passes. See commentary/claims.py for why a key claim can be
-- supported by the engraved key but never contradicted by it.

CREATE TABLE IF NOT EXISTS passage_claims (
    id              SERIAL PRIMARY KEY,
    passage_id      INT NOT NULL REFERENCES text_passages(id) ON DELETE CASCADE,
    claim_type      TEXT NOT NULL CHECK (claim_type IN ('key', 'bar_ref', 'form')),
    value           TEXT NOT NULL,          -- "D-flat major", "measure 19", "second theme"
    quote           TEXT NOT NULL,          -- the sentence it was read from
    check_status    TEXT NOT NULL CHECK (check_status IN
                        ('supported', 'agrees_with_estimate', 'disagrees_with_estimate', 'not_checkable')),
    check_evidence  JSONB NOT NULL DEFAULT '{}',
    UNIQUE (passage_id, claim_type, value)
);
CREATE INDEX IF NOT EXISTS passage_claims_status_idx ON passage_claims (claim_type, check_status);
