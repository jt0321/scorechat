-- 006: drop the retrieval layer.
--
-- score_segments held 4,519 prose summaries of measure windows, embedded to a
-- vector column; text_sources held the same idea for external prose. Nothing reads
-- either any more: answers come from tool calls over the canonical layer, in
-- which the measures are given by the question or found as an edge in
-- span_relations -- neither of which a similarity search over prose can reach.
--
-- Dropping them also drops the schema's only vector columns, and with them the
-- lock that made changing embedding provider a migration plus a full re-embed.
-- The score itself is untouched: it lives in score_measures.symbolic_data, and
-- score_segments never held anything that was not derived from it.

DROP TABLE IF EXISTS score_segments;
DROP TABLE IF EXISTS text_sources;

-- The extension is no longer used by any column. Left installed rather than
-- dropped, so an existing database keeps working if something outside this
-- schema depends on it; drop it by hand if nothing does:
--   DROP EXTENSION IF EXISTS vector;
