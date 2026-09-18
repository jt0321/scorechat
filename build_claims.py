"""
build_claims.py
Read claims out of the commentary passages and check them against the score.

DB-only, like build_relations.py: it reads text_passages and the stored
analysis, and rebuilds passage_claims wholesale. Key claims are checked --
first against each movement's engraved key, then against the estimated key
regions -- and the result is reported both ways: how the commentary fares
against the score, and, where the estimator disagrees with every commentator,
where the estimator may be wrong.

Usage:
    python build_claims.py                 # rebuild and summarise
    python build_claims.py --disagreements # list key claims the estimate does not bear out
"""

import json
from collections import Counter

import click
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text

from commentary.claims import MovementKeys, check_claim, extract_claims, movements_for
from commentary.store import load_catalogue
from db.session import session_scope
from pipeline.analysis_api import get_key_plan


def movement_keys() -> dict[int, list[MovementKeys]]:
    """Every sonata's movements with their engraved key and estimated regions."""
    with session_scope() as session:
        rows = session.execute(text("""
            SELECT w.id, w.work_number, w.movement_number,
                   (SELECT ma.analysis_data->>'global_key'
                    FROM score_measures sm JOIN measure_analyses ma ON ma.measure_id = sm.id
                    WHERE sm.work_id = w.id ORDER BY sm.measure_index LIMIT 1) AS declared
            FROM works w WHERE w.work_number IS NOT NULL ORDER BY w.work_number, w.movement_number
        """)).mappings().all()
    by_sonata: dict[int, list[MovementKeys]] = {}
    for row in rows:
        plan = get_key_plan(row["id"])
        by_sonata.setdefault(row["work_number"], []).append(MovementKeys(
            row["id"], row["movement_number"], row["declared"], tuple(plan.get("regions") or ())))
    return by_sonata


@click.command()
@click.option("--disagreements", is_flag=True, help="List key claims the estimate does not bear out")
def main(disagreements):
    keys = movement_keys()
    with session_scope() as session:
        passages = session.execute(text("""
            SELECT p.id, p.content, p.sonata, p.work_ids, p.anchor_status, p.page, p.line,
                   d.author FROM text_passages p JOIN text_documents d ON d.id = p.document_id
            ORDER BY p.id""")).mappings().all()

    catalogue = load_catalogue()
    rows = []
    for passage in passages:
        passage = dict(passage)
        sonata_movements = keys.get(passage["sonata"], [])
        sonata = catalogue.sonatas.get(passage["sonata"])
        for claim in extract_claims(passage["content"]):
            movements, placed_by = movements_for(claim, passage, sonata_movements, sonata)
            check = check_claim(claim, passage, movements)
            if claim.claim_type == "key" and check.status != "not_checkable":
                check.evidence["placed_by"] = placed_by
            rows.append({"passage": passage, "claim": claim, "check": check})

    with session_scope() as session:
        session.execute(text("DELETE FROM passage_claims"))
        for row in rows:
            session.execute(text("""
                INSERT INTO passage_claims (passage_id, claim_type, value, quote, check_status, check_evidence)
                VALUES (:pid, :type, :value, :quote, :status, CAST(:evidence AS jsonb))"""),
                {"pid": row["passage"]["id"], "type": row["claim"].claim_type,
                 "value": row["claim"].value, "quote": row["claim"].quote,
                 "status": row["check"].status, "evidence": json.dumps(row["check"].evidence)})
        session.commit()

    by_type = Counter(r["claim"].claim_type for r in rows)
    click.echo(f"{len(rows)} claims from {len(passages)} passages: "
               + ", ".join(f"{t} {n}" for t, n in by_type.most_common()))
    key_checks = Counter(r["check"].status for r in rows if r["claim"].claim_type == "key")
    click.echo("key claims: " + ", ".join(f"{s} {n}" for s, n in key_checks.most_common()))

    if disagreements:
        click.echo("\nKey claims the estimated key plan does not bear out:")
        for row in rows:
            if row["check"].status == "disagrees_with_estimate":
                p, c, e = row["passage"], row["claim"], row["check"].evidence
                where = f"p. {p['page']}" if p["page"] else f"line {p['line']}"
                rel = "  (relative of an estimated key)" if e.get("relative_of") else ""
                click.echo(f"  sonata {p['sonata']:>2}  {p['author'].split()[-1]:9} {where:>8}  "
                           f"{c.value:16} estimated {e['estimated_keys']}{rel}")


if __name__ == "__main__":
    main()
