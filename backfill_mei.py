"""
backfill_mei.py
---------------
Copy already-rendered MEI files into `score_assets.content` (migration 009).

The MEI a browser renders was stored as a *path* into data/mei, which is derived
and gitignored: only a machine that has run a full ingest can serve a score. This
moves the text into the database so a deployed instance needs no filesystem. It
re-reads nothing and re-renders nothing — the file on disk is the file already
recorded — so it is safe to re-run.

    python backfill_mei.py
    python backfill_mei.py --check     # report coverage, write nothing
"""

import click
from dotenv import load_dotenv

load_dotenv()

from db.models import ScoreAsset
from db.session import session_scope
from db.store import resolve_repo_path


@click.command()
@click.option("--check", is_flag=True, help="Report coverage without writing.")
def main(check: bool) -> None:
    with session_scope() as session:
        assets = (
            session.query(ScoreAsset)
            .filter(ScoreAsset.asset_type == "mei")
            .order_by(ScoreAsset.id)
            .all()
        )
        stored = sum(1 for a in assets if a.content)
        pending = [a for a in assets if not a.content]
        click.echo(f"{len(assets)} MEI assets: {stored} already in the database, "
                   f"{len(pending)} to backfill.")
        if check or not pending:
            return

        missing = []
        for asset in pending:
            path = resolve_repo_path(asset.file_path)
            if not path.exists():
                missing.append(asset.file_path)
                continue
            asset.content = path.read_text(encoding="utf-8")
        session.commit()

    click.echo(f"✓ Backfilled {len(pending) - len(missing)} assets.")
    if missing:
        # An absent file is not recoverable here: re-render it with
        # `python ingest_scores.py`, which writes the content directly.
        click.echo(f"✗ {len(missing)} files were not on disk:")
        for path in missing[:10]:
            click.echo(f"    {path}")


if __name__ == "__main__":
    main()
