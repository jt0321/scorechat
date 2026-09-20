"""db/session.py — SQLAlchemy session factory."""
import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_engine = None
_Session = None


def normalise_url(url: str) -> str:
    """Accept the `postgres://` URL hosted providers hand out.

    Neon, Supabase, Heroku and fly all print connection strings with the
    `postgres://` scheme, but SQLAlchemy 2 dropped that alias and fails with
    `Can't load plugin: sqlalchemy.dialects:postgres` — which reads like a
    missing driver rather than a one-word difference in a secret.
    """
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url


def get_session():
    global _engine, _Session
    if _engine is None:
        url = normalise_url(os.environ["DATABASE_URL"])
        _engine = create_engine(url, pool_pre_ping=True)
        _Session = sessionmaker(bind=_engine)
    return _Session()


@contextmanager
def session_scope():
    """Yield a session and always close it, so short-lived requests
    (each API call, each ingest step) don't leak idle-in-transaction
    connections that can end up blocking DDL/migrations."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()
