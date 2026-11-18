"""db/session.py — SQLAlchemy session factory."""
import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_engine = None
_Session = None


def normalise_url(url: str) -> str:
    """Name the driver, and accept the `postgres://` URL hosted providers hand out.

    Neon, Supabase, Heroku and fly all print connection strings with the
    `postgres://` scheme, but SQLAlchemy 2 dropped that alias and fails with
    `Can't load plugin: sqlalchemy.dialects:postgres` — which reads like a
    missing driver rather than a one-word difference in a secret.

    A bare `postgresql://` is not safe either: SQLAlchemy 2.1 changed its
    default driver from psycopg2 to psycopg 3, so an unpinned URL fails with
    `No module named 'psycopg'` on a fresh install. We depend on psycopg2, so
    say so. A URL that already names a driver (`postgresql+psycopg://`) is left
    alone.
    """
    for scheme in ("postgres://", "postgresql://"):
        if url.startswith(scheme):
            return "postgresql+psycopg2://" + url[len(scheme):]
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
