"""db/session.py — SQLAlchemy session factory."""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_engine = None
_Session = None


def get_session():
    global _engine, _Session
    if _engine is None:
        url = os.environ["DATABASE_URL"]
        _engine = create_engine(url, pool_pre_ping=True)
        _Session = sessionmaker(bind=_engine)
    return _Session()
