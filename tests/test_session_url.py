"""db.session.normalise_url: the driver must be named, not left to SQLAlchemy's default."""
import pytest

from db.session import normalise_url


@pytest.mark.parametrize("url", ["postgresql://u:p@h:5432/d", "postgres://u:p@h:5432/d"])
def test_bare_schemes_get_the_psycopg2_driver(url):
    # SQLAlchemy 2.1 defaults a bare postgresql:// to psycopg 3, which we do not install.
    assert normalise_url(url) == "postgresql+psycopg2://u:p@h:5432/d"


def test_a_url_that_names_a_driver_is_left_alone():
    url = "postgresql+psycopg://u:p@h/d"
    assert normalise_url(url) == url
