"""
tests/conftest.py
Session setup shared by every test.

The DB-backed tests read DATABASE_URL from the environment, which lives in
`.env` for every other entry point -- each CLI calls load_dotenv() itself. A
plain `pytest` did not, so twenty tests failed with KeyError: 'DATABASE_URL'
unless the caller had exported it by hand. Loading it here makes the suite work
the way the rest of the project does. Existing environment variables win, so a
CI run or a deliberate `DATABASE_URL=... pytest` is not overridden.
"""

from dotenv import load_dotenv

load_dotenv(override=False)
