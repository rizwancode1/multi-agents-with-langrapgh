"""
Shared pytest configuration.

Points SQLAlchemy at an isolated temporary SQLite database for the whole
test session. This must happen in conftest.py (imported before any test
module) so app.config caches Settings with the test DATABASE_URL before
app.db creates its engine.
"""

import os
import tempfile

_TEMP_DB = os.path.join(tempfile.gettempdir(), "multi_agents_test.db")
if os.path.exists(_TEMP_DB):
    os.remove(_TEMP_DB)
os.environ["DATABASE_URL"] = f"sqlite:///{_TEMP_DB}"
