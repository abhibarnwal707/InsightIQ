import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.cache import store


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """Redirect the SQLite budget/cache DB to a throwaway file for this test."""
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(store, "_DB_PATH", db_file)
    store.init_db()
    return db_file
