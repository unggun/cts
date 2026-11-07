"""Paper traders pick up promoted versions without a restart."""
import os
import tempfile

import pytest

import src.execution.paper_trader as pt
import src.learning.filter_rules as fr
from src.data.database import get_connection, init_db, save_parameters


@pytest.fixture
def db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    fr.init_filter_rules_table(get_connection(path))
    monkeypatch.setattr(pt, "get_connection", lambda: get_connection(path))
    monkeypatch.setattr(fr, "get_connection", lambda: get_connection(path))
    yield path
    os.unlink(path)


def _trader():
    t = object.__new__(pt.PaperTrader)
    t.strategy = "macd"
    t.config = {"risk": {"max_positions": 3}}
    t.params = {"macd": {"fast_period": 3}, "risk": {"max_positions": 3}}
    t.param_version = 0
    t.filters = []
    t._notify = lambda msg: None
    return t


def test_reloads_params_when_new_version_saved(db):
    t = _trader()
    t._refresh_learned_state()
    assert t.params["macd"]["fast_period"] == 3          # nothing saved yet

    conn = get_connection(db)
    save_parameters(conn, "macd", {"macd": {"fast_period": 5}})
    t._refresh_learned_state()
    assert t.params["macd"]["fast_period"] == 5
    assert t.params["risk"] == {"max_positions": 3}      # risk comes from config
    assert t.param_version == 1


def test_refreshes_filters(db):
    t = _trader()
    fr.save_filter_rules("macd", [{"feature": "atr_pct", "operator": ">=", "threshold": 0.02,
                                   "effect_size": 0.8}], source="analyzer")
    t._refresh_learned_state()
    assert [f["feature"] for f in t.filters] == ["atr_pct"]
