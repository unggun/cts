import json
import os
import tempfile
from datetime import datetime, timedelta

import pytest

from src.data.database import get_connection, init_db
from src.execution import live_readiness as lr


@pytest.fixture
def tmp_db():
    """Create a fresh SQLite DB with the real schema for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    yield path
    os.unlink(path)


def _insert_mc(db_path, strategy, verdict, ruin, num_trades, when=None):
    when = when or datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO simulation_results (run_id, strategy, num_paths, "
        "initial_capital, probability_of_ruin, verdict, details_json, created_at) "
        "VALUES (?, ?, 10000, 10000000, ?, ?, ?, ?)",
        ("test", strategy, ruin, verdict,
         json.dumps({"num_source_trades": num_trades}), when),
    )
    conn.commit()
    conn.close()


def _insert_session(db_path, strategy, confidence, when=None):
    when = when or datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO learning_sessions (session_type, strategy, num_trades_analyzed, "
        "recommendations_json, created_at) VALUES (?, ?, 100, ?, ?)",
        ("claude_review", strategy, json.dumps({"confidence": confidence}), when),
    )
    conn.commit()
    conn.close()


def _insert_params(db_path, strategy, version, when):
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO strategy_parameters (strategy, version, parameters_json, created_at) "
        "VALUES (?, ?, ?, ?)",
        (strategy, version, "{}", when),
    )
    conn.commit()
    conn.close()


def _insert_paper_trade(db_path, strategy, entry_time, pnl_pct, closed=True):
    exit_time = entry_time if closed else None
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO trade_journal (mode, pair, strategy, entry_time, entry_price, "
        "exit_time, pnl_pct) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("paper", "BTC/IDR", strategy, entry_time, 1000000.0, exit_time,
         pnl_pct if closed else None),
    )
    conn.commit()
    conn.close()


def test_smoke_imports():
    assert lr.KNOWN_STRATEGIES
    assert lr.MIN_PAPER_TRADES == 30


def test_load_mc_result_returns_latest(tmp_db):
    _insert_mc(tmp_db, "dbw", "RISKY", 0.092, 166, when="2026-04-17 10:00:00")
    _insert_mc(tmp_db, "dbw", "ROBUST", 0.018, 200, when="2026-04-18 22:00:00")
    result = lr.load_mc_result("dbw", db_path=tmp_db)
    assert result["verdict"] == "ROBUST"
    assert result["probability_of_ruin"] == 0.018
    assert result["num_source_trades"] == 200


def test_load_mc_result_none_when_missing(tmp_db):
    assert lr.load_mc_result("unknown", db_path=tmp_db) is None


def test_load_confidence_picks_latest_claude_review(tmp_db):
    _insert_session(tmp_db, "dbw", "medium", when="2026-04-17 22:00:00")
    _insert_session(tmp_db, "dbw", "high",   when="2026-04-18 22:00:00")
    assert lr.load_confidence("dbw", db_path=tmp_db) == "high"


def test_load_confidence_none_when_missing(tmp_db):
    assert lr.load_confidence("dbw", db_path=tmp_db) is None


def test_load_param_cutover_returns_latest(tmp_db):
    _insert_params(tmp_db, "dbw", 1, when="2026-04-10 10:00:00")
    _insert_params(tmp_db, "dbw", 2, when="2026-04-18 22:00:00")
    assert lr.load_param_cutover("dbw", db_path=tmp_db) == "2026-04-18 22:00:00"


def test_load_param_cutover_none_when_missing(tmp_db):
    assert lr.load_param_cutover("dbw", db_path=tmp_db) is None


def test_load_paper_stats_splits_around_cutover(tmp_db):
    cutover = "2026-04-18 22:00:00"
    # Pre-update: 2 closed, 1 win
    _insert_paper_trade(tmp_db, "dbw", "2026-04-15 10:00:00", 0.05)
    _insert_paper_trade(tmp_db, "dbw", "2026-04-16 10:00:00", -0.02)
    # Current: 3 closed, 2 wins
    _insert_paper_trade(tmp_db, "dbw", "2026-04-19 08:00:00", 0.03)
    _insert_paper_trade(tmp_db, "dbw", "2026-04-19 09:00:00", 0.04)
    _insert_paper_trade(tmp_db, "dbw", "2026-04-19 10:00:00", -0.01)

    stats = lr.load_paper_stats("dbw", cutover, db_path=tmp_db)
    assert stats["pre_update"]["closed"] == 2
    assert stats["pre_update"]["wins"] == 1
    assert stats["pre_update"]["win_rate"] == pytest.approx(0.5)
    assert stats["pre_update"]["total_pnl_pct"] == pytest.approx(0.03)
    assert stats["current"]["closed"] == 3
    assert stats["current"]["wins"] == 2
    assert stats["current"]["win_rate"] == pytest.approx(2 / 3)
    assert stats["current"]["total_pnl_pct"] == pytest.approx(0.06)


def test_load_paper_stats_no_cutover_everything_current(tmp_db):
    _insert_paper_trade(tmp_db, "dbw", "2026-04-19 10:00:00", 0.03)
    stats = lr.load_paper_stats("dbw", None, db_path=tmp_db)
    assert stats["current"]["closed"] == 1
    assert stats["pre_update"]["closed"] == 0


def test_load_paper_stats_ignores_open_trades(tmp_db):
    _insert_paper_trade(tmp_db, "dbw", "2026-04-19 10:00:00", 0.0, closed=False)
    stats = lr.load_paper_stats("dbw", None, db_path=tmp_db)
    assert stats["current"]["closed"] == 0
    assert stats["current"]["win_rate"] is None


def test_load_paper_stats_excludes_trades_older_than_30_days(tmp_db, monkeypatch):
    # Freeze "now" to 2026-04-19
    class FakeDT:
        @classmethod
        def utcnow(cls):
            return datetime(2026, 4, 19, 0, 0, 0)
    monkeypatch.setattr(lr, "datetime", FakeDT)
    _insert_paper_trade(tmp_db, "dbw", "2026-03-01 10:00:00", 0.05)  # ~49 days old
    _insert_paper_trade(tmp_db, "dbw", "2026-04-18 10:00:00", 0.05)  # 1 day old
    stats = lr.load_paper_stats("dbw", None, db_path=tmp_db)
    assert stats["current"]["closed"] == 1
