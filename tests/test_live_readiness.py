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
