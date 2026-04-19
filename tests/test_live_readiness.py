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
    original_dt = datetime
    class FakeDT:
        @classmethod
        def utcnow(cls):
            return original_dt(2026, 4, 19, 0, 0, 0)

        # Preserve fromisoformat for _parse_ts
        @staticmethod
        def fromisoformat(s):
            return original_dt.fromisoformat(s)
    monkeypatch.setattr(lr, "datetime", FakeDT)
    _insert_paper_trade(tmp_db, "dbw", "2026-03-01 10:00:00", 0.05)  # ~49 days old
    _insert_paper_trade(tmp_db, "dbw", "2026-04-18 10:00:00", 0.05)  # 1 day old
    stats = lr.load_paper_stats("dbw", None, db_path=tmp_db)
    assert stats["current"]["closed"] == 1


def test_load_paper_stats_handles_mixed_timestamp_formats(tmp_db):
    """entry_time is ISO-T format (from paper_trader); cutover is space format (from SQLite default).
    Comparison must be datetime-based, not lexicographic."""
    cutover_space = "2026-04-18 22:01:17"
    # Entry BEFORE cutover (05:00 < 22:01), T-format — must land in pre_update
    _insert_paper_trade(tmp_db, "dbw", "2026-04-18T05:00:00.123456", 0.04)
    # Entry AFTER cutover (23:53 > 22:01), T-format — must land in current
    _insert_paper_trade(tmp_db, "dbw", "2026-04-18T23:53:04.375191", 0.03)

    stats = lr.load_paper_stats("dbw", cutover_space, db_path=tmp_db)
    assert stats["pre_update"]["closed"] == 1
    assert stats["current"]["closed"] == 1


def test_load_mc_result_handles_malformed_details_json(tmp_db):
    conn = get_connection(tmp_db)
    conn.execute(
        "INSERT INTO simulation_results (run_id, strategy, num_paths, "
        "initial_capital, probability_of_ruin, verdict, details_json, created_at) "
        "VALUES (?, ?, 10000, 10000000, ?, ?, ?, ?)",
        ("test", "dbw", 0.02, "ROBUST", "NOT VALID JSON", "2026-04-18 22:00:00"),
    )
    conn.commit()
    conn.close()
    result = lr.load_mc_result("dbw", db_path=tmp_db)
    assert result["verdict"] == "ROBUST"
    assert result["num_source_trades"] == 0  # falls back to 0 on parse failure


def test_load_confidence_handles_malformed_recommendations_json(tmp_db):
    conn = get_connection(tmp_db)
    conn.execute(
        "INSERT INTO learning_sessions (session_type, strategy, num_trades_analyzed, "
        "recommendations_json, created_at) VALUES (?, ?, 100, ?, ?)",
        ("claude_review", "dbw", "NOT VALID JSON", "2026-04-18 22:00:00"),
    )
    conn.commit()
    conn.close()
    assert lr.load_confidence("dbw", db_path=tmp_db) is None


def _mc(verdict="ROBUST", ruin=0.01, trades=200):
    return {"verdict": verdict, "probability_of_ruin": ruin,
            "num_source_trades": trades, "created_at": "2026-04-18 22:00:00"}


def _paper(closed=30, wins=20, pnl=0.15):
    wr = wins / closed if closed else None
    return {
        "current": {"closed": closed, "wins": wins, "win_rate": wr, "total_pnl_pct": pnl},
        "pre_update": {"closed": 0, "wins": 0, "win_rate": None, "total_pnl_pct": 0.0},
    }


def test_evaluate_ready_all_gates_pass():
    r = lr.evaluate_strategy(_mc(), "high", _paper())
    assert r["status"] == "READY"


def test_evaluate_insufficient_data_when_few_mc_trades():
    r = lr.evaluate_strategy(_mc(trades=19), "high", _paper())
    assert r["status"] == "INSUFFICIENT_DATA"


def test_evaluate_insufficient_data_when_no_mc():
    r = lr.evaluate_strategy(None, "high", _paper())
    assert r["status"] == "INSUFFICIENT_DATA"


def test_evaluate_blocked_on_fragile_verdict():
    r = lr.evaluate_strategy(_mc(verdict="FRAGILE"), "high", _paper())
    assert r["status"] == "BLOCKED"


def test_evaluate_blocked_on_risky_verdict():
    r = lr.evaluate_strategy(_mc(verdict="RISKY"), "high", _paper())
    assert r["status"] == "BLOCKED"


def test_evaluate_blocked_on_high_ruin():
    r = lr.evaluate_strategy(_mc(ruin=0.05), "high", _paper())
    assert r["status"] == "BLOCKED"


def test_evaluate_blocked_on_low_confidence():
    r = lr.evaluate_strategy(_mc(), "low", _paper())
    assert r["status"] == "BLOCKED"


def test_evaluate_blocked_on_missing_confidence():
    r = lr.evaluate_strategy(_mc(), None, _paper())
    assert r["status"] == "BLOCKED"


def test_evaluate_blocked_on_negative_pnl_with_enough_samples():
    r = lr.evaluate_strategy(_mc(), "high", _paper(closed=30, wins=10, pnl=-0.05))
    assert r["status"] == "BLOCKED"


def test_evaluate_watch_when_sample_too_small():
    r = lr.evaluate_strategy(_mc(), "high", _paper(closed=5, wins=3, pnl=0.02))
    assert r["status"] == "WATCH"


def test_evaluate_watch_on_medium_confidence():
    r = lr.evaluate_strategy(_mc(), "medium", _paper())
    assert r["status"] == "WATCH"


def test_evaluate_watch_on_low_wr_but_positive_pnl():
    r = lr.evaluate_strategy(_mc(), "high", _paper(closed=30, wins=12, pnl=0.04))
    assert r["status"] == "WATCH"


def test_evaluate_gates_list_has_pass_flags():
    r = lr.evaluate_strategy(_mc(), "high", _paper())
    gate_names = {g["name"] for g in r["gates"]}
    assert "mc_verdict" in gate_names
    assert "probability_of_ruin" in gate_names
    assert "mc_source_trades" in gate_names
    assert "confidence" in gate_names
    assert "paper_trades" in gate_names
    assert "paper_win_rate" in gate_names
    assert "paper_total_pnl" in gate_names
    assert all(isinstance(g["pass"], bool) for g in r["gates"])
