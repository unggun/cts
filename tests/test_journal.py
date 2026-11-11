"""Research journal: holdout predictions vs paper results."""
import json
import os
import tempfile
import time
from datetime import datetime

import pytest

import src.learning.journal as j
from src.data.database import get_connection, init_db, save_parameters

CFG = {"simulation": {"initial_capital": 1_000_000}}
NOW = datetime(2026, 12, 1)


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    yield path
    os.unlink(path)


@pytest.fixture
def berlin_tz():
    old = os.environ.get("TZ")
    os.environ["TZ"] = "Europe/Berlin"
    time.tzset()
    yield
    if old is None:
        os.environ.pop("TZ")
    else:
        os.environ["TZ"] = old
    time.tzset()


def _version(conn, created_at, holdout=None, strategy="s"):
    perf = {"holdout": holdout} if holdout else {"win_rate": 0.5}
    v = save_parameters(conn, strategy, {"x": 1}, source="claude_auto_review", performance=perf)
    conn.execute("UPDATE strategy_parameters SET created_at = ? WHERE strategy = ? AND version = ?",
                 (created_at, strategy, v))
    conn.commit()
    return v, conn.execute("SELECT id FROM strategy_parameters WHERE strategy = ? AND version = ?",
                           (strategy, v)).fetchone()["id"]


def _trade(conn, entry, exit_=None, pnl=None, param_id=None, strategy="s", capital=1_000_000):
    conn.execute(
        "INSERT INTO trade_journal (mode, pair, strategy, entry_time, entry_price, exit_time, "
        "pnl_pct, pnl_absolute, param_id) VALUES ('paper', 'BTC/IDR', ?, ?, 1, ?, ?, ?, ?)",
        (strategy, entry, exit_, pnl, None if pnl is None else pnl * capital * 0.1, param_id))
    conn.commit()


# ── linking trades to versions ─────────────────────────────────────────

def test_param_id_wins_and_legacy_trades_are_split_by_local_time(db, berlin_tz):
    conn = get_connection(db)
    _, id1 = _version(conn, "2026-11-01 10:00:00")
    _, id2 = _version(conn, "2026-11-08 10:00:00")   # 11:00 local (CET)
    _trade(conn, "2026-11-08T10:30:00")              # local 10:30: before v2 started
    _trade(conn, "2026-11-08T11:30:00")              # local 11:30: after v2 started
    _trade(conn, "2026-11-02T09:00:00", param_id=id2)  # explicit link wins over time
    versions = j.load_versions(conn, "s")
    got = j.assign_versions(j.load_paper_trades(conn, "s"), versions)
    assert [t["entry_time"] for t in got[id1]] == ["2026-11-08T10:30:00"]
    assert sorted(t["entry_time"] for t in got[id2]) == ["2026-11-02T09:00:00",
                                                         "2026-11-08T11:30:00"]


# ── stats and verdict ──────────────────────────────────────────────────

def _closed(pnls, start="2026-11-10"):
    return [{"entry_time": f"{start}T00:00:00", "exit_time": f"{start}T{i % 24:02d}:00:00",
             "pnl_pct": p, "pnl_absolute": p * 100_000, "param_id": None}
            for i, p in enumerate(pnls)]


def test_paper_stats():
    trades = _closed([0.02, -0.01, 0.03, -0.02]) + [
        {"entry_time": "2026-11-11T00:00:00", "exit_time": None, "pnl_pct": None,
         "pnl_absolute": None, "param_id": None}]
    s = j.paper_stats(trades, datetime(2026, 11, 3), datetime(2026, 11, 17), 1_000_000)
    assert s["closed"] == 4 and s["open"] == 1
    assert s["expectancy"] == pytest.approx(0.005)
    assert s["win_rate"] == 0.5
    assert s["trades_per_week"] == pytest.approx(2.5)
    # equity 1.002M -> 1.001M -> 1.004M -> 1.002M: worst drop 2k from 1.004M
    assert s["max_drawdown"] == pytest.approx(2_000 / 1_004_000)


def test_verdicts():
    few = j.paper_stats(_closed([0.01] * 5), datetime(2026, 11, 1), None, 1e6, NOW)
    assert j.verdict(None, few) == "NO_PREDICTION"
    assert j.verdict({"expectancy": 0.003}, few) == "TOO_FEW"
    losing = j.paper_stats(_closed([0.004, -0.008] * 15), datetime(2026, 11, 1), None, 1e6, NOW)
    assert j.verdict({"expectancy": 0.003}, losing) == "BEHIND"
    noisy = j.paper_stats(_closed([0.05, -0.05] * 15), datetime(2026, 11, 1), None, 1e6, NOW)
    assert j.verdict({"expectancy": 0.003}, noisy) == "ON_TRACK"


def test_hypothesis_is_parsed_defensively():
    assert j.hypothesis({}) == {"idea": None, "expected_expectancy": None, "expected_trades": None}
    h = j.hypothesis({"hypothesis": {"idea": "skip strong trends",
                                     "expected_expectancy_pct": "0.25", "expected_trades": 80.0}})
    assert h == {"idea": "skip strong trends", "expected_expectancy": 0.0025,
                 "expected_trades": 80}
    bad = j.hypothesis({"hypothesis": {"idea": 3, "expected_expectancy_pct": "nan",
                                       "expected_trades": "lots"}})
    assert bad == {"idea": None, "expected_expectancy": None, "expected_trades": None}
    assert j.hypothesis({"hypothesis": "text"})["idea"] is None


# ── end to end ─────────────────────────────────────────────────────────

def test_build_and_format_journal(db):
    conn = get_connection(db)
    _version(conn, "2026-11-01 00:00:00")
    holdout = {"trades": 90, "expectancy": 0.003, "max_drawdown": 0.1}
    v2, id2 = _version(conn, "2026-11-08 00:00:00", holdout=holdout)
    for p in [0.004, -0.008] * 12:
        _trade(conn, "2026-11-09T00:00:00", "2026-11-10T00:00:00", p, param_id=id2)
    rec = {"confidence": "medium", "hypothesis": {"idea": "counter-trend fails in strong trends",
                                                  "expected_expectancy_pct": 0.25,
                                                  "expected_trades": 80}}
    ev = {"promoted": True, "version": v2, "param_id": id2, "reason": "better", "candidate": holdout,
          "incumbent": {"trades": 60, "expectancy": -0.001, "max_drawdown": 0.07}}
    for r, e in [(rec, ev), ({"confidence": "low"}, {"promoted": False, "reason": "nothing"}),
                 ({"confidence": "high"}, None)]:
        conn.execute("INSERT INTO learning_sessions (session_type, strategy, recommendations_json, "
                     "evaluation_json) VALUES ('claude_review', 's', ?, ?)",
                     (json.dumps(r), json.dumps(e) if e else None))
    conn.commit()

    out = j.build_journal("s", db_path=db, config=CFG, now=NOW)
    assert out["current"]["version"] == v2 and out["current"]["verdict"] == "BEHIND"
    assert [r["outcome"] for r in out["reviews"]] == [
        "no holdout test (before promotion testing)", "not evaluated: nothing",
        f"promoted → v{v2}"]
    text = j.format_strategy(out)
    assert "Idea: counter-trend fails in strong trends" in text
    assert "Predicted: +0.25%/trade, 80 trades" in text
    assert "Holdout (90d): 90 trades, +0.30%/trade, max DD 10.0%" in text
    assert "→ BEHIND" in text


def _review(conn, ev, created_at):
    ev = dict(ev, candidate={"trades": 90, "expectancy": 0.003}, incumbent={})
    conn.execute("INSERT INTO learning_sessions (session_type, strategy, recommendations_json, "
                 "evaluation_json, created_at) VALUES ('claude_review', 's', '{}', ?, ?)",
                 (json.dumps(ev), created_at))
    conn.commit()


def test_reviews_link_to_the_saved_row_not_a_reused_version_number(db):
    conn = get_connection(db)
    v1, old_id = _version(conn, "2026-10-01 00:00:30")
    _review(conn, {"promoted": True, "version": v1, "param_id": old_id, "reason": "old"},
            "2026-10-01 00:00:00")
    _review(conn, {"promoted": True, "version": v1, "reason": "legacy"}, "2026-10-01 00:00:00")
    # --reset-params deletes the rows; numbering restarts at 1 with a new row id.
    conn.execute("DELETE FROM strategy_parameters WHERE strategy = 's'")
    conn.commit()
    new_v, new_id = _version(conn, "2026-11-08 00:00:30", holdout={"trades": 90, "expectancy": 0.003})
    assert new_v == v1 and new_id != old_id
    _review(conn, {"promoted": True, "version": new_v, "param_id": new_id, "reason": "new"},
            "2026-11-08 00:00:00")
    _review(conn, {"promoted": True, "version": new_v, "reason": "new, legacy"},
            "2026-11-08 00:00:10")

    out = j.build_journal("s", db_path=db, config=CFG, now=NOW)
    linked = {r["session_id"]: r["param_id"] for r in out["reviews"]}
    assert sorted(linked.values(), key=str) == [new_id, new_id, None, None]
    text = j.format_strategy(out)
    assert text.count("Paper since:") == 2


def test_overview_and_behind_list(db, monkeypatch):
    monkeypatch.setattr(j, "KNOWN_STRATEGIES", ["s", "empty"])
    conn = get_connection(db)
    _, vid = _version(conn, "2026-11-08 00:00:00", holdout={"trades": 90, "expectancy": 0.003})
    for p in [0.004, -0.008] * 12:
        _trade(conn, "2026-11-09T00:00:00", "2026-11-10T00:00:00", p, param_id=vid)
    rows = j.overview(db_path=db, config=CFG, now=NOW)
    assert [r["strategy"] for r in rows] == ["s"]
    assert "BEHIND" in j.format_overview(rows)
    assert [r["strategy"] for r in j.behind_versions(db_path=db, config=CFG)] == ["s"]


# ── prompt, Telegram ───────────────────────────────────────────────────

def test_prompt_asks_for_hypothesis_and_prediction():
    from src.learning.claude_review import build_analysis_prompt
    prompt = build_analysis_prompt({"total_trades": 50}, {}, {"a": 1}, "macd", holdout_days=90)
    assert '"hypothesis": {"idea"' in prompt and "expected_expectancy_pct" in prompt


def test_weekly_summary_lists_behind_versions(monkeypatch):
    import src.run_learning_cycle as c
    behind = [{"strategy": "mean_reversion", "current": {
        "version": 4, "paper": {"expectancy": -0.004, "closed": 25},
        "holdout": {"expectancy": 0.003}}}]
    monkeypatch.setattr(j, "behind_versions", lambda: behind)
    text = "\n".join(c._journal_lines())
    assert "mean\\_reversion v4: paper -0.40% on 25 trades vs holdout +0.30%" in text
    monkeypatch.setattr(j, "behind_versions", lambda: [])
    assert c._journal_lines() == []


def test_telegram_journal_command(monkeypatch):
    from src.execution import telegram_bot as tb
    bot = object.__new__(tb.TelegramBot)
    monkeypatch.setattr(tb, "init_db", lambda: None)
    assert bot._cmd_journal(["nope"]).startswith("Unknown strategy: nope")
    monkeypatch.setattr(j, "build_journal", lambda name, limit: {"x": name})
    monkeypatch.setattr(j, "format_strategy", lambda d: "use `x` " + "y" * 5000)
    out = bot._cmd_journal(["cvd"])
    assert out.startswith("```\n") and out.endswith("\n```")
    assert out.count("`") == 6 and len(out) < 4096
