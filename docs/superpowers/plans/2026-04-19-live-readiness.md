# Live-Readiness Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only CLI (`python -m src.execution.live_readiness`) that evaluates per-strategy readiness to flip from paper to live trading, based on MC verdict, Claude confidence, and current-param paper-trade performance.

**Architecture:** One module, `src/execution/live_readiness.py`, with three layers kept as separate functions for testability: (1) DB loaders that fetch raw rows by strategy, (2) a pure `evaluate_strategy(...)` function that maps inputs to a status dict, (3) formatters and `main()`. Tests use a temporary SQLite file populated with fixture rows; the project already exposes `get_connection(db_path=...)` and `init_db(db_path=...)` for this.

**Tech Stack:** Python 3, stdlib sqlite3 (already used everywhere in the project), pytest (existing test runner), argparse (matches `positions.py`).

---

## File Structure

- Create: `src/execution/live_readiness.py` — all loaders, evaluator, formatters, CLI entry
- Create: `tests/test_live_readiness.py` — unit tests for evaluator + integration tests against temp DB

All logic lives in one module because it's small (~200 lines) and mirrors `src/execution/positions.py`. Keep functions separable so each is unit-testable.

## Module Interface

```python
# Public API of src/execution/live_readiness.py

KNOWN_STRATEGIES = [
    "darvas", "dbw", "candlestick", "macd", "ma_crossover",
    "mean_reversion", "sr_breakout", "cvd", "bb_squeeze", "rsi_vwap",
]

def load_mc_result(strategy: str, db_path: str | None = None) -> dict | None:
    """Return latest simulation_results row for strategy as:
       {"verdict": str, "probability_of_ruin": float, "num_source_trades": int, "created_at": str}
       or None if no row exists."""

def load_confidence(strategy: str, db_path: str | None = None) -> str | None:
    """Return 'high'|'medium'|'low' from latest claude_review learning_session
       for strategy, or None if no session exists."""

def load_param_cutover(strategy: str, db_path: str | None = None) -> str | None:
    """Return the `created_at` (ISO string) of the most recent
       strategy_parameters row, or None if none exists."""

def load_paper_stats(strategy: str, cutover: str | None, db_path: str | None = None) -> dict:
    """Return {'current': {...}, 'pre_update': {...}} where each dict has
       closed, wins, win_rate (None if closed==0), total_pnl_pct.
       Trades older than 30 days are excluded. If cutover is None, all trades go to 'current'."""

def evaluate_strategy(mc: dict | None, confidence: str | None, paper_stats: dict) -> dict:
    """Pure function: given inputs, returns:
       {"status": "READY"|"WATCH"|"BLOCKED"|"INSUFFICIENT_DATA",
        "gates": [ {"name": str, "value": any, "pass": bool, "detail": str} ]}."""

def build_report(strategy: str, db_path: str | None = None) -> dict:
    """Load everything for one strategy and run evaluate_strategy.
       Return dict with keys: strategy, status, gates, mc, confidence, paper, cutover."""

def format_text(report: dict) -> str: ...
def format_summary(reports: list[dict]) -> str: ...
def main(): ...
```

---

## Task 1: Create empty module and test file, establish fixtures

**Files:**
- Create: `src/execution/live_readiness.py`
- Create: `tests/test_live_readiness.py`

- [ ] **Step 1: Create the module stub**

Write `src/execution/live_readiness.py`:

```python
"""Evaluate per-strategy readiness to flip from paper to live trading."""
import json
from datetime import datetime, timedelta

from src.data.database import get_connection, init_db

KNOWN_STRATEGIES = [
    "darvas", "dbw", "candlestick", "macd", "ma_crossover",
    "mean_reversion", "sr_breakout", "cvd", "bb_squeeze", "rsi_vwap",
]

MIN_MC_TRADES = 50
MAX_RUIN_PROB = 0.03
MIN_PAPER_TRADES = 30
MIN_PAPER_WR = 0.50
PAPER_LOOKBACK_DAYS = 30
```

- [ ] **Step 2: Create the test file with DB fixture helper**

Write `tests/test_live_readiness.py`:

```python
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
```

- [ ] **Step 3: Run the smoke test**

Run: `pytest tests/test_live_readiness.py::test_smoke_imports -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/execution/live_readiness.py tests/test_live_readiness.py
git commit -m "feat: scaffold live_readiness module and test fixtures"
```

---

## Task 2: Implement DB loader functions

**Files:**
- Modify: `src/execution/live_readiness.py`
- Modify: `tests/test_live_readiness.py`

- [ ] **Step 1: Write failing tests for `load_mc_result`**

Append to `tests/test_live_readiness.py`:

```python
def test_load_mc_result_returns_latest(tmp_db):
    _insert_mc(tmp_db, "dbw", "RISKY", 0.092, 166, when="2026-04-17 10:00:00")
    _insert_mc(tmp_db, "dbw", "ROBUST", 0.018, 200, when="2026-04-18 22:00:00")
    result = lr.load_mc_result("dbw", db_path=tmp_db)
    assert result["verdict"] == "ROBUST"
    assert result["probability_of_ruin"] == 0.018
    assert result["num_source_trades"] == 200


def test_load_mc_result_none_when_missing(tmp_db):
    assert lr.load_mc_result("unknown", db_path=tmp_db) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_live_readiness.py::test_load_mc_result_returns_latest -v`
Expected: FAIL with `AttributeError: module 'src.execution.live_readiness' has no attribute 'load_mc_result'`.

- [ ] **Step 3: Implement `load_mc_result`**

Append to `src/execution/live_readiness.py`:

```python
def load_mc_result(strategy: str, db_path: str = None) -> dict | None:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT verdict, probability_of_ruin, details_json, created_at "
        "FROM simulation_results WHERE strategy = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (strategy,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    details = json.loads(row["details_json"]) if row["details_json"] else {}
    return {
        "verdict": row["verdict"],
        "probability_of_ruin": row["probability_of_ruin"],
        "num_source_trades": int(details.get("num_source_trades", 0)),
        "created_at": row["created_at"],
    }
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `pytest tests/test_live_readiness.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Write failing tests for `load_confidence` and `load_param_cutover`**

Append to `tests/test_live_readiness.py`:

```python
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
```

- [ ] **Step 6: Run to verify failure**

Run: `pytest tests/test_live_readiness.py -v`
Expected: 4 new tests fail with AttributeError.

- [ ] **Step 7: Implement both loaders**

Append to `src/execution/live_readiness.py`:

```python
def load_confidence(strategy: str, db_path: str = None) -> str | None:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT recommendations_json FROM learning_sessions "
        "WHERE strategy = ? AND session_type = 'claude_review' "
        "ORDER BY created_at DESC LIMIT 1",
        (strategy,),
    ).fetchone()
    conn.close()
    if row is None or not row["recommendations_json"]:
        return None
    try:
        recs = json.loads(row["recommendations_json"])
    except (json.JSONDecodeError, TypeError):
        return None
    conf = recs.get("confidence")
    if conf in ("high", "medium", "low"):
        return conf
    return None


def load_param_cutover(strategy: str, db_path: str = None) -> str | None:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT created_at FROM strategy_parameters "
        "WHERE strategy = ? ORDER BY version DESC LIMIT 1",
        (strategy,),
    ).fetchone()
    conn.close()
    return row["created_at"] if row else None
```

- [ ] **Step 8: Run tests**

Run: `pytest tests/test_live_readiness.py -v`
Expected: all tests PASS.

- [ ] **Step 9: Write failing tests for `load_paper_stats`**

Append to `tests/test_live_readiness.py`:

```python
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
```

- [ ] **Step 10: Implement `load_paper_stats`**

Append to `src/execution/live_readiness.py`:

```python
def _empty_bucket() -> dict:
    return {"closed": 0, "wins": 0, "win_rate": None, "total_pnl_pct": 0.0}


def load_paper_stats(strategy: str, cutover: str | None, db_path: str = None) -> dict:
    cutoff = (datetime.utcnow() - timedelta(days=PAPER_LOOKBACK_DAYS)).isoformat(sep=" ")
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT entry_time, exit_time, pnl_pct FROM trade_journal "
        "WHERE mode = 'paper' AND strategy = ? AND entry_time >= ?",
        (strategy, cutoff),
    ).fetchall()
    conn.close()

    buckets = {"current": _empty_bucket(), "pre_update": _empty_bucket()}
    for row in rows:
        if row["exit_time"] is None:
            continue
        bucket_name = "current" if (cutover is None or row["entry_time"] >= cutover) else "pre_update"
        b = buckets[bucket_name]
        b["closed"] += 1
        pnl = row["pnl_pct"] or 0.0
        b["total_pnl_pct"] += pnl
        if pnl > 0:
            b["wins"] += 1

    for b in buckets.values():
        if b["closed"] > 0:
            b["win_rate"] = b["wins"] / b["closed"]
    return buckets
```

- [ ] **Step 11: Run tests**

Run: `pytest tests/test_live_readiness.py -v`
Expected: all tests PASS.

- [ ] **Step 12: Commit**

```bash
git add src/execution/live_readiness.py tests/test_live_readiness.py
git commit -m "feat: add DB loaders for live-readiness check"
```

---

## Task 3: Implement pure `evaluate_strategy` function

**Files:**
- Modify: `src/execution/live_readiness.py`
- Modify: `tests/test_live_readiness.py`

- [ ] **Step 1: Write failing tests covering every tier**

Append to `tests/test_live_readiness.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_live_readiness.py -v -k evaluate`
Expected: all new tests fail with AttributeError.

- [ ] **Step 3: Implement `evaluate_strategy`**

Append to `src/execution/live_readiness.py`:

```python
def evaluate_strategy(mc: dict | None, confidence: str | None, paper_stats: dict) -> dict:
    cur = paper_stats["current"]
    gates = []

    mc_verdict = mc["verdict"] if mc else None
    mc_ruin = mc["probability_of_ruin"] if mc else None
    mc_trades = mc["num_source_trades"] if mc else 0

    gates.append({"name": "mc_verdict", "value": mc_verdict,
                  "pass": mc_verdict == "ROBUST", "detail": "need ROBUST"})
    gates.append({"name": "probability_of_ruin", "value": mc_ruin,
                  "pass": mc_ruin is not None and mc_ruin < MAX_RUIN_PROB,
                  "detail": f"need < {MAX_RUIN_PROB:.0%}"})
    gates.append({"name": "mc_source_trades", "value": mc_trades,
                  "pass": mc_trades >= MIN_MC_TRADES,
                  "detail": f"need >= {MIN_MC_TRADES}"})
    gates.append({"name": "confidence", "value": confidence,
                  "pass": confidence == "high", "detail": "need high"})
    gates.append({"name": "paper_trades", "value": cur["closed"],
                  "pass": cur["closed"] >= MIN_PAPER_TRADES,
                  "detail": f"need >= {MIN_PAPER_TRADES} closed (current params)"})
    wr_pass = cur["win_rate"] is not None and cur["win_rate"] >= MIN_PAPER_WR
    gates.append({"name": "paper_win_rate", "value": cur["win_rate"],
                  "pass": wr_pass, "detail": f"need >= {MIN_PAPER_WR:.0%}"})
    gates.append({"name": "paper_total_pnl", "value": cur["total_pnl_pct"],
                  "pass": cur["total_pnl_pct"] > 0, "detail": "need > 0"})

    # Status tiers — first match wins.
    if mc is None or mc_trades < MIN_MC_TRADES:
        status = "INSUFFICIENT_DATA"
    elif (
        mc_verdict in ("FRAGILE", "RISKY")
        or (mc_ruin is not None and mc_ruin >= MAX_RUIN_PROB)
        or confidence in (None, "low")
        or (cur["closed"] >= MIN_PAPER_TRADES and cur["total_pnl_pct"] <= 0)
    ):
        status = "BLOCKED"
    elif (
        cur["closed"] < MIN_PAPER_TRADES
        or confidence == "medium"
        or (cur["win_rate"] is not None and cur["win_rate"] < MIN_PAPER_WR
            and cur["total_pnl_pct"] > 0)
    ):
        status = "WATCH"
    else:
        status = "READY"

    return {"status": status, "gates": gates}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_live_readiness.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/execution/live_readiness.py tests/test_live_readiness.py
git commit -m "feat: add evaluate_strategy pure function with full tier coverage"
```

---

## Task 4: Implement `build_report`, formatters, and CLI

**Files:**
- Modify: `src/execution/live_readiness.py`
- Modify: `tests/test_live_readiness.py`

- [ ] **Step 1: Write failing tests for `build_report` and formatters**

Append to `tests/test_live_readiness.py`:

```python
def test_build_report_integrates_all_sources(tmp_db):
    _insert_mc(tmp_db, "ma_crossover", "ROBUST", 0.018, 300)
    _insert_session(tmp_db, "ma_crossover", "high")
    _insert_params(tmp_db, "ma_crossover", 1, when="2026-04-18 22:00:00")
    report = lr.build_report("ma_crossover", db_path=tmp_db)
    assert report["strategy"] == "ma_crossover"
    assert report["status"] in {"READY", "WATCH", "BLOCKED", "INSUFFICIENT_DATA"}
    assert "gates" in report
    assert report["mc"]["verdict"] == "ROBUST"
    assert report["confidence"] == "high"


def test_build_report_no_data_is_insufficient_data(tmp_db):
    report = lr.build_report("unknown_strategy", db_path=tmp_db)
    assert report["status"] == "INSUFFICIENT_DATA"
    assert report["mc"] is None


def test_format_text_contains_strategy_and_status():
    report = {
        "strategy": "dbw", "status": "BLOCKED",
        "mc": {"verdict": "RISKY", "probability_of_ruin": 0.09, "num_source_trades": 166},
        "confidence": "high",
        "paper": {"current": {"closed": 0, "wins": 0, "win_rate": None, "total_pnl_pct": 0.0},
                  "pre_update": {"closed": 5, "wins": 3, "win_rate": 0.6, "total_pnl_pct": 0.1}},
        "cutover": "2026-04-18 22:00:00",
        "gates": [{"name": "mc_verdict", "value": "RISKY", "pass": False, "detail": "need ROBUST"}],
    }
    out = lr.format_text(report)
    assert "dbw" in out
    assert "BLOCKED" in out
    assert "RISKY" in out


def test_format_summary_counts_by_status():
    reports = [
        {"strategy": "a", "status": "READY"},
        {"strategy": "b", "status": "WATCH"},
        {"strategy": "c", "status": "BLOCKED"},
        {"strategy": "d", "status": "BLOCKED"},
        {"strategy": "e", "status": "INSUFFICIENT_DATA"},
    ]
    out = lr.format_summary(reports)
    assert "1 READY" in out
    assert "1 WATCH" in out
    assert "2 BLOCKED" in out
    assert "1 INSUFFICIENT_DATA" in out
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_live_readiness.py -v -k "build_report or format"`
Expected: all 4 new tests fail.

- [ ] **Step 3: Implement `build_report` and formatters**

Append to `src/execution/live_readiness.py`:

```python
def build_report(strategy: str, db_path: str = None) -> dict:
    mc = load_mc_result(strategy, db_path=db_path)
    confidence = load_confidence(strategy, db_path=db_path)
    cutover = load_param_cutover(strategy, db_path=db_path)
    paper = load_paper_stats(strategy, cutover, db_path=db_path)
    evaluation = evaluate_strategy(mc, confidence, paper)
    return {
        "strategy": strategy,
        "status": evaluation["status"],
        "gates": evaluation["gates"],
        "mc": mc,
        "confidence": confidence,
        "paper": paper,
        "cutover": cutover,
    }


_STATUS_ICON = {"READY": "🟢", "WATCH": "🟡", "BLOCKED": "🔴", "INSUFFICIENT_DATA": "⚪"}


def _fmt_pct(value):
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def _fmt_bucket(bucket: dict) -> str:
    c = bucket["closed"]
    if c == 0:
        return "0 closed"
    return f"{c} closed, {_fmt_pct(bucket['win_rate'])} WR, {bucket['total_pnl_pct'] * 100:+.2f}% PnL"


def format_text(report: dict) -> str:
    lines = []
    icon = _STATUS_ICON.get(report["status"], "")
    lines.append(f"\nStrategy: {report['strategy']:<25s}  Status: {icon} {report['status']}")
    lines.append("  " + "─" * 56)
    for g in report["gates"]:
        mark = "✓" if g["pass"] else "✗"
        value_str = g["value"] if g["value"] is not None else "n/a"
        if isinstance(g["value"], float):
            value_str = f"{g['value']:.4f}" if g["name"] != "probability_of_ruin" else _fmt_pct(g["value"])
        lines.append(f"  {g['name']:<22s} {str(value_str):<12s} {mark}  ({g['detail']})")
    lines.append(f"  Paper (current params)  {_fmt_bucket(report['paper']['current'])}")
    lines.append(f"  Paper (pre-update)      {_fmt_bucket(report['paper']['pre_update'])}  (informational)")
    if report["cutover"]:
        lines.append(f"  Param version since ... {report['cutover']}")
    return "\n".join(lines)


def format_summary(reports: list) -> str:
    counts = {"READY": 0, "WATCH": 0, "BLOCKED": 0, "INSUFFICIENT_DATA": 0}
    for r in reports:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return (f"\nSummary: {counts['READY']} READY | {counts['WATCH']} WATCH | "
            f"{counts['BLOCKED']} BLOCKED | {counts['INSUFFICIENT_DATA']} INSUFFICIENT_DATA")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_live_readiness.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Add `main()` and CLI entrypoint**

Append to `src/execution/live_readiness.py`:

```python
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Per-strategy live-trading readiness check")
    parser.add_argument("--strategy", help="Evaluate a single strategy")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    init_db()
    strategies = [args.strategy] if args.strategy else KNOWN_STRATEGIES
    reports = [build_report(s) for s in strategies]

    if args.json:
        payload = {
            "reports": reports,
            "summary": {
                status: sum(1 for r in reports if r["status"] == status)
                for status in ("READY", "WATCH", "BLOCKED", "INSUFFICIENT_DATA")
            },
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        for r in reports:
            print(format_text(r))
        print(format_summary(reports))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run CLI smoke test on the live DB**

Run: `python3 -m src.execution.live_readiness`
Expected: script prints one report block per strategy ending with a summary line. No exceptions. Verify that `candlestick` shows `BLOCKED` (because yesterday's MC verdict was FRAGILE) and `ma_crossover` shows `WATCH` or `BLOCKED` (ROBUST but with tiny current-param paper sample).

- [ ] **Step 7: Run CLI smoke test with `--strategy` filter**

Run: `python3 -m src.execution.live_readiness --strategy dbw`
Expected: exactly one strategy report for dbw, then summary line showing `1` total.

- [ ] **Step 8: Run CLI smoke test with `--json` flag**

Run: `python3 -m src.execution.live_readiness --json | python3 -m json.tool > /dev/null && echo OK`
Expected: prints `OK` (output is valid JSON).

- [ ] **Step 9: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all pre-existing tests still pass, plus all live_readiness tests pass.

- [ ] **Step 10: Commit**

```bash
git add src/execution/live_readiness.py tests/test_live_readiness.py
git commit -m "feat: add live-readiness CLI (python -m src.execution.live_readiness)"
```

---

## Task 5: Update CLAUDE.md with the new command

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Read current command section**

Read the "Commands" block of `/opt/cts/CLAUDE.md`, specifically the area around the `positions.py` commands.

- [ ] **Step 2: Add the new command alongside `positions`**

Insert after the `python3 -m src.execution.positions --json` line:

```bash
# Check live-trading readiness (per strategy: READY / WATCH / BLOCKED / INSUFFICIENT_DATA)
python3 -m src.execution.live_readiness
python3 -m src.execution.live_readiness --strategy dbw
python3 -m src.execution.live_readiness --json
```

- [ ] **Step 3: Add an Architecture subsection**

Insert a new subsection near "Position Queries":

```markdown
### Live-Readiness Check

`src/execution/live_readiness.py` evaluates per-strategy readiness to flip from paper to live trading. Read-only, checks four gates: Monte Carlo verdict/ruin probability, Claude confidence, and current-param paper-trade stats (split around the latest `strategy_parameters.created_at` cutover; trades older than 30 days excluded). Status tiers: `INSUFFICIENT_DATA` (MC sample < 50) → `BLOCKED` → `WATCH` → `READY`. Flipping to live remains manual.
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document live-readiness CLI in CLAUDE.md"
```

---

## Self-Review Notes

Spec coverage check:
- CLI shape, `--strategy`, `--json` → Task 4 steps 5–8.
- Four data sources → Task 2 loaders (1 per source).
- Paper-trade partitioning (30-day lookback, cutover split) → Task 2 step 9–11.
- Seven gates with thresholds → Task 3 `evaluate_strategy` builds 7-gate list.
- Status tiers including `INSUFFICIENT_DATA` and medium-confidence → WATCH rule → Task 3 tests cover every tier.
- Output format showing per-strategy block + footer summary → Task 4 `format_text` + `format_summary`.
- Scope boundary "no schema changes / read-only / no auto-flip" → no migration step; `main()` only reads.
- Testing approach from spec (smoke test on live DB, single-strategy filter, --json parseable) → Task 4 steps 6–8.

No placeholders, no "TBD" strings, all code shown verbatim, types consistent (`dict | None`, same field names across loaders and evaluator).
