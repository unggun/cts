"""Propose -> verify -> promote."""
import json
import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import src.backtest.engine as eng_mod
import src.learning.promotion as pr
from src.data.database import get_connection, init_db

CFG = pr.promotion_config({})


def _m(trades, exp, dd=0.10):
    return {"trades": trades, "expectancy": exp, "win_rate": 0.5, "max_drawdown": dd}


# ── decide ──────────────────────────────────────────────────────────────

def test_promotes_clear_improvement():
    ok, reason = pr.decide(_m(40, -0.002), _m(35, 0.001), CFG, changed=True)
    assert ok, reason


def test_rejects_small_improvement_below_margin():
    ok, reason = pr.decide(_m(40, 0.0010), _m(40, 0.0012), CFG, changed=True)
    assert not ok and "expectancy" in reason


def test_rejects_too_few_holdout_trades():
    ok, reason = pr.decide(_m(40, -0.01), _m(10, 0.02), CFG, changed=True)
    assert not ok and "holdout trades" in reason


def test_rejects_filters_that_starve_trading():
    ok, reason = pr.decide(_m(100, -0.001), _m(30, 0.005), CFG, changed=True)
    assert not ok and "%" in reason


def test_rejects_worse_drawdown():
    ok, reason = pr.decide(_m(40, 0.0, dd=0.10), _m(40, 0.002, dd=0.20), CFG, changed=True)
    assert not ok and "drawdown" in reason


def test_incumbent_without_trades_needs_positive_expectancy():
    assert pr.decide(_m(3, 0.0), _m(25, 0.001), CFG, changed=True)[0]
    assert not pr.decide(_m(3, 0.0), _m(25, -0.001), CFG, changed=True)[0]


def test_rejects_unchanged_bundle():
    ok, reason = pr.decide(_m(40, -0.01), _m(40, 0.01), CFG, changed=False)
    assert not ok and "no change" in reason


# ── config / split ──────────────────────────────────────────────────────

def test_promotion_config_overrides():
    cfg = pr.promotion_config({"learning": {"promotion": {"holdout_days": 60}}})
    assert cfg["holdout_days"] == 60 and cfg["min_holdout_trades"] == 20


def test_holdout_start():
    now = datetime(2026, 11, 7, 12, 0)
    assert pr.holdout_start({"learning": {"promotion": {"holdout_days": 10}}}, now) \
        == pd.Timestamp("2026-10-28 12:00")


# ── candidate filters / bundle change ───────────────────────────────────

def test_candidate_filters_merges_sources_and_caps():
    review = {"filter_rules": [{"rule": "Skip when rsi > 70", "type": "skip"}]}
    findings = {"suggested_filters": [
        {"rule": "atr_pct >= 0.02", "feature": "atr_pct", "effect_size": 0.9},
        {"rule": "rsi >= 65", "feature": "rsi", "effect_size": 0.6},
        {"rule": "hour >= 12", "feature": "hour", "effect_size": 0.3},
    ]}
    rules = pr.candidate_filters(review, findings, limit=2)
    assert [r["feature"] for r in rules] == ["atr_pct", "rsi"]
    assert all(r["rule_type"] == "skip" and r["source"] for r in rules)


def test_bundle_changed():
    f = [{"feature": "a", "operator": ">=", "threshold": 1.0}]
    assert not pr.bundle_changed({"x": 1}, f, {"x": 1}, [dict(f[0], threshold="1.0")])
    assert pr.bundle_changed({"x": 1}, f, {"x": 2}, f)
    assert pr.bundle_changed({"x": 1}, f, {"x": 1}, [])


# ── evaluate_bundle ─────────────────────────────────────────────────────

@pytest.fixture
def fake_strategy(monkeypatch):
    def detect(df, params):
        df = df.copy()
        df["sig"] = None
        df.iloc[::10, df.columns.get_loc("sig")] = "buy"
        return df

    def levels(df, i, params):
        c = df.iloc[i]["close"]
        return {"entry": c, "stop_loss": c - 5, "target": c + params.get("tp", 0.5),
                "risk": 5, "risk_reward": 0.1}

    monkeypatch.setitem(eng_mod.STRATEGY_REGISTRY, "fake", (detect, "sig", levels))
    monkeypatch.setattr(eng_mod, "load_active_filters", lambda s: [])
    monkeypatch.setattr(eng_mod, "extract_features", lambda df, i: {"x": 1.0})


def _dfs(n=400):
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    c = np.full(n, 100.0)
    df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0},
                      index=idx)
    return {"BTC/IDR": df, "ETH/IDR": df.copy()}


def test_evaluate_bundle_only_counts_holdout_and_charges_costs(fake_strategy):
    dfs = _dfs()
    start = dfs["BTC/IDR"].index[300]
    params = {"fake": {"min_risk_reward": 0.0}, "risk": {"max_position_pct": 0.05}}
    m = pr.evaluate_bundle("fake", params, [], dfs, "1h", start,
                           {"fee_pct": 0.001, "slippage_pct": 0.0}, 1_000_000)
    assert m["trades"] == 2 * 10          # bars 300..399, every 10th, two pairs
    # +0.5 on 100 minus 0.2% fees
    assert m["expectancy"] == pytest.approx(0.005 - 0.002, abs=2e-4)
    json.dumps(m)                         # serializable for evaluation_json


# ── DB: filters + evaluation history ────────────────────────────────────

@pytest.fixture
def db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    monkeypatch.setattr(pr, "get_connection", lambda: get_connection(path))
    import src.learning.filter_rules as fr
    monkeypatch.setattr(fr, "get_connection", lambda: get_connection(path))
    fr.init_filter_rules_table(get_connection(path))
    yield path
    os.unlink(path)


def test_replace_active_filters_makes_set_exclusive(db):
    from src.learning.filter_rules import load_active_filters, save_filter_rules
    save_filter_rules("s", [{"feature": "old", "operator": ">=", "threshold": 1}], source="analyzer")
    pr.replace_active_filters("s", [{"feature": "new", "operator": "<=", "threshold": 2,
                                      "rule_type": "skip", "effect_size": 0.5,
                                      "rationale": "r", "source": "claude_review"}])
    assert [f["feature"] for f in load_active_filters("s", limit=None)] == ["new"]


def test_record_and_load_recent_evaluations(db):
    conn = get_connection(db)
    for i in range(4):
        cur = conn.execute(
            "INSERT INTO learning_sessions (session_type, strategy, recommendations_json) "
            "VALUES ('claude_review', 's', ?)",
            (json.dumps({"parameter_changes": [{"param": "p", "old": i, "new": i + 1}],
                         "filter_rules": [{"rule": f"Skip when x > {i}"}]}),))
        conn.commit()
        pr.record_evaluation(cur.lastrowid, {"promoted": i == 3, "reason": f"r{i}",
                                             "incumbent": _m(40, 0.0), "candidate": _m(40, 0.001)})
    hist = pr.load_recent_evaluations("s", n=3)
    assert [h["reason"] for h in hist] == ["r3", "r2", "r1"]
    assert hist[0]["promoted"] is True
    assert hist[0]["parameter_changes"][0]["new"] == 4
    row = conn.execute("SELECT applied FROM learning_sessions ORDER BY id DESC LIMIT 1").fetchone()
    assert row["applied"] == 1


def test_promotion_telegram_lines(monkeypatch):
    import src.run_learning_cycle as c
    monkeypatch.setattr(c, "format_param_diff", lambda s: "sl_atr_multiplier: 7 → 9")
    promoted = {"promotion": {"promoted": True, "version": 4, "reason": "expectancy +0.1%",
                              "candidate_filters": ["atr_pct >= 0.02"]}}
    text = "\n".join(c._promotion_lines("macd", promoted))
    assert "Promoted v4" in text and "sl_atr_multiplier" in text and "atr\\_pct" in text
    rejected = {"promotion": {"promoted": False, "reason": "only 5 holdout trades"}}
    assert "Not promoted: only 5 holdout trades" in c._promotion_lines("macd", rejected)[0]
    assert c._promotion_lines("macd", {}) == []
