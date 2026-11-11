"""Fixes from the PR #2 review: atomic promotion, safe proposals, consistent reloads."""
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import src.backtest.engine as eng_mod
import src.backtest.runner as runner
import src.learning.filter_rules as fr
import src.learning.promotion as pr
from src.backtest.engine import BacktestEngine
from src.data.database import get_connection, init_db, save_parameters, upsert_ohlcv
from src.learning.claude_review import build_analysis_prompt, format_history


@pytest.fixture
def db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    fr.init_filter_rules_table(get_connection(path))
    monkeypatch.setattr(pr, "get_connection", lambda: get_connection(path))
    monkeypatch.setattr(fr, "get_connection", lambda: get_connection(path))
    yield path
    os.unlink(path)


@pytest.fixture
def fake_strategy(monkeypatch):
    """Buy every 10th bar; target ``tp`` above entry (hit next bar if <= 1)."""
    def detect(df, params):
        df = df.copy()
        df["sig"] = None
        df.iloc[::10, df.columns.get_loc("sig")] = "buy"
        return df

    def levels(df, i, params):
        if params.get("boom"):
            raise ValueError("bad params")
        c = df.iloc[i]["close"]
        return {"entry": c, "stop_loss": c - 5, "target": c + params.get("tp", 0.5),
                "risk": 5, "risk_reward": 0.1}

    monkeypatch.setitem(eng_mod.STRATEGY_REGISTRY, "fake", (detect, "sig", levels))
    monkeypatch.setattr(eng_mod, "extract_features", lambda df, i: {"x": 1.0})


def _candles(n, end=None, freq_h=1):
    """Flat 100-price candles; ``end`` is the open time of the last one."""
    end = end or pd.Timestamp("2026-03-01")
    idx = pd.date_range(end=end, periods=n, freq=f"{freq_h}h")
    return [[int(t.value // 10**6), 100.0, 101.0, 99.0, 100.0, 1.0] for t in idx], idx


CONFIG = {
    "trading": {"pairs": ["BTC/IDR"], "timeframes": ["1h"]},
    "strategy": {"fake": {"min_risk_reward": 0.0, "tp": 0.5}},
    "risk": {"max_position_pct": 0.05},
    "costs": {"fee_pct": 0.0, "slippage_pct": 0.0},
}


def _seed(db, n=600, end=None):
    rows, idx = _candles(n, end)
    upsert_ohlcv(get_connection(db), "BTC/IDR", "1h", rows)
    return idx


def _versions(db):
    return get_connection(db).execute("SELECT COUNT(*) FROM strategy_parameters").fetchone()[0]


# ── promotion ───────────────────────────────────────────────────────────

def test_promotion_commits_params_and_filters_together(db, fake_strategy, monkeypatch):
    idx = _seed(db)
    review = {"updated_parameters": {"min_risk_reward": 0.0, "tp": 1.0}}
    real_replace = pr.replace_active_filters

    def fail(*a, **kw):
        raise RuntimeError("disk full")
    monkeypatch.setattr(pr, "replace_active_filters", fail)
    with pytest.raises(RuntimeError):
        pr.run_promotion("fake", review, {}, CONFIG, idx[300])
    assert _versions(db) == 0          # the parameter row was rolled back

    monkeypatch.setattr(pr, "replace_active_filters", real_replace)
    ev = pr.run_promotion("fake", review, {}, CONFIG, idx[300])
    assert ev["promoted"] and _versions(db) == 1
    row = get_connection(db).execute("SELECT id, version FROM strategy_parameters").fetchone()
    assert (ev["param_id"], ev["version"]) == (row[0], row[1])


def test_invalid_parameters():
    inc = {"rsi_vwap": {"rsi_period": 14, "atr_mult": 1.5, "threshold": 0, "on": True}}
    assert pr.invalid_parameters(inc, {"rsi_vwap": {"rsi_period": -2}}) == ["rsi_vwap.rsi_period"]
    assert pr.invalid_parameters(inc, {"rsi_vwap": {"rsi_period": 0}}) == ["rsi_vwap.rsi_period"]
    assert pr.invalid_parameters(inc, {"rsi_vwap": {"rsi_period": 13.5}}) == ["rsi_vwap.rsi_period"]
    assert pr.invalid_parameters(inc, {"rsi_vwap": {"atr_mult": "wide"}}) == ["rsi_vwap.atr_mult"]
    assert pr.invalid_parameters(inc, {"rsi_vwap": {"rsi_period": 21, "atr_mult": 2.0,
                                                    "threshold": -1, "on": False}}) == []


def test_rejects_invalid_or_crashing_proposals(db, fake_strategy):
    idx = _seed(db)
    bad = pr.run_promotion("fake", {"updated_parameters": {"min_risk_reward": 0.0, "tp": -1}},
                           {}, CONFIG, idx[300])
    assert not bad["promoted"] and "invalid parameter values: tp" in bad["reason"]
    boom = pr.run_promotion("fake", {"updated_parameters": {"tp": 1.0, "boom": True}},
                            {}, CONFIG, idx[300])
    assert not boom["promoted"] and "candidate backtest failed" in boom["reason"]
    assert _versions(db) == 0


def test_promotion_without_regime_data(db, fake_strategy, monkeypatch):
    idx = _seed(db)
    config = dict(CONFIG, trading=dict(CONFIG["trading"], regime_pair="ETH/IDR"))
    seen = []
    real = pr.evaluate_bundle
    monkeypatch.setattr(pr, "evaluate_bundle",
                        lambda *a, **kw: seen.append(kw["btc_df"]) or real(*a, **kw))
    btc_rule = {"filter_rules": [{"rule": "Skip when btc_return_24h <= -0.02",
                                  "type": "skip"}]}
    ev = pr.run_promotion("fake", btc_rule, {}, config, idx[300])
    # No ETH/IDR candles: no BTC/IDR stand-in, and a btc_* rule can't be tested.
    assert seen and all(b.empty for b in seen)
    assert not ev["promoted"] and "no ETH/IDR candles to evaluate btc_return_24h" in ev["reason"]
    assert _versions(db) == 0


def test_promotion_ignores_forming_candle(db, fake_strategy, monkeypatch):
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).floor("h")
    idx = _seed(db, n=100, end=now)           # last candle opened this hour: still forming
    seen = []
    monkeypatch.setattr(pr, "evaluate_bundle",
                        lambda s, p, f, dfs, *a, **kw: seen.append(dfs["BTC/IDR"].index[-1])
                        or {"trades": 0, "expectancy": 0.0, "win_rate": 0.0,
                            "max_drawdown": 0.0, "per_pair": {}})
    pr.run_promotion("fake", {}, {}, CONFIG, idx[50], dry_run=True)
    assert seen and all(t == idx[-2] for t in seen)


def test_history_shows_evaluated_filters_and_unevaluated_proposals(db):
    import json
    conn = get_connection(db)
    for rec, ev in [
        ({"filter_rules": [{"rule": "Skip when rsi > 70"}]},
         {"promoted": False, "reason": "expectancy", "incumbent": {}, "candidate": {},
          "candidate_filters": ["atr_pct >= 0.02", "rsi > 70"]}),
        ({"filter_rules": [{"rule": "Skip when vol < 1"}]},
         {"promoted": False, "reason": "Claude confidence is low"}),
    ]:
        cur = conn.execute("INSERT INTO learning_sessions (session_type, strategy, "
                           "recommendations_json) VALUES ('claude_review', 's', ?)",
                           (json.dumps(rec),))
        conn.commit()
        pr.record_evaluation(cur.lastrowid, ev)
    low, evaluated = pr.load_recent_evaluations("s")
    assert evaluated["filter_rules"] == ["atr_pct >= 0.02", "rsi > 70"]
    assert low["evaluated"] is False
    text = format_history([low, evaluated])
    assert "NOT EVALUATED — Claude confidence is low" in text
    assert "atr_pct >= 0.02" in text


def test_prompt_states_effective_promotion_rule():
    cfg = pr.promotion_config({"learning": {"promotion": {"min_holdout_trades": 40,
                                                          "min_expectancy_gain": 0.001}}})
    prompt = build_analysis_prompt({}, {}, {}, "macd", holdout_days=90,
                                   promotion_rule=pr.describe_rule(cfg))
    assert "at least 40 holdout trades" in prompt and "0.10%" in prompt
    assert "only needs positive expectancy" in prompt
    default = build_analysis_prompt({}, {}, {}, "macd", holdout_days=90)
    assert "at least 20 holdout trades" in default


# ── engine / runner ─────────────────────────────────────────────────────

def _df(n=200):
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    c = np.full(n, 100.0)
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0},
                        index=idx)


def test_drawdown_includes_liquidation_costs(monkeypatch):
    def detect(df, params):
        df = df.copy()
        df["sig"] = None
        df.iloc[-5, df.columns.get_loc("sig")] = "buy"
        return df
    monkeypatch.setitem(eng_mod.STRATEGY_REGISTRY, "hold", (
        detect, "sig",
        lambda df, i, p: {"entry": 100.0, "stop_loss": 50.0, "target": 200.0,
                          "risk": 50.0, "risk_reward": 2.0}))
    monkeypatch.setattr(eng_mod, "extract_features", lambda df, i: {})
    e = BacktestEngine("hold", {"hold": {"min_risk_reward": 0.0}}, 1_000_000,
                       costs={"fee_pct": 0.01, "slippage_pct": 0.0}, filters=[])
    res = e.run(_df(), "BTC/IDR", "1h")
    assert e.trades[0]["exit_reason"] == "end_of_data"
    assert e.equity_curve[-1]["equity"] == pytest.approx(e.capital)
    assert res["max_drawdown"] > 0           # exit fee shows up as drawdown


def test_indicators_warm_up_before_holdout(monkeypatch):
    """A signal that needs pre-holdout bars is still found once trading starts."""
    def detect(df, params):
        df = df.copy()
        sma = df["close"].rolling(100).mean()
        df["sig"] = np.where(sma.notna() & ~sma.shift(1).notna(), "buy", None)
        return df
    monkeypatch.setitem(eng_mod.STRATEGY_REGISTRY, "warm", (
        detect, "sig",
        lambda df, i, p: {"entry": 100.0, "stop_loss": 95.0, "target": 100.5,
                          "risk": 5.0, "risk_reward": 0.1}))
    monkeypatch.setattr(eng_mod, "extract_features", lambda df, i: {})
    df = _df()
    e = BacktestEngine("warm", {"warm": {"min_risk_reward": 0.0}}, 1_000_000,
                       costs={"fee_pct": 0.0, "slippage_pct": 0.0}, filters=[])
    e.run(df, "BTC/IDR", "1h", trade_start=df.index[99])
    assert [pd.Timestamp(t["entry_time"]) for t in e.trades] == [df.index[99]]


def test_empty_filter_override_ignores_db_rules(monkeypatch):
    skip_all = [{"feature": "x", "operator": ">=", "threshold": 0.0, "rule_type": "skip",
                 "effect_size": 1.0, "rationale": ""}]
    monkeypatch.setattr(eng_mod, "load_active_filters", lambda s: skip_all)
    monkeypatch.setitem(eng_mod.STRATEGY_REGISTRY, "every", (
        lambda df, p: df.assign(sig=["buy" if i % 10 == 0 else None for i in range(len(df))]),
        "sig",
        lambda df, i, p: {"entry": 100.0, "stop_loss": 95.0, "target": 100.5,
                          "risk": 5.0, "risk_reward": 0.1}))
    monkeypatch.setattr(eng_mod, "extract_features", lambda df, i: {"x": 1.0})
    e = BacktestEngine("every", {"every": {"min_risk_reward": 0.0}}, 1_000_000,
                       costs={"fee_pct": 0.0, "slippage_pct": 0.0}, filters=[])
    res = e.run(_df(), "BTC/IDR", "1h")
    assert e.trades and res["filtered_trades"] == 0


def test_runner_handles_pair_without_candles(monkeypatch):
    monkeypatch.setattr(runner, "init_db", lambda: None)
    monkeypatch.setattr(runner, "get_connection", lambda: type("C", (), {"close": lambda s: None})())
    monkeypatch.setattr(runner, "load_ohlcv", lambda conn, pair, tf: pd.DataFrame())
    assert runner.run_backtest("darvas", "NEW/IDR", "1h", config={},
                               end=pd.Timestamp("2026-08-01")) == {}


def test_history_survives_null_or_malformed_claude_fields(db):
    import json
    conn = get_connection(db)
    for rec in [{"filter_rules": None, "parameter_changes": None, "confidence": "low"},
                {"filter_rules": ["Skip when rsi > 70", 3], "parameter_changes": ["x", None]}]:
        cur = conn.execute("INSERT INTO learning_sessions (session_type, strategy, "
                           "recommendations_json) VALUES ('claude_review', 's', ?)",
                           (json.dumps(rec),))
        conn.commit()
        pr.record_evaluation(cur.lastrowid, {"promoted": False,
                                             "reason": "Claude returned no parameters or filter rules"})
    history = pr.load_recent_evaluations("s")
    assert [h["filter_rules"] for h in history] == [["Skip when rsi > 70", "3"], []]
    assert all(h["parameter_changes"] == [] for h in history)
    assert "NOT EVALUATED" in format_history(history)


def test_history_skips_bad_rows_and_still_fills_n(db):
    import json
    conn = get_connection(db)
    good = {"promoted": False, "reason": "ok", "incumbent": {"trades": 40, "expectancy": 0.0},
            "candidate": {"trades": 40, "expectancy": 0.001}}
    rows = [({"filter_rules": [{"rule": "Skip when a > 1"}]},
             dict(good, candidate_filters=["a > 1", None])),
            ({"filter_rules": [{"rule": "Skip when b > 1"}]}, dict(good, incumbent=None)),
            ({"filter_rules": [{"rule": None}, None]}, {"promoted": False, "reason": "skipped"}),
            ({}, dict(good, candidate=None)),
            ({}, [1, 2])]
    for rec, ev in rows:
        cur = conn.execute("INSERT INTO learning_sessions (session_type, strategy, "
                           "recommendations_json, evaluation_json) VALUES ('claude_review', "
                           "'s', ?, ?)", (json.dumps(rec), json.dumps(ev)))
        conn.commit()
    history = pr.load_recent_evaluations("s", n=3)
    # The two newest rows are malformed; older usable rows fill the history.
    assert [h["reason"] for h in history] == ["skipped", "ok", "ok"]
    assert history[0]["filter_rules"] == []
    assert history[1]["incumbent"] == {}
    assert history[2]["filter_rules"] == ["a > 1"]
    format_history(history)
