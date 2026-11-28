"""Per-strategy regime gate: signal, storage, decision rule, and where it applies."""
import json
import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import src.backtest.engine as eng_mod
import src.execution.paper_trader as pt
import src.learning.filter_rules as fr
import src.learning.promotion as pr
import src.learning.regime_gate as rg
import src.run_learning_cycle as cycle
from src.backtest.engine import BacktestEngine
from src.data.database import get_connection, init_db, save_parameters, upsert_ohlcv
from src.indicators.regime import add_regime_columns
from src.learning.claude_review import build_analysis_prompt
from tests.test_review_feedback import CONFIG, _candles

CFG = rg.gate_config({})


@pytest.fixture
def db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    fr.init_filter_rules_table(get_connection(path))
    for mod in (pr, fr, rg, pt):
        monkeypatch.setattr(mod, "get_connection", lambda: get_connection(path))
    yield path
    os.unlink(path)


@pytest.fixture
def btc_trend(monkeypatch):
    """A strategy buying every 10th bar; btc_st_1d_up comes from ``state['trend']``."""
    state = {"trend": 0.0}

    def detect(df, params):
        df = df.copy()
        df["sig"] = None
        df.iloc[::10, df.columns.get_loc("sig")] = "buy"
        return df

    def levels(df, i, params):
        c = df.iloc[i]["close"]
        return {"entry": c, "stop_loss": c - 5, "target": c + 0.5, "risk": 5, "risk_reward": 0.1}

    def features(df, i):
        f = {"x": 1.0}
        if state["trend"] is not None:
            f["btc_st_1d_up"] = state["trend"]
        return f

    monkeypatch.setitem(eng_mod.STRATEGY_REGISTRY, "fake", (detect, "sig", levels))
    monkeypatch.setattr(eng_mod, "extract_features", features)
    return state


def _df(n=400):
    rows, idx = _candles(n)
    return pd.DataFrame([r[1:] for r in rows], index=idx,
                        columns=["open", "high", "low", "close", "volume"])


# ── signal ──────────────────────────────────────────────────────────────

def test_btc_daily_trend_feature_uses_closed_candles_only():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=24 * 60, freq="1h")
    close = 100 + np.cumsum(rng.normal(0, 1, len(idx)))
    btc = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close,
                        "volume": 1.0}, index=idx)
    full = add_regime_columns(btc, btc, "1h")
    assert full["btc_st_1d_up"].iloc[-1] in (0.0, 1.0)
    cut = 24 * 45 + 7                          # mid-day: today's daily candle is not closed
    part = add_regime_columns(btc.iloc[:cut], btc.iloc[:cut], "1h")
    pd.testing.assert_series_equal(part["btc_st_1d_up"], full["btc_st_1d_up"].iloc[:cut])
    assert add_regime_columns(btc, None, "1h")["btc_st_1d_up"].isna().all()


def test_gate_blocks_only_a_known_downtrend():
    assert rg.gate_blocks({"btc_st_1d_up": 0.0})
    assert not rg.gate_blocks({"btc_st_1d_up": 1.0})
    assert not rg.gate_blocks({})                              # stale BTC data
    assert not rg.gate_blocks({"btc_st_1d_up": float("nan")})


# ── storage ─────────────────────────────────────────────────────────────

def test_state_is_stored_and_survives_promotion(db):
    assert rg.load_gate_state("fake") is None and not rg.gate_enabled("fake", {})
    rg.save_gate_state("fake", True, "on: test", {"gated": {}})
    assert rg.gate_enabled("fake", {})
    pr.replace_active_filters("fake", [])               # what a promotion does
    assert rg.gate_enabled("fake", {})
    assert not rg.gate_enabled("fake", {"learning": {"regime_gate": {"enabled": False}}})
    rg.save_gate_state("fake", False, "off: test", {})
    assert rg.load_gate_state("fake")["reason"] == "off: test"


# ── engine ──────────────────────────────────────────────────────────────

def test_engine_skips_entries_only_when_gated(db, btc_trend):
    df = _df()
    def trades(**kw):
        eng = BacktestEngine("fake", {"min_risk_reward": 0.0}, costs={"fee_pct": 0,
                             "slippage_pct": 0}, filters=[], **kw)
        res = eng.run(df.copy(), "ETH/IDR", "1h")
        return len(eng.trades), res.get("gate_blocked")
    n, _ = trades(gate=False)
    assert n > 0
    assert trades(gate=True) == (0, n)
    btc_trend["trend"] = 1.0
    assert trades(gate=True)[0] == n
    btc_trend["trend"] = None                              # missing feature: not blocked
    assert trades(gate=True)[0] == n


def test_engine_default_follows_the_stored_state(db, btc_trend):
    df = _df()
    def n(**kw):
        eng = BacktestEngine("fake", {"min_risk_reward": 0.0}, **kw)
        eng.run(df.copy(), "ETH/IDR", "1h")
        return len(eng.trades)
    assert n() > 0
    rg.save_gate_state("fake", True, "on", {})
    assert n() == 0                                    # the deployed version is gated
    assert n(use_filters=False) > 0                    # the analyzer's unfiltered run is not
    assert n(filters=[]) > 0                           # explicit filters: caller decides


# ── decision rule ───────────────────────────────────────────────────────

def _ev(g, u, g1, g2, u1, u2, gn=100):
    s = lambda e, a, b, n: {"trades": n, "expectancy": e,
                            "first_half": {"trades": n // 2, "expectancy": a},
                            "second_half": {"trades": n // 2, "expectancy": b}}
    return {"gated": s(g, g1, g2, gn), "ungated": s(u, u1, u2, 2 * gn)}


def test_turns_on_only_when_both_halves_improve():
    on, why = rg.decide(False, _ev(0.0026, -0.0035, 0.002, 0.003, -0.004, -0.003), CFG)
    assert on and why.startswith("on")
    on, why = rg.decide(False, _ev(0.0018, -0.0004, -0.0024, 0.0038, -0.001, 0.0003), CFG)
    assert not on and "both halves" in why                 # the pooled case in the spec
    on, why = rg.decide(False, _ev(-0.001, -0.004, -0.0005, -0.0015, -0.004, -0.004), CFG)
    assert not on and "not positive" in why
    on, why = rg.decide(False, _ev(0.003, 0.0, 0.003, 0.003, 0.0, 0.0, gn=10), CFG)
    assert not on and "only 10 gated trades" in why


def test_hysteresis_keeps_an_on_gate_until_it_is_worse_overall():
    kept, why = rg.decide(True, _ev(0.001, 0.0005, -0.001, 0.003, 0.0, 0.001), CFG)
    assert kept and why.startswith("kept on")
    off, why = rg.decide(True, _ev(0.0004, 0.0009, 0.0, 0.001, 0.001, 0.001), CFG)
    assert not off and why.startswith("off")
    assert rg.decide(True, {"gated": {}, "ungated": {}}, CFG)[0] is True


def test_review_stores_the_decision_unless_dry(db, monkeypatch):
    monkeypatch.setattr(rg, "gather_evidence", lambda *a, **k: _ev(
        0.0026, -0.0035, 0.002, 0.003, -0.004, -0.003))
    dry = rg.review_gate("fake", {}, apply=False)
    assert dry["enabled"] and dry["changed"] and rg.load_gate_state("fake") is None
    r = rg.review_gate("fake", {}, apply=True)
    assert r["changed"] and rg.gate_enabled("fake", {})
    assert rg.review_gate("fake", {}, apply=True)["changed"] is False
    assert "🛡 ON" in rg.format_review(r)


# ── promotion, prompt, paper trader, Telegram ───────────────────────────

def test_promotion_judges_both_versions_under_the_gate(db, btc_trend):
    rows, idx = _candles(600)
    upsert_ohlcv(get_connection(db), "BTC/IDR", "1h", rows)
    rg.save_gate_state("fake", True, "on", {})
    ev = pr.run_promotion("fake", {"updated_parameters": {"min_risk_reward": 0.0, "tp": 1.0}},
                          {}, CONFIG, idx[300])
    assert ev["regime_gate"] is True
    assert ev["incumbent"]["trades"] == 0 and ev["candidate"]["trades"] == 0


def test_prompt_states_the_gate():
    on = build_analysis_prompt({}, {}, {}, "macd", regime_gate={"enabled": True, "reason": "on: x"})
    assert "currently BLOCKED" in on and "Do not propose a filter" in on
    off = build_analysis_prompt({}, {}, {}, "macd", regime_gate={"enabled": False, "reason": None})
    assert "NOT gated" in off
    assert "Regime Gate" not in build_analysis_prompt({}, {}, {}, "macd")


def test_paper_trader_picks_up_and_announces_gate_changes(db):
    sent = []
    t = object.__new__(pt.PaperTrader)
    t.strategy, t.config = "macd", {"risk": {}}
    t.params, t.param_version, t.param_row_id, t.filters = {"risk": {}}, 0, None, []
    t._notify = sent.append
    t._refresh_learned_state()
    assert t.gate is False and not sent
    rg.save_gate_state("macd", True, "on", {})
    t._refresh_learned_state()
    assert t.gate is True and "regime gate ON" in sent[-1]
    rg.save_gate_state("macd", False, "off", {})
    t._refresh_learned_state()
    assert t.gate is False and "regime gate OFF" in sent[-1]


def test_telegram_lines():
    changed = {"regime_gate": {"enabled": True, "changed": True, "reason": "on: gated_x"}}
    lines = cycle._gate_lines(changed)
    assert "Regime gate ON" in lines[0] and "gated\\_x" in lines[1]
    assert cycle._gate_lines({"regime_gate": {"enabled": True, "changed": False}}) == [
        "\n🛡 Regime gate on"]
    assert cycle._gate_lines({"regime_gate": {"enabled": False, "changed": False}}) == []
    assert cycle._gate_lines({}) == []
