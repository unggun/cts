"""run_backtest: train-window only, budgeted, recorded; and the review's tool loop."""
import json
import os
import tempfile
from types import SimpleNamespace

import pandas as pd
import pytest

import src.backtest.engine as eng_mod
import src.learning.backtest_tool as bt
import src.learning.filter_rules as fr
import src.learning.promotion as pr
from src.data.database import get_connection, init_db, save_parameters, upsert_ohlcv
from src.learning import journal
from src.learning.claude_review import LAST_TURN_NOTE, build_analysis_prompt, converse
from tests.test_review_feedback import CONFIG, _candles

TRAIN_END = pd.Timestamp("2026-02-20")     # candles run to 2026-03-01


@pytest.fixture
def db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    fr.init_filter_rules_table(get_connection(path))
    for mod in (pr, fr, bt):
        monkeypatch.setattr(mod, "get_connection", lambda: get_connection(path))
    rows, _ = _candles(600)
    upsert_ohlcv(get_connection(path), "BTC/IDR", "1h", rows)
    yield path
    os.unlink(path)


@pytest.fixture
def seen(monkeypatch):
    """A strategy that buys every 10th bar and records what reaches the engine."""
    record = {"last_bar": [], "last_btc_bar": [], "params": []}

    def detect(df, params):
        record["last_bar"].append(df.index.max())
        record["params"].append(dict(params))
        df = df.copy()
        df["sig"] = None
        df.iloc[::10, df.columns.get_loc("sig")] = "buy"
        return df

    def levels(df, i, params):
        c = df.iloc[i]["close"]
        return {"entry": c, "stop_loss": c - 5, "target": c + params.get("tp", 0.5),
                "risk": 5, "risk_reward": 0.1}

    real_regime = eng_mod.add_regime_columns

    def regime(df, btc_df, timeframe):
        if btc_df is not None and not btc_df.empty:
            record["last_btc_bar"].append(btc_df.index.max())
        return real_regime(df, btc_df, timeframe)

    monkeypatch.setitem(eng_mod.STRATEGY_REGISTRY, "fake", (detect, "sig", levels))
    monkeypatch.setattr(eng_mod, "extract_features", lambda df, i: {"x": 1.0})
    monkeypatch.setattr(eng_mod, "add_regime_columns", regime)
    return record


def _tool(db, **kw):
    save_parameters(get_connection(db), "fake", {"min_risk_reward": 0.0, "tp": 0.5, "period": 10})
    return bt.BacktestTool("fake", CONFIG, TRAIN_END, **kw)


# ── executor ────────────────────────────────────────────────────────────

def test_no_holdout_candle_reaches_the_engine(db, seen):
    tool = _tool(db)
    text, err = tool.run({"parameter_changes": {"tp": 0.6}, "filter_rules": []})
    assert not err and json.loads(text)["trades"] > 0
    assert seen["last_bar"] and max(seen["last_bar"]) < TRAIN_END
    assert seen["last_btc_bar"] and max(seen["last_btc_bar"]) < TRAIN_END

    # Even when the caller hands over newer candles, they are cut.
    rows, idx = _candles(600)
    df = pd.DataFrame([r[1:] for r in rows], index=idx,
                      columns=["open", "high", "low", "close", "volume"])
    seen["last_bar"].clear()
    bt.BacktestTool("fake", CONFIG, TRAIN_END, dfs={"BTC/IDR": df}, btc_df=df).run(
        {"parameter_changes": {}, "filter_rules": []})
    assert max(seen["last_bar"]) < TRAIN_END


def test_merges_partial_parameters_and_rejects_invalid_ones(db, seen):
    tool = _tool(db)
    _, err = tool.run({"parameter_changes": {"tp": 0.7}, "filter_rules": []})
    assert not err
    assert seen["params"][-1]["tp"] == 0.7 and seen["params"][-1]["period"] == 10

    text, err = tool.run({"parameter_changes": {"period": -3}, "filter_rules": []})
    assert err and "period" in text
    text, err = tool.run({"parameter_changes": "tp=1", "filter_rules": []})
    assert err
    text, _ = tool.run({"parameter_changes": {"new_knob": 2}, "filter_rules": []})
    assert "new_knob" in json.loads(text)["warning"]
    assert len(tool.calls) == 2        # errors don't use the budget


def test_filters_are_capped_like_the_proposal(db, seen):
    tool = _tool(db)
    rules = [{"rule": f"Skip when {f} > 5", "type": "skip"} for f in ("x", "y", "z", "w")]
    out = json.loads(tool.run({"parameter_changes": {}, "filter_rules": rules})[0])
    assert len(out["applied_filters"]) == 3
    assert out["dropped_filters"] == ["w > 5"]
    skip_all = json.loads(tool.run({"parameter_changes": {},
                                    "filter_rules": [{"rule": "Skip when x > 0.5"}]})[0])
    assert skip_all["trades"] == 0 and out["trades"] > 0


def test_result_compares_with_current_version_and_splits_halves(db, seen):
    tool = _tool(db)
    out = json.loads(tool.run({"parameter_changes": {}, "filter_rules": []})[0])
    halves = out["first_half"]["trades"] + out["second_half"]["trades"]
    assert halves == out["trades"] and out["first_half"]["trades"] > 0 \
        and out["second_half"]["trades"] > 0
    assert out["current_version"]["trades"] == out["trades"]
    assert out["calls_left"] == bt.DEFAULT_MAX_CALLS - 1
    n = len(seen["last_bar"])
    tool.run({"parameter_changes": {"tp": 0.6}, "filter_rules": []})
    assert len(seen["last_bar"]) == n + 1      # the baseline is computed once


def test_budget_is_enforced_and_recorded(db, seen):
    tool = _tool(db, max_calls=2)
    for _ in range(2):
        assert not tool.run({"parameter_changes": {}, "filter_rules": []})[1]
    text, err = tool.run({"parameter_changes": {}, "filter_rules": []})
    assert err and "budget used up" in text.lower()
    summary = tool.summary()
    assert len(summary["calls"]) == 2 and summary["refused"] == 1
    assert summary["train_end"].startswith("2026-02-20")


def test_tool_config():
    assert bt.tool_config({}) == {"enabled": True, "max_calls": 6}
    assert bt.tool_config({"learning": {"backtest_tool": {"max_calls": 0}}})["enabled"] is False
    assert bt.tool_config({"learning": {"backtest_tool": {"enabled": False}}})["enabled"] is False


# ── conversation loop ───────────────────────────────────────────────────

def _msg(stop, *blocks):
    return SimpleNamespace(stop_reason=stop, content=list(blocks),
                           usage=SimpleNamespace(input_tokens=100, output_tokens=10,
                                                 cache_creation_input_tokens=0,
                                                 cache_read_input_tokens=50))


def _use(i):
    return SimpleNamespace(type="tool_use", id=f"t{i}", name="run_backtest",
                           input={"parameter_changes": {"tp": i}, "filter_rules": []})


class FakeClient:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.replies.pop(0)


class FakeTool:
    def __init__(self, max_calls):
        self.max_calls, self.inputs = max_calls, []

    def run(self, tool_input):
        self.inputs.append(tool_input)
        if len(self.inputs) > self.max_calls:
            return "budget used up", True
        return json.dumps({"trades": 5}), False


def test_loop_runs_tools_then_returns_the_answer():
    thinking = SimpleNamespace(type="thinking", thinking="hmm", signature="sig")
    answer = SimpleNamespace(type="text", text='{"confidence": "medium"}')
    client = FakeClient([_msg("tool_use", thinking, _use(1), _use(2)),
                         _msg("tool_use", _use(3)),
                         _msg("end_turn", answer)])
    tool = FakeTool(6)
    message, usage = converse(client, "m", "prompt", tool)
    assert message.content == [answer] and len(tool.inputs) == 3
    second = client.requests[1]
    assert second["tools"][0]["name"] == "run_backtest"
    assert second["cache_control"] == {"type": "ephemeral"}
    assert second["messages"][1]["content"][0] is thinking      # sent back unchanged
    results = second["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert usage["turns"] == 3 and usage["input_tokens"] == 300


def test_loop_stops_at_the_turn_limit():
    client = FakeClient([_msg("tool_use", _use(i)) for i in range(10)])
    tool = FakeTool(2)
    message, usage = converse(client, "m", "prompt", tool)
    assert usage["turns"] == 4 and len(client.requests) == 4
    last_results = client.requests[3]["messages"][-1]["content"]
    assert last_results[-1] == {"type": "text", "text": LAST_TURN_NOTE}
    assert message.stop_reason == "tool_use"      # no JSON: the cycle keeps the version


def test_without_tool_it_is_one_plain_request():
    client = FakeClient([_msg("end_turn", SimpleNamespace(type="text", text="{}"))])
    converse(client, "m", "prompt", None)
    assert len(client.requests) == 1 and "tools" not in client.requests[0]


def test_prompt_mentions_the_tool_only_when_offered():
    assert "run_backtest" not in build_analysis_prompt({}, {}, {}, "macd", holdout_days=90)
    prompt = build_analysis_prompt({}, {}, {}, "macd", holdout_days=90, backtest_budget=6)
    assert "up to 6 times" in prompt and "never the holdout" in prompt


# ── journal ─────────────────────────────────────────────────────────────

def test_journal_lists_backtests_defensively():
    rec = {"backtests": {"calls": [
        {"parameter_changes": {"tp": 0.7}, "applied_filters": ["x > 5"],
         "result": {"trades": 40, "expectancy_pct": 0.12,
                    "first_half": {"expectancy_pct": 0.2}, "second_half": None}},
        "junk", {"result": "bad"}]}}
    lines = journal.backtests(rec)
    assert lines[0] == "tp=0.7; skip x > 5 → 40 trades, +0.12%/trade (halves +0.20% / —)"
    assert lines[1].startswith("current params, no filters → — trades")
    assert journal.backtests({}) == [] and journal.backtests({"backtests": "x"}) == []
