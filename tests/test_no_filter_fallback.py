"""When the proposal loses, removing all filters gets the same holdout test."""
import os
import tempfile

import pytest

import src.backtest.engine as eng_mod
import src.learning.filter_rules as fr
import src.learning.promotion as pr
from src.data.database import get_connection, init_db, upsert_ohlcv
from tests.test_review_feedback import CONFIG, _candles


@pytest.fixture
def db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    fr.init_filter_rules_table(get_connection(path))
    monkeypatch.setattr(pr, "get_connection", lambda: get_connection(path))
    monkeypatch.setattr(fr, "get_connection", lambda: get_connection(path))
    rows, idx = _candles(600)
    upsert_ohlcv(get_connection(path), "BTC/IDR", "1h", rows)
    yield path, idx
    os.unlink(path)


@pytest.fixture(autouse=True)
def winning_strategy(monkeypatch):
    """Buys every 10th bar and always hits its target (+0.5%) on the next bar."""
    def detect(df, params):
        df = df.copy()
        df["sig"] = None
        df.iloc[::10, df.columns.get_loc("sig")] = "buy"
        return df

    def levels(df, i, params):
        c = df.iloc[i]["close"]
        return {"entry": c, "stop_loss": c - 5, "target": c + 0.5, "risk": 5,
                "risk_reward": 0.1}

    monkeypatch.setitem(eng_mod.STRATEGY_REGISTRY, "fake", (detect, "sig", levels))
    monkeypatch.setattr(eng_mod, "extract_features", lambda df, i: {"x": 1.0})


SKIP_ALL = {"feature": "x", "operator": ">=", "threshold": 0.5, "rule_type": "skip"}
SKIP_NONE = {"feature": "x", "operator": ">=", "threshold": 5.0, "rule_type": "skip"}
LOSING_PROPOSAL = {"filter_rules": [{"rule": "Skip when x >= 0.9", "type": "skip"}]}


def _state(path):
    conn = get_connection(path)
    versions = conn.execute("SELECT source FROM strategy_parameters").fetchall()
    active = conn.execute("SELECT COUNT(*) FROM filter_rules WHERE strategy='fake' "
                          "AND active=1").fetchone()[0]
    return [v[0] for v in versions], active


def test_starving_filters_are_removed_when_no_filters_passes(db):
    path, idx = db
    pr.replace_active_filters("fake", [SKIP_ALL])         # current filters block everything
    ev = pr.run_promotion("fake", LOSING_PROPOSAL, {}, CONFIG, idx[300])
    assert ev["candidate"]["trades"] == 0                 # the proposal starves too
    assert ev["no_filters"]["trades"] == 30 and ev["no_filters"]["expectancy"] > 0
    assert ev["promoted"] and ev["filters_removed"]
    assert "all filters removed" in ev["reason"] and "only 0 holdout trades" in ev["reason"]
    assert _state(path) == (["no_filter_fallback"], 0)


def test_no_filters_must_beat_the_current_filters(db):
    path, idx = db
    pr.replace_active_filters("fake", [SKIP_NONE])        # harmless filter: same 30 trades
    ev = pr.run_promotion("fake", LOSING_PROPOSAL, {}, CONFIG, idx[300])
    assert ev["no_filters"]["trades"] == 30 and not ev["promoted"]
    assert not ev.get("filters_removed")
    assert "only 0 holdout trades" in ev["reason"]
    assert _state(path) == ([], 1)


def test_no_fallback_without_active_filters_or_when_already_tested(db):
    path, idx = db
    ev = pr.run_promotion("fake", LOSING_PROPOSAL, {}, CONFIG, idx[300])
    assert "no_filters" not in ev                          # nothing to remove
    pr.replace_active_filters("fake", [SKIP_NONE])
    ev = pr.run_promotion("fake", {}, {}, CONFIG, idx[300])  # proposal *is* "no filters"
    assert "no_filters" not in ev and not ev["promoted"]


def test_dry_run_reports_but_keeps_filters(db):
    path, idx = db
    pr.replace_active_filters("fake", [SKIP_ALL])
    ev = pr.run_promotion("fake", LOSING_PROPOSAL, {}, CONFIG, idx[300], dry_run=True)
    assert ev["would_promote"] and ev["filters_removed"] and not ev["promoted"]
    assert _state(path) == ([], 1)


def test_removed_filters_in_history_and_telegram(db, monkeypatch):
    import json
    import src.run_learning_cycle as c
    from src.learning.claude_review import format_history
    path, idx = db
    pr.replace_active_filters("fake", [SKIP_ALL])
    ev = pr.run_promotion("fake", LOSING_PROPOSAL, {}, CONFIG, idx[300])
    conn = get_connection(path)
    cur = conn.execute("INSERT INTO learning_sessions (session_type, strategy, "
                       "recommendations_json) VALUES ('claude_review', 'fake', ?)",
                       (json.dumps(LOSING_PROPOSAL),))
    conn.commit()
    pr.record_evaluation(cur.lastrowid, ev)
    (h,) = pr.load_recent_evaluations("fake")
    assert not h["promoted"] and h["filters_removed"]     # Claude's proposal itself lost
    text = format_history([h])
    assert "REJECTED" in text and "all filters were removed instead" in text

    monkeypatch.setattr(c, "format_param_diff", lambda s: "")
    lines = "\n".join(c._promotion_lines("fake", {"promotion": ev}))
    assert "Promoted v1" in lines and "Filters now: none" in lines
