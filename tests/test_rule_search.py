"""Skip rules are judged by what they remove from the unfiltered trades."""
import numpy as np
import pandas as pd

from src.learning.claude_review import build_analysis_prompt
from src.learning.filter_rules import apply_filters, parse_analyzer_filters
from src.learning.rule_search import search_rules

CFG = {"min_removed": 30, "min_kept_ratio": 0.5, "min_gain": 0.0005, "time_evidence_mult": 2}


def _interleave(*groups):
    """Spread groups of (feature dict, pnl) evenly through time."""
    rows = [r for g in groups for r in g]
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    return pd.DataFrame([f for f, _ in rows]), np.array([p for _, p in rows])


def test_picks_rule_that_removes_losers():
    good = [({"atr": 0.01, "noise": i % 7}, 0.01) for i in range(100)]
    bad = [({"atr": 0.05, "noise": i % 7}, -0.01) for i in range(60)]
    feats, pnl = _interleave(good, bad)
    rules = search_rules(feats, pnl, limit=3, cfg=CFG)
    assert len(rules) == 1
    r = rules[0]
    assert r["feature"] == "atr" and r["operator"] == ">=" and 0.01 < r["threshold"] <= 0.05
    assert r["removed"] == 60 and r["kept"] == 100
    assert r["expectancy_after"] > r["expectancy_before"]
    assert "removes 60 of 160 trades" in r["rationale"]


def test_rejects_rule_that_cuts_big_winners():
    """Removes 40 losers and 20 winners (better win rate) but the winners were big."""
    keep = [({"x": 0.0}, 0.002) for _ in range(100)]
    cut_losers = [({"x": 1.0}, -0.005) for _ in range(40)]
    cut_winners = [({"x": 1.0}, 0.03) for _ in range(20)]
    feats, pnl = _interleave(keep, cut_losers, cut_winners)
    assert search_rules(feats, pnl, limit=3, cfg=CFG) == []


def test_needs_enough_removed_trades():
    good = [({"atr": 0.01}, 0.01) for _ in range(100)]
    bad = [({"atr": 0.05}, -0.02) for _ in range(20)]       # only 20 < 30
    feats, pnl = _interleave(good, bad)
    assert search_rules(feats, pnl, limit=3, cfg=CFG) == []


def test_time_rules_need_double_evidence():
    good = [({"hour": 3}, 0.01) for _ in range(150)]
    bad = [({"hour": 20}, -0.01) for _ in range(45)]        # 45 < 2 x 30
    feats, pnl = _interleave(good, bad)
    assert search_rules(feats, pnl, limit=3, cfg=CFG) == []
    more_bad = [({"hour": 20}, -0.01) for _ in range(70)]
    feats, pnl = _interleave(good, more_bad)
    assert [r["feature"] for r in search_rules(feats, pnl, limit=3, cfg=CFG)] == ["hour"]


def test_greedy_set_skips_redundant_rules():
    """b duplicates a; after a's rule the b rule removes nothing new."""
    good = [({"a": 0, "b": 0, "c": 0}, 0.01) for _ in range(120)]
    bad_a = [({"a": 1, "b": 1, "c": 0}, -0.01) for _ in range(40)]
    bad_c = [({"a": 0, "b": 0, "c": 1}, -0.01) for _ in range(40)]
    feats, pnl = _interleave(good, bad_a, bad_c)
    rules = search_rules(feats, pnl, limit=3, cfg=CFG)
    assert sorted(r["feature"] for r in rules) in (["a", "c"], ["b", "c"])
    assert rules[-1]["kept"] == 120


def test_rule_must_hold_in_both_halves():
    early = [({"x": 1.0}, -0.02) for _ in range(40)] + [({"x": 0.0}, 0.0) for _ in range(60)]
    late = [({"x": 1.0}, 0.02) for _ in range(40)] + [({"x": 0.0}, 0.0) for _ in range(60)]
    # Removing x == 1 helps a lot in the first half and hurts in the second.
    feats = pd.DataFrame([f for f, _ in early + late])
    pnl = np.array([p for _, p in early + late])
    assert search_rules(feats, pnl, limit=3, cfg=CFG) == []


def test_keeps_at_least_half_the_trades():
    good = [({"a": 0}, 0.01) for _ in range(50)]
    bad = [({"a": 1}, -0.01) for _ in range(60)]             # skipping keeps 45%
    feats, pnl = _interleave(good, bad)
    assert search_rules(feats, pnl, limit=3, cfg=CFG) == []


def test_categorical_rule_round_trips_into_filters():
    good = [({"session": "asia"}, 0.01) for _ in range(100)]
    bad = [({"session": "us"}, -0.01) for _ in range(70)]
    feats, pnl = _interleave(good, bad)
    rules = search_rules(feats, pnl, limit=3, cfg=CFG)
    assert rules[0]["operator"] == "==" and rules[0]["threshold"] == "us"
    parsed = parse_analyzer_filters(rules)
    live = [dict(parsed[0], rule_type="skip")]
    assert apply_filters({"session": "us"}, live)[0]
    assert not apply_filters({"session": "asia"}, live)[0]


def test_prompt_shows_active_filters_and_scored_rules():
    findings = {"total_trades": 160, "suggested_filters": [
        {"rule": "atr >= 0.05", "rationale": "removes 60 of 160 trades; expectancy "
                                             "+0.250% → +1.000%", "effect_size": 1.2}]}
    active = [{"feature": "hour", "operator": ">=", "threshold": "12"}]
    prompt = build_analysis_prompt(findings, {}, {}, "macd", active_filters=active)
    assert "all signals, no filters applied" in prompt
    assert "Skip when hour >= 12" in prompt
    assert "Skip when atr >= 0.05: removes 60 of 160 trades" in prompt
    assert "- none" in build_analysis_prompt(findings, {}, {}, "macd", active_filters=[])
