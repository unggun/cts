"""Filter rules must stay few enough that strategies can still trade."""
from src.learning.filter_rules import select_filters


def _rule(feature, effect, rule_type="skip", op=">=", thr=1.0):
    return {"feature": feature, "operator": op, "threshold": thr,
            "rule_type": rule_type, "effect_size": effect, "rationale": ""}


def test_select_filters_caps_to_strongest():
    rules = [_rule("a", 0.3), _rule("b", -0.9), _rule("c", 0.6), _rule("d", 0.1)]
    picked = select_filters(rules, limit=2)
    assert [r["feature"] for r in picked] == ["b", "c"]


def test_select_filters_one_rule_per_feature():
    # Contradictory day_of_week rules from two sources must not both apply.
    rules = [_rule("day_of_week", 0.8, op="<=", thr=2.9),
             _rule("day_of_week", None, op=">=", thr=4.0),
             _rule("hour", 0.5)]
    picked = select_filters(rules, limit=3)
    assert [r["feature"] for r in picked] == ["day_of_week", "hour"]
    assert picked[0]["operator"] == "<="


def test_select_filters_ignores_non_skip_and_ranks_missing_effect_last():
    rules = [_rule("p", 0.9, rule_type="prefer"), _rule("x", None), _rule("y", 0.2)]
    picked = select_filters(rules, limit=5)
    assert [r["feature"] for r in picked] == ["y", "x"]


def test_select_filters_no_limit_keeps_all_unique():
    rules = [_rule("a", 0.3), _rule("b", 0.2)]
    assert len(select_filters(rules, limit=None)) == 2
