"""Search for skip rules by what they remove from the unfiltered trades.

A rule is judged by the trades it would skip: it qualifies only if it
removes enough trades, keeps enough of them, and raises the kept trades'
expectancy (mean net pnl per trade) in both halves of the train window.
Rules are chosen greedily: after each pick the rest are re-scored on the
trades that remain, so the set never stacks overlapping or contradictory
rules. Promotion (holdout) still makes the final call.
"""
import numpy as np
import pandas as pd

DEFAULTS = {
    "min_removed": 30,          # trades a rule must skip
    "min_kept_ratio": 0.5,      # share of all trades the full set must keep
    "min_gain": 0.0005,         # expectancy gain per rule (0.05% per trade)
    "time_evidence_mult": 2,    # day/hour rules need this many times the evidence
    "max_categories": 12,
}
TIME_FEATURES = {"hour", "day_of_week", "is_weekend", "session"}
QUANTILES = np.linspace(0.1, 0.9, 9)


def rule_search_config(config: dict) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update((config.get("learning", {}) or {}).get("rule_search", {}) or {})
    return cfg


def _candidates(values: pd.Series, cfg: dict):
    """(operator, threshold, skip_mask) for one feature over the current trades."""
    present = values.dropna()
    if present.empty:
        return
    numeric = pd.api.types.is_numeric_dtype(present) and not pd.api.types.is_bool_dtype(present)
    if numeric:
        # Keep full precision: the threshold is stored as text and parsed back
        # exactly, so the evidence here and the live rule (apply_filters) test
        # the same predicate. Rounding could move a cutoff past the data.
        thresholds = sorted({float(t) for t in present.quantile(QUANTILES).to_numpy()})
        for thr in thresholds:
            yield ">=", thr, (values >= thr).fillna(False).to_numpy()
            yield "<=", thr, (values <= thr).fillna(False).to_numpy()
    else:
        cats = present.astype(str).unique()
        if len(cats) > cfg["max_categories"]:
            return
        as_str = values.astype(str).where(values.notna())
        for cat in cats:
            yield "==", cat, (as_str == cat).fillna(False).to_numpy()


def _holds_in_both_halves(pnl: np.ndarray, skip: np.ndarray) -> bool:
    """Kept expectancy must not fall below the current one in either half."""
    half = len(pnl) // 2
    for sl in (slice(0, half), slice(half, None)):
        p, s = pnl[sl], skip[sl]
        if (~s).sum() == 0 or p[~s].mean() < p.mean():
            return False
    return True


def search_rules(features: pd.DataFrame, pnl, limit: int, cfg: dict = None,
                 effect_sizes: dict = None) -> list:
    """Greedy skip-rule set for trades in chronological order.

    Args:
        features: one row per trade (entry-time features), oldest first
        pnl: net pnl_pct per trade, same order
        limit: max rules (``learning.max_filter_rules``)
        effect_sizes: feature -> Cohen's d, carried on the rule for ranking
    """
    cfg = dict(DEFAULTS, **(cfg or {}))
    pnl = np.asarray(pnl, dtype=float)
    features = features.reset_index(drop=True)
    n_total = len(pnl)
    kept = np.ones(n_total, dtype=bool)
    rules, used = [], set()

    for _ in range(limit or 0):
        cur_pnl = pnl[kept]
        cur_exp = cur_pnl.mean() if len(cur_pnl) else 0.0
        cur_feats = features[kept].reset_index(drop=True)
        best = None
        for col in cur_feats.columns:
            if col in used:
                continue
            need = cfg["min_removed"] * (cfg["time_evidence_mult"] if col in TIME_FEATURES else 1)
            for op, thr, skip in _candidates(cur_feats[col], cfg):
                removed = int(skip.sum())
                if removed < need:
                    continue
                if kept.sum() - removed < cfg["min_kept_ratio"] * n_total:
                    continue
                after = cur_pnl[~skip].mean()
                gain = after - cur_exp
                if gain < cfg["min_gain"] or not _holds_in_both_halves(cur_pnl, skip):
                    continue
                if best is None or (gain, -removed) > (best["gain"], -best["removed"]):
                    best = {"feature": col, "operator": op, "threshold": thr,
                            "removed": removed, "gain": gain, "skip": skip,
                            "before": cur_exp, "after": after}
        if best is None:
            break
        idx = np.flatnonzero(kept)
        kept[idx[best["skip"]]] = False
        used.add(best["feature"])
        d = (effect_sizes or {}).get(best["feature"])
        rules.append({
            "feature": best["feature"],
            "operator": best["operator"],
            "threshold": best["threshold"],
            "type": "skip",
            "rule": f"{best['feature']} {best['operator']} {best['threshold']}",
            "effect_size": d,
            "removed": best["removed"],
            "kept": int(kept.sum()),
            "expectancy_before": float(best["before"]),
            "expectancy_after": float(best["after"]),
            "rationale": (f"removes {best['removed']} of {len(cur_pnl)} trades; expectancy "
                          f"{best['before']:+.3%} → {best['after']:+.3%}"),
        })
    return rules
