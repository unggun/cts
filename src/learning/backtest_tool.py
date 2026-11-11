"""``run_backtest``: Claude tests a bundle on the train window during its review.

Candles at or after ``train_end`` (the holdout start the learning cycle uses)
are cut off before they reach the engine, so nothing this tool returns depends
on holdout data. The holdout stays the only judge of promotion.

Design: docs/superpowers/specs/2026-11-11-run-backtest-tool-design.md

Usage (no API call; runs the tool once on the current version)::

    python -m src.learning.backtest_tool --strategy macd
"""

import argparse
import copy
import json
import time

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.data.database import get_connection, load_latest_parameters
from src.learning.filter_rules import load_active_filters, parse_claude_filter_rules

TOOL_NAME = "run_backtest"
DEFAULT_MAX_CALLS = 6

TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Backtest a parameter/filter bundle for this strategy on the train window: the same "
        "data as the analysis in the prompt, never the holdout. Fees and slippage are "
        "included. Returns trades, expectancy per trade, win rate, max drawdown, expectancy "
        "in each chronological half of the window, and the same numbers for the current "
        "version. Filters are combined exactly like your final proposal: the analyzer's "
        "skip rules are added to yours, then only the strongest skip rules (one per "
        "feature) are applied; the result lists what was applied and what was dropped. "
        "Each call takes about half a minute and the number of calls is limited."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "parameter_changes": {
                "type": "object",
                "description": ("Only the parameters to change, e.g. {\"atr_period\": 21}; "
                                "merged onto the current version. {} keeps them all."),
            },
            "filter_rules": {
                "type": "array",
                "description": ("Filter rules in the same form as in your final answer, e.g. "
                                "[{\"rule\": \"Skip when atr_pct > 0.02\", \"type\": \"skip\"}]. "
                                "[] means only the analyzer's rules."),
                "items": {
                    "type": "object",
                    "properties": {"rule": {"type": "string"},
                                   "type": {"type": "string", "enum": ["skip", "prefer"]}},
                    "required": ["rule"],
                },
            },
        },
        "required": ["parameter_changes", "filter_rules"],
    },
}

BUDGET_USED_UP = ("Backtest budget used up ({n} calls). No more backtests are available; "
                  "reply now with your final JSON answer.")


def tool_config(config: dict) -> dict:
    cfg = (config.get("learning", {}) or {}).get("backtest_tool", {}) or {}
    try:
        max_calls = max(0, int(cfg.get("max_calls", DEFAULT_MAX_CALLS)))
    except (TypeError, ValueError):
        max_calls = DEFAULT_MAX_CALLS
    return {"enabled": bool(cfg.get("enabled", True)) and max_calls > 0,
            "max_calls": max_calls}


def merge_params(base: dict, changes: dict) -> dict:
    """``base`` with ``changes`` applied; nested dicts merge key by key."""
    out = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_params(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _rule_text(rule: dict) -> str:
    thr = rule.get("threshold")
    if isinstance(thr, float):
        thr = f"{thr:.6g}"
    return f"{rule.get('feature')} {rule.get('operator')} {thr}"


def _pct(x):
    return None if x is None else round(float(x) * 100, 3)


class BacktestTool:
    """Runs ``run_backtest`` calls for one review and records them.

    ``dfs``/``btc_df`` are cut at ``train_end`` here, whatever the caller passes.
    """

    def __init__(self, strategy: str, config: dict, train_end, findings: dict = None,
                 dfs: dict = None, btc_df: pd.DataFrame = None, max_calls: int = None):
        from src.backtest.costs import trading_costs
        from src.learning.filter_rules import _configured_filter_limit
        from src.learning.promotion import load_closed_candles

        self.strategy = strategy
        self.train_end = pd.Timestamp(train_end)
        self.findings = findings or {}
        self.max_calls = tool_config(config)["max_calls"] if max_calls is None else max_calls
        self.timeframe = config["trading"]["timeframes"][0]
        self.costs = trading_costs(config)
        self.capital = config.get("simulation", {}).get("initial_capital", 10_000_000)
        self.risk = config.get("risk", {})
        self.filter_limit = _configured_filter_limit()

        if dfs is None:
            dfs, btc_df = load_closed_candles(config, end=self.train_end)
        self.dfs = {p: d[d.index < self.train_end] for p, d in dfs.items()}
        self.dfs = {p: d for p, d in self.dfs.items() if not d.empty}
        btc_df = btc_df if btc_df is not None else pd.DataFrame()
        self.btc_df = btc_df[btc_df.index < self.train_end] if not btc_df.empty else btc_df
        starts = [d.index[0] for d in self.dfs.values()]
        self.midpoint = (min(starts) + (self.train_end - min(starts)) / 2) if starts else None

        conn = get_connection()
        try:
            params = load_latest_parameters(conn, strategy)
        finally:
            conn.close()
        if params is None:
            params = copy.deepcopy(config.get("strategy", {}).get(strategy, {}))
        self.current_params = {k: v for k, v in params.items() if k != "risk"}
        self.current_filters = load_active_filters(strategy)
        self.calls = []          # one record per executed backtest
        self.refused = 0         # calls after the budget ran out
        self._baseline = None

    # ── evaluation ──────────────────────────────────────────────────────

    def evaluate(self, params: dict, filters: list) -> dict:
        trades, drawdowns, per_pair = [], [], {}
        for pair, df in self.dfs.items():
            engine = BacktestEngine(self.strategy, dict(copy.deepcopy(params), risk=self.risk),
                                    self.capital, costs=self.costs, filters=filters)
            res = engine.run(df.copy(), pair, self.timeframe, btc_df=self.btc_df)
            per_pair[pair] = len(engine.trades)
            trades.extend(engine.trades)
            if engine.trades:
                drawdowns.append(float(res.get("max_drawdown", 0.0)))
        pnls = [t["pnl_pct"] for t in trades]

        def half(first: bool):
            sel = [t["pnl_pct"] for t in trades
                   if (pd.Timestamp(t["entry_time"]) < self.midpoint) == first]
            return {"trades": len(sel),
                    "expectancy_pct": _pct(np.mean(sel)) if sel else None}

        return {
            "trades": len(pnls),
            "expectancy_pct": _pct(np.mean(pnls)) if pnls else None,
            "win_rate_pct": _pct(np.mean([p > 0 for p in pnls])) if pnls else None,
            "max_drawdown_pct": _pct(max(drawdowns)) if drawdowns else 0.0,
            "first_half": half(True) if self.midpoint is not None else None,
            "second_half": half(False) if self.midpoint is not None else None,
            "per_pair": per_pair,
        }

    def baseline(self) -> dict:
        """The current version (params + active filters) on the train window, once."""
        if self._baseline is None:
            self._baseline = self.evaluate(self.current_params, self.current_filters)
        return self._baseline

    # ── tool calls ──────────────────────────────────────────────────────

    def run(self, tool_input) -> tuple[str, bool]:
        """Execute one call. Returns ``(text for the tool_result, is_error)``."""
        from src.learning.promotion import candidate_filters, invalid_parameters

        if len(self.calls) >= self.max_calls:
            self.refused += 1
            return BUDGET_USED_UP.format(n=self.max_calls), True
        if not self.dfs:
            return "Error: no train-window candles are available; answer without backtests.", True
        if not isinstance(tool_input, dict):
            return "Error: input must be an object with parameter_changes and filter_rules.", True
        changes = tool_input.get("parameter_changes")
        rules = tool_input.get("filter_rules")
        if changes is None:
            changes = {}
        if not isinstance(changes, dict) or not isinstance(rules, list) \
                or not all(isinstance(r, dict) for r in rules):
            return ("Error: parameter_changes must be an object and filter_rules a list of "
                    "{\"rule\": ..., \"type\": ...} objects."), True
        changes = {k: v for k, v in changes.items() if k != "risk"}

        params = merge_params(self.current_params, changes)
        bad = invalid_parameters(self.current_params, params)
        if bad:
            return (f"Error: invalid parameter values: {', '.join(bad[:5])}. Periods, "
                    f"multipliers and thresholds must stay positive numbers."), True
        unknown = sorted(k for k in changes if k not in self.current_params)

        applied = candidate_filters({"filter_rules": rules}, self.findings, self.filter_limit)
        btc_rules = sorted({f["feature"] for f in applied if f["feature"].startswith("btc_")})
        if btc_rules and self.btc_df.empty:
            return f"Error: no regime-pair candles to evaluate {', '.join(btc_rules)}.", True
        parsed = [_rule_text(r) for r in parse_claude_filter_rules(rules)]
        applied_text = [_rule_text(r) for r in applied]
        dropped = [r for r in parsed if r not in applied_text]

        started = time.monotonic()
        try:
            baseline = self.baseline()
            result = self.evaluate(params, applied)
        except Exception as e:  # a malformed bundle must not end the review
            return f"Error: backtest failed: {type(e).__name__}: {e}", True
        seconds = round(time.monotonic() - started, 1)

        out = dict(result)
        out["applied_filters"] = applied_text
        if dropped:
            out["dropped_filters"] = dropped
        if unknown:
            out["warning"] = (f"not parameters of the current version: {', '.join(unknown)} "
                              f"(the strategy may ignore them)")
        out["current_version"] = {k: baseline[k] for k in
                                  ("trades", "expectancy_pct", "win_rate_pct",
                                   "max_drawdown_pct", "first_half", "second_half")}
        self.calls.append({"parameter_changes": changes,
                           "filter_rules": [r.get("rule") for r in rules],
                           "applied_filters": applied_text,
                           "result": {k: result[k] for k in
                                      ("trades", "expectancy_pct", "win_rate_pct",
                                       "max_drawdown_pct", "first_half", "second_half")},
                           "seconds": seconds})
        out["calls_left"] = self.max_calls - len(self.calls)
        return json.dumps(out), False

    def summary(self) -> dict:
        """What gets stored with the review."""
        return {"calls": self.calls, "refused": self.refused, "max_calls": self.max_calls,
                "train_end": str(self.train_end),
                "current_version": self._baseline and {
                    k: self._baseline[k] for k in ("trades", "expectancy_pct",
                                                   "first_half", "second_half")}}


def main():
    from src.config import load_config
    from src.data.database import init_db
    from src.learning.promotion import holdout_start

    parser = argparse.ArgumentParser(description="Run the run_backtest tool once (no API call)")
    parser.add_argument("--strategy", required=True)
    args = parser.parse_args()
    init_db()
    config = load_config()
    tool = BacktestTool(args.strategy, config, holdout_start(config))
    print(f"Train window ends {tool.train_end:%Y-%m-%d %H:%M} UTC; "
          f"pairs: {', '.join(tool.dfs)}; budget {tool.max_calls} calls")
    started = time.monotonic()
    text, is_error = tool.run({"parameter_changes": {}, "filter_rules": []})
    print(f"({time.monotonic() - started:.1f}s, including the current-version baseline)")
    print(json.dumps(json.loads(text), indent=2) if not is_error else text)


if __name__ == "__main__":
    main()
