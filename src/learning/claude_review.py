"""Claude API integration for automated trade analysis and parameter optimization.

This is the "brain" of the auto-learning loop. It takes the pattern analysis
findings and uses Claude to:
1. Interpret the statistical findings in trading context
2. Suggest specific parameter changes
3. Generate updated filter rules
4. Track the evolution of strategy performance
"""
import json

from src.config import load_config
from src.data.database import get_connection, load_latest_parameters
from src.learning.analyzer import analyze_winners_vs_losers, analyze_time_patterns
from src.learning.filter_rules import load_active_filters


DEFAULT_MODEL = "claude-sonnet-5-5"


def extract_review(message) -> dict:
    """Parse Claude's reply into a review dict.

    Current models can open the response with a ``thinking`` block, so only
    ``text`` blocks are read. A refusal comes back as HTTP 200 with
    ``stop_reason == "refusal"`` and is reported as an error.
    """
    if getattr(message, "stop_reason", None) == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        return {"error": f"Claude declined the request (category: {category})"}

    response_text = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )
    json_start = response_text.find("{")
    json_end = response_text.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        try:
            return json.loads(response_text[json_start:json_end])
        except json.JSONDecodeError:
            pass
    return {"raw_response": response_text}


def format_history(history: list) -> str:
    """Render past proposals and their holdout outcomes for the prompt."""
    if not history:
        return ""
    out = "\n## Your Previous Proposals and Their Holdout Results (newest first)\n"
    for h in history:
        if not h.get("evaluated", True):
            out += f"- {h['created_at']}: NOT EVALUATED — {h['reason']}\n"
        else:
            verdict = "PROMOTED" if h["promoted"] else "REJECTED"
            cand, inc = h.get("candidate", {}), h.get("incumbent", {})
            out += (f"- {h['created_at']}: {verdict} — {h['reason']} "
                    f"(candidate {cand.get('trades', 0)} trades, "
                    f"exp {cand.get('expectancy', 0):+.3%}; "
                    f"current {inc.get('trades', 0)} trades, "
                    f"exp {inc.get('expectancy', 0):+.3%})\n")
        changes = ", ".join(f"{c.get('param')}: {c.get('old')}→{c.get('new')}"
                            for c in h.get("parameter_changes", [])[:6])
        if changes:
            out += f"  - changes: {changes}\n"
        if h.get("filter_rules"):
            out += f"  - filters: {'; '.join(h['filter_rules'][:4])}\n"
        if h.get("filters_removed"):
            out += "  - all filters were removed instead: no filters beat the current set\n"
    out += "Do not re-propose changes that were rejected unless the data now clearly differs.\n"
    return out


def build_analysis_prompt(findings: dict, time_patterns: dict,
                          current_params: dict, strategy: str,
                          sim_results: dict = None, history: list = None,
                          holdout_days: int = None, promotion_rule: str = None,
                          active_filters: list = None) -> str:
    """Build the prompt for Claude to analyze trading patterns.

    ``promotion_rule`` describes the effective acceptance criteria
    (``promotion.describe_rule``); defaults to the built-in thresholds.
    """
    if promotion_rule is None:
        from src.learning.promotion import DEFAULTS, describe_rule
        promotion_rule = describe_rule(DEFAULTS)

    prompt = f"""You are a quantitative trading analyst reviewing backtest results for a '{strategy}' strategy on crypto (Tokocrypto exchange, IDR pairs).

## Current Strategy Parameters
```json
{json.dumps(current_params, indent=2)}
```

## Backtest Analysis Findings (all signals, no filters applied)
- Total trades: {findings.get('total_trades', 0)}
- Win rate: {findings.get('overall_win_rate', 0):.1%}
- Average win: {findings.get('avg_win_pct', 0):.2%}
- Average loss: {findings.get('avg_loss_pct', 0):.2%}

## Feature Analysis (Winner vs Loser differences)
Top features by effect size:
"""
    # Add exit reason breakdown if available
    if findings.get("exit_reason_stats"):
        prompt += "\n## Exit Reason Breakdown\n"
        for reason, stats in findings["exit_reason_stats"].items():
            prompt += f"- {reason}: {stats['count']} trades, {stats['win_rate']:.0%} win rate\n"

    prompt += "\n"
    # Add top features
    if findings.get("feature_analysis"):
        sorted_features = sorted(
            findings["feature_analysis"].items(),
            key=lambda x: abs(x[1]["effect_size"]),
            reverse=True
        )[:20]
        for name, analysis in sorted_features:
            prompt += (f"- {name}: winners avg={analysis['winner_mean']}, "
                      f"losers avg={analysis['loser_mean']}, "
                      f"effect_size={analysis['effect_size']}\n")

    if active_filters is not None:
        prompt += "\n## Currently Active Filters (the version your proposal must beat)\n"
        if active_filters:
            for f in active_filters:
                prompt += f"- Skip when {f['feature']} {f['operator']} {f['threshold']}\n"
        else:
            prompt += "- none\n"

    # Add suggested filters
    if findings.get("suggested_filters"):
        prompt += ("\n## Skip Rules Found by the Analyzer\n"
                   "Chosen one at a time on the trades above by how much they raise expectancy "
                   "per trade after fees (each holds in both halves of the data):\n")
        for f in findings["suggested_filters"][:10]:
            prompt += f"- Skip when {f['rule']}: {f['rationale']}\n"

    # Add warnings
    if findings.get("warnings"):
        prompt += "\n## Warnings\n"
        for w in findings["warnings"]:
            prompt += f"- {w}\n"

    # Add time patterns
    if time_patterns.get("by_day"):
        prompt += "\n## Win Rate by Day of Week\n"
        for day, stats in time_patterns["by_day"].items():
            prompt += f"- {day}: {stats['win_rate']:.0%} win rate ({stats['trades']} trades)\n"

    if time_patterns.get("by_session"):
        prompt += "\n## Win Rate by Session\n"
        for session, stats in time_patterns["by_session"].items():
            prompt += f"- {session}: {stats['win_rate']:.0%} win rate ({stats['trades']} trades)\n"

    # Add Monte Carlo results if available
    if sim_results:
        prompt += f"""
## Monte Carlo Simulation Results
- Probability of ruin: {sim_results.get('probability_of_ruin', 'N/A')}
- Median final equity: {sim_results.get('median_final_equity', 'N/A')}
- Max drawdown (median): {sim_results.get('max_drawdown_median', 'N/A')}
- Verdict: {sim_results.get('verdict', 'N/A')}
"""

    prompt += format_history(history)

    if holdout_days:
        prompt += f"""
## How Your Proposal Is Judged
The data above excludes the most recent {holdout_days} days. Your updated parameters and
filter rules are backtested on those unseen {holdout_days} days, after trading fees and
slippage, against the current version. {promotion_rule}
Filter rules that remove most trades will be rejected. Propose only changes you expect to
generalize. Every proposal is tested this way, whatever confidence you report; confidence
reflects how likely you think it is to win here and to hold up in live trading.
"""

    prompt += """
## Your Task

Based on this analysis, provide:

1. **Key Insights**: What are the 3-5 most important patterns you see? Which features most strongly predict winners vs losers?

2. **Parameter Recommendations**: Suggest specific changes to the strategy parameters. For each change, explain why and what improvement you expect.

3. **Filter Rules**: Suggest concrete, machine-parseable filter rules. Each rule must reference an exact feature name from the feature analysis above and use a comparison operator.
   Format: "Skip when feature_name > threshold" or "Prefer when feature_name < threshold"
   Use actual feature names (e.g., rsi_bull_divergence, upper_wick_ratio, mixed_ema_signals). Numeric features take
   numeric thresholds; categorical features (session, rsi_zone) take == or != with their text value.

4. **Risk Assessment**: Based on the Monte Carlo results (if available) and win/loss patterns, should the position sizing be adjusted?

5. **Stop-Loss Analysis**: CRITICAL — if the stop-loss win rate is below 10%, the stops are likely too tight. Consider:
   - Widening stop_loss_pct or sl_atr_multiplier significantly (e.g., 1.5x to 3x current value)
   - Using ATR-based stops instead of fixed percentages
   - The current stop-loss win rate by exit reason is shown above — use this to calibrate

6. **Updated Parameters**: Output the complete updated parameters as a JSON block that can be directly loaded into the system.

IMPORTANT constraints:
- Do NOT skip more than 1 day of the week. Skipping multiple days kills too many opportunities.
- Parameter changes should be incremental (10-30% adjustments), not dramatic swings.
- If stop_loss win rate is 0%, the top priority is widening stops — other optimizations are secondary.
- Effect sizes in this data are usually small (well under 0.5); that alone is not a reason
  for low confidence. When analyzer skip rules are listed above, prefer them: each removes
  enough trades, raises expectancy after fees, and holds in both halves of the data.
- Categorical features take == or !=, e.g. "Skip when session == european".

Be specific and actionable. Include the exact threshold values for filters.
Respond with valid JSON in the following structure:
```json
{
  "key_insights": ["insight1", "insight2", ...],
  "parameter_changes": [{"param": "name", "old": value, "new": value, "reason": "..."}],
  "filter_rules": [{"rule": "Skip when feature_name > 0.15", "type": "skip", "expected_impact": "..."}],
  "risk_assessment": "...",
  "updated_parameters": { ... complete parameter dict ... },
  "confidence": "low|medium|high",
  "notes": "..."
}
```"""

    return prompt


def run_claude_review(strategy: str = None, run_id: str = None,
                      sim_results: dict = None, dry_run: bool = False,
                      run_ids: list = None, history: list = None,
                      holdout_days: int = None) -> dict:
    """Run a Claude-powered analysis of recent trading performance.

    Args:
        strategy: Strategy to analyze
        run_id: Specific backtest run to analyze
        sim_results: Optional Monte Carlo results to include
        dry_run: If True, just build the prompt without calling API
        history: Past proposals with holdout outcomes, shown in the prompt
        holdout_days: Size of the holdout window Claude's proposal is judged on
        run_ids: Restrict analysis to these run_ids (e.g. just the
            current learning cycle's pair-level runs). Preferred over
            ``run_id`` when a cycle produces multiple runs.

    Returns:
        dict with Claude's proposal. Nothing is applied here; ``_session_id``
        is the learning_sessions row the evaluation gets recorded on.
    """
    from src.learning.promotion import describe_rule, promotion_config

    config = load_config()
    learning_cfg = config.get("learning", {})
    api_key = learning_cfg.get("claude_api_key", "")
    model = learning_cfg.get("claude_model", DEFAULT_MODEL)

    # Get findings
    findings = analyze_winners_vs_losers(run_id=run_id, strategy=strategy,
                                         run_ids=run_ids)
    if "error" in findings:
        return findings

    time_patterns = analyze_time_patterns(run_id=run_id, strategy=strategy,
                                          run_ids=run_ids)

    # Get current parameters
    conn = get_connection()
    current_params = load_latest_parameters(conn, strategy or "darvas")
    if current_params is None:
        current_params = config.get("strategy", {}).get(strategy or "darvas", {})
    conn.close()

    # Build prompt
    prompt = build_analysis_prompt(
        findings, time_patterns, current_params,
        strategy or "darvas", sim_results, history=history,
        holdout_days=holdout_days, promotion_rule=describe_rule(promotion_config(config)),
        active_filters=load_active_filters(strategy or "darvas"),
    )

    if dry_run:
        print("=" * 60)
        print("DRY RUN — Prompt that would be sent to Claude:")
        print("=" * 60)
        print(prompt)
        return {"dry_run": True, "prompt_length": len(prompt)}

    # Call Claude API
    if not api_key or api_key == "YOUR_CLAUDE_API_KEY":
        print("Claude API key not configured. Run with --dry-run to see the prompt.")
        return {"error": "Claude API key not configured"}

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Thinking is on by default on current models; max_tokens must
        # cover thinking plus the JSON reply.
        message = client.messages.create(
            model=model,
            max_tokens=16000,
            output_config={"effort": "high"},
            messages=[{"role": "user", "content": prompt}],
        )

        review = extract_review(message)
        if "error" in review:
            print(review["error"])
            return review
        if message.stop_reason == "max_tokens":
            print("⚠️ Claude response hit max_tokens — output may be truncated")

        # Save the learning session and results in a single connection
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO learning_sessions
                (session_type, strategy, num_trades_analyzed, findings_json,
                 recommendations_json)
                VALUES (?, ?, ?, ?, ?)
            """, (
                "claude_review", strategy or "all",
                findings.get("total_trades", 0),
                json.dumps(findings),
                json.dumps(review),
            ))

            # Nothing is applied here: the learning cycle backtests the
            # proposal on held-out data and promotes it only if it wins
            # (src/learning/promotion.py).
            session_id = cur.lastrowid
            conn.commit()
        finally:
            conn.close()

        review["_session_id"] = session_id
        return review

    except ImportError:
        print("anthropic package not installed. Run: pip install anthropic")
        return {"error": "anthropic package not installed"}
    except Exception as e:
        print(f"Claude API error: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Claude-powered trade analysis")
    parser.add_argument("--strategy", default="darvas")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompt without calling API")
    args = parser.parse_args()

    result = run_claude_review(
        strategy=args.strategy,
        run_id=args.run_id,
        dry_run=args.dry_run,
    )

    if not args.dry_run and "error" not in result:
        print("\n" + "=" * 60)
        print("CLAUDE ANALYSIS RESULTS")
        print("=" * 60)
        if "key_insights" in result:
            print("\nKey Insights:")
            for i, insight in enumerate(result["key_insights"], 1):
                print(f"  {i}. {insight}")
        if "filter_rules" in result:
            print("\nFilter Rules:")
            for rule in result["filter_rules"]:
                print(f"  [{rule.get('type', '?')}] {rule['rule']}")
        if "risk_assessment" in result:
            print(f"\nRisk Assessment: {result['risk_assessment']}")
        if "confidence" in result:
            print(f"Confidence: {result['confidence']}")
