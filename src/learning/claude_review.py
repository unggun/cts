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
from src.data.database import get_connection, save_parameters, load_latest_parameters
from src.learning.analyzer import analyze_winners_vs_losers, analyze_time_patterns
from src.learning.filter_rules import (
    init_filter_rules_table, save_filter_rules,
    parse_analyzer_filters, parse_claude_filter_rules,
)


def build_analysis_prompt(findings: dict, time_patterns: dict,
                          current_params: dict, strategy: str,
                          sim_results: dict = None) -> str:
    """Build the prompt for Claude to analyze trading patterns."""

    prompt = f"""You are a quantitative trading analyst reviewing backtest results for a '{strategy}' strategy on crypto (Tokocrypto exchange, IDR pairs).

## Current Strategy Parameters
```json
{json.dumps(current_params, indent=2)}
```

## Backtest Analysis Findings
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

    # Add suggested filters
    if findings.get("suggested_filters"):
        prompt += "\n## Auto-Generated Filter Suggestions\n"
        for f in findings["suggested_filters"][:10]:
            prompt += f"- {f['rule']} (effect: {f['effect_size']}, reason: {f['rationale']})\n"

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

    prompt += """
## Your Task

Based on this analysis, provide:

1. **Key Insights**: What are the 3-5 most important patterns you see? Which features most strongly predict winners vs losers?

2. **Parameter Recommendations**: Suggest specific changes to the strategy parameters. For each change, explain why and what improvement you expect.

3. **Filter Rules**: Suggest concrete, machine-parseable filter rules. Each rule must reference an exact feature name from the feature analysis above and use a comparison operator.
   Format: "Skip when feature_name > threshold" or "Prefer when feature_name < threshold"
   Use actual feature names (e.g., rsi_bull_divergence, upper_wick_ratio, mixed_ema_signals) and numeric thresholds.

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
- Filter rules should target features with effect_size > 0.5 to avoid over-filtering.

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
                      block_param_updates: bool = False) -> dict:
    """Run a Claude-powered analysis of recent trading performance.

    Args:
        strategy: Strategy to analyze
        run_id: Specific backtest run to analyze
        sim_results: Optional Monte Carlo results to include
        dry_run: If True, just build the prompt without calling API
        block_param_updates: If True, skip parameter updates regardless of confidence

    Returns:
        dict with Claude's analysis and recommendations
    """
    config = load_config()
    learning_cfg = config.get("learning", {})
    api_key = learning_cfg.get("claude_api_key", "")
    model = learning_cfg.get("claude_model", "claude-sonnet-4-20250514")

    # Get findings
    findings = analyze_winners_vs_losers(run_id=run_id, strategy=strategy)
    if "error" in findings:
        return findings

    time_patterns = analyze_time_patterns(run_id=run_id, strategy=strategy)

    # Get current parameters
    conn = get_connection()
    current_params = load_latest_parameters(conn, strategy or "darvas")
    if current_params is None:
        current_params = config.get("strategy", {}).get(strategy or "darvas", {})
    conn.close()

    # Build prompt
    prompt = build_analysis_prompt(
        findings, time_patterns, current_params,
        strategy or "darvas", sim_results
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

        message = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = message.content[0].text

        # Try to parse JSON from response
        try:
            # Find JSON block in response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                review = json.loads(response_text[json_start:json_end])
            else:
                review = {"raw_response": response_text}
        except json.JSONDecodeError:
            review = {"raw_response": response_text}

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

            # Confidence-gated parameter updates
            confidence = review.get("confidence", "medium")
            strat_name = strategy or "darvas"

            if "updated_parameters" in review:
                if block_param_updates:
                    print(f"⚠️ Skipping parameter update — quality gates failed")
                elif confidence == "low":
                    print(f"⚠️ Skipping parameter update — Claude confidence is LOW")
                    print(f"  Reason: {review.get('notes', 'insufficient data or unclear patterns')}")
                else:
                    version = save_parameters(
                        conn, strat_name,
                        review["updated_parameters"],
                        source="claude_auto_review",
                        performance={"win_rate": findings.get("overall_win_rate")},
                        notes=json.dumps(review.get("key_insights", []))
                    )
                    print(f"Saved updated parameters as version {version} (confidence: {confidence})")

            # Save filter rules from Claude (reuse connection)
            if review.get("filter_rules"):
                init_filter_rules_table(conn)
                structured_rules = parse_claude_filter_rules(review["filter_rules"])
                if structured_rules:
                    save_filter_rules(strat_name, structured_rules,
                                      source="claude_review", conn=conn)
                    print(f"Saved {len(structured_rules)} filter rules from Claude review")

            # Also save analyzer's suggested filters (reuse connection)
            if findings.get("suggested_filters"):
                init_filter_rules_table(conn)
                analyzer_rules = parse_analyzer_filters(findings["suggested_filters"])
                if analyzer_rules:
                    save_filter_rules(strat_name, analyzer_rules,
                                      source="analyzer", conn=conn)
                    print(f"Saved {len(analyzer_rules)} filter rules from analyzer")

            conn.commit()
        finally:
            conn.close()

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
