# run_backtest tool for the Claude review — Design

## Why

Claude proposes blind. It reads the analyzer's statistics, writes one bundle
(parameters + filter rules), and learns only a week later whether that bundle
wins on the holdout. A bad idea costs a full weekly cycle. Examples: a filter that
removes almost every trade, or a parameter change whose effect Claude guessed
wrong.

With a `run_backtest` tool, Claude can check an idea on the **train window**
before proposing it. It can see how many trades a change keeps, what it does to
expectancy, and whether the effect holds in both halves of the window. Then it
proposes the version it has actually seen work.

## Goals

- Claude can backtest candidate bundles on the train window during its review.
- The holdout stays untouched and remains the only judge of promotion.
- Every tool call and its result is recorded, so the journal can show what Claude
  tried before it proposed.

## Non-goals

- No change to the promotion rule, the holdout, or what gets saved.
- No backtests on the holdout window, under any input.
- No new strategies or indicator code written by Claude. The tool runs existing
  strategies with different parameters and filters only.
- No Managed Agents or memory (that is P3).

## Anti-goals (would make this a failure even if it works)

- **Holdout leakage.** If any holdout candle reaches the tool's backtest, the
  holdout stops being an honest test. This is the one hard guarantee.
- **Train-window overfitting dressed up as research.** Ten tries at small
  variations, followed by proposing the best one, is multiple testing. The
  result would look strong on train and fail on the holdout.
  - Evidence that this risk is real: the current versions differ sharply between
    the two windows. On 2026-11-11, sr_breakout was −0.50% on train vs +0.85% on
    the holdout. Candlestick (no filters) was −0.44% vs +0.41%.
  - The budget and the half-split below are there to limit this. The holdout
    still bounds the damage: at worst, nothing gets promoted.

## Design

### The tool

```
run_backtest(parameter_changes: object, filter_rules: array) -> result
```

- **`parameter_changes`** holds only the keys to change. They are merged onto the
  current version's parameters. The final proposal still gives the complete set.
  - Keys the current version doesn't have produce a warning in the result, not an
    error, because some strategies have defaults in code.
  - Invalid values are rejected with the same `invalid_parameters` check promotion
    uses, and the error goes back to Claude as the tool result.
- **`filter_rules`** use the same shape as the proposal's `filter_rules`, and are
  required.
  - They go through `candidate_filters`, the same function promotion uses. The
    analyzer's skip rules are added, then the live cap applies (3 skip rules, one
    per feature). So Claude tests exactly what its final proposal would apply.
  - The result lists the rules that were applied and those dropped by the cap.
  - `[]` means only the analyzer's rules.
- **The window** is the train window only. The cut-off is the holdout start the
  learning cycle already computed (`train_end=trade_start`). It is not recomputed:
  recomputing "now − 90 days" minutes later would move the boundary past a
  holdout candle. Candles are cut there **before** they reach the engine, not just at entry time, so indicators can't
  see holdout bars either. Warm-up uses the earlier history, as the train
  backtests already do.
- **The result:**
  - trades, expectancy after costs, win rate, max drawdown
  - trades per pair
  - expectancy in each chronological half of the train window
  - the same numbers for the current version, so every result reads as a
    comparison (computed once per review and reused)
- Costs, the regime pair and the closed-candle rule are the same as in
  `evaluate_bundle`, which the tool reuses with a train-window cut.

### Budget

- At most `learning.backtest_tool.max_calls` calls per review (default **6**).
- After that, the tool returns "budget used up, give your final answer" as an
  error result. The tool stays in the request, because a conversation that
  contains `tool_use` blocks needs it defined. Calls that fail validation don't
  use up the budget.
- A train backtest takes about 15–30 s on the VPS (measured 2026-11-11: 16–28 s).
  - Six calls add up to about 3 minutes per strategy.
  - Across 10 strategies, the weekly run grows by up to about 30 minutes.
  - Running the pairs in parallel could cut this later. That is not part of this
    change.

### The conversation loop

- `run_claude_review` switches from one `messages.create` call to a manual
  tool-use loop:
  - call the model
  - run any `run_backtest` calls
  - send back the `tool_result`s
  - repeat until Claude stops calling the tool or a turn limit (`max_calls + 2`)
    is reached. Before the last turn, a note says that no more backtests are
    available.
- The assistant's content blocks, including thinking blocks, are sent back
  unchanged.
- The prompt prefix is marked for prompt caching, because it is re-sent on every
  turn.
- The final message must contain the same JSON as today. `extract_review` reads
  it from the last assistant message.
- If the loop ends without valid JSON, that is the existing "Claude returned no
  parameters or filter rules" path. The strategy keeps its current version.

### Prompt changes

- Explain the tool, its budget, and that it sees the train window only.
- Say plainly that the holdout decides, and that small train gains are usually
  noise. A change worth proposing should help in **both halves** and keep enough
  trades.
- Ask Claude to use backtests to test its main hypothesis (the journal's
  `hypothesis.idea`), not to fine-tune numbers.

### Recording

- Each call (input, result, duration) is stored in
  `recommendations_json["backtests"]`. Token totals for the conversation are stored
  in `["api_usage"]`. No schema change is needed.
- `/journal <strategy>` and the CLI detail view show the number of backtests per
  review. The detail view lists them as one line each:
  `changes → trades / expectancy / halves`.
- The learning-cycle Telegram message adds "n backtests" to the review line.

### Dry run

`--dry-run` still shows the prompt, including the tool section, and makes no API
call. `python -m src.learning.backtest_tool --strategy s` runs the tool once on the
current version. That lets an operator see the output format and timing without
calling the API.

## Testing

- **Leakage:** with a fake engine that records the candles it receives, no
  timestamp ≥ `holdout_start` ever reaches it, whatever the input.
- **Executor:** merges partial parameters, rejects invalid values, caps filters
  and reports which ones were dropped, computes both halves, and reuses the
  current version's numbers.
- **Loop:** a fake client that calls the tool twice and then answers gives two
  recorded backtests and the final JSON. A fake client that keeps calling stops
  at the budget, and the extra calls get the "budget used up" result. Thinking
  blocks are passed back unchanged.
- **No regression:** a reply without tool calls behaves exactly as today.

## Decided by the builder (Claude), open to change

- Partial `parameter_changes` rather than full parameter sets: it keeps the calls
  short and makes it clear what changed.
- A budget of 6: enough for a hypothesis, a variant and a check, without
  inviting a parameter sweep.
- Show both halves, not a p-value. This matches the analyzer's rule search, which
  Claude already reads.

## Left to the builder

- The exact tool description text and the result formatting.
- The module layout. Suggested: `src/learning/backtest_tool.py` for the executor,
  with the loop in `claude_review.py`.
