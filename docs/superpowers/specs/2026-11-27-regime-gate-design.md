# Regime gate (per strategy) — Design

## Why

Every strategy only buys. From 2026-11-11 to 11-27, BTC/IDR fell 14.8%, and the
paper traders lost about 7.0M IDR on 73 closed trades. Only cvd made money. A
backtest of the same dates lost just as much (candlestick −2.63% per trade vs
−1.82% on paper; mean_reversion −0.83% vs −0.91%), so this is the market, not a
paper-trading bug.

A year of backtested trades (2025-12-08 to 2026-11-27, 1,921 trades from the 8
paper strategies, deployed versions) shows the following for a gate that only
allows buys while **BTC's daily Supertrend(10, 3) is up**:

- It would have blocked all 92 trades from Nov 11 to 27 (−0.80% per trade).
- With all strategies together, allowed trades average +0.18% and blocked trades
  −0.19%. But it doesn't hold over the year: in the first half (Dec–Jun), the
  allowed trades did worse than average (−0.24% vs −0.10%).
- Per strategy, it improves expectancy in **both** halves for candlestick (−0.35%
  → +0.26%), mean_reversion (+0.13% → +0.50%) and rsi_vwap (+0.39% → +0.97%). It
  hurts ma_crossover and macd.

Each daily gate blocks about 60% of trades. The analyzer's rule search only
accepts rules that keep at least 50% of trades, and Claude's filters are capped
at 3 and judged on a 90-day holdout that may contain no downtrend. So the
learning loop can't find this on its own.

**Decision (user, 2026-11-27):** gate per strategy, only where the data supports
it. Not one gate for every strategy.

## Goals

- A strategy with the gate on opens no new positions while BTC's daily trend is
  down.
- Whether each strategy has the gate is decided by data, with a fixed rule.
  Claude doesn't decide it, and it is re-checked on every learning run.
- Backtests, promotion, Claude's backtest tool and the paper trader all apply the
  same gate, so the numbers they report stay comparable.

## Non-goals

- No closing of open positions when the trend turns. Exits are unchanged.
- No shorting, and no position sizing by regime. The allocator (P2) is separate
  and comes later.
- No change to the live trader. It doesn't compute regime features yet, so the
  gate doesn't apply to live trades. This is the same limitation the regime
  filters already have, and it is documented.

## Anti-goals

- **Flip-flopping.** A gate that turns on and off every few days makes the paper
  results impossible to read. The rule below has hysteresis.
- **A gate that starves a strategy.** If a gated strategy would make almost no
  trades in a year, it should be off, or the strategy should be dropped. That's
  for the user to decide; the gate doesn't hide it.

## Design

### Signal

- New regime feature **`btc_st_1d_up`**: BTC's daily Supertrend(10, 3)
  direction, as of the newest daily candle that had **closed** when the bar
  closed.
- It is built from the 1h candles like the other regime features, so there is no
  look-ahead.
- It is NaN when the candles are stale or missing.
- It is added to `REGIME_COLUMNS`, so the analyzer, the rule search and Claude
  also see it as an ordinary feature.

### The gate

- The gate is one fixed skip rule, `btc_st_1d_up < 0.5`, applied **on top of** a
  strategy's filter rules.
  - It doesn't count toward the 3-rule cap.
  - Promotion never replaces it.
  - When the value is missing (NaN), the gate doesn't block, which matches
    `apply_filters`. *(My call: blocking on missing data would stop trading
    whenever the BTC feed lags.)*
- State is kept in a new table `regime_gates`:
  - one row per decision: strategy, `enabled`, `evidence_json`, `decided_at`
  - the newest row per strategy is the current state, and the history stays for
    the journal
- `load_regime_gate(strategy)` returns the gate's skip rule, or nothing.
  - `BacktestEngine` adds it to its filters unless told otherwise (new `gate`
    argument, default "use the stored state").
  - `evaluate_bundle` and the backtest tool pass the stored state explicitly.

### The decision rule (each learning run, per strategy)

1. After promotion, backtest the version that will trade, with its filters, over
   **all stored closed candles** (about a year), with and without the gate.
2. Split the trades at the midpoint of the window.
3. **Turn on** when all of these hold:
   - in both halves, gated expectancy ≥ ungated + `min_gain` (0.05%)
   - gated expectancy over the window > 0
   - gated trades ≥ `min_trades` (30)
4. **Turn off** when gated expectancy over the whole window is **below**
   ungated.
5. Otherwise keep the current state. *(My call: this hysteresis keeps a gate from
   flipping on small changes.)*
6. Store the decision with its evidence: trades, expectancy and halves, gated and
   not.

**Settings:** `learning.regime_gate`: `enabled` (true), `min_gain` (0.0005),
`min_trades` (30). With `enabled: false`, no gate is applied anywhere and none is
decided.

**On holdout overlap:** the decision uses the full history, including the 90-day
holdout. That's acceptable because the gate is not a Claude proposal: promotion
then judges the current version and the candidate **both** with the same gate
state, so the comparison stays fair.

### Where the gate applies

- **The cycle's train backtests**, the ones run with active filters: yes. They
  describe the deployed version.
- **The analyzer's unfiltered train backtest:** no. It stays unfiltered, so its
  blind spots stay visible.
- **Promotion:** the current version and the candidate are both evaluated with
  the current gate state.
- **Backtest tool:** the same as promotion, and each result says whether the gate
  was on.
- **Claude's prompt:** one line with the gate state and its evidence, plus "it is
  decided separately; don't propose a filter for it".
- **Paper trader:** it reloads the gate each loop, like its filters. A blocked
  entry is logged as `regime gate: BTC daily trend down`, and a change of state
  is announced on Telegram.
- **Live trader:** unchanged (see Non-goals).

### Reporting

- **Learning-cycle Telegram summary:** each strategy line shows 🛡 when its gate
  is on. Changes are listed, for example "candlestick: regime gate ON (gated
  +0.26% vs −0.35% ungated, both halves)".
- **`python -m src.learning.regime_gate [--strategy s] [--apply]`:** shows the
  evidence. With `--apply`, it stores the decisions. This lets the gate go live
  right after deploy without waiting for Sunday.
- **`/journal <strategy>`:** shows the current gate state.

## Testing

- **Feature:** `btc_st_1d_up` only uses closed daily candles. A test changes
  future candles and checks that past values don't change.
- **Gate:** a gated engine skips entries when `btc_st_1d_up` = 0 and allows them
  when it is 1 or NaN. Promotion's `replace_active_filters` doesn't remove it,
  and it doesn't count toward the cap.
- **Decision:**
  - both halves better → on
  - one half worse → no change
  - whole window worse → off
  - too few gated trades → no change (or off if it was on and is now worse)
- **Promotion and the tool:** they evaluate with the stored state and report it.
- **Paper trader:** a blocked entry is logged, and a gate change is picked up
  without a restart.

## Rollout

1. Merge, then restart the paper traders and the bot.
2. Run `python -m src.learning.regime_gate` (no `--apply`) on production and
   check the decisions against the numbers above.
3. Apply them with the user's OK, or wait for the next learning run.

## Left to the builder

The module layout (suggested: `src/learning/regime_gate.py`), message wording,
and how the evidence is formatted.
