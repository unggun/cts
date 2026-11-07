# Market-regime features (Supertrend, ADX, BTC trend)

## Why

Every strategy traded on 1h signals with no view of the bigger trend. The
4h/1d candles were downloaded but unused (and their stored copies lag the
1h data by days).

## Design (defaults chosen without user input)

- `src/indicators/regime.py` builds 4h and daily candles from the 1h data
  (UTC boundaries) and adds per 1h bar: `st_4h_up`, `st_1d_up`
  (Supertrend 10/3 direction), `adx_4h` (ADX 14), `btc_st_4h_up`,
  `btc_return_24h`. BTC is `trading.regime_pair` (default `BTC/IDR`).
- **No look-ahead:** a 1h bar only sees higher-timeframe candles whose
  close time is ≤ its own close time. Tests check that truncating the data
  never changes past values and that a spiking forming candle doesn't move
  any closed bar.
- The same function runs in the engine (backtests, holdout promotion) and
  the paper trader. The live trader is unchanged: regime filter rules don't
  apply there (`apply_filters` ignores missing features).
- **No hand-wired trend switch.** The values are ordinary features: the
  rule search and Claude can use them, and promotion on the 90-day holdout
  decides.

## Evidence (2026-11-07, production data)

Rule search on the same unfiltered train trades, holdout trades /
expectancy per trade:

| strategy | no filters | without regime | with regime | rule picked |
|---|---|---|---|---|
| macd | 151 / −0.05% | 95 / −0.38% | 75 / −0.03% | `adx_4h >= 21.8` |
| cvd | 211 / +0.00% | 145 / +0.04% | 122 / +0.06% | `adx_4h >= 22.9` |
| other 7 | | unchanged | unchanged | non-regime rule wins |

A fixed "skip when daily Supertrend is down" rule would have helped the
holdout in 8 of 9 strategies (e.g. sr_breakout +0.51% → +0.84%, cvd
+0.00% → +0.28%), but on the train window it removed ~70% of trades and
*lowered* expectancy in 6 of 8. The effect flips between periods, so
the rule search correctly didn't adopt it; hard-coding it would fit the
last 90 days.

## Later

Supertrend as a trailing exit; regime features in the live trader; a
regime-aware evaluation over several holdout windows.
