# OCO Protection — Manual Integration Runbook

Use this checklist before flipping `risk.protection.enabled: true` in production.
Requires a small live IDR balance (Rp 100k is enough).

## Pre-flight

- [ ] `python -m src.execution.live_trader --preflight` — all gates ✅
- [ ] `risk.protection.enabled: false` deploy first; confirm bot behaves
      identically to today (polling-only) for at least 24h on one strategy.

## Smoke test

- [ ] Set `risk.protection.enabled: true`, single pair (BTC/IDR), single
      strategy (any non-Darvas), `risk.max_position_pct: 0.005` (small).
- [ ] Wait for an entry signal. After buy fills:
  - [ ] Telegram message includes `Protection: oco`.
  - [ ] Tokocrypto web UI shows an OCO with both legs at the spec'd prices.
  - [ ] DB row in `trade_journal` has `protection_mode='oco'`.
- [ ] Manually cancel the OCO from the Tokocrypto web UI:
  - [ ] Within one loop tick, Telegram sends "protection cancelled externally".
  - [ ] DB position remains open; `protection_mode` flips to `polling` in
        memory and is persisted to DB.
- [ ] Restart the bot mid-position:
  - [ ] On startup, "Adopted active oco for BTC/IDR" log line appears.
  - [ ] No duplicate OCO placed.
- [ ] Force a stop fire by setting an entry's `stop_loss` close to current
      price (or wait for natural exit):
  - [ ] Telegram "SELL" message appears with the right reason.
  - [ ] DB row updated with `exit_time`, `exit_price`, `exit_reason`. The
        `exit_time` should match the actual exchange fill time, not bot
        detection time.

## Darvas-ATH smoke test

- [ ] Switch to the Darvas strategy. After an ATH entry fills:
  - [ ] Telegram includes `Protection: stop_only`.
  - [ ] Tokocrypto UI shows a single STOP_LOSS_LIMIT, no OCO.
- [ ] Watch over several candles as price rises:
  - [ ] When the trailing stop bumps past `min_trail_ratchet_pct`, log
        line "trailing stop ratcheted to ..." appears.
  - [ ] UI shows the old order cancelled and a new STOP_LOSS_LIMIT at the
        higher price.
  - [ ] DB row `protection_order_id` is updated to the new id.

## Rollback

If anything goes sideways, set `risk.protection.enabled: false` and restart
the bot. All open positions revert to polling protection on next loop tick.
On restart, OCO/stop_only positions in DB are reconciled — any orders still
on the exchange are cancelled is up to the operator (the bot leaves them in
place by default; they continue to protect the position).
