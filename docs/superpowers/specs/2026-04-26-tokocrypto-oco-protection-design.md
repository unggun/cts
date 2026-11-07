# Tokocrypto Exchange-Side Protection (OCO + Stop-Only)

**Date:** 2026-04-26
**Status:** Design approved, pending implementation plan
**Scope:** `src/execution/live_trader.py` and supporting modules

## Problem

The current live trader places a real limit-buy on Tokocrypto, but stop-loss, take-profit, and trailing stops are all enforced **client-side** by polling candle high/low and firing fresh sell orders when levels cross (`live_trader.py:238–263`). This creates two failure modes:

1. **Polling gap risk.** Between loop iterations (default ~10 minutes for 1h timeframe), price can blow through the stop-loss with no order standing on the book. The bot only reacts on the next candle.
2. **Bot-down risk.** If the process dies, positions have zero protection on the exchange. `_restore_positions` recovers state from DB but the position was naked the whole time the bot was offline.

Tokocrypto exposes `POST /open/v1/orders/oco` (HMAC SHA256). CCXT 4.5.44's `tokocrypto` class exposes it via the raw `privatePostOpenV1OrdersOco` method (no high-level wrapper). We can place protective orders directly on the exchange and eliminate both failure modes for fixed-target strategies.

## Goal

Live positions are protected by exchange-side orders whenever possible. The current polling-based protection becomes a fallback path, never the default.

## Non-Goals

- Migrating paper_trader's exit simulation. Paper already triggers on candle high/low, which is a closer simulation of OCO than of polling. No behavior change there.
- Refactoring shared entry/exit logic between paper and live traders. Out of scope.
- Adding new strategies or changing how `calculate_*_levels` work.

## Design Decisions

Four decisions made during brainstorming, in order:

1. **Darvas-ATH path: stop-loss-limit only, no OCO.** ATH-breakout positions have no fixed target — they trail. Placing an OCO with a fake target leg is dishonest and adds API surface. Use a single `STOP_LOSS_LIMIT` order; cancel-and-replace it when the trail ratchets up.
2. **Buy-fill protection failure: 3× retry then fall back to polling.** If protection placement fails after the buy filled, retry with `[1s, 2s, 4s]` backoff, then fall back to local polling and alert via Telegram. Never auto-flatten a position over a transient API error.
3. **Restart reconciliation: query exchange.** On bot restart, for each restored position, check `fetch_open_orders` and `fetch_my_trades` to determine if the protection is still active, already filled while down, or naked. Three explicit code paths.
4. **Paper trader: `protection_mode` field added for observability, no logic change.** A new column on `trade_journal` makes the protection state visible everywhere; paper writes `"simulated"`, live writes the real mode.

## Architecture

### Module layout

A new `src/execution/protection.py` owns all protection-order logic. `LiveTrader` calls into it. The trader stays focused on the loop, sizing, and journaling.

```
src/execution/
├── live_trader.py     # unchanged structurally; calls protection.* helpers
├── paper_trader.py    # writes protection_mode="simulated" to journal
└── protection.py      # NEW
```

`protection.py` exposes four functions:

| Function | Purpose |
|---|---|
| `place_protection(exchange, pair, size, stop_loss, target, is_ath, retry, backoff_seconds) -> ProtectionResult` | Branch on `is_ath`; place OCO or stop-only; retry; return mode + ids. |
| `replace_trailing_stop(exchange, pair, old_order_id, new_stop, size) -> ProtectionResult` | Cancel-and-replace the stop for ATH ratchet. |
| `check_protection_fill(exchange, pair, position) -> FillResult \| LostProtection \| None` | Per-loop status check. |
| `reconcile_on_restart(exchange, pair, position) -> Active \| ClosedWhileDown \| Naked` | Restart state recovery. |

### State model

**Position dict** (in-memory, in `LiveTrader.positions[pair]`) gains:

```python
"protection_mode": "oco" | "stop_only" | "polling"
"protection_order_id": str | None        # OCO list-id or single stop order id
"protection_child_ids": tuple[str, str] | None   # (TP_id, SL_id) for OCO; None for stop-only
```

**Schema change.** One column on `trade_journal`:

```sql
ALTER TABLE trade_journal ADD COLUMN protection_mode TEXT;
```

NULL means "legacy or unknown" — existing rows and any path that doesn't set the field will work unchanged. The migration is an idempotent `ALTER TABLE ... ADD COLUMN` wrapped in try/except in `src/data/database.py:init_db()`.

## Place Flow

The existing `_execute_buy` logic (lines 313–355) is unchanged through fill confirmation. After the fill is confirmed and before the journal write, a new step runs:

```python
result = place_protection(
    self.exchange, pair, filled_size,
    stop_loss=levels["stop_loss"],
    target=levels.get("target"),
    is_ath=levels.get("is_ath", False),
    retry=3, backoff_seconds=[1, 2, 4],
)
self.positions[pair]["protection_mode"] = result.mode
self.positions[pair]["protection_order_id"] = result.order_id
self.positions[pair]["protection_child_ids"] = result.child_ids
```

### Inside `place_protection`

**Non-ATH branch — OCO.** Build params:

```python
{
  "symbol": tokocrypto_symbol(pair),       # e.g., "BTC_IDR"
  "side": "SELL",
  "quantity": size,
  "price": target,                         # take-profit limit
  "stopPrice": stop_loss,                  # stop trigger
  "stopLimitPrice": stop_loss * 0.995,     # 0.5% slip room on stop fill
  "stopLimitTimeInForce": "GTC",
  "listClientOrderId": f"cts-{strategy}-{uuid4().hex[:8]}",
}
```

POST via `exchange.privatePostOpenV1OrdersOco(params)`. Response includes `orderListId` and an `orders` array with the two child `orderId` values — capture both.

**ATH branch — stop-loss-limit only.** Build params:

```python
{
  "symbol": tokocrypto_symbol(pair),
  "side": "SELL",
  "type": "STOP_LOSS_LIMIT",
  "quantity": size,
  "stopPrice": stop_loss,
  "price": stop_loss * 0.995,
  "timeInForce": "GTC",
  "newClientOrderId": f"cts-{strategy}-stop-{uuid4().hex[:8]}",
}
```

POST via `exchange.private_post_open_v1_orders(params)`.

**Retry loop.** 3 attempts with `[1s, 2s, 4s]` backoff. Catch `ccxt.NetworkError`, `ccxt.ExchangeNotAvailable`, and generic `ccxt.ExchangeError`. **Do not retry** on these permanent errors — surface them immediately:

- `ccxt.InsufficientFunds` — fill amount disagrees with what we own (a bug, not a transient).
- `ccxt.InvalidOrder` — params malformed (size below minimum, price outside band, etc.).
- `ccxt.AuthenticationError` — credentials wrong; retrying won't help.

### Outcomes

| Outcome | `ProtectionResult` | Side effects |
|---|---|---|
| Any retry succeeds | `mode="oco"` or `"stop_only"`, ids populated | Normal flow; journal write proceeds with `protection_mode` set. |
| All 3 retries fail | `mode="polling"`, ids None | Telegram: `⚠️ {pair} protection placement failed — falling back to local polling`. Position kept; existing `_check_exit` watches it. |
| `InsufficientFunds` | `mode="error"` (sentinel) | Telegram critical alert. Position is naked. Operator decides — bot does **not** auto-flatten. |

## Trailing Path (Darvas-ATH)

Inside `_check_exit`, when `position["is_ath"]` and `protection_mode == "stop_only"`:

```python
new_trail = high - 2 * ATR(14)
ratchet_gate = position["stop_loss"] * (1 + min_ratchet_pct)   # default 0.5%
if new_trail > ratchet_gate:
    result = replace_trailing_stop(
        self.exchange, pair,
        position["protection_order_id"], new_trail,
        position["position_size"],
    )
    if result.mode == "polling":
        # cancel succeeded but new placement failed → fall through
        position["protection_mode"] = "polling"
        position["protection_order_id"] = None
        self._notify(f"⚠️ {pair} trail replace failed — polling fallback")
    elif result.mode == "kept":
        # cancel failed; old stop still active. Don't update in-memory state.
        pass
    else:  # mode == "stop_only" (success)
        position["stop_loss"] = new_trail
        position["protection_order_id"] = result.order_id
```

`min_ratchet_pct` is config-tunable (`risk.protection.min_trail_ratchet_pct`, default `0.005`). Without this gate, every candle could nudge the trail by a few sats and burn API quota with hundreds of cancel+replace cycles per day.

`replace_trailing_stop` is `cancel_order(old_id) → place_protection(is_ath=True, ...)`. If cancel fails entirely, leave the old stop in place and return `mode="kept"` — neither stop_loss nor order_id should change in memory; the next ratchet attempt will retry.

## Exit Detection

For each position with `protection_mode in ("oco", "stop_only")`, each loop iteration calls `check_protection_fill`.

### OCO mode

Tokocrypto's `fetchOrder` does not reliably accept the OCO list-id. Use:

1. `fetch_open_orders(pair)` — match against `position["protection_child_ids"]`.
2. **Both children open** → `None` (still active).
3. **Neither child open** → `fetch_my_trades(pair, since=position.entry_time, limit=20)`. Find a SELL trade whose `order.clientOrderId` matches `child_ids[0]` (TP) or `child_ids[1]` (SL). Return `FillResult(price=trade.average, reason="target" | "stop_loss", time=trade.timestamp)`.
4. **One child open, one gone** → same `fetch_my_trades` lookup; the gone child either filled (FillResult) or was auto-cancelled by OCO when the other one filled (same lookup path covers both, since the filled one shows up in trades).

### Stop-only mode (ATH)

Simpler — use `fetch_order(protection_order_id, pair)`:

| Status | Result |
|---|---|
| `open` | `None` |
| `closed` / `filled` | `FillResult(price=order.average, reason="stop_loss")` |
| `canceled` (and we didn't cancel it) | `LostProtection` — operator cancelled outside the bot, or exchange did. Trader flips `protection_mode = "polling"`, alerts, falls through to `_check_exit` next tick. |

### When `FillResult` returns

The trader:
- Computes PnL exactly like `_execute_sell` does today (`live_trader.py:426–435`).
- Calls `_log_exit` with the actual fill price and reason.
- Sends Telegram with the existing `📉 *SELL {pair} ({reason})*` format.
- Removes from `self.positions`.

**No `_execute_sell` call** — the exchange already filled the sell. This is the key behavioral difference from today.

## Restart Reconciliation

`_restore_positions` now reads `protection_mode` from the DB row and dispatches:

```python
for pair, pos in restored_rows:
    if pos["protection_mode"] in ("oco", "stop_only"):
        result = reconcile_on_restart(self.exchange, pair, pos)
        match result:
            case Active(order_id, child_ids):
                pos["protection_order_id"] = order_id
                pos["protection_child_ids"] = child_ids
                self.positions[pair] = pos
            case ClosedWhileDown(exit_price, exit_reason, exit_time):
                self._log_exit(conn, pair, exit_price, exit_reason, ...)
                self._notify(
                    f"ℹ️ {pair} closed while bot was down: "
                    f"{exit_reason} @ {exit_price:,.0f} (at {exit_time})"
                )
                # not added to self.positions
            case Naked():
                self._notify(f"⚠️ {pair} restored without protection — re-placing")
                result = place_protection(...)  # full retry+fallback flow
                pos["protection_mode"] = result.mode
                pos["protection_order_id"] = result.order_id
                self.positions[pair] = pos
    else:  # protection_mode == "polling" or NULL (legacy)
        self.positions[pair] = pos  # restore as today
```

### Inside `reconcile_on_restart`

1. `fetch_open_orders(pair)` — match against `position["protection_order_id"]` (stop-only) or `position["protection_child_ids"]` (OCO).
2. **Match found** → `Active(order_id, child_ids)`.
3. **No match** → `fetch_my_trades(pair, since=position.entry_time, limit=50)`. Look for a SELL trade whose `order.clientOrderId` matches the stored ids and whose amount is within 1% of `position.position_size` (fees and rounding can shave it). If found → `ClosedWhileDown(price, reason, time)`. Reason inferred from which child id matched (TP → `"target"`, SL → `"stop_loss"`).
4. **No match and no exit trade** → `Naked()`.

This handles: planned restarts, crash during a fill, manual order cancellation from the Tokocrypto UI, and OCOs that fired in the gap.

## Failure Modes (Recap)

| Scenario | Behavior |
|---|---|
| Buy fills, OCO POST fails 3× | Position kept, `mode="polling"`, Telegram warn. |
| OCO placed but later cancelled externally | Detected next loop via `LostProtection`, `mode` flipped to `polling`, alert. |
| Trailing replace: cancel OK, new place fails 3× | `mode` flipped to `polling`, alert. |
| Trailing replace: cancel itself fails | Old stop kept, no-op; next ratchet retries. |
| Bot restart, OCO already filled while down | `ClosedWhileDown` — close position in DB with real exit, alert. |
| Bot restart, OCO still active | `Active` — adopt orderId, continue. |
| Bot restart, OCO gone but no exit fill | `Naked` — re-place protection via full retry+fallback flow. |
| `STOP_TRADING` kill switch flipped | Loop exits. Open OCOs/stops **remain on exchange** — actually safer than today. Document in kill-switch comment. |
| `InsufficientFunds` on protection placement | No retry (it's a bug indicator). Critical alert. Position naked, operator decides. |

## Configuration

New `risk` config keys (`config.yaml`), all optional with sensible defaults:

```yaml
risk:
  protection:
    enabled: true                  # master switch; false = legacy polling-only
    stop_limit_slip_pct: 0.005     # how far below stopPrice the limit price sits
    min_trail_ratchet_pct: 0.005   # ATH ratchet gate
    retry_attempts: 3
    retry_backoff_seconds: [1, 2, 4]
```

The `enabled: false` switch is the primary rollback. Flipping it makes the bot behave exactly as it does today (polling-only) without code redeploy.

## Pre-Flight Gate

Add to `preflight_check`:

```python
# OCO capability check
has_oco = hasattr(exchange, "privatePostOpenV1OrdersOco")
checks.append(("CCXT supports Tokocrypto OCO", has_oco))
```

If a future CCXT upgrade regresses this method's name, preflight fails loudly before any live order is placed.

## Files Touched

| File | Change |
|---|---|
| `src/execution/protection.py` | NEW — all protection logic. |
| `src/execution/live_trader.py` | Wire `place_protection`, `check_protection_fill`, `replace_trailing_stop`, `reconcile_on_restart` into the loop. Update `_execute_buy`, `_check_exit`, `_restore_positions`. Drop `_execute_sell` calls from the OCO/stop paths. |
| `src/data/database.py` | Add `protection_mode TEXT` column to `trade_journal` via idempotent migration in `init_db()`. |
| `src/execution/paper_trader.py` | Pass `protection_mode="simulated"` to journal writes. |
| `src/execution/positions.py` | Surface `protection_mode` in CLI output. |
| `src/execution/telegram_bot.py` | Surface `protection_mode` in `/positions` reply. |
| `config.example.yaml` | Document new `risk.protection.*` keys. |
| `tests/execution/test_protection.py` | NEW — unit tests (see Testing). |
| `requirements.txt` / `pyproject.toml` | Add `pytest` if not already present. |

## Testing

Tight scope — this is critical-path code touching real money. Verification matters more than coverage breadth.

### Unit tests (`tests/execution/test_protection.py`)

- OCO param builder produces correctly-formatted body for known input (snapshot fixture).
- Retry loop:
  - 2 fails then a success → returns `mode="oco"`.
  - 3 fails → returns `mode="polling"`.
  - `InsufficientFunds` on first attempt → no retry, returns `mode="error"`.
- `check_protection_fill` decoder: mocked `fetch_open_orders` + `fetch_my_trades` responses for target-fill, stop-fill, lost-protection, and still-open cases.
- Reconciler: three scenarios (`Active`, `ClosedWhileDown`, `Naked`) each with mocked exchange responses.
- Trailing replace: ratchet gate suppresses sub-threshold bumps, fires above threshold.

Use plain `unittest.mock` against CCXT methods. The codebase has no existing test framework (`find /opt/cts/tests` returns nothing), so this PR sets up `pytest`.

### Integration test (manual)

On Tokocrypto live with a tiny IDR position (~Rp 50k):

1. Open one BTC/IDR position.
2. Verify OCO appears in the Tokocrypto web UI with both legs.
3. Manually cancel the OCO from the web UI → next loop should flip to `polling` and alert via Telegram.
4. Restart the bot mid-position → reconciliation should adopt the existing OCO.
5. Set `stop_loss` close to current price to trigger a fill → verify exit appears in `trade_journal` with correct `protection_mode` and reason.

### Pre-deploy checks

- Preflight gate passes (capability check).
- Run with `risk.protection.enabled: false` first to confirm no behavior change vs. current main.
- Flip to `true` and run a single pair in paper mode for one week before any live position.

## Rollback

`risk.protection.enabled: false` reverts behavior to today's polling-only model in one config change. The new code paths short-circuit when the flag is off; the polling path is preserved unchanged.

If a more catastrophic rollback is needed, revert the PR — the schema change is additive (NULL-able column) so revert doesn't lose data.
