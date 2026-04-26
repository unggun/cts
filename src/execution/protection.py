"""Exchange-side protection orders for live_trader.

Wraps the Tokocrypto raw OCO and stop-loss-limit endpoints (CCXT 4.5+
exposes them as `privatePostOpenV1OrdersOco` and
`private_post_open_v1_orders`), plus the per-loop fill detection and
restart reconciliation logic.

See docs/superpowers/specs/2026-04-26-tokocrypto-oco-protection-design.md
"""
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import ccxt


# ─── Result types ────────────────────────────────────────────

@dataclass
class ProtectionResult:
    """Outcome of placing or replacing a protective order.

    `mode` is the source of truth that the trader writes to position state
    and the trade_journal column.

      "oco"        — full OCO placed; child_ids populated
      "stop_only"  — single stop-loss-limit placed (Darvas-ATH path)
      "polling"    — placement failed retries; trader falls back to
                     client-side _check_exit
      "kept"       — only used by replace_trailing_stop when cancel
                     failed: old protection still active, in-memory
                     state must NOT change
      "error"      — a permanent error (InsufficientFunds, InvalidOrder,
                     AuthenticationError); operator must intervene
    """
    mode: str
    order_id: Optional[str] = None
    child_ids: Optional[tuple[str, str]] = None  # (TP_id, SL_id)


@dataclass
class FillResult:
    """Protection order has been filled by the exchange."""
    price: float
    reason: str  # "target" | "stop_loss"
    time: str


class LostProtection:
    """Sentinel: protection cancelled out from under us (operator
    cancelled it on the exchange UI, or the exchange did)."""
    def __repr__(self):
        return "LostProtection()"


# ─── Reconcile sentinels ─────────────────────────────────────

@dataclass
class Active:
    """Reconcile result: protection still on the book; bot adopts it."""
    order_id: str
    child_ids: Optional[tuple[str, str]]


@dataclass
class ClosedWhileDown:
    """Reconcile result: protection fired while the bot was down."""
    price: float
    reason: str
    time: str


class Naked:
    def __repr__(self):
        return "Naked()"


# ─── Helpers ─────────────────────────────────────────────────

def tokocrypto_symbol(ccxt_pair: str) -> str:
    """Convert CCXT-format pair "BTC/IDR" to Tokocrypto raw-endpoint
    format "BTC_IDR". Already-underscored inputs pass through."""
    return ccxt_pair.replace("/", "_")


# ─── OCO placement ───────────────────────────────────────────

def _build_oco_params(pair, size, stop_loss, target, strategy, stop_limit_slip_pct):
    return {
        "symbol": tokocrypto_symbol(pair),
        "side": "SELL",
        "quantity": size,
        "price": target,
        "stopPrice": stop_loss,
        "stopLimitPrice": stop_loss * (1 - stop_limit_slip_pct),
        "stopLimitTimeInForce": "GTC",
        "listClientOrderId": f"cts-{strategy}-{uuid.uuid4().hex[:8]}",
    }


def _is_permanent_error(exc) -> bool:
    """Errors that mean retrying won't help — surface immediately."""
    return isinstance(exc, (
        ccxt.InsufficientFunds,
        ccxt.InvalidOrder,
        ccxt.AuthenticationError,
    ))


def place_protection(
    exchange, pair, size, stop_loss, target, is_ath, strategy,
    retry=3, backoff_seconds=(1, 2, 4), stop_limit_slip_pct=0.005,
) -> ProtectionResult:
    """Place an OCO (non-ATH) or single STOP_LOSS_LIMIT (ATH) sell order
    to protect a long position. Retries transient errors, falls back to
    `mode="polling"` on persistent transient failures.
    """
    if is_ath:
        # ATH branch implemented in the next task.
        raise NotImplementedError("ATH stop-only path implemented in Task 4")

    params = _build_oco_params(pair, size, stop_loss, target, strategy,
                               stop_limit_slip_pct)

    last_exc = None
    for attempt in range(retry):
        try:
            resp = exchange.privatePostOpenV1OrdersOco(params)
            list_id = str(resp.get("orderListId") or resp.get("listClientOrderId"))
            children = resp.get("orders", [])
            if len(children) >= 2:
                tp_id = str(children[0]["orderId"])
                sl_id = str(children[1]["orderId"])
            else:
                tp_id = sl_id = None
            return ProtectionResult(
                mode="oco", order_id=list_id,
                child_ids=(tp_id, sl_id) if tp_id and sl_id else None,
            )
        except Exception as exc:
            if _is_permanent_error(exc):
                return ProtectionResult(mode="error")
            last_exc = exc
            if attempt < retry - 1:
                time.sleep(backoff_seconds[attempt])

    return ProtectionResult(mode="polling")
