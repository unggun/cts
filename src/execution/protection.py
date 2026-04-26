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


def _sleep_backoff(backoff_seconds, attempt: int) -> None:
    """Sleep based on the attempt index, clamped to the last entry of
    backoff_seconds. Empty backoff_seconds means no sleep."""
    if not backoff_seconds:
        return
    idx = min(attempt, len(backoff_seconds) - 1)
    time.sleep(backoff_seconds[idx])


# ─── OCO placement ───────────────────────────────────────────

def _build_oco_params(
    pair: str, size: float, stop_loss: float, target: float,
    strategy: str, stop_limit_slip_pct: float,
) -> dict:
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


def _build_stop_only_params(
    pair: str, size: float, stop_loss: float,
    strategy: str, stop_limit_slip_pct: float,
) -> dict:
    return {
        "symbol": tokocrypto_symbol(pair),
        "side": "SELL",
        "type": "STOP_LOSS_LIMIT",
        "quantity": size,
        "stopPrice": stop_loss,
        "price": stop_loss * (1 - stop_limit_slip_pct),
        "timeInForce": "GTC",
        "newClientOrderId": f"cts-{strategy}-stop-{uuid.uuid4().hex[:8]}",
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
        params = _build_stop_only_params(
            pair, size, stop_loss, strategy, stop_limit_slip_pct,
        )
        endpoint = exchange.private_post_open_v1_orders
        success_mode = "stop_only"
    else:
        params = _build_oco_params(
            pair, size, stop_loss, target, strategy, stop_limit_slip_pct,
        )
        endpoint = exchange.privatePostOpenV1OrdersOco
        success_mode = "oco"

    for attempt in range(retry):
        try:
            resp = endpoint(params)
            if success_mode == "oco":
                list_id_raw = resp.get("orderListId") or resp.get("listClientOrderId")
                list_id = str(list_id_raw) if list_id_raw is not None else None
                children = resp.get("orders", [])
                if len(children) >= 2:
                    # Tokocrypto/Binance-compatible OCO returns the limit
                    # (TP) leg at index 0 and the stop-limit (SL) leg at
                    # index 1. Tasks 6/7 rely on this ordering.
                    tp_id = str(children[0]["orderId"])
                    sl_id = str(children[1]["orderId"])
                else:
                    tp_id = sl_id = None
                return ProtectionResult(
                    mode="oco", order_id=list_id,
                    child_ids=(tp_id, sl_id) if tp_id and sl_id else None,
                )
            else:
                order_id_raw = resp.get("orderId")
                return ProtectionResult(
                    mode="stop_only",
                    order_id=str(order_id_raw) if order_id_raw is not None else None,
                )
        except ccxt.BaseError as exc:
            if _is_permanent_error(exc):
                return ProtectionResult(mode="error")
            if attempt < retry - 1:
                _sleep_backoff(backoff_seconds, attempt)

    return ProtectionResult(mode="polling")
