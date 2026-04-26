"""Exchange-side protection orders for live_trader.

Wraps the Tokocrypto raw OCO and stop-loss-limit endpoints (CCXT 4.5+
exposes them as `privatePostOpenV1OrdersOco` and
`private_post_open_v1_orders`), plus the per-loop fill detection and
restart reconciliation logic.

See docs/superpowers/specs/2026-04-26-tokocrypto-oco-protection-design.md
"""
from dataclasses import dataclass
from typing import Optional


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
    order_id: str
    child_ids: Optional[tuple[str, str]]


@dataclass
class ClosedWhileDown:
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
