# Tokocrypto OCO/Stop Exchange-Side Protection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace client-side polling for SL/TP/trailing stops in `live_trader` with exchange-side OCO orders (fixed-target strategies) and stop-loss-limit orders (Darvas-ATH trailing). Polling becomes the fallback path, never the default.

**Architecture:** Carve a new `src/execution/protection.py` module that owns all OCO/stop placement, fill detection, trailing replace, and restart reconciliation. `LiveTrader` orchestrates by calling these helpers. Add a `protection_mode` column to `trade_journal` to make the protection state observable. Master switch (`risk.protection.enabled`) allows config-only rollback to today's polling behavior.

**Tech Stack:** Python 3, CCXT 4.5+ (`privatePostOpenV1OrdersOco` and `private_post_open_v1_orders` raw endpoints), stdlib sqlite3, pytest with `unittest.mock`. Spec: `docs/superpowers/specs/2026-04-26-tokocrypto-oco-protection-design.md`.

---

## File Structure

- Create: `src/execution/protection.py` — all protection logic (~400 lines)
- Modify: `src/execution/live_trader.py` — wire protection helpers into the loop
- Modify: `src/data/database.py` — add `protection_mode` column migration to `init_db()`
- Modify: `src/execution/paper_trader.py` — write `protection_mode="simulated"` to journal
- Modify: `src/execution/positions.py` — surface `protection_mode` in CLI output
- Modify: `src/execution/telegram_bot.py` — surface `protection_mode` in `/positions` reply
- Modify: `config.example.yaml` — document new `risk.protection.*` keys
- Create: `tests/test_protection.py` — unit tests
- Modify: `requirements.txt` — pytest is already used by existing tests, but add it explicitly if missing

The module decomposition keeps `live_trader.py` from growing further (already 660 lines) and isolates the logic that's most affected by exchange API quirks, making it easier to unit-test with mocked CCXT.

## Module Interface

```python
# Public API of src/execution/protection.py

from dataclasses import dataclass

@dataclass
class ProtectionResult:
    """Result of placing or replacing protection."""
    mode: str  # "oco" | "stop_only" | "polling" | "kept" | "error"
    order_id: str | None = None
    child_ids: tuple[str, str] | None = None  # (TP_id, SL_id) for OCO

@dataclass
class FillResult:
    """Protection order has been filled by the exchange."""
    price: float
    reason: str  # "target" | "stop_loss"
    time: str    # ISO

@dataclass
class LostProtection:
    """Protection order was cancelled outside of our control."""
    pass

@dataclass
class Active:
    """Restart reconcile: protection still on the exchange."""
    order_id: str
    child_ids: tuple[str, str] | None

@dataclass
class ClosedWhileDown:
    """Restart reconcile: protection fired while bot was down."""
    price: float
    reason: str
    time: str

@dataclass
class Naked:
    """Restart reconcile: position has no protection."""
    pass

def place_protection(
    exchange, pair: str, size: float,
    stop_loss: float, target: float | None, is_ath: bool,
    strategy: str,
    retry: int = 3, backoff_seconds: list[int] = (1, 2, 4),
    stop_limit_slip_pct: float = 0.005,
) -> ProtectionResult: ...

def replace_trailing_stop(
    exchange, pair: str, old_order_id: str,
    new_stop: float, size: float, strategy: str,
    stop_limit_slip_pct: float = 0.005,
) -> ProtectionResult: ...

def check_protection_fill(
    exchange, pair: str, position: dict,
) -> FillResult | LostProtection | None: ...

def reconcile_on_restart(
    exchange, pair: str, position: dict,
) -> Active | ClosedWhileDown | Naked: ...

def tokocrypto_symbol(ccxt_pair: str) -> str:
    """Convert "BTC/IDR" → "BTC_IDR" for raw endpoint params."""
    return ccxt_pair.replace("/", "_")
```

---

## Task 1: Add `protection_mode` column to `trade_journal`

**Files:**
- Modify: `src/data/database.py:128-150` (after the `CREATE TABLE` for `trade_journal`)
- Test: `tests/test_protection.py` (new file, will grow over the plan)

- [ ] **Step 1: Write the failing test**

Create `tests/test_protection.py`:

```python
"""Unit tests for src/execution/protection.py and protection-related schema."""
import os
import tempfile
import pytest

from src.data.database import get_connection, init_db


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    yield path
    os.unlink(path)


class TestSchema:
    def test_trade_journal_has_protection_mode_column(self, tmp_db):
        conn = get_connection(tmp_db)
        cur = conn.execute("PRAGMA table_info(trade_journal)")
        cols = {row["name"] for row in cur.fetchall()}
        conn.close()
        assert "protection_mode" in cols

    def test_init_db_is_idempotent_when_column_exists(self, tmp_db):
        # Running init_db twice on the same DB must not raise.
        init_db(tmp_db)
        init_db(tmp_db)
        conn = get_connection(tmp_db)
        cur = conn.execute("PRAGMA table_info(trade_journal)")
        cols = [row["name"] for row in cur.fetchall()]
        conn.close()
        # Should not have duplicate columns.
        assert cols.count("protection_mode") == 1

    def test_can_insert_with_protection_mode(self, tmp_db):
        conn = get_connection(tmp_db)
        conn.execute("""
            INSERT INTO trade_journal
            (mode, pair, strategy, direction, entry_time, entry_price, protection_mode)
            VALUES ('live', 'BTC/IDR', 'darvas', 'long', '2026-04-26T10:00:00', 1000000, 'oco')
        """)
        conn.commit()
        cur = conn.execute("SELECT protection_mode FROM trade_journal")
        rows = cur.fetchall()
        conn.close()
        assert rows[0]["protection_mode"] == "oco"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /opt/cts && python -m pytest tests/test_protection.py::TestSchema -v
```

Expected: FAIL — `assert 'protection_mode' in cols` fails because column does not exist yet.

- [ ] **Step 3: Add the migration to `init_db`**

In `src/data/database.py`, after the `cur.execute("""CREATE TABLE IF NOT EXISTS trade_journal ...""")` block (after line 150) and before `conn.commit()`, add:

```python
    # ── Migration: add protection_mode column if missing ──
    # Used by src/execution/live_trader.py to track exchange-side protection state.
    cur.execute("PRAGMA table_info(trade_journal)")
    existing_cols = {row["name"] for row in cur.fetchall()}
    if "protection_mode" not in existing_cols:
        cur.execute("ALTER TABLE trade_journal ADD COLUMN protection_mode TEXT")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /opt/cts && python -m pytest tests/test_protection.py::TestSchema -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /opt/cts && git add src/data/database.py tests/test_protection.py
git commit -m "feat(db): add protection_mode column to trade_journal

Idempotent migration runs on init_db. NULL means legacy/unknown,
preserving existing rows.
"
```

---

## Task 2: Skeleton `protection.py` with result dataclasses + symbol helper

**Files:**
- Create: `src/execution/protection.py`
- Test: `tests/test_protection.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_protection.py`:

```python
from src.execution.protection import (
    ProtectionResult, FillResult, LostProtection,
    Active, ClosedWhileDown, Naked,
    tokocrypto_symbol,
)


class TestDataclasses:
    def test_protection_result_defaults(self):
        r = ProtectionResult(mode="oco", order_id="abc",
                             child_ids=("c1", "c2"))
        assert r.mode == "oco"
        assert r.order_id == "abc"
        assert r.child_ids == ("c1", "c2")

    def test_protection_result_polling_has_no_ids(self):
        r = ProtectionResult(mode="polling")
        assert r.order_id is None
        assert r.child_ids is None

    def test_fill_result(self):
        f = FillResult(price=1000000.0, reason="target", time="2026-04-26T10:00:00")
        assert f.reason in ("target", "stop_loss")

    def test_reconcile_variants(self):
        # Sentinels — exact construction patterns we use later.
        Active(order_id="x", child_ids=None)
        ClosedWhileDown(price=1.0, reason="target", time="t")
        Naked()
        LostProtection()


class TestSymbolHelper:
    def test_btc_idr_to_underscore(self):
        assert tokocrypto_symbol("BTC/IDR") == "BTC_IDR"

    def test_already_underscore_passthrough(self):
        assert tokocrypto_symbol("ETH_IDR") == "ETH_IDR"

    def test_lowercase_preserved(self):
        # Tokocrypto accepts uppercase; we don't try to normalize case.
        assert tokocrypto_symbol("btc/idr") == "btc_idr"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /opt/cts && python -m pytest tests/test_protection.py::TestDataclasses tests/test_protection.py::TestSymbolHelper -v
```

Expected: FAIL — `ImportError: No module named 'src.execution.protection'`.

- [ ] **Step 3: Create `src/execution/protection.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /opt/cts && python -m pytest tests/test_protection.py::TestDataclasses tests/test_protection.py::TestSymbolHelper -v
```

Expected: 7 passed (3 dataclass + 3 symbol helper + the two existing schema tests still pass).

- [ ] **Step 5: Commit**

```bash
cd /opt/cts && git add src/execution/protection.py tests/test_protection.py
git commit -m "feat(protection): scaffold module with result types and symbol helper"
```

---

## Task 3: `place_protection` — non-ATH OCO branch

**Files:**
- Modify: `src/execution/protection.py`
- Test: `tests/test_protection.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_protection.py`:

```python
from unittest.mock import MagicMock
import ccxt

from src.execution.protection import place_protection


def _mock_exchange_oco_success(list_id="L1", tp_id="TP1", sl_id="SL1"):
    """Build a CCXT mock whose privatePostOpenV1OrdersOco returns
    a Tokocrypto-shaped success response."""
    ex = MagicMock()
    ex.privatePostOpenV1OrdersOco.return_value = {
        "orderListId": list_id,
        "listClientOrderId": "cts-darvas-abcd1234",
        "orders": [
            {"orderId": tp_id, "clientOrderId": "tp-cli", "side": "SELL"},
            {"orderId": sl_id, "clientOrderId": "sl-cli", "side": "SELL"},
        ],
    }
    return ex


class TestPlaceProtectionOCO:
    def test_oco_success_first_try(self):
        ex = _mock_exchange_oco_success()
        r = place_protection(
            ex, "BTC/IDR", size=0.001,
            stop_loss=900_000_000.0, target=1_100_000_000.0,
            is_ath=False, strategy="darvas",
        )
        assert r.mode == "oco"
        assert r.order_id == "L1"
        assert r.child_ids == ("TP1", "SL1")

    def test_oco_param_shape(self):
        ex = _mock_exchange_oco_success()
        place_protection(
            ex, "BTC/IDR", size=0.001,
            stop_loss=1_000_000.0, target=1_200_000.0,
            is_ath=False, strategy="darvas",
            stop_limit_slip_pct=0.005,
        )
        call_args = ex.privatePostOpenV1OrdersOco.call_args
        params = call_args[0][0] if call_args[0] else call_args[1]
        assert params["symbol"] == "BTC_IDR"
        assert params["side"] == "SELL"
        assert params["quantity"] == 0.001
        assert params["price"] == 1_200_000.0
        assert params["stopPrice"] == 1_000_000.0
        assert params["stopLimitPrice"] == 1_000_000.0 * 0.995
        assert params["stopLimitTimeInForce"] == "GTC"
        assert params["listClientOrderId"].startswith("cts-darvas-")

    def test_oco_retries_on_network_error_then_succeeds(self):
        ex = _mock_exchange_oco_success()
        # First call raises, second succeeds.
        ex.privatePostOpenV1OrdersOco.side_effect = [
            ccxt.NetworkError("boom"),
            {
                "orderListId": "L1",
                "orders": [
                    {"orderId": "TP1"}, {"orderId": "SL1"},
                ],
            },
        ]
        r = place_protection(
            ex, "BTC/IDR", size=0.001,
            stop_loss=900_000.0, target=1_100_000.0,
            is_ath=False, strategy="darvas",
            backoff_seconds=[0, 0, 0],  # speed up test
        )
        assert r.mode == "oco"
        assert ex.privatePostOpenV1OrdersOco.call_count == 2

    def test_oco_falls_back_to_polling_after_3_failures(self):
        ex = MagicMock()
        ex.privatePostOpenV1OrdersOco.side_effect = ccxt.NetworkError("down")
        r = place_protection(
            ex, "BTC/IDR", size=0.001,
            stop_loss=900_000.0, target=1_100_000.0,
            is_ath=False, strategy="darvas",
            backoff_seconds=[0, 0, 0],
        )
        assert r.mode == "polling"
        assert r.order_id is None
        assert ex.privatePostOpenV1OrdersOco.call_count == 3

    def test_oco_does_not_retry_on_insufficient_funds(self):
        ex = MagicMock()
        ex.privatePostOpenV1OrdersOco.side_effect = ccxt.InsufficientFunds("nope")
        r = place_protection(
            ex, "BTC/IDR", size=0.001,
            stop_loss=900_000.0, target=1_100_000.0,
            is_ath=False, strategy="darvas",
            backoff_seconds=[0, 0, 0],
        )
        assert r.mode == "error"
        assert ex.privatePostOpenV1OrdersOco.call_count == 1

    def test_oco_does_not_retry_on_invalid_order(self):
        ex = MagicMock()
        ex.privatePostOpenV1OrdersOco.side_effect = ccxt.InvalidOrder("bad")
        r = place_protection(
            ex, "BTC/IDR", size=0.001,
            stop_loss=900_000.0, target=1_100_000.0,
            is_ath=False, strategy="darvas",
            backoff_seconds=[0, 0, 0],
        )
        assert r.mode == "error"
        assert ex.privatePostOpenV1OrdersOco.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /opt/cts && python -m pytest tests/test_protection.py::TestPlaceProtectionOCO -v
```

Expected: FAIL — `ImportError` because `place_protection` is not yet defined.

- [ ] **Step 3: Implement `place_protection` with OCO branch**

Append to `src/execution/protection.py`:

```python
import time
import uuid
import ccxt


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /opt/cts && python -m pytest tests/test_protection.py::TestPlaceProtectionOCO -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd /opt/cts && git add src/execution/protection.py tests/test_protection.py
git commit -m "feat(protection): place_protection OCO path with retry/fallback"
```

---

## Task 4: `place_protection` — ATH stop-only branch

**Files:**
- Modify: `src/execution/protection.py`
- Test: `tests/test_protection.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_protection.py`:

```python
class TestPlaceProtectionStopOnly:
    def test_stop_only_success(self):
        ex = MagicMock()
        ex.private_post_open_v1_orders.return_value = {
            "orderId": "S1", "clientOrderId": "cts-darvas-stop-aaaa",
            "type": "STOP_LOSS_LIMIT",
        }
        r = place_protection(
            ex, "BTC/IDR", size=0.002,
            stop_loss=950_000.0, target=None,
            is_ath=True, strategy="darvas",
        )
        assert r.mode == "stop_only"
        assert r.order_id == "S1"
        assert r.child_ids is None
        ex.private_post_open_v1_orders.assert_called_once()

    def test_stop_only_param_shape(self):
        ex = MagicMock()
        ex.private_post_open_v1_orders.return_value = {"orderId": "S1"}
        place_protection(
            ex, "BTC/IDR", size=0.002,
            stop_loss=1_000_000.0, target=None,
            is_ath=True, strategy="darvas",
            stop_limit_slip_pct=0.005,
        )
        call_args = ex.private_post_open_v1_orders.call_args
        params = call_args[0][0] if call_args[0] else call_args[1]
        assert params["symbol"] == "BTC_IDR"
        assert params["side"] == "SELL"
        assert params["type"] == "STOP_LOSS_LIMIT"
        assert params["quantity"] == 0.002
        assert params["stopPrice"] == 1_000_000.0
        assert params["price"] == 1_000_000.0 * 0.995
        assert params["timeInForce"] == "GTC"
        assert params["newClientOrderId"].startswith("cts-darvas-stop-")

    def test_stop_only_retries_then_polling(self):
        ex = MagicMock()
        ex.private_post_open_v1_orders.side_effect = ccxt.NetworkError("down")
        r = place_protection(
            ex, "BTC/IDR", size=0.001,
            stop_loss=900_000.0, target=None,
            is_ath=True, strategy="darvas",
            backoff_seconds=[0, 0, 0],
        )
        assert r.mode == "polling"
        assert ex.private_post_open_v1_orders.call_count == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /opt/cts && python -m pytest tests/test_protection.py::TestPlaceProtectionStopOnly -v
```

Expected: FAIL — `NotImplementedError` from the placeholder branch.

- [ ] **Step 3: Implement the ATH branch**

In `src/execution/protection.py`, add a `_build_stop_only_params` helper after `_build_oco_params`:

```python
def _build_stop_only_params(pair, size, stop_loss, strategy, stop_limit_slip_pct):
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
```

Then replace the `if is_ath: raise NotImplementedError(...)` block with the full ATH path. The full body of `place_protection` becomes:

```python
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
            else:
                return ProtectionResult(
                    mode="stop_only",
                    order_id=str(resp.get("orderId")),
                )
        except Exception as exc:
            if _is_permanent_error(exc):
                return ProtectionResult(mode="error")
            if attempt < retry - 1:
                time.sleep(backoff_seconds[attempt])

    return ProtectionResult(mode="polling")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /opt/cts && python -m pytest tests/test_protection.py -v
```

Expected: all previous tests still pass + 3 new stop-only tests pass.

- [ ] **Step 5: Commit**

```bash
cd /opt/cts && git add src/execution/protection.py tests/test_protection.py
git commit -m "feat(protection): place_protection ATH stop-only branch"
```

---

## Task 5: `replace_trailing_stop`

**Files:**
- Modify: `src/execution/protection.py`
- Test: `tests/test_protection.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_protection.py`:

```python
from src.execution.protection import replace_trailing_stop


class TestReplaceTrailingStop:
    def test_cancel_then_place_success(self):
        ex = MagicMock()
        ex.cancel_order.return_value = {"id": "old", "status": "canceled"}
        ex.private_post_open_v1_orders.return_value = {"orderId": "new1"}

        r = replace_trailing_stop(
            ex, "BTC/IDR", old_order_id="old",
            new_stop=1_050_000.0, size=0.001, strategy="darvas",
        )
        assert r.mode == "stop_only"
        assert r.order_id == "new1"
        ex.cancel_order.assert_called_once_with("old", "BTC/IDR")
        ex.private_post_open_v1_orders.assert_called_once()

    def test_cancel_failure_returns_kept(self):
        ex = MagicMock()
        ex.cancel_order.side_effect = ccxt.OrderNotFound("gone")

        r = replace_trailing_stop(
            ex, "BTC/IDR", old_order_id="old",
            new_stop=1_050_000.0, size=0.001, strategy="darvas",
        )
        assert r.mode == "kept"
        # Place should NOT be attempted if cancel failed.
        ex.private_post_open_v1_orders.assert_not_called()

    def test_place_failure_returns_polling(self):
        ex = MagicMock()
        ex.cancel_order.return_value = {"id": "old", "status": "canceled"}
        ex.private_post_open_v1_orders.side_effect = ccxt.NetworkError("down")

        r = replace_trailing_stop(
            ex, "BTC/IDR", old_order_id="old",
            new_stop=1_050_000.0, size=0.001, strategy="darvas",
            backoff_seconds=[0, 0, 0],
        )
        assert r.mode == "polling"
        ex.cancel_order.assert_called_once()
        assert ex.private_post_open_v1_orders.call_count == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /opt/cts && python -m pytest tests/test_protection.py::TestReplaceTrailingStop -v
```

Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement `replace_trailing_stop`**

`replace_trailing_stop` accepts a `backoff_seconds` kwarg that is forwarded to `place_protection` so tests can run without sleeping. Append to `src/execution/protection.py`:

```python
def replace_trailing_stop(
    exchange, pair, old_order_id, new_stop, size, strategy,
    stop_limit_slip_pct=0.005,
    backoff_seconds=(1, 2, 4),
) -> ProtectionResult:
    """Cancel the existing stop-loss-limit order and place a new one at
    `new_stop`. If cancel fails, returns mode="kept" — caller must NOT
    update its in-memory stop_loss because the old order is still active.
    If cancel succeeds but the new placement fails after retries,
    returns mode="polling" — position is now unprotected.
    """
    try:
        exchange.cancel_order(old_order_id, pair)
    except Exception:
        # Cancel failed — old protection still on the book. Treat as
        # transient; next ratchet cycle will retry.
        return ProtectionResult(mode="kept")

    return place_protection(
        exchange, pair, size,
        stop_loss=new_stop, target=None, is_ath=True,
        strategy=strategy,
        retry=3, backoff_seconds=backoff_seconds,
        stop_limit_slip_pct=stop_limit_slip_pct,
    )
```

Now refine the third test to use `backoff_seconds=[0,0,0]`:

Replace the `test_place_failure_returns_polling` test with:

```python
    def test_place_failure_returns_polling(self):
        ex = MagicMock()
        ex.cancel_order.return_value = {"id": "old", "status": "canceled"}
        ex.private_post_open_v1_orders.side_effect = ccxt.NetworkError("down")

        r = replace_trailing_stop(
            ex, "BTC/IDR", old_order_id="old",
            new_stop=1_050_000.0, size=0.001, strategy="darvas",
            backoff_seconds=[0, 0, 0],
        )
        assert r.mode == "polling"
        ex.cancel_order.assert_called_once()
        assert ex.private_post_open_v1_orders.call_count == 3
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /opt/cts && python -m pytest tests/test_protection.py::TestReplaceTrailingStop -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /opt/cts && git add src/execution/protection.py tests/test_protection.py
git commit -m "feat(protection): replace_trailing_stop with kept/polling fallbacks"
```

---

## Task 6: `check_protection_fill` — stop-only mode

**Files:**
- Modify: `src/execution/protection.py`
- Test: `tests/test_protection.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_protection.py`:

```python
from src.execution.protection import check_protection_fill


class TestCheckProtectionFillStopOnly:
    def _pos(self, order_id="S1"):
        return {
            "protection_mode": "stop_only",
            "protection_order_id": order_id,
            "protection_child_ids": None,
            "entry_time": "2026-04-26T10:00:00",
        }

    def test_open_returns_none(self):
        ex = MagicMock()
        ex.fetch_order.return_value = {"id": "S1", "status": "open"}
        result = check_protection_fill(ex, "BTC/IDR", self._pos())
        assert result is None

    def test_filled_returns_fill_result(self):
        ex = MagicMock()
        ex.fetch_order.return_value = {
            "id": "S1", "status": "closed",
            "average": 950_000.0,
            "lastTradeTimestamp": 1_700_000_000_000,
        }
        result = check_protection_fill(ex, "BTC/IDR", self._pos())
        assert isinstance(result, FillResult)
        assert result.price == 950_000.0
        assert result.reason == "stop_loss"

    def test_canceled_returns_lost_protection(self):
        ex = MagicMock()
        ex.fetch_order.return_value = {"id": "S1", "status": "canceled"}
        result = check_protection_fill(ex, "BTC/IDR", self._pos())
        assert isinstance(result, LostProtection)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /opt/cts && python -m pytest tests/test_protection.py::TestCheckProtectionFillStopOnly -v
```

Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement stop-only branch**

Append to `src/execution/protection.py`:

```python
from datetime import datetime


def _ts_to_iso(ts_ms) -> str:
    """Convert a Tokocrypto/CCXT millisecond timestamp to ISO string."""
    if ts_ms is None:
        return datetime.utcnow().isoformat()
    return datetime.utcfromtimestamp(ts_ms / 1000).isoformat()


def check_protection_fill(exchange, pair, position):
    """Per-loop check: has the protective order filled, gone missing, or
    is it still active? Returns FillResult, LostProtection, or None."""
    mode = position.get("protection_mode")
    if mode == "stop_only":
        return _check_stop_only_fill(exchange, pair, position)
    elif mode == "oco":
        return _check_oco_fill(exchange, pair, position)
    else:
        return None  # polling positions are handled by _check_exit


def _check_stop_only_fill(exchange, pair, position):
    order_id = position["protection_order_id"]
    order = exchange.fetch_order(order_id, pair)
    status = order.get("status")
    if status == "open":
        return None
    if status in ("closed", "filled"):
        return FillResult(
            price=float(order.get("average") or order.get("price")),
            reason="stop_loss",
            time=_ts_to_iso(order.get("lastTradeTimestamp")),
        )
    if status == "canceled":
        return LostProtection()
    # Unknown status — treat as still-active to avoid double-firing.
    return None
```

(The `_check_oco_fill` body is added in the next task; for now, leave it as a stub that raises so the test failure is clear if accidentally hit.)

```python
def _check_oco_fill(exchange, pair, position):
    raise NotImplementedError("OCO fill detection — Task 7")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /opt/cts && python -m pytest tests/test_protection.py::TestCheckProtectionFillStopOnly -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /opt/cts && git add src/execution/protection.py tests/test_protection.py
git commit -m "feat(protection): check_protection_fill stop-only branch"
```

---

## Task 7: `check_protection_fill` — OCO mode

**Files:**
- Modify: `src/execution/protection.py`
- Test: `tests/test_protection.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_protection.py`:

```python
class TestCheckProtectionFillOCO:
    def _pos(self):
        return {
            "protection_mode": "oco",
            "protection_order_id": "L1",
            "protection_child_ids": ("TP1", "SL1"),
            "entry_time": "2026-04-26T10:00:00",
        }

    def test_both_children_open_returns_none(self):
        ex = MagicMock()
        ex.fetch_open_orders.return_value = [
            {"id": "TP1", "status": "open"},
            {"id": "SL1", "status": "open"},
        ]
        result = check_protection_fill(ex, "BTC/IDR", self._pos())
        assert result is None

    def test_target_filled_returns_target_reason(self):
        ex = MagicMock()
        ex.fetch_open_orders.return_value = []  # neither child open
        ex.fetch_my_trades.return_value = [
            {
                "side": "sell",
                "order": "TP1",
                "price": 1_200_000.0,
                "average": 1_200_000.0,
                "amount": 0.001,
                "timestamp": 1_700_000_000_000,
            }
        ]
        result = check_protection_fill(ex, "BTC/IDR", self._pos())
        assert isinstance(result, FillResult)
        assert result.reason == "target"
        assert result.price == 1_200_000.0

    def test_stop_filled_returns_stop_loss_reason(self):
        ex = MagicMock()
        ex.fetch_open_orders.return_value = []
        ex.fetch_my_trades.return_value = [
            {
                "side": "sell",
                "order": "SL1",
                "price": 950_000.0,
                "average": 950_000.0,
                "amount": 0.001,
                "timestamp": 1_700_000_000_000,
            }
        ]
        result = check_protection_fill(ex, "BTC/IDR", self._pos())
        assert isinstance(result, FillResult)
        assert result.reason == "stop_loss"

    def test_one_child_open_one_gone_uses_trade_lookup(self):
        # TP filled, SL auto-cancelled by OCO.
        ex = MagicMock()
        ex.fetch_open_orders.return_value = [
            # SL was auto-cancelled, doesn't appear in open orders.
        ]
        ex.fetch_my_trades.return_value = [
            {
                "side": "sell",
                "order": "TP1",
                "average": 1_200_000.0,
                "amount": 0.001,
                "timestamp": 1_700_000_000_000,
            }
        ]
        result = check_protection_fill(ex, "BTC/IDR", self._pos())
        assert isinstance(result, FillResult)
        assert result.reason == "target"

    def test_no_match_returns_none_to_avoid_double_fire(self):
        # Edge: orders not in open list, but no matching trade either.
        # Could happen with API lag. Return None and try again next loop.
        ex = MagicMock()
        ex.fetch_open_orders.return_value = []
        ex.fetch_my_trades.return_value = []
        result = check_protection_fill(ex, "BTC/IDR", self._pos())
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /opt/cts && python -m pytest tests/test_protection.py::TestCheckProtectionFillOCO -v
```

Expected: FAIL — `NotImplementedError` from the stub.

- [ ] **Step 3: Implement `_check_oco_fill`**

In `src/execution/protection.py`, replace the stub with:

```python
def _check_oco_fill(exchange, pair, position):
    tp_id, sl_id = position["protection_child_ids"]

    open_orders = exchange.fetch_open_orders(pair)
    open_ids = {str(o.get("id")) for o in open_orders}
    tp_open = tp_id in open_ids
    sl_open = sl_id in open_ids

    if tp_open and sl_open:
        return None  # both legs still active

    # At least one leg is gone. Look up which one filled via my_trades.
    # Use a `since` slightly before entry_time to handle clock skew.
    since_iso = position.get("entry_time")
    since_ms = None
    if since_iso:
        try:
            since_ms = int(datetime.fromisoformat(since_iso).timestamp() * 1000)
        except (ValueError, TypeError):
            since_ms = None

    trades = exchange.fetch_my_trades(pair, since=since_ms, limit=20) or []
    for trade in trades:
        if trade.get("side", "").lower() != "sell":
            continue
        order_ref = str(trade.get("order") or "")
        if order_ref == tp_id:
            return FillResult(
                price=float(trade.get("average") or trade.get("price")),
                reason="target",
                time=_ts_to_iso(trade.get("timestamp")),
            )
        if order_ref == sl_id:
            return FillResult(
                price=float(trade.get("average") or trade.get("price")),
                reason="stop_loss",
                time=_ts_to_iso(trade.get("timestamp")),
            )

    # Orders gone but no trade found yet — exchange API is lagging.
    # Return None so we retry next loop instead of double-firing.
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /opt/cts && python -m pytest tests/test_protection.py::TestCheckProtectionFillOCO -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /opt/cts && git add src/execution/protection.py tests/test_protection.py
git commit -m "feat(protection): check_protection_fill OCO branch with my_trades lookup"
```

---

## Task 8: `reconcile_on_restart`

**Files:**
- Modify: `src/execution/protection.py`
- Test: `tests/test_protection.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_protection.py`:

```python
from src.execution.protection import reconcile_on_restart


class TestReconcileOnRestart:
    def _stop_only_pos(self):
        return {
            "protection_mode": "stop_only",
            "protection_order_id": "S1",
            "protection_child_ids": None,
            "entry_time": "2026-04-26T10:00:00",
            "position_size": 0.001,
        }

    def _oco_pos(self):
        return {
            "protection_mode": "oco",
            "protection_order_id": "L1",
            "protection_child_ids": ("TP1", "SL1"),
            "entry_time": "2026-04-26T10:00:00",
            "position_size": 0.001,
        }

    def test_active_stop_only(self):
        ex = MagicMock()
        ex.fetch_open_orders.return_value = [{"id": "S1", "status": "open"}]
        result = reconcile_on_restart(ex, "BTC/IDR", self._stop_only_pos())
        assert isinstance(result, Active)
        assert result.order_id == "S1"
        assert result.child_ids is None

    def test_active_oco(self):
        ex = MagicMock()
        ex.fetch_open_orders.return_value = [
            {"id": "TP1", "status": "open"},
            {"id": "SL1", "status": "open"},
        ]
        result = reconcile_on_restart(ex, "BTC/IDR", self._oco_pos())
        assert isinstance(result, Active)
        assert result.order_id == "L1"
        assert result.child_ids == ("TP1", "SL1")

    def test_closed_while_down_oco_target(self):
        ex = MagicMock()
        ex.fetch_open_orders.return_value = []
        ex.fetch_my_trades.return_value = [
            {
                "side": "sell", "order": "TP1",
                "average": 1_200_000.0, "amount": 0.001,
                "timestamp": 1_700_000_000_000,
            }
        ]
        result = reconcile_on_restart(ex, "BTC/IDR", self._oco_pos())
        assert isinstance(result, ClosedWhileDown)
        assert result.reason == "target"
        assert result.price == 1_200_000.0

    def test_closed_while_down_stop_only(self):
        ex = MagicMock()
        ex.fetch_open_orders.return_value = []
        ex.fetch_my_trades.return_value = [
            {
                "side": "sell", "order": "S1",
                "average": 950_000.0, "amount": 0.001,
                "timestamp": 1_700_000_000_000,
            }
        ]
        result = reconcile_on_restart(ex, "BTC/IDR", self._stop_only_pos())
        assert isinstance(result, ClosedWhileDown)
        assert result.reason == "stop_loss"
        assert result.price == 950_000.0

    def test_naked_no_orders_no_trades(self):
        ex = MagicMock()
        ex.fetch_open_orders.return_value = []
        ex.fetch_my_trades.return_value = []
        result = reconcile_on_restart(ex, "BTC/IDR", self._stop_only_pos())
        assert isinstance(result, Naked)

    def test_amount_tolerance_within_1pct(self):
        # Trade amount slightly less than position size due to fees.
        ex = MagicMock()
        ex.fetch_open_orders.return_value = []
        ex.fetch_my_trades.return_value = [
            {
                "side": "sell", "order": "S1",
                "average": 950_000.0, "amount": 0.000999,  # 0.1% short
                "timestamp": 1_700_000_000_000,
            }
        ]
        result = reconcile_on_restart(ex, "BTC/IDR", self._stop_only_pos())
        assert isinstance(result, ClosedWhileDown)

    def test_amount_mismatch_beyond_tolerance_returns_naked(self):
        # Trade amount more than 1% off — different trade, not ours.
        ex = MagicMock()
        ex.fetch_open_orders.return_value = []
        ex.fetch_my_trades.return_value = [
            {
                "side": "sell", "order": "S1",
                "average": 950_000.0, "amount": 0.0005,  # 50% short
                "timestamp": 1_700_000_000_000,
            }
        ]
        result = reconcile_on_restart(ex, "BTC/IDR", self._stop_only_pos())
        assert isinstance(result, Naked)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /opt/cts && python -m pytest tests/test_protection.py::TestReconcileOnRestart -v
```

Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement `reconcile_on_restart`**

Append to `src/execution/protection.py`:

```python
def reconcile_on_restart(exchange, pair, position):
    """On bot restart, determine the state of an exchange-protected
    position. Returns Active, ClosedWhileDown, or Naked.

    Three-way decision:
      Active           — protection still on the book; bot adopts it.
      ClosedWhileDown  — protection fired while bot was down; trader
                         closes the position in DB with the real exit.
      Naked            — protection is gone but no exit trade found;
                         trader must re-place protection.
    """
    mode = position.get("protection_mode")
    target_ids = []
    if mode == "stop_only":
        target_ids = [position["protection_order_id"]]
    elif mode == "oco":
        target_ids = list(position["protection_child_ids"])
    else:
        return Naked()

    # 1. Check open orders.
    open_orders = exchange.fetch_open_orders(pair) or []
    open_ids = {str(o.get("id")) for o in open_orders}
    if mode == "stop_only" and target_ids[0] in open_ids:
        return Active(order_id=target_ids[0], child_ids=None)
    if mode == "oco" and all(t in open_ids for t in target_ids):
        return Active(
            order_id=position["protection_order_id"],
            child_ids=tuple(target_ids),
        )

    # 2. Check trade history.
    since_iso = position.get("entry_time")
    since_ms = None
    if since_iso:
        try:
            since_ms = int(datetime.fromisoformat(since_iso).timestamp() * 1000)
        except (ValueError, TypeError):
            since_ms = None

    trades = exchange.fetch_my_trades(pair, since=since_ms, limit=50) or []
    expected_size = float(position.get("position_size") or 0)

    for trade in trades:
        if trade.get("side", "").lower() != "sell":
            continue
        order_ref = str(trade.get("order") or "")
        if order_ref not in target_ids:
            continue
        # Tolerance: fees can shave the amount by a small fraction.
        amount = float(trade.get("amount") or 0)
        if expected_size > 0 and abs(amount - expected_size) / expected_size > 0.01:
            continue
        if mode == "stop_only":
            reason = "stop_loss"
        else:
            tp_id, sl_id = target_ids
            reason = "target" if order_ref == tp_id else "stop_loss"
        return ClosedWhileDown(
            price=float(trade.get("average") or trade.get("price")),
            reason=reason,
            time=_ts_to_iso(trade.get("timestamp")),
        )

    return Naked()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /opt/cts && python -m pytest tests/test_protection.py::TestReconcileOnRestart -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
cd /opt/cts && git add src/execution/protection.py tests/test_protection.py
git commit -m "feat(protection): reconcile_on_restart with 3-way state decision"
```

---

## Task 9: Wire `place_protection` into `LiveTrader._execute_buy` + DB write

**Files:**
- Modify: `src/execution/live_trader.py:312-368`
- Modify: `src/execution/live_trader.py:453-467` (`_log_entry`)

This task has no unit test — it's wiring against real CCXT. Verification is via integration testing in Task 14.

- [ ] **Step 1: Update `_log_entry` signature to include `protection_mode`**

In `src/execution/live_trader.py`, change `_log_entry` (lines 453–467) to accept and write `protection_mode`:

```python
    def _log_entry(self, conn, pair, price, levels, size, features, protection_mode):
        """Log trade entry to journal."""
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trade_journal
            (mode, pair, strategy, direction, entry_time, entry_price,
             stop_loss, take_profit, position_size, features_json,
             protection_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "live", pair, self.strategy, "long",
            datetime.now().isoformat(), price,
            levels["stop_loss"], levels.get("target"), size,
            json.dumps(features), protection_mode,
        ))
        conn.commit()
```

- [ ] **Step 2: Add the `place_protection` import at the top of `live_trader.py`**

Find the import block at lines 14–30 and add:

```python
from src.execution import protection as prot
```

- [ ] **Step 3: Modify `_execute_buy` to place protection after fill confirmation**

In `_execute_buy` (around line 344, inside the `if status in ("closed", "filled"):` block), **before** the `self.positions[pair] = {...}` assignment, add the protection placement:

```python
            if status in ("closed", "filled"):
                filled_size = order.get("filled", size)

                # ── Place exchange-side protection ──
                protection_cfg = self.config.get("risk", {}).get("protection", {})
                if protection_cfg.get("enabled", True):
                    prot_result = prot.place_protection(
                        self.exchange, pair, filled_size,
                        stop_loss=levels["stop_loss"],
                        target=levels.get("target"),
                        is_ath=levels.get("is_ath", False),
                        strategy=self.strategy,
                        retry=protection_cfg.get("retry_attempts", 3),
                        backoff_seconds=protection_cfg.get(
                            "retry_backoff_seconds", [1, 2, 4]
                        ),
                        stop_limit_slip_pct=protection_cfg.get(
                            "stop_limit_slip_pct", 0.005
                        ),
                    )
                    if prot_result.mode == "polling":
                        self._notify(
                            f"⚠️ {pair} protection placement failed — "
                            "falling back to local polling"
                        )
                    elif prot_result.mode == "error":
                        self._notify(
                            f"🚨 {pair} protection placement returned "
                            "permanent error — position is unprotected, "
                            "manual intervention required"
                        )
                        prot_result = prot.ProtectionResult(mode="polling")
                else:
                    prot_result = prot.ProtectionResult(mode="polling")

                self.positions[pair] = {
                    "entry_time": datetime.now().isoformat(),
                    "entry_price": filled_price,
                    "stop_loss": levels["stop_loss"],
                    "target": levels.get("target"),
                    "position_size": filled_size,
                    "is_ath": levels.get("is_ath", False),
                    "order_id": order_id,
                    "features": features,
                    "protection_mode": prot_result.mode,
                    "protection_order_id": prot_result.order_id,
                    "protection_child_ids": prot_result.child_ids,
                }
                print(f"     ✅ FILLED @ {filled_price:,.0f} "
                      f"(protection: {prot_result.mode})")

                target_str = f"{levels['target']:,.0f}" if levels.get("target") else "ATH trailing"
                self._notify(
                    f"📈 *BUY {pair}*\n"
                    f"Entry: {filled_price:,.0f}\n"
                    f"Stop loss: {levels['stop_loss']:,.0f}\n"
                    f"Target: {target_str}\n"
                    f"Size: {filled_size:.8f}\n"
                    f"Protection: {prot_result.mode}"
                )

                # Log to journal with protection_mode
                self._log_entry(conn, pair, filled_price, levels,
                                filled_size, features, prot_result.mode)
```

- [ ] **Step 4: Smoke-test the unit suite still passes**

```bash
cd /opt/cts && python -m pytest tests/test_protection.py -v
```

Expected: all previous tests still pass.

- [ ] **Step 5: Smoke-test the import doesn't break the live trader's preflight**

```bash
cd /opt/cts && python -m src.execution.live_trader --preflight
```

Expected: preflight runs without import errors. (It may exit with "issues to resolve" — that's fine for now.)

- [ ] **Step 6: Commit**

```bash
cd /opt/cts && git add src/execution/live_trader.py
git commit -m "feat(live_trader): place exchange-side protection after buy fills

Calls protection.place_protection after every confirmed buy fill.
Master switch risk.protection.enabled=false short-circuits to today's
polling-only behavior. _log_entry now writes protection_mode column."
```

---

## Task 10: Wire `check_protection_fill` into the main loop

**Files:**
- Modify: `src/execution/live_trader.py:194-236` (`_process_pair`)
- Modify: `src/execution/live_trader.py:238-263` (`_check_exit`)

- [ ] **Step 1: Add a new method `_check_protection` to `LiveTrader`**

In `src/execution/live_trader.py`, after the `_check_exit` method (after line 263), add:

```python
    def _check_protection(self, pair, conn):
        """For positions protected by exchange-side OCO/stop, check if
        the protective order has filled. If so, close the position with
        the actual exit price/reason. Returns True if the position was
        closed, False otherwise."""
        pos = self.positions.get(pair)
        if not pos:
            return False
        mode = pos.get("protection_mode")
        if mode not in ("oco", "stop_only"):
            return False

        try:
            result = prot.check_protection_fill(self.exchange, pair, pos)
        except Exception as e:
            print(f"  Error checking protection for {pair}: {e}")
            return False

        if result is None:
            return False

        if isinstance(result, prot.LostProtection):
            # Operator cancelled outside the bot. Fall back to polling.
            print(f"  ⚠️ {pair} protection lost — falling back to polling")
            self._notify(f"⚠️ {pair} protection cancelled externally — polling fallback")
            pos["protection_mode"] = "polling"
            pos["protection_order_id"] = None
            pos["protection_child_ids"] = None
            return False

        # FillResult: exchange already executed the sell. Close locally.
        pnl_per_unit = result.price - pos["entry_price"]
        pnl_absolute = pnl_per_unit * pos["position_size"]
        pnl_pct = pnl_per_unit / pos["entry_price"]
        self.daily_pnl += pnl_absolute
        self.trade_count_today += 1

        emoji = "✅" if pnl_pct > 0 else "❌"
        print(f"\n  {emoji} {pair} CLOSED via {result.reason} "
              f"@ {result.price:,.0f} | PnL: {pnl_pct:+.2%} "
              f"({pnl_absolute:+,.0f} IDR)")

        self._notify(
            f"{emoji} *SELL {pair}* ({result.reason})\n"
            f"Exit: {result.price:,.0f}\n"
            f"PnL: {pnl_pct:+.2%} ({pnl_absolute:+,.0f} IDR)"
        )

        self._log_exit(conn, pair, result.price, result.reason,
                       pnl_pct, pnl_absolute, pos)
        del self.positions[pair]
        return True
```

- [ ] **Step 2: Modify `_process_pair` to call `_check_protection` first**

In `_process_pair` (around line 226–234), replace the exit/entry block:

```python
        # ── Check exits ──
        if pair in self.positions:
            # Exchange-side protection check first; falls through to
            # candle-based polling if protection_mode != oco/stop_only.
            closed = self._check_protection(pair, conn)
            if not closed and pair in self.positions:
                self._check_exit(df, last_idx, pair, conn)

        # ── Check entries ──
        if pair not in self.positions:
            if df.iloc[last_idx].get(signal_col) == "buy":
                self._try_entry(df, last_idx, pair, conn)
```

- [ ] **Step 3: Update `_check_exit` to skip OCO/stop-only positions**

The candle-based exit logic must NOT fire for positions already protected on the exchange (avoid double-sell). Modify `_check_exit` (line 238) to short-circuit:

```python
    def _check_exit(self, df, idx, pair, conn):
        """Check and execute exits via candle-based polling. Used for
        positions in protection_mode='polling' or legacy positions."""
        pos = self.positions[pair]
        if pos.get("protection_mode") in ("oco", "stop_only"):
            # Exchange-side protection is active; _check_protection
            # handles fills. Don't double-trigger from candles.
            return

        low = df.iloc[idx]["low"]
        high = df.iloc[idx]["high"]

        exit_price = None
        exit_reason = None

        if low <= pos["stop_loss"]:
            exit_price = pos["stop_loss"]
            exit_reason = "stop_loss"
        elif pos.get("target") and high >= pos["target"]:
            exit_price = pos["target"]
            exit_reason = "target"

        # ATH trailing stop (polling fallback path; the OCO/stop_only
        # ratchet path is handled in Task 11)
        if pos.get("is_ath") and exit_price is None:
            from src.indicators.features import _atr
            atr = _atr(df["high"].values, df["low"].values, df["close"].values, 14)
            trailing = high - (2 * atr)
            if trailing > pos["stop_loss"]:
                pos["stop_loss"] = trailing

        if exit_price:
            self._execute_sell(pair, exit_price, exit_reason, conn)
```

- [ ] **Step 4: Verify unit tests still pass**

```bash
cd /opt/cts && python -m pytest tests/ -v
```

Expected: all tests still pass (we didn't add any new ones in this task; just wired existing logic).

- [ ] **Step 5: Commit**

```bash
cd /opt/cts && git add src/execution/live_trader.py
git commit -m "feat(live_trader): per-loop check_protection_fill, polling skipped for OCO

_process_pair now calls _check_protection before _check_exit. Positions
with protection_mode='oco' or 'stop_only' bypass the candle-based exit
logic so the exchange-side fill is the source of truth."
```

---

## Task 11: Trailing replace path for Darvas-ATH

**Files:**
- Modify: `src/execution/live_trader.py` (extend `_check_protection` or add a new method)

- [ ] **Step 1: Add ATH ratchet logic to `_check_protection`**

The ratchet check needs the latest candle data, which `_check_protection` doesn't currently see. Restructure: pass the dataframe and index in, then handle ratchet inside.

Modify the `_process_pair` exit block:

```python
        # ── Check exits ──
        if pair in self.positions:
            closed = self._check_protection(pair, df, last_idx, conn)
            if not closed and pair in self.positions:
                self._check_exit(df, last_idx, pair, conn)
```

Update the `_check_protection` signature and add a ratchet section after the `LostProtection` branch and before the `FillResult` handling:

```python
    def _check_protection(self, pair, df, idx, conn):
        pos = self.positions.get(pair)
        if not pos:
            return False
        mode = pos.get("protection_mode")
        if mode not in ("oco", "stop_only"):
            return False

        try:
            result = prot.check_protection_fill(self.exchange, pair, pos)
        except Exception as e:
            print(f"  Error checking protection for {pair}: {e}")
            return False

        if isinstance(result, prot.LostProtection):
            print(f"  ⚠️ {pair} protection lost — falling back to polling")
            self._notify(f"⚠️ {pair} protection cancelled externally — polling fallback")
            pos["protection_mode"] = "polling"
            pos["protection_order_id"] = None
            pos["protection_child_ids"] = None
            return False

        if result is None:
            # Active and unfilled. For ATH stop-only, check if we should
            # ratchet the trailing stop up.
            if mode == "stop_only" and pos.get("is_ath"):
                self._maybe_ratchet_trailing_stop(pair, pos, df, idx)
            return False

        # FillResult: exchange already executed the sell.
        pnl_per_unit = result.price - pos["entry_price"]
        pnl_absolute = pnl_per_unit * pos["position_size"]
        pnl_pct = pnl_per_unit / pos["entry_price"]
        self.daily_pnl += pnl_absolute
        self.trade_count_today += 1

        emoji = "✅" if pnl_pct > 0 else "❌"
        print(f"\n  {emoji} {pair} CLOSED via {result.reason} "
              f"@ {result.price:,.0f} | PnL: {pnl_pct:+.2%} "
              f"({pnl_absolute:+,.0f} IDR)")

        self._notify(
            f"{emoji} *SELL {pair}* ({result.reason})\n"
            f"Exit: {result.price:,.0f}\n"
            f"PnL: {pnl_pct:+.2%} ({pnl_absolute:+,.0f} IDR)"
        )

        self._log_exit(conn, pair, result.price, result.reason,
                       pnl_pct, pnl_absolute, pos)
        del self.positions[pair]
        return True
```

- [ ] **Step 2: Add `_maybe_ratchet_trailing_stop`**

Add this method right after `_check_protection`:

```python
    def _maybe_ratchet_trailing_stop(self, pair, pos, df, idx):
        """For ATH stop-only positions: compute new trailing stop from
        recent candles. If it's higher than the current stop by at least
        min_ratchet_pct, cancel-and-replace the exchange order."""
        from src.indicators.features import _atr

        protection_cfg = self.config.get("risk", {}).get("protection", {})
        min_ratchet = protection_cfg.get("min_trail_ratchet_pct", 0.005)

        atr = _atr(df["high"].values, df["low"].values,
                   df["close"].values, 14)
        high = df.iloc[idx]["high"]
        new_trail = high - 2 * atr

        ratchet_threshold = pos["stop_loss"] * (1 + min_ratchet)
        if new_trail <= ratchet_threshold:
            return  # not enough movement to bother

        result = prot.replace_trailing_stop(
            self.exchange, pair,
            old_order_id=pos["protection_order_id"],
            new_stop=new_trail,
            size=pos["position_size"],
            strategy=self.strategy,
            stop_limit_slip_pct=protection_cfg.get("stop_limit_slip_pct", 0.005),
        )

        if result.mode == "polling":
            print(f"  ⚠️ {pair} trail replace failed — polling fallback")
            self._notify(f"⚠️ {pair} trail replace failed — polling fallback")
            pos["protection_mode"] = "polling"
            pos["protection_order_id"] = None
        elif result.mode == "kept":
            # Cancel failed; old order still on the book. Don't change
            # in-memory state. Next ratchet attempt will retry.
            pass
        else:  # "stop_only"
            pos["stop_loss"] = new_trail
            pos["protection_order_id"] = result.order_id
            print(f"  ↗ {pair} trailing stop ratcheted to {new_trail:,.0f}")
```

- [ ] **Step 3: Verify unit tests still pass**

```bash
cd /opt/cts && python -m pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
cd /opt/cts && git add src/execution/live_trader.py
git commit -m "feat(live_trader): cancel-and-replace ATH trailing stop on the exchange

When protection_mode is stop_only and is_ath, each loop computes a new
trailing stop from recent candles. If it exceeds the current stop by
min_trail_ratchet_pct (default 0.5%), the old stop is cancelled and a
new one placed. kept mode (cancel failed) leaves state unchanged."
```

---

## Task 12: Wire `reconcile_on_restart` into `_restore_positions`

**Files:**
- Modify: `src/execution/live_trader.py:485-509`

- [ ] **Step 1: Update the SELECT to include `protection_mode`**

Replace `_restore_positions` (lines 485–509) with:

```python
    def _restore_positions(self):
        """Restore open positions from DB on startup. For positions
        protected by exchange-side OCO/stop, reconcile state with the
        exchange (still active / closed-while-down / naked)."""
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT pair, entry_time, entry_price, stop_loss, take_profit,
                   position_size, features_json, protection_mode
            FROM trade_journal
            WHERE mode='live' AND strategy=? AND exit_time IS NULL
        """, (self.strategy,))
        rows = cur.fetchall()

        for row in rows:
            # `is_ath` is not persisted in trade_journal, but it's
            # equivalent to `protection_mode == "stop_only"` by
            # construction — stop-only protection only exists for
            # ATH-breakout positions. This keeps the trailing ratchet
            # working after a restart.
            protection_mode = row["protection_mode"]
            pos = {
                "entry_time": row["entry_time"],
                "entry_price": row["entry_price"],
                "stop_loss": row["stop_loss"],
                "target": row["take_profit"],
                "position_size": row["position_size"],
                "is_ath": protection_mode == "stop_only",
                "features": json.loads(row["features_json"] or "{}"),
                "protection_mode": protection_mode,
                "protection_order_id": None,
                "protection_child_ids": None,
            }
            pair = row["pair"]

            if pos["protection_mode"] in ("oco", "stop_only"):
                self._reconcile_position(pair, pos, conn)
            else:
                # polling mode or legacy NULL — restore as today.
                self.positions[pair] = pos

        conn.close()
        if self.positions:
            print(f"  Restored {len(self.positions)} open positions: "
                  f"{', '.join(self.positions.keys())}")
```

- [ ] **Step 2: Add `_reconcile_position`**

Add after `_restore_positions`:

```python
    def _reconcile_position(self, pair, pos, conn):
        """Check exchange state for a restored position. Three outcomes:
          Active           — adopt the open order id; restore position.
          ClosedWhileDown  — close the position in DB with real exit;
                             do not restore.
          Naked            — re-place protection (or fall back to polling).
        """
        try:
            result = prot.reconcile_on_restart(self.exchange, pair, pos)
        except Exception as e:
            print(f"  ⚠️ Could not reconcile {pair}: {e}. "
                  "Falling back to polling.")
            pos["protection_mode"] = "polling"
            self.positions[pair] = pos
            return

        if isinstance(result, prot.Active):
            pos["protection_order_id"] = result.order_id
            pos["protection_child_ids"] = result.child_ids
            self.positions[pair] = pos
            print(f"  Adopted active {pos['protection_mode']} for {pair} "
                  f"(order {result.order_id})")

        elif isinstance(result, prot.ClosedWhileDown):
            print(f"  ℹ️ {pair} closed while bot was down: "
                  f"{result.reason} @ {result.price:,.0f}")
            self._notify(
                f"ℹ️ {pair} closed while bot was down: "
                f"{result.reason} @ {result.price:,.0f} (at {result.time})"
            )
            pnl_per_unit = result.price - pos["entry_price"]
            pnl_absolute = pnl_per_unit * pos["position_size"]
            pnl_pct = pnl_per_unit / pos["entry_price"]
            self._log_exit(conn, pair, result.price, result.reason,
                           pnl_pct, pnl_absolute, pos)
            # Do NOT add to self.positions — it's closed.

        elif isinstance(result, prot.Naked):
            print(f"  ⚠️ {pair} restored without protection — re-placing")
            self._notify(f"⚠️ {pair} restored without protection — re-placing")

            protection_cfg = self.config.get("risk", {}).get("protection", {})
            prot_result = prot.place_protection(
                self.exchange, pair, pos["position_size"],
                stop_loss=pos["stop_loss"],
                target=pos["target"],
                is_ath=pos["is_ath"],
                strategy=self.strategy,
                retry=protection_cfg.get("retry_attempts", 3),
                backoff_seconds=protection_cfg.get(
                    "retry_backoff_seconds", [1, 2, 4]
                ),
                stop_limit_slip_pct=protection_cfg.get(
                    "stop_limit_slip_pct", 0.005
                ),
            )
            pos["protection_mode"] = (
                prot_result.mode if prot_result.mode != "error" else "polling"
            )
            pos["protection_order_id"] = prot_result.order_id
            pos["protection_child_ids"] = prot_result.child_ids
            self.positions[pair] = pos
```

- [ ] **Step 3: Verify the live trader still imports cleanly**

```bash
cd /opt/cts && python -c "from src.execution import live_trader; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Run all tests**

```bash
cd /opt/cts && python -m pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /opt/cts && git add src/execution/live_trader.py
git commit -m "feat(live_trader): reconcile_on_restart wired into _restore_positions

For positions protected by OCO/stop_only, on startup the trader queries
the exchange and either adopts the active order, closes a position that
fired while down, or re-places protection on a naked position."
```

---

## Task 13: Paper trader writes `protection_mode="simulated"`

**Files:**
- Modify: `src/execution/paper_trader.py:223-236` (the INSERT statement)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_protection.py`:

```python
class TestPaperTraderProtectionMode:
    def test_paper_insert_uses_simulated_mode(self, tmp_db):
        """Paper trader's journal writes must include protection_mode."""
        conn = get_connection(tmp_db)
        # Mimic the INSERT shape from paper_trader._open_position
        conn.execute("""
            INSERT INTO trade_journal
            (mode, pair, strategy, direction, entry_time, entry_price,
             stop_loss, take_profit, position_size, features_json,
             protection_mode)
            VALUES ('paper', 'BTC/IDR', 'darvas', 'long',
                    '2026-04-26T10:00:00', 1000000.0,
                    900000.0, 1100000.0, 0.001, '{}',
                    'simulated')
        """)
        conn.commit()
        cur = conn.execute(
            "SELECT protection_mode FROM trade_journal WHERE mode='paper'"
        )
        assert cur.fetchone()["protection_mode"] == "simulated"
        conn.close()
```

This test passes immediately because the schema already supports it. It documents intent. The real verification is the source change.

- [ ] **Step 2: Update `paper_trader._open_position` INSERT**

In `src/execution/paper_trader.py` (lines 224–235), update the INSERT:

```python
        # Persist to DB
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trade_journal
            (mode, pair, strategy, direction, entry_time, entry_price,
             stop_loss, take_profit, position_size, features_json,
             protection_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "paper", pair, self.strategy, "long",
            self.positions[pair]["entry_time"], levels["entry"],
            levels["stop_loss"], levels.get("target"), position_size,
            json.dumps(features), "simulated",
        ))
        conn.commit()
```

- [ ] **Step 3: Run the test suite**

```bash
cd /opt/cts && python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Smoke-test paper trader imports**

```bash
cd /opt/cts && python -c "from src.execution import paper_trader; print('ok')"
```

Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
cd /opt/cts && git add src/execution/paper_trader.py tests/test_protection.py
git commit -m "feat(paper_trader): write protection_mode='simulated' to journal"
```

---

## Task 14: Surface `protection_mode` in `positions.py` and `telegram_bot.py`

**Files:**
- Modify: `src/execution/positions.py`
- Modify: `src/execution/telegram_bot.py`

- [ ] **Step 1: Inspect current display logic**

Read the relevant section so you know where to plug in:

```bash
cd /opt/cts && grep -n "take_profit\|target\|stop_loss" src/execution/positions.py | head -10
cd /opt/cts && grep -n "take_profit\|target\|stop_loss\|positions" src/execution/telegram_bot.py | head -20
```

- [ ] **Step 2: Update the SELECT in `positions.py`**

Find the query that reads from `trade_journal` (look for `SELECT.*FROM trade_journal`) and add `protection_mode` to the column list. Then add `protection_mode` to the displayed output.

Example pattern (find the actual query and adapt):

```python
# Original:
"SELECT mode, pair, strategy, entry_time, entry_price, stop_loss, "
"take_profit, position_size FROM trade_journal WHERE exit_time IS NULL"

# Updated:
"SELECT mode, pair, strategy, entry_time, entry_price, stop_loss, "
"take_profit, position_size, protection_mode FROM trade_journal "
"WHERE exit_time IS NULL"
```

In the row-formatting code, append a `Prot:` field. The text format (line 49 area) should look like:

```python
prot = pos.get("protection_mode") or "—"
target_str = f"{pos['take_profit']:,.0f}" if pos.get("take_profit") else "ATH trail"
print(f"  {pos['pair']:<12} {pos['mode']:<6} {pos['strategy']:<14} "
      f"entry={pos['entry_price']:,.0f} sl={pos['stop_loss']:,.0f} "
      f"tp={target_str} prot={prot}")
```

For JSON output (`--json` flag), `protection_mode` is automatically included if you added it to the SELECT and the row dict.

- [ ] **Step 3: Update `telegram_bot.py` /positions reply**

Find the `/positions` handler (around line 220–230) and append `protection_mode` to each line:

```python
prot = r.get("protection_mode") or "—"
target = f"{r['take_profit']:,.0f}" if r.get("take_profit") else "ATH"
lines.append(
    f"• {r['pair']} ({r['strategy']}) "
    f"entry {r['entry_price']:,.0f} sl {r['stop_loss']:,.0f} "
    f"tp {target} [{prot}]"
)
```

(Adapt to the actual existing line shape — the goal is just to add `[{prot}]` somewhere visible.)

Make sure the SELECT in the telegram bot also includes `protection_mode`.

- [ ] **Step 4: Manual verification**

```bash
cd /opt/cts && python -m src.execution.positions --json
```

Expected: each row includes a `protection_mode` key (NULL or `simulated` for any existing rows).

- [ ] **Step 5: Run unit tests**

```bash
cd /opt/cts && python -m pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /opt/cts && git add src/execution/positions.py src/execution/telegram_bot.py
git commit -m "feat: surface protection_mode in /positions CLI and Telegram bot"
```

---

## Task 15: Config keys, master switch validation, preflight gate

**Files:**
- Modify: `config.example.yaml`
- Modify: `src/execution/live_trader.py:554-638` (`preflight_check`)

- [ ] **Step 1: Add config keys to `config.example.yaml`**

Find the `risk:` section in `config.example.yaml` and add a `protection:` subsection:

```yaml
risk:
  # ... existing keys ...

  # Exchange-side protection orders (Tokocrypto OCO + STOP_LOSS_LIMIT).
  # See docs/superpowers/specs/2026-04-26-tokocrypto-oco-protection-design.md
  protection:
    enabled: true                    # master switch; false = legacy polling-only
    stop_limit_slip_pct: 0.005       # how far below stopPrice the limit price sits
    min_trail_ratchet_pct: 0.005     # ATH ratchet gate (0.5% min bump)
    retry_attempts: 3
    retry_backoff_seconds: [1, 2, 4]
```

- [ ] **Step 2: Add a preflight check for OCO endpoint availability**

In `src/execution/live_trader.py`, in `preflight_check` (line 554), after the existing checks but before the print loop (around line 622), add:

```python
    # 8. CCXT exposes the OCO endpoint
    protection_cfg = config.get("risk", {}).get("protection", {})
    if protection_cfg.get("enabled", True):
        try:
            from src.data.downloader import create_exchange
            ex = create_exchange(config, authenticated=False)
            has_oco = hasattr(ex, "privatePostOpenV1OrdersOco")
            has_stop = hasattr(ex, "private_post_open_v1_orders")
            checks.append(("CCXT supports Tokocrypto OCO", has_oco))
            checks.append(("CCXT supports Tokocrypto stop-loss-limit", has_stop))
        except Exception as e:
            checks.append((f"CCXT OCO capability check ({e})", False))
```

- [ ] **Step 3: Smoke-test preflight**

```bash
cd /opt/cts && python -m src.execution.live_trader --preflight
```

Expected: the new "CCXT supports Tokocrypto OCO" check appears and shows ✅.

- [ ] **Step 4: Run unit tests**

```bash
cd /opt/cts && python -m pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /opt/cts && git add config.example.yaml src/execution/live_trader.py
git commit -m "feat: config keys for risk.protection.* + preflight OCO capability check"
```

---

## Task 16: README/CLAUDE.md update + integration smoke checklist

**Files:**
- Modify: `CLAUDE.md`
- Create: `docs/superpowers/specs/2026-04-26-tokocrypto-oco-protection-runbook.md`

- [ ] **Step 1: Add a section to CLAUDE.md**

Find the "### Paper & Live Trading" section in `CLAUDE.md` and append:

```markdown
### Exchange-Side Protection (OCO / Stop-Loss-Limit)

`src/execution/protection.py` places real protective sell orders on Tokocrypto
after a buy fill: an **OCO** for fixed-target strategies, a **STOP_LOSS_LIMIT**
for Darvas ATH-breakout positions (which trail). Polling-based exits in
`live_trader._check_exit` are now a fallback path used only when placement
fails (`protection_mode="polling"`) or for legacy positions.

Protection state is observable via the `protection_mode` column on
`trade_journal` (`oco` | `stop_only` | `polling` | `simulated`) and via
`/positions` and the Telegram bot.

Master switch: `risk.protection.enabled` in `config.yaml` — set to `false`
to revert to today's polling-only behavior without code changes.

On bot restart, `_restore_positions` reconciles each restored position with
the exchange (still active / closed-while-down / naked) and re-places
protection if it's missing.
```

- [ ] **Step 2: Write a manual integration runbook**

Create `docs/superpowers/specs/2026-04-26-tokocrypto-oco-protection-runbook.md`:

```markdown
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
        memory (DB column is not updated, by design — only set on entry).
- [ ] Restart the bot mid-position:
  - [ ] On startup, "Adopted active oco for BTC/IDR" log line appears.
  - [ ] No duplicate OCO placed.
- [ ] Force a stop fire by setting an entry's `stop_loss` close to current
      price (or wait for natural exit):
  - [ ] Telegram "SELL" message appears with the right reason.
  - [ ] DB row updated with exit_time, exit_price, exit_reason.

## Darvas-ATH smoke test

- [ ] Switch to the Darvas strategy. After an ATH entry fills:
  - [ ] Telegram includes `Protection: stop_only`.
  - [ ] Tokocrypto UI shows a single STOP_LOSS_LIMIT, no OCO.
- [ ] Watch over several candles as price rises:
  - [ ] When the trailing stop bumps past `min_trail_ratchet_pct`, log
        line "trailing stop ratcheted to ..." appears.
  - [ ] UI shows the old order cancelled and a new STOP_LOSS_LIMIT at the
        higher price.

## Rollback

If anything goes sideways, set `risk.protection.enabled: false` and restart
the bot. All open positions revert to polling protection on next loop tick.
```

- [ ] **Step 3: Commit**

```bash
cd /opt/cts && git add CLAUDE.md docs/superpowers/specs/2026-04-26-tokocrypto-oco-protection-runbook.md
git commit -m "docs: OCO protection notes in CLAUDE.md + manual integration runbook"
```

---

## Final verification

- [ ] **Run the full test suite one more time**

```bash
cd /opt/cts && python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Verify the live_trader imports and preflight runs cleanly**

```bash
cd /opt/cts && python -m src.execution.live_trader --preflight
```

Expected: preflight runs to completion, OCO capability check passes.

- [ ] **Verify `git log` shows the expected commit progression**

```bash
cd /opt/cts && git log --oneline -20
```

Expected: 16 commits in the order of the tasks above, each with a clear feat/docs prefix.

The feature is now ready for the manual integration runbook (Task 16) on Tokocrypto with a small position before flipping the master switch in production.
