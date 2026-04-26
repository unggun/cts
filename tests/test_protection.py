"""Unit and schema tests for OCO/stop exchange-side protection.

TestSchema covers the trade_journal protection_mode column.
TestDataclasses and TestSymbolHelper cover src/execution/protection.py.
"""
import os
import tempfile
from unittest.mock import MagicMock

import ccxt
import pytest

from src.data.database import get_connection, init_db
from src.execution.protection import (
    Active, ClosedWhileDown, FillResult, LostProtection,
    Naked, ProtectionResult,
    place_protection, tokocrypto_symbol,
)


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    yield path
    os.unlink(path)


class TestSchema:
    def test_trade_journal_has_protection_mode_column(self, tmp_db):
        with get_connection(tmp_db) as conn:
            cols = {row["name"] for row in
                    conn.execute("PRAGMA table_info(trade_journal)").fetchall()}
        assert "protection_mode" in cols

    def test_init_db_is_idempotent_when_column_exists(self, tmp_db):
        # Fixture already called init_db once; calling it twice more
        # exercises the idempotency guard (3 invocations total).
        init_db(tmp_db)
        init_db(tmp_db)
        with get_connection(tmp_db) as conn:
            cols = [row["name"] for row in
                    conn.execute("PRAGMA table_info(trade_journal)").fetchall()]
        assert cols.count("protection_mode") == 1

    def test_can_insert_with_protection_mode(self, tmp_db):
        with get_connection(tmp_db) as conn:
            conn.execute("""
                INSERT INTO trade_journal
                (mode, pair, strategy, direction, entry_time, entry_price, protection_mode)
                VALUES ('live', 'BTC/IDR', 'darvas', 'long', '2026-04-26T10:00:00', 1000000, 'oco')
            """)
            conn.commit()
            rows = conn.execute("SELECT protection_mode FROM trade_journal").fetchall()
        assert rows[0]["protection_mode"] == "oco"


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
        assert f.price == 1000000.0
        assert f.reason == "target"
        assert f.time == "2026-04-26T10:00:00"

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
            backoff_seconds=[0, 0, 0],
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

    def test_oco_with_empty_backoff_does_not_crash(self):
        # Empty backoff_seconds should be treated as "no sleep" rather
        # than raising IndexError.
        ex = MagicMock()
        ex.privatePostOpenV1OrdersOco.side_effect = ccxt.NetworkError("down")
        r = place_protection(
            ex, "BTC/IDR", size=0.001,
            stop_loss=900_000.0, target=1_100_000.0,
            is_ath=False, strategy="darvas",
            backoff_seconds=[],
        )
        assert r.mode == "polling"
        assert ex.privatePostOpenV1OrdersOco.call_count == 3


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

    def test_stop_only_returns_none_order_id_when_response_missing_id(self):
        # Defensive: if the exchange somehow returns a response without
        # orderId, we must return order_id=None, not the string "None".
        ex = MagicMock()
        ex.private_post_open_v1_orders.return_value = {}  # no orderId
        r = place_protection(
            ex, "BTC/IDR", size=0.001,
            stop_loss=900_000.0, target=None,
            is_ath=True, strategy="darvas",
        )
        assert r.mode == "stop_only"
        assert r.order_id is None
