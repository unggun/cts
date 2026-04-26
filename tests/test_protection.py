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
    place_protection, replace_trailing_stop, tokocrypto_symbol,
    check_protection_fill, reconcile_on_restart,
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

    def test_trade_journal_has_protection_id_columns(self, tmp_db):
        with get_connection(tmp_db) as conn:
            cols = {row["name"] for row in
                    conn.execute("PRAGMA table_info(trade_journal)").fetchall()}
        assert "protection_order_id" in cols
        assert "protection_child_ids" in cols

    def test_can_insert_with_protection_ids(self, tmp_db):
        with get_connection(tmp_db) as conn:
            conn.execute("""
                INSERT INTO trade_journal
                (mode, pair, strategy, direction, entry_time, entry_price,
                 protection_mode, protection_order_id, protection_child_ids)
                VALUES ('live', 'BTC/IDR', 'darvas', 'long',
                        '2026-04-26T10:00:00', 1000000, 'oco', 'L1',
                        '["TP1", "SL1"]')
            """)
            conn.commit()
            row = conn.execute(
                "SELECT protection_order_id, protection_child_ids FROM trade_journal"
            ).fetchone()
        assert row["protection_order_id"] == "L1"
        assert row["protection_child_ids"] == '["TP1", "SL1"]'


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

    def test_filled_with_missing_timestamp_returns_empty_time(self):
        # If the exchange omits lastTradeTimestamp, _ts_to_iso must return
        # an empty string rather than silently substituting wall-clock time.
        ex = MagicMock()
        ex.fetch_order.return_value = {
            "id": "S1", "status": "closed",
            "average": 950_000.0,
            # no lastTradeTimestamp
        }
        result = check_protection_fill(ex, "BTC/IDR", self._pos())
        assert isinstance(result, FillResult)
        assert result.time == ""

    def test_filled_with_missing_price_returns_none(self):
        # Defensive: malformed response with no price → return None,
        # don't crash with float(None).
        ex = MagicMock()
        ex.fetch_order.return_value = {
            "id": "S1", "status": "closed",
            # no average, no price
            "lastTradeTimestamp": 1_700_000_000_000,
        }
        result = check_protection_fill(ex, "BTC/IDR", self._pos())
        assert result is None


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
        # TP filled, SL auto-cancelled by OCO. SL doesn't appear in open
        # orders; TP doesn't either (it filled). Trade record reveals
        # which one filled.
        ex = MagicMock()
        ex.fetch_open_orders.return_value = [
            # Imagine a different OCO's leg is also live on this pair —
            # we should NOT match it.
            {"id": "OTHER_TP", "status": "open"},
            {"id": "OTHER_SL", "status": "open"},
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
