"""Unit and schema tests for OCO/stop exchange-side protection.

TestSchema covers the trade_journal protection_mode column.
TestDataclasses and TestSymbolHelper cover src/execution/protection.py.
"""
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
