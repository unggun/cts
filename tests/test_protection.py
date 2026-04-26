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
