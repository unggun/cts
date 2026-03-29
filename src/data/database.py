"""SQLite database schema and operations for the trading system."""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import get_db_path


def get_connection(db_path: str = None) -> sqlite3.Connection:
    """Get a database connection with row factory."""
    if db_path is None:
        db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str = None):
    """Create all tables if they don't exist."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    # ── OHLCV candle data ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            UNIQUE(pair, timeframe, timestamp)
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup
        ON ohlcv(pair, timeframe, timestamp)
    """)

    # ── Backtest trades ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS backtest_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            strategy TEXT NOT NULL,
            pair TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            direction TEXT NOT NULL DEFAULT 'long',
            entry_time TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_time TEXT,
            exit_price REAL,
            stop_loss REAL,
            take_profit REAL,
            position_size REAL,
            pnl_pct REAL,
            pnl_absolute REAL,
            exit_reason TEXT,
            features_json TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Strategy parameters (versioned) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS strategy_parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL,
            version INTEGER NOT NULL,
            parameters_json TEXT NOT NULL,
            performance_json TEXT,
            source TEXT DEFAULT 'manual',
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(strategy, version)
        )
    """)

    # ── Auto-learning analysis results ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS learning_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_type TEXT NOT NULL,
            strategy TEXT NOT NULL,
            pair TEXT,
            num_trades_analyzed INTEGER,
            findings_json TEXT,
            recommendations_json TEXT,
            applied INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Monte Carlo simulation results ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS simulation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            strategy TEXT NOT NULL,
            num_paths INTEGER,
            initial_capital REAL,
            median_final_equity REAL,
            mean_final_equity REAL,
            worst_final_equity REAL,
            best_final_equity REAL,
            probability_of_ruin REAL,
            max_drawdown_median REAL,
            max_drawdown_worst REAL,
            sharpe_ratio REAL,
            sortino_ratio REAL,
            profit_factor REAL,
            verdict TEXT,
            details_json TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Paper/live trade journal ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trade_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT NOT NULL DEFAULT 'paper',
            pair TEXT NOT NULL,
            strategy TEXT NOT NULL,
            direction TEXT NOT NULL DEFAULT 'long',
            entry_time TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_time TEXT,
            exit_price REAL,
            stop_loss REAL,
            take_profit REAL,
            position_size REAL,
            pnl_pct REAL,
            pnl_absolute REAL,
            exit_reason TEXT,
            features_json TEXT,
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path or get_db_path()}")


# ── OHLCV operations ──

def upsert_ohlcv(conn: sqlite3.Connection, pair: str, timeframe: str,
                  candles: list[list]):
    """Insert or update OHLCV candles. Each candle = [timestamp, O, H, L, C, V]."""
    cur = conn.cursor()
    cur.executemany("""
        INSERT INTO ohlcv (pair, timeframe, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pair, timeframe, timestamp) DO UPDATE SET
            open=excluded.open, high=excluded.high, low=excluded.low,
            close=excluded.close, volume=excluded.volume
    """, [(pair, timeframe, c[0], c[1], c[2], c[3], c[4], c[5]) for c in candles])
    conn.commit()
    return cur.rowcount


def load_ohlcv(conn: sqlite3.Connection, pair: str, timeframe: str,
               start_ts: int = None, end_ts: int = None):
    """Load OHLCV data as a pandas DataFrame."""
    import pandas as pd
    query = "SELECT timestamp, open, high, low, close, volume FROM ohlcv WHERE pair=? AND timeframe=?"
    params = [pair, timeframe]
    if start_ts:
        query += " AND timestamp >= ?"
        params.append(start_ts)
    if end_ts:
        query += " AND timestamp <= ?"
        params.append(end_ts)
    query += " ORDER BY timestamp ASC"

    df = pd.read_sql_query(query, conn, params=params)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)
    return df


# ── Trade operations ──

def save_backtest_trades(conn: sqlite3.Connection, trades: list[dict]):
    """Save a batch of backtest trades."""
    cur = conn.cursor()
    for t in trades:
        cur.execute("""
            INSERT INTO backtest_trades
            (run_id, strategy, pair, timeframe, direction, entry_time, entry_price,
             exit_time, exit_price, stop_loss, take_profit, position_size,
             pnl_pct, pnl_absolute, exit_reason, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            t["run_id"], t["strategy"], t["pair"], t["timeframe"],
            t.get("direction", "long"), t["entry_time"], t["entry_price"],
            t.get("exit_time"), t.get("exit_price"), t.get("stop_loss"),
            t.get("take_profit"), t.get("position_size"),
            t.get("pnl_pct"), t.get("pnl_absolute"), t.get("exit_reason"),
            json.dumps(t.get("features", {}))
        ))
    conn.commit()


def load_backtest_trades(conn: sqlite3.Connection, run_id: str = None,
                         strategy: str = None):
    """Load backtest trades as DataFrame."""
    import pandas as pd
    query = "SELECT * FROM backtest_trades WHERE 1=1"
    params = []
    if run_id:
        query += " AND run_id = ?"
        params.append(run_id)
    if strategy:
        query += " AND strategy = ?"
        params.append(strategy)
    query += " ORDER BY entry_time ASC"
    return pd.read_sql_query(query, conn, params=params)


# ── Parameter versioning ──

def save_parameters(conn: sqlite3.Connection, strategy: str,
                    parameters: dict, source: str = "manual",
                    performance: dict = None, notes: str = None) -> int:
    """Save a new version of strategy parameters. Returns version number."""
    cur = conn.cursor()
    cur.execute(
        "SELECT COALESCE(MAX(version), 0) FROM strategy_parameters WHERE strategy=?",
        (strategy,)
    )
    next_version = cur.fetchone()[0] + 1

    cur.execute("""
        INSERT INTO strategy_parameters (strategy, version, parameters_json,
                                         performance_json, source, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (strategy, next_version, json.dumps(parameters),
          json.dumps(performance) if performance else None, source, notes))
    conn.commit()
    return next_version


def load_latest_parameters(conn: sqlite3.Connection, strategy: str) -> Optional[dict]:
    """Load the latest parameter version for a strategy."""
    cur = conn.cursor()
    cur.execute("""
        SELECT parameters_json FROM strategy_parameters
        WHERE strategy=? ORDER BY version DESC LIMIT 1
    """, (strategy,))
    row = cur.fetchone()
    return json.loads(row[0]) if row else None


if __name__ == "__main__":
    init_db()
