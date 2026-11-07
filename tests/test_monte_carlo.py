"""Monte Carlo runner must not crash on tiny trade samples."""
import pandas as pd

import src.simulation.monte_carlo as mc


def test_run_simulation_with_few_trades_returns_insufficient(monkeypatch):
    trades = pd.DataFrame({"pnl_pct": [0.01, -0.02, 0.03]})
    monkeypatch.setattr(mc, "init_db", lambda: None)
    monkeypatch.setattr(mc, "get_connection", lambda: type("C", (), {"close": lambda self: None})())
    monkeypatch.setattr(mc, "load_backtest_trades", lambda *a, **k: trades)

    result = mc.run_simulation(strategy="x", run_ids=["r1"])

    assert result["verdict"] == "INSUFFICIENT_DATA"
