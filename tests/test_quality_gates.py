from src.run_learning_cycle import check_quality_gates


def test_quality_gates_pass():
    results = {
        "BTC/IDR": {
            "total_trades": 80,
            "winners": 48,
            "win_rate": 0.60,
            "profit_factor": 1.8,
            "max_drawdown": 0.15,
        },
        "ETH/IDR": {
            "total_trades": 50,
            "winners": 28,
            "win_rate": 0.55,
            "profit_factor": 1.6,
            "max_drawdown": 0.10,
        },
    }
    passed, failures = check_quality_gates(results)
    assert passed is True
    assert len(failures) == 0


def test_quality_gates_fail_win_rate():
    results = {
        "BTC/IDR": {
            "total_trades": 100,
            "winners": 40,
            "win_rate": 0.40,
            "profit_factor": 1.8,
            "max_drawdown": 0.15,
        },
    }
    passed, failures = check_quality_gates(results)
    assert passed is False
    assert any("win_rate" in f.lower() or "Win rate" in f for f in failures)


def test_quality_gates_fail_drawdown():
    results = {
        "BTC/IDR": {
            "total_trades": 100,
            "winners": 60,
            "win_rate": 0.60,
            "profit_factor": 1.8,
            "max_drawdown": 0.30,
        },
    }
    passed, failures = check_quality_gates(results)
    assert passed is False
    assert any("drawdown" in f.lower() for f in failures)


def test_quality_gates_fail_min_trades():
    results = {
        "BTC/IDR": {
            "total_trades": 20,
            "winners": 12,
            "win_rate": 0.60,
            "profit_factor": 1.8,
            "max_drawdown": 0.10,
        },
    }
    passed, failures = check_quality_gates(results)
    assert passed is False
    assert any("trades" in f.lower() for f in failures)


def test_quality_gates_fail_profit_factor():
    results = {
        "BTC/IDR": {
            "total_trades": 100,
            "winners": 60,
            "win_rate": 0.60,
            "profit_factor": 1.1,
            "max_drawdown": 0.10,
        },
    }
    passed, failures = check_quality_gates(results)
    assert passed is False
    assert any("profit" in f.lower() for f in failures)


def test_quality_gates_skip_empty_pairs():
    results = {
        "BTC/IDR": {
            "total_trades": 100,
            "winners": 60,
            "win_rate": 0.60,
            "profit_factor": 1.8,
            "max_drawdown": 0.10,
        },
        "DOGE/IDR": {
            "total_trades": 0,
            "message": "No trades generated",
        },
    }
    passed, failures = check_quality_gates(results)
    assert passed is True
