# run_backtest.py
"""
Run a full backtest on GBPUSD M5.
Usage:  python run_backtest.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.engine import BacktestEngine

if __name__ == "__main__":
    engine = BacktestEngine(
        ml_min_confidence = 0.40,
        signal_score      = 3,
        style             = "scalper",
    )

    report = engine.run(
        symbol          = "GBPUSD",
        timeframe       = 5,
        bars            = 10000,
        initial_balance = 200.0,
    )

    engine.export_report(report)
    print(f"\n✅ Full trade log exported to reports/ folder")
