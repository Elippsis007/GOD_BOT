# diagnose_labels.py  — run from forex_system/
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from core.mt5_connector import MT5Connector
from core.indicators    import IndicatorEngine
from config.settings    import CONFIG

connector  = MT5Connector()
indicators = IndicatorEngine()
CONFIG.SCALPER_TF_SELECTED = 5

connector.connect()
df_raw = connector.get_ohlcv("EURUSD", 5, bars=50000)
df     = indicators.compute_all(df_raw)
connector.disconnect()

df.columns = [c.lower() for c in df.columns]
FORWARD_BARS = 12

fwd_ret  = df["close"].shift(-FORWARD_BARS) / df["close"] - 1
abs_move = fwd_ret.abs().dropna()

print(f"\nForward {FORWARD_BARS}-bar absolute move statistics:")
print(f"  mean  : {abs_move.mean():.6f}  ({abs_move.mean()*10000:.2f} pips)")
print(f"  median: {abs_move.median():.6f}  ({abs_move.median()*10000:.2f} pips)")
print(f"  25th % : {abs_move.quantile(0.25):.6f}  ({abs_move.quantile(0.25)*10000:.2f} pips)")
print(f"  75th % : {abs_move.quantile(0.75):.6f}  ({abs_move.quantile(0.75)*10000:.2f} pips)")
print(f"  90th % : {abs_move.quantile(0.90):.6f}  ({abs_move.quantile(0.90)*10000:.2f} pips)")

print(f"\nLabel distribution at different thresholds:")
for thresh in [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012]:
    buy  = ((fwd_ret >  thresh) & (abs_move >  thresh)).sum()
    sell = ((fwd_ret < -thresh) & (abs_move > thresh)).sum()
    hold = len(abs_move) - buy - sell
    total = len(abs_move)
    print(f"  thresh={thresh:.4f} ({thresh*10000:.1f} pips) — "
          f"HOLD {hold/total*100:.1f}%  "
          f"BUY {buy/total*100:.1f}%  "
          f"SELL {sell/total*100:.1f}%")
