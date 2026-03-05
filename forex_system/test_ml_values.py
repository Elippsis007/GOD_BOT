# test_ml_values.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import joblib
from pathlib import Path
from core.mt5_connector import MT5Connector
from core.indicators    import IndicatorEngine
from signals.ml_model   import MLSignalModel

connector = MT5Connector()
connector.connect()

df_raw     = connector.get_ohlcv('EURUSD', 1, bars=1000)
indicators = IndicatorEngine()
df         = indicators.compute_all(df_raw)

model = MLSignalModel()
model.load('EURUSD')

X        = model._build_features(df)
X_last   = X.iloc[[-1]]
scaler   = joblib.load(Path("models/scaler_EURUSD.pkl"))
X_scaled = scaler.transform(X_last)

print("=== RAW FEATURE VALUES (last bar) ===")
for col, val in zip(X.columns, X_last.values[0]):
    print(f"  {col:<25} = {val:.6f}")

print("\n=== SCALED VALUES (what model sees) ===")
for col, val in zip(X.columns, X_scaled[0]):
    flag = " ⚠️  EXTREME" if abs(val) > 3.0 else ""
    print(f"  {col:<25} = {val:+.4f}{flag}")

print("\n=== NaN CHECK ===")
nan_cols = X_last.columns[X_last.isna().any()].tolist()
if nan_cols:
    print(f"  ❌ NaN values found in: {nan_cols}")
else:
    print("  ✅ No NaN values")

print("\n=== INF CHECK ===")
inf_cols = X_last.columns[np.isinf(X_last.values).any(axis=0)].tolist()
if inf_cols:
    print(f"  ❌ Inf values found in: {inf_cols}")
else:
    print("  ✅ No Inf values")
