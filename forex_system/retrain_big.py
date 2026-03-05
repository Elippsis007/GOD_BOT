# retrain_big.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.mt5_connector import MT5Connector
from core.indicators    import IndicatorEngine
from signals.ml_model   import MLSignalModel

print("Connecting to MT5...")
connector = MT5Connector()
connector.connect()

print("Fetching 50,000 bars of EURUSD M1 history...")
df_raw = connector.get_ohlcv('EURUSD', 1, bars=50000)
print(f"Got {len(df_raw)} bars")

print("Computing indicators...")
indicators = IndicatorEngine()
df = indicators.compute_all(df_raw)
print(f"Indicators computed — {len(df)} usable bars")

print("Training model (this will take 3-5 minutes)...")
import os
for f in ['models/rf_EURUSD.pkl', 'models/gb_EURUSD.pkl', 'models/scaler_EURUSD.pkl']:
    if os.path.exists(f):
        os.remove(f)
        print(f"  Deleted old {f}")

model = MLSignalModel()
results = model.train(df, symbol='EURUSD')

print("\n=== TRAINING COMPLETE ===")
print(f"RF  mean CV accuracy: {results['mean_accuracy_rf']:.3f}")
print(f"GB  mean CV accuracy: {results['mean_accuracy_gb']:.3f}")