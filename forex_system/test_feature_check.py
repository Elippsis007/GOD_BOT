# test_feature_check.py
import sys, os, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.mt5_connector import MT5Connector
from core.indicators    import IndicatorEngine
from signals.ml_model   import MLSignalModel
from pathlib import Path

connector = MT5Connector()
connector.connect()

df_raw     = connector.get_ohlcv('EURUSD', 1, bars=1000)
indicators = IndicatorEngine()
df         = indicators.compute_all(df_raw)

model = MLSignalModel()
X     = model._build_features(df)

print(f"Features _build_features() produces ({len(X.columns)}):")
for col in sorted(X.columns):
    print(f"   {col}")

# Check what the saved scaler was trained on
scaler_path = Path("models/scaler_EURUSD.pkl")
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    if hasattr(scaler, "feature_names_in_"):
        saved = list(scaler.feature_names_in_)
        print(f"\nFeatures saved scaler expects ({len(saved)}):")
        for col in sorted(saved):
            print(f"   {col}")
        new_set    = set(X.columns)
        saved_set  = set(saved)
        missing    = saved_set - new_set
        extra      = new_set - saved_set
        if missing:
            print(f"\n❌ Missing from current features (saved expects these): {missing}")
        if extra:
            print(f"\n⚠️  Extra in current features (saved doesn't know these): {extra}")
        if not missing and not extra:
            print("\n✅ Feature sets match perfectly")
    else:
        print("\n⚠️  Scaler has no feature_names_in_ — sklearn version mismatch")
