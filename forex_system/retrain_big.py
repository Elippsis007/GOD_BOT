# retrain_big.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.mt5_connector import MT5Connector
from core.indicators    import IndicatorEngine
from signals.ml_model   import MLSignalModel
from config.settings    import CONFIG

connector  = MT5Connector()
indicators = IndicatorEngine()

print("Connecting to MT5...")
connector.connect()

# EURUSD trains on M1 (50k bars = ~35 days of granular data)
# GBPUSD trains on M5 (50k bars = ~175 days, matches live trading timeframe)
SYMBOL_TIMEFRAMES = {
    "EURUSD": 1,
}

for symbol in CONFIG.WATCHLIST:
    timeframe = SYMBOL_TIMEFRAMES.get(symbol, 5)
    tf_label  = f"M{timeframe}"

    print(f"\n{'='*50}")
    print(f"  Training ML model for {symbol} ({tf_label})")
    print(f"{'='*50}")

    print(f"Fetching 50,000 bars of {symbol} {tf_label} history...")
    df_raw = connector.get_ohlcv(symbol, timeframe, bars=50000)
    print(f"Got {len(df_raw)} bars")

    print("Computing indicators...")
    df = indicators.compute_all(df_raw)
    print(f"Indicators computed — {len(df)} usable bars")

    print("Deleting old model files...")
    for f in [
        f'models/xgb_{symbol}.pkl',
        f'models/lgbm_{symbol}.pkl',
        f'models/scaler_{symbol}.pkl',
        f'models/features_{symbol}.pkl',
        f'models/rf_{symbol}.pkl',
        f'models/gb_{symbol}.pkl',
    ]:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Deleted {f}")

    print(f"Training {symbol} model (3-5 minutes)...")
    model   = MLSignalModel()
    results = model.train(df, symbol=symbol)

    print(f"\n  ✅ {symbol} training complete:")
    print(f"     XGB mean CV accuracy : {results['mean_accuracy_xgb']:.3f}")
    print(f"     LGBM mean CV accuracy: {results['mean_accuracy_lgbm']:.3f}")
    print(f"     Features used        : {results['n_features']}")
    print(f"     Samples trained on   : {results['n_samples']}")

print(f"\n{'='*50}")
print("  ✅ All models trained successfully!")
print(f"{'='*50}\n")
