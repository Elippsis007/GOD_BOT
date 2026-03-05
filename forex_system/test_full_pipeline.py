# test_full_pipeline.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.mt5_connector import MT5Connector
from core.indicators    import IndicatorEngine
from signals.signal_engine import SignalEngine
from signals.ml_model   import MLSignalModel
from config.settings    import CONFIG

connector  = MT5Connector()
connector.connect()

df_raw     = connector.get_ohlcv('EURUSD', 1, bars=1000)
indicators = IndicatorEngine()
df         = indicators.compute_all(df_raw)

print(f"✅ Data loaded: {len(df)} bars")
print(f"   Columns: {list(df.columns)}")

engine = SignalEngine(trading_style="scalper")
signal = engine.evaluate(df, "EURUSD")

if signal:
    print(f"\n✅ Signal fired: {signal.signal.value}")
    print(f"   Score confidence : {signal.confidence:.0%}")
    print(f"   Entry: {signal.entry}  SL: {signal.sl}  TP: {signal.tp}")
else:
    print("\n⚠️  No signal this candle (score too low)")

model = MLSignalModel()
model.load("EURUSD")
result = model.predict(df)

print(f"\n✅ ML prediction: {['HOLD','BUY','SELL'][result['label']]}")
print(f"   ML confidence : {result['confidence']:.0%}")
print(f"   HOLD={result['probabilities']['HOLD']:.0%}  "
      f"BUY={result['probabilities']['BUY']:.0%}  "
      f"SELL={result['probabilities']['SELL']:.0%}")

print(f"\n   ML_MIN_CONFIDENCE  : {CONFIG.ML_MIN_CONFIDENCE:.0%}")
print(f"   ALERT_MIN_CONFIDENCE: {CONFIG.ALERT_MIN_CONFIDENCE:.0%}")

if result['confidence'] >= CONFIG.ML_MIN_CONFIDENCE:
    print("✅ ML gate: PASS — signal would proceed to sentiment/COT gates")
else:
    print(f"❌ ML gate: BLOCKED — confidence {result['confidence']:.0%} < "
          f"threshold {CONFIG.ML_MIN_CONFIDENCE:.0%}")
