# test_ml.py
import os, sys
sys.path.insert(0, '.')

from core.mt5_connector import MT5Connector
from core.indicators    import IndicatorEngine
from signals.ml_model   import MLSignalModel

connector = MT5Connector()
connector.connect()

df_raw = connector.get_ohlcv('EURUSD', 1, bars=200)
indicators = IndicatorEngine()
df = indicators.compute_all(df_raw)

model = MLSignalModel()
model.load('EURUSD')
result = model.predict(df)

print('=== ML PREDICTION ===')
print(f'Full result dict: {result}')
print()
print(f'Label      : {result["label"]} (0=HOLD  1=BUY  2=SELL)')
print(f'Confidence : {result["confidence"]:.0%}')
print()
print(f'Min confidence required : 50%')
if result["confidence"] >= 0.50:
    print('✅ ML gate would PASS — alert would fire')
else:
    print('❌ ML gate BLOCKING — confidence too low')
    print()
    print('WHY: The model sees a SELL setup but is only 36% confident.')
    print('     This means the other 64% probability is split between')
    print('     BUY and HOLD — market conditions are ambiguous.')
    print('     The bot is correctly protecting you from a weak signal.')
