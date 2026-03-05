# test_scalper.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.mt5_connector import MT5Connector, MT5_TIMEFRAME_MAP
import MetaTrader5 as mt5

print("=" * 60)
print("  SCALPER TIMEFRAME CHECK")
print("=" * 60)

connector = MT5Connector()
if not connector.connect():
    print("❌ MT5 connection failed")
    sys.exit(1)

# All timeframes used by scalper and day trader modes
timeframes = {
    # Scalper
    1:     "M1  — Scalper Primary",
    5:     "M5  — Scalper Confirm",
    # Day Trader
    16385: "H1  — DayTrader Primary",
    16388: "H4  — DayTrader Confirm",
    # Bonus check
    15:    "M15 — General",
    60:    "H1  — General (seconds format)",
    900:   "M15 — General (seconds format)",
}

print("\n[1] Timeframe Map Entries...")
for tf_val, label in timeframes.items():
    mapped = MT5_TIMEFRAME_MAP.get(tf_val)
    if mapped is not None:
        print(f"    ✅ {tf_val:>6} → {label} (mt5={mapped})")
    else:
        print(f"    ❌ {tf_val:>6} → {label} — MISSING FROM MAP")

print("\n[2] Live Data Fetch for Each Timeframe...")
for tf_val, label in timeframes.items():
    df = connector.get_ohlcv("EURUSD", tf_val, bars=5)
    if df is not None and len(df) > 0:
        print(f"    ✅ {label:<35} close={df['close'].iloc[-1]:.5f}")
    else:
        print(f"    ❌ {label:<35} NO DATA RETURNED")

print("\n[3] Spread Check (used by scalper gate)...")
info = mt5.symbol_info("EURUSD")
if info:
    spread_pips = info.spread * info.point * 10
    print(f"    ✅ Current spread : {info.spread} points")
    print(f"    ✅ Spread in pips : {spread_pips:.1f}")
    print(f"    ✅ Scalper max    : 1.5 pips")
    if spread_pips <= 1.5:
        print(f"    ✅ Spread OK for scalping")
    else:
        print(f"    ⚠️  Spread too wide for scalper right now "
              f"(normal outside London/NY hours)")
else:
    print("    ❌ Could not read symbol info")

print("\n" + "=" * 60)
print("  Paste output here if any ❌ appears")
print("=" * 60)
