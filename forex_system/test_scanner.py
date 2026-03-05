# test_scanner.py
import MetaTrader5 as mt5
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import CONFIG
from core.mt5_connector import MT5Connector, MT5_TIMEFRAME_MAP
from research.calendar_scanner import CalendarScanner

print("=" * 60)
print("  GODBOT DIAGNOSTIC CHECK")
print("=" * 60)

# ── Test 1: MT5 Connection ─────────────────────────────────────
print("\n[1] MT5 Connection...")
connector = MT5Connector()
if connector.connect():
    print("    ✅ Connected to MT5")
    info = connector.get_account_info()
    print(f"    ✅ Account  : {info.get('login')}")
    print(f"    ✅ Balance  : {info.get('balance')} {info.get('currency')}")
    print(f"    ✅ Server   : {info.get('server')}")
else:
    print("    ❌ MT5 connection failed — stop here and fix MT5 first")
    sys.exit(1)

# ── Test 2: OHLCV Data ─────────────────────────────────────────
print("\n[2] OHLCV Data Fetch (EURUSD H1)...")
tf = 16385  # H1 internal constant
df = connector.get_ohlcv("EURUSD", tf, bars=10)
if df is not None and len(df) > 0:
    print(f"    ✅ Got {len(df)} bars")
    print(f"    ✅ Columns : {list(df.columns)}")
    print(f"    ✅ Latest  : {df.index[-1]}  close={df['close'].iloc[-1]}")
else:
    print("    ❌ No OHLCV data returned — timeframe map not applied")

# ── Test 3: OHLCV with Scalper timeframe ──────────────────────
print("\n[3] OHLCV Data Fetch (EURUSD M1 — scalper tf=1)...")
df2 = connector.get_ohlcv("EURUSD", 1, bars=10)
if df2 is not None and len(df2) > 0:
    print(f"    ✅ Got {len(df2)} bars on M1")
else:
    print("    ❌ No M1 data returned")

# ── Test 4: Timeframe Map ──────────────────────────────────────
print("\n[4] Timeframe Map Check...")
test_cases = {
    16385: "H1",
    16388: "H4",
    900:   "M15",
    1:     "M1",
    60:    "H1",
}
all_ok = True
for val, label in test_cases.items():
    result = MT5_TIMEFRAME_MAP.get(val)
    if result is not None:
        print(f"    ✅ {val:>6} → {label} ({result})")
    else:
        print(f"    ❌ {val:>6} → NOT FOUND in map")
        all_ok = False
if all_ok:
    print("    ✅ All timeframe mappings OK")

# ── Test 5: Calendar Scanner ───────────────────────────────────
print("\n[5] Calendar Scanner...")
scanner = CalendarScanner()
result  = scanner.is_safe_to_trade("EURUSD")
print(f"    ✅ is_safe_to_trade returned without error")
print(f"    ✅ Safe     : {result['safe']}")
print(f"    ✅ Reason   : {result['reason']}")
print(f"    ✅ Events   : {len(result['events'])} nearby")

# ── Test 6: No MT5 Calendar Error ─────────────────────────────
print("\n[6] Checking _get_mt5_events returns None cleanly...")
mt5_events = scanner._get_mt5_events("EURUSD")
if mt5_events is None:
    print("    ✅ _get_mt5_events() returns None — no AttributeError")
else:
    print("    ⚠️  _get_mt5_events() returned data unexpectedly")

# ── Test 7: Today's Events ─────────────────────────────────────
print("\n[7] Today's Calendar Events...")
scanner.print_todays_events()

# ── Summary ───────────────────────────────────────────────────
print("=" * 60)
print("  If all items show ✅ — run python main.py")
print("  If any show ❌  — paste this output here for a fix")
print("=" * 60)
