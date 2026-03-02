# test_data.py
import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime

# ── Connect ───────────────────────────────────────────────
print("Connecting to MT5...")

# Step 1: Attach to already-running terminal (no path needed)
if not mt5.initialize():
    # Step 2: Fallback — launch terminal explicitly
    if not mt5.initialize(
        path=r"C:\Program Files\StoneX Europe MT5 Terminal\terminal64.exe",
        login=5046802311,
        password="-8LblzDe",
        server="MetaQuotes-Demo"
    ):
        print(f"❌ Connection FAILED: {mt5.last_error()}")
        quit()

print("✅ Connected!\n")

# ── Symbols to test ───────────────────────────────────────
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]

# ── Fetch OHLCV for each symbol ───────────────────────────
utc_now = datetime.now(pytz.utc)

print("=" * 55)
print(f"{'SYMBOL':<10} {'BARS':>6} {'LAST CLOSE':>12} {'STATUS'}")
print("=" * 55)

for symbol in SYMBOLS:
    if not mt5.symbol_select(symbol, True):
        print(f"{symbol:<10} {'N/A':>6} {'N/A':>12} ❌ Not available")
        continue

    rates = mt5.copy_rates_from(
        symbol,
        mt5.TIMEFRAME_H1,
        utc_now,
        100
    )

    if rates is None or len(rates) == 0:
        print(f"{symbol:<10} {'0':>6} {'N/A':>12} ❌ No data")
        continue

    df         = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    last_close = df["close"].iloc[-1]
    bar_count  = len(df)

    print(f"{symbol:<10} {bar_count:>6} {last_close:>12.5f} ✅ OK")

print("=" * 55)

# ── Show last 5 candles for EURUSD ────────────────────────
print("\n📊 Last 5 EURUSD H1 Candles:")
print("-" * 65)

rates      = mt5.copy_rates_from("EURUSD", mt5.TIMEFRAME_H1, utc_now, 100)
df         = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
df         = df[["time", "open", "high", "low", "close", "tick_volume"]]
df.columns = ["Time", "Open", "High", "Low", "Close", "Volume"]

print(df.tail(5).to_string(index=False))
print("-" * 65)

# ── Spread Check ──────────────────────────────────────────
print("\n💰 Current Spreads:")
print("-" * 40)

for symbol in SYMBOLS:
    tick     = mt5.symbol_info_tick(symbol)
    sym_info = mt5.symbol_info(symbol)
    if tick and sym_info:
        spread = (tick.ask - tick.bid) / sym_info.point / 10
        print(f"  {symbol:<10} {spread:.1f} pips")
    else:
        print(f"  {symbol:<10} N/A")

print("-" * 40)

# ── Disconnect ────────────────────────────────────────────
mt5.shutdown()
print("\n✅ Data test complete — connection closed")