# test_debug.py — diagnose intermarket and COT issues
import yfinance as yf
import pandas as pd
import os

print("=" * 55)
print("  🔍 DIAGNOSTIC TEST")
print("=" * 55)

# ── Test yfinance ─────────────────────────────────────────
print("\n📊 YFINANCE VERSION & DOWNLOAD TEST")
print("-" * 40)
print(f"yfinance version: {yf.__version__}")

test_tickers = ["SPY", "GLD", "EURUSD=X", "^GSPC"]
for ticker in test_tickers:
    try:
        data = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
        if data is not None and len(data) > 0:
            print(f"  ✅ {ticker:<12} — {len(data)} rows, last close: {data['Close'].iloc[-1].values[0]:.4f}")
        else:
            print(f"  ❌ {ticker:<12} — empty dataframe")
    except Exception as e:
        print(f"  ❌ {ticker:<12} — error: {e}")

# ── Test COT columns ──────────────────────────────────────
print("\n📋 COT COLUMN NAMES")
print("-" * 40)
cache_file = "data/cot/cot_latest.csv"
if os.path.exists(cache_file):
    df = pd.read_csv(cache_file, nrows=2, low_memory=False)
    print(f"Total columns: {len(df.columns)}")
    print("\nAll column names:")
    for i, col in enumerate(df.columns):
        print(f"  {i:>3}. {col}")
else:
    print("❌ No COT cache file found — run test_research.py first")

print("\n✅ Diagnostic complete")