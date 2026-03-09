# retrain_big.py
"""
Retrains the XGBoost + LightGBM ensemble ML models for every symbol
in CONFIG.WATCHLIST using historical OHLCV data fetched from MT5.

Run from the forex_system/ directory:
    python retrain_big.py

Requirements:
  - MetaTrader5 terminal must be open and logged in before running.
  - models/ directory is created automatically if it does not exist.
  - Old model files for each symbol are deleted before retraining
    so stale pickle files never mix with freshly trained ones.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.mt5_connector import MT5Connector
from core.indicators    import IndicatorEngine
from signals.ml_model   import MLSignalModel
from config.settings    import CONFIG

# ── Ensure models directory exists ────────────────────────────────────────────
os.makedirs("models", exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
#  Interactive M1 / M5 prompt
#  Must run BEFORE connecting to MT5 so the user sees it immediately.
# ────────────────────────────────────────────────────────────────────────────
def _ask_timeframe() -> int:
    """
    Ask the user which scalper timeframe to train for.
    Sets CONFIG.SCALPER_TF_SELECTED so get_scalper_profile() returns
    the correct M1 or M5 block for the rest of the script.

    Returns the timeframe integer (1 or 5).
    """
    print("\n" + "=" * 55)
    print("  🤖  GODBOT — ML Model Retraining")
    print("=" * 55)
    print()
    print("  Select timeframe to train for:")
    print()
    print("    1 = M1  (1-minute scalping)")
    print("         50,000 bars ≈ 35 days of data")
    print("         RSI 7 | EMA 5/13/34 | MACD 5/13/4 | ATR 7")
    print("         TP ~6 pips | SL ~3 pips | Max spread 0.8 pips")
    print()
    print("    2 = M5  (5-minute scalping)  [recommended]")
    print("         50,000 bars ≈ 175 days of data")
    print("         RSI 9 | EMA 8/21/50 | MACD 8/21/5 | ATR 10")
    print("         TP ~12 pips | SL ~6 pips | Max spread 1.2 pips")
    print()

    while True:
        choice = input("  Timeframe (1 or 2): ").strip()
        if choice == "1":
            tf = 1
            break
        elif choice == "2":
            tf = 5
            break
        else:
            print("  ⚠️  Please enter 1 or 2")

    # Write to CONFIG so get_scalper_profile() resolves the correct block
    CONFIG.SCALPER_TF_SELECTED = tf

    # Load and display the active profile so the user can confirm
    try:
        p = CONFIG.get_scalper_profile()
        print(f"\n  ✅ M{tf} profile loaded — active settings:")
        print(f"     RSI period      : {p.get('rsi_period',    '?')}")
        print(f"     EMA fast/slow   : {p.get('ema_fast','?')} / {p.get('ema_slow','?')}")
        print(f"     MACD fast/slow  : {p.get('macd_fast','?')} / {p.get('macd_slow','?')}")
        print(f"     ATR period      : {p.get('atr_period',    '?')}")
        print(f"     ADX threshold   : {p.get('adx_threshold', '?')}")
        print(f"     TP pips         : {p.get('tp_pips',       '?')}")
        print(f"     SL pips         : {p.get('sl_pips',       '?')}")
        print(f"     Max spread      : {p.get('max_spread',    '?')} pips")
        print(f"     Scan interval   : {p.get('scan_secs',     '?')} seconds")
        print(f"     Signal score    : {p.get('signal_score',  '?')} / 9")
    except Exception as e:
        print(f"\n  ⚠️  Could not display profile settings: {e}")

    print()
    return tf


# ── Ask the user BEFORE doing anything else ───────────────────────────────────
selected_tf = _ask_timeframe()
tf_label    = f"M{selected_tf}"

# Build the per-symbol timeframe map from the user's choice.
# Every symbol in the watchlist uses the same selected TF so the
# trained model always matches the live timeframe in main.py.
SYMBOL_TIMEFRAMES = {sym: selected_tf for sym in CONFIG.WATCHLIST}

# Also cover common symbols not yet in the watchlist in case the user
# adds them later — they will default to the selected TF.
for sym in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"):
    SYMBOL_TIMEFRAMES.setdefault(sym, selected_tf)


# ── Connect to MT5 ────────────────────────────────────────────────────────────
connector  = MT5Connector()
indicators = IndicatorEngine()

print("Connecting to MT5...")
connected = connector.connect()
if not connected:
    print("\n❌ Could not connect to MetaTrader5.")
    print("   Make sure the MT5 terminal is open and logged in, then try again.")
    sys.exit(1)

print("✅ MT5 connected\n")


# ── Train each symbol ─────────────────────────────────────────────────────────
trained = []
failed  = []

for symbol in CONFIG.WATCHLIST:

    # ── Resolve timeframe ──────────────────────────────────────────────────
    if symbol not in SYMBOL_TIMEFRAMES:
        print(
            f"⚠️  {symbol} not in SYMBOL_TIMEFRAMES map — "
            f"defaulting to {tf_label}.  Add it to the map for best results."
        )
    timeframe = SYMBOL_TIMEFRAMES.get(symbol, selected_tf)
    sym_tf_label = f"M{timeframe}"

    print(f"\n{'='*55}")
    print(f"  Training ML model: {symbol} ({sym_tf_label})")
    print(f"{'='*55}")

    # ── Fetch raw OHLCV ────────────────────────────────────────────────────
    bars = 50_000
    print(
        f"  Fetching {bars:,} bars of {symbol} {sym_tf_label} history "
        f"(≈{'35 days' if timeframe == 1 else '175 days'})..."
    )
    try:
        df_raw = connector.get_ohlcv(symbol, timeframe, bars=bars)
    except Exception as e:
        print(f"  ❌ Data fetch error for {symbol}: {e} — skipping")
        failed.append(symbol)
        continue

    if df_raw is None or df_raw.empty:
        print(
            f"  ❌ No data returned for {symbol}. "
            f"Check that the symbol is available on your broker and "
            f"that the market has sufficient history — skipping."
        )
        failed.append(symbol)
        continue

    print(f"  ✅ Got {len(df_raw):,} bars")

    # ── Compute indicators ─────────────────────────────────────────────────
    print("  Computing indicators...")
    try:
        df = indicators.compute_all(df_raw)
    except Exception as e:
        print(f"  ❌ Indicator computation error for {symbol}: {e} — skipping")
        failed.append(symbol)
        continue

    if df is None or df.empty:
        print(
            f"  ❌ Indicator engine returned no usable bars for {symbol}. "
            f"The dataset may have too many NaN values — skipping."
        )
        failed.append(symbol)
        continue

    print(f"  ✅ Indicators computed — {len(df):,} usable bars")

    # ── Delete stale model files ───────────────────────────────────────────
    print(f"  Deleting old model files for {symbol}...")
    stale_files = [
        f"models/xgb_{symbol}.pkl",
        f"models/lgbm_{symbol}.pkl",
        f"models/scaler_{symbol}.pkl",
        f"models/features_{symbol}.pkl",
        f"models/rf_{symbol}.pkl",     # legacy format
        f"models/gb_{symbol}.pkl",     # legacy format
    ]
    for filepath in stale_files:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"    Deleted {filepath}")

    # ── Train ──────────────────────────────────────────────────────────────
    print(
        f"  Training {symbol} model on {sym_tf_label} data "
        f"(this takes 5–10 minutes)..."
    )
    print("  Steps: Optuna tuning → 5-fold CV → final fit → save")
    try:
        model   = MLSignalModel()
        results = model.train(df, symbol=symbol)
    except Exception as e:
        print(f"  ❌ Training error for {symbol}: {e} — skipping")
        failed.append(symbol)
        continue

    # ── Validate results ───────────────────────────────────────────────────
    if not results:
        print(
            f"  ❌ Training returned no results for {symbol}. "
            f"Not enough usable samples after label generation — skipping."
        )
        failed.append(symbol)
        continue

    # ── Print results ──────────────────────────────────────────────────────
    xgb_acc  = results.get("mean_accuracy_xgb",  0.0)
    lgbm_acc = results.get("mean_accuracy_lgbm", 0.0)
    n_feat   = results.get("n_features",          0)
    n_samp   = results.get("n_samples",           0)

    print(f"\n  ✅ {symbol} ({sym_tf_label}) training complete:")
    print(
        f"     XGB  mean CV accuracy : {xgb_acc:.3f}  "
        f"{'✅' if xgb_acc  >= 0.55 else '⚠️  below 0.55 — check data quality'}"
    )
    print(
        f"     LGBM mean CV accuracy : {lgbm_acc:.3f}  "
        f"{'✅' if lgbm_acc >= 0.55 else '⚠️  below 0.55 — check data quality'}"
    )
    print(f"     Features used         : {n_feat}")
    print(f"     Samples trained on    : {n_samp:,}")
    print(f"     Timeframe             : {sym_tf_label}")

    if xgb_acc > 0.80 or lgbm_acc > 0.80:
        print(
            f"\n  ⚠️  WARNING: CV accuracy > 80% is unusually high for "
            f"live forex data.\n"
            f"     This may indicate overfitting or a data quality issue.\n"
            f"     Consider reviewing the label threshold "
            f"(ATR_MULTIPLIER) in ml_model.py."
        )

    trained.append(symbol)


# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  RETRAINING COMPLETE  ({tf_label})")
print(f"{'='*55}")
print(f"  ✅ Trained  : {len(trained)}  — {', '.join(trained) if trained else 'none'}")
print(f"  ❌ Failed   : {len(failed)}   — {', '.join(failed)  if failed  else 'none'}")

if failed:
    print(
        f"\n  ⚠️  Some symbols failed. Check the error messages above.\n"
        f"     Common causes:\n"
        f"       — Symbol not available on your broker\n"
        f"       — MT5 has insufficient history for that timeframe\n"
        f"       — Not enough non-NaN bars after indicator computation"
    )

if trained:
    print(f"\n  Models saved to models/ directory:")
    for sym in trained:
        print(f"    models/xgb_{sym}.pkl")
        print(f"    models/lgbm_{sym}.pkl")
        print(f"    models/scaler_{sym}.pkl")
        print(f"    models/features_{sym}.pkl")
    print(
        f"\n  ✅ Models trained on {tf_label} data — "
        f"start the bot and select {tf_label} scalper mode to match."
    )

print(f"\n  Run the bot: python main.py\n")
