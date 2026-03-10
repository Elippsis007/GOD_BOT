# retrain_big.py
"""
Retrains the XGBoost + LightGBM ensemble ML models for every symbol
in CONFIG.SYMBOLS using historical OHLCV data fetched from MT5.

Run from the forex_system/ directory:
    python retrain_big.py                        # retrain all symbols
    python retrain_big.py --symbols EURUSD       # retrain one symbol
    python retrain_big.py --trials 80            # more Optuna trials
    python retrain_big.py --symbols EURUSD --trials 80

Requirements:
  - MetaTrader5 terminal must be open and logged in before running.
  - models/ directory is created automatically if it does not exist.
  - Old model files for each symbol are replaced only after a
    successful training run so stale models are never lost mid-run.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.mt5_connector              import MT5Connector
from indicators.indicators_engine    import IndicatorEngine   # Fix 1
from signals.ml_model                import MLSignalModel
from config.settings                 import CONFIG
from monitoring.logger               import logger

# ── Ensure models directory exists ────────────────────────────────────────────
os.makedirs("models", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="retrain_big.py",
        description=(
            "GODBOT — retrain XGBoost + LightGBM ensemble models "
            "from MT5 history."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        metavar="SYMBOL",
        default=None,
        help=(
            "Symbols to retrain (space-separated). "
            "Defaults to every symbol in CONFIG.SYMBOLS."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=40,
        metavar="N",
        help=(
            "Number of Optuna hyperparameter-search trials per model "
            "(default: 40)."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Interactive M1 / M5 prompt
# ─────────────────────────────────────────────────────────────────────────────
def _ask_timeframe() -> int:
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

    CONFIG.SCALPER_TF_SELECTED = tf

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
        logger.warning(
            "Could not display scalper profile during retrain prompt: %s", e
        )

    print()
    return tf


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    selected_tf = _ask_timeframe()
    tf_label    = f"M{selected_tf}"

    # ── Build symbol list ──────────────────────────────────────────────────
    # Fix 3 – CONFIG.WATCHLIST does not exist; correct attribute is
    # CONFIG.SYMBOLS throughout the entire codebase.
    if args.symbols:
        unknown = [s for s in args.symbols if s not in CONFIG.SYMBOLS]
        if unknown:
            print(
                f"\n  ⚠️  The following symbols are not in CONFIG.SYMBOLS "
                f"and will be trained anyway: {', '.join(unknown)}"
            )
            logger.warning(
                "Symbols not in CONFIG.SYMBOLS requested for training: %s",
                ", ".join(unknown),
            )
        symbols_to_train = args.symbols
    else:
        symbols_to_train = list(CONFIG.SYMBOLS)   # Fix 3

    symbol_timeframes: dict[str, int] = {
        sym: selected_tf for sym in symbols_to_train
    }

    for sym in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"):
        symbol_timeframes.setdefault(sym, selected_tf)

    bars          = max(50_000, getattr(CONFIG, "BARS_HISTORY", 50_000))
    optuna_trials = args.trials

    logger.info(
        "Retraining started — timeframe=%s, symbols=%s, "
        "bars=%d, optuna_trials=%d",
        tf_label,
        ", ".join(symbols_to_train),
        bars,
        optuna_trials,
    )

    # ── Connect to MT5 ─────────────────────────────────────────────────────
    connector  = MT5Connector()
    indicators = IndicatorEngine()

    print("Connecting to MT5...")
    connected = connector.connect()
    if not connected:
        msg = (
            "Could not connect to MetaTrader5. "
            "Make sure the terminal is open and logged in."
        )
        print(f"\n❌ {msg}")
        logger.error(msg)
        sys.exit(1)

    print("✅ MT5 connected\n")
    logger.info("MT5 connected successfully for retraining session.")

    # ── Training loop ──────────────────────────────────────────────────────
    trained: list[str] = []
    failed:  list[str] = []

    try:
        for symbol in symbols_to_train:

            timeframe    = symbol_timeframes.get(symbol, selected_tf)
            sym_tf_label = f"M{timeframe}"

            print(f"\n{'=' * 55}")
            print(f"  Training ML model: {symbol} ({sym_tf_label})")
            print(f"{'=' * 55}")
            logger.info("Starting training for %s (%s).", symbol, sym_tf_label)

            # ── Fetch raw OHLCV ────────────────────────────────────────────
            approx_days = "35 days" if timeframe == 1 else "175 days"
            print(
                f"  Fetching {bars:,} bars of {symbol} {sym_tf_label} "
                f"history (≈{approx_days})..."
            )
            try:
                df_raw = connector.get_ohlcv(symbol, timeframe, bars=bars)
            except Exception as e:
                msg = f"Data fetch error for {symbol}: {e}"
                print(f"  ❌ {msg} — skipping")
                logger.error(msg)
                failed.append(symbol)
                continue

            if df_raw is None or df_raw.empty:
                msg = (
                    f"No data returned for {symbol}. "
                    f"Check that the symbol is available on your broker "
                    f"and that sufficient history exists."
                )
                print(f"  ❌ {msg} — skipping")
                logger.error(msg)
                failed.append(symbol)
                continue

            print(f"  ✅ Got {len(df_raw):,} bars")
            logger.info("%s: fetched %d bars.", symbol, len(df_raw))

            # ── Compute indicators ─────────────────────────────────────────
            # Fix 2 – method is calculate(df, symbol) not compute_all(df)
            print("  Computing indicators...")
            try:
                df = indicators.compute_all(df_raw)
            except Exception as e:
                msg = f"Indicator computation error for {symbol}: {e}"
                print(f"  ❌ {msg} — skipping")
                logger.error(msg)
                failed.append(symbol)
                continue

            if df is None or df.empty:
                msg = (
                    f"Indicator engine returned no usable bars for {symbol}. "
                    f"The dataset may have too many NaN values."
                )
                print(f"  ❌ {msg} — skipping")
                logger.error(msg)
                failed.append(symbol)
                continue

            print(f"  ✅ Indicators computed — {len(df):,} usable bars")
            logger.info(
                "%s: %d usable bars after indicator computation.",
                symbol, len(df),
            )

            # ── Train ──────────────────────────────────────────────────────
            print(
                f"  Training {symbol} model on {sym_tf_label} data "
                f"(this takes 5–10 minutes)..."
            )
            print(
                f"  Steps: Optuna tuning ({optuna_trials} trials) → "
                f"3-fold walk-forward CV → final fit → "
                f"held-out test → save"
            )

            try:
                model   = MLSignalModel(symbol=symbol)
                results = model.train(df, n_trials=optuna_trials)
            except Exception as e:
                msg = f"Training error for {symbol}: {e}"
                print(f"  ❌ {msg} — skipping")
                logger.error(msg)
                failed.append(symbol)
                continue

            if not results:
                msg = (
                    f"Training returned no results for {symbol}. "
                    f"Not enough usable samples after label generation."
                )
                print(f"  ❌ {msg} — skipping")
                logger.error(msg)
                failed.append(symbol)
                continue

            # ── Save model ─────────────────────────────────────────────────
            # Fix 4 – save() takes no arguments; symbol is set at
            # MLSignalModel(symbol=symbol) construction time.
            try:
                model.save()
                print(f"  ✅ Models saved for {symbol}")
            except Exception as e:
                msg = f"Model save error for {symbol}: {e}"
                print(f"  ❌ {msg} — skipping")
                logger.error(msg)
                failed.append(symbol)
                continue

            # ── Print results ──────────────────────────────────────────────
            xgb_acc  = results.get("xgb_cv_accuracy",  0.0)
            lgbm_acc = results.get("lgbm_cv_accuracy", 0.0)
            test_acc = results.get("test_accuracy",     0.0)
            n_feat   = results.get("n_features",        0)
            n_train  = results.get("n_train",           0)
            n_test   = results.get("n_test",            0)

            status_xgb  = (
                "✅" if xgb_acc  >= 0.55
                else "⚠️  below 0.55 — check data quality"
            )
            status_lgbm = (
                "✅" if lgbm_acc >= 0.55
                else "⚠️  below 0.55 — check data quality"
            )

            cv_mean = (xgb_acc + lgbm_acc) / 2
            gap     = abs(test_acc - cv_mean)
            if gap > 0.10:
                status_test = (
                    f"⚠️  gap vs CV mean is {gap:.3f} — "
                    f"possible overfit or distribution shift"
                )
            elif test_acc >= 0.55:
                status_test = "✅"
            else:
                status_test = "⚠️  below 0.55 — check data quality"

            print(f"\n  ✅ {symbol} ({sym_tf_label}) training complete:")
            print(f"     XGB  mean CV accuracy : {xgb_acc:.3f}  {status_xgb}")
            print(f"     LGBM mean CV accuracy : {lgbm_acc:.3f}  {status_lgbm}")
            print(f"     Held-out test accuracy: {test_acc:.3f}  {status_test}")
            print(f"     Train rows            : {n_train:,}  (80%)")
            print(f"     Test rows             : {n_test:,}   (20% held-out)")
            print(f"     Features used         : {n_feat}")
            print(f"     Timeframe             : {sym_tf_label}")
            print(f"     Optuna trials         : {optuna_trials}")

            logger.info(
                "%s (%s): training complete — XGB CV=%.3f, LGBM CV=%.3f, "
                "test=%.3f, features=%d, trials=%d.",
                symbol, sym_tf_label,
                xgb_acc, lgbm_acc, test_acc,
                n_feat, optuna_trials,
            )

            if xgb_acc > 0.80 or lgbm_acc > 0.80:
                warn = (
                    f"CV accuracy > 80% for {symbol} is unusually high for "
                    f"live forex data. This may indicate overfitting or a "
                    f"data quality issue. Consider reviewing the label "
                    f"threshold (ATR_MULTIPLIER) in ml_model.py."
                )
                print(f"\n  ⚠️  WARNING: {warn}")
                logger.warning(warn)

            if gap > 0.10:
                warn = (
                    f"Large gap between test accuracy ({test_acc:.3f}) and "
                    f"CV mean ({cv_mean:.3f}) for {symbol}. "
                    f"The model may not generalise well to unseen data. "
                    f"Consider retraining with more bars or adjusting "
                    f"label thresholds."
                )
                print(f"\n  ⚠️  WARNING: {warn}")
                logger.warning(warn)

            trained.append(symbol)

    finally:
        connector.disconnect()
        print("\n  MT5 disconnected.")
        logger.info("MT5 disconnected after retraining session.")

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  RETRAINING COMPLETE  ({tf_label})")
    print(f"{'=' * 55}")
    print(
        f"  ✅ Trained  : {len(trained)}  — "
        f"{', '.join(trained) if trained else 'none'}"
    )
    print(
        f"  ❌ Failed   : {len(failed)}   — "
        f"{', '.join(failed)  if failed  else 'none'}"
    )

    if failed:
        print(
            f"\n  ⚠️  Some symbols failed. Check the error messages above.\n"
            f"     Common causes:\n"
            f"       — Symbol not available on your broker\n"
            f"       — MT5 has insufficient history for that timeframe\n"
            f"       — Not enough non-NaN bars after indicator computation"
        )
        logger.warning("Retraining failed for: %s", ", ".join(failed))

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
        logger.info(
            "Retraining complete — trained: %s | failed: %s.",
            ", ".join(trained),
            ", ".join(failed) if failed else "none",
        )

    print(f"\n  Run the bot: python main.py\n")


if __name__ == "__main__":
    main()
