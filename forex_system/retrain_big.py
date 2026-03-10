# retrain_big.py
"""
Retrains the triple-specialist ML models (BUY / SELL / REGIME) for every
symbol in CONFIG.SYMBOLS using historical OHLCV data fetched from MT5.

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

from core.mt5_connector           import MT5Connector
from indicators.indicators_engine import IndicatorEngine
from signals.ml_model             import MLSignalModel
from config.settings              import CONFIG
from monitoring.logger            import logger

# ── Ensure models directory exists ────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

# ── Bar count ─────────────────────────────────────────────────────────────────
_TARGET_BARS = 100_000


# ─────────────────────────────────────────────────────────────────────────────
#  CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="retrain_big.py",
        description=(
            "GODBOT — retrain triple-specialist ML models "
            "(BUY / SELL / REGIME) from MT5 history."
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
            "Number of Optuna hyperparameter-search trials per specialist "
            "(default: 40). Total trials = N × 3 specialists."
        ),
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=_TARGET_BARS,
        metavar="N",
        help=(
            f"Number of OHLCV bars to fetch per symbol "
            f"(default: {_TARGET_BARS:,}). "
            f"Override if your broker has less history available."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Approximate calendar days helper
# ─────────────────────────────────────────────────────────────────────────────
def _approx_days(bars: int, timeframe_minutes: int) -> str:
    """
    Returns a human-readable string such as '~347 days (~1 year)'.
    Accounts for the fact that forex markets are open ~24 h/day Mon–Fri.
    """
    trading_minutes_per_day = 1_440
    total_minutes  = bars * timeframe_minutes
    calendar_days  = total_minutes / trading_minutes_per_day
    trading_days   = calendar_days * (5 / 7)

    if trading_days >= 300:
        years = trading_days / 252
        return (
            f"~{int(calendar_days)} calendar days "
            f"(~{years:.1f} year{'s' if years >= 1.95 else ''})"
        )
    elif trading_days >= 60:
        months = trading_days / 21
        return (
            f"~{int(calendar_days)} calendar days (~{months:.0f} months)"
        )
    else:
        weeks = trading_days / 5
        return (
            f"~{int(calendar_days)} calendar days (~{weeks:.0f} weeks)"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Results display helper  ← UPDATED for triple-specialist architecture
# ─────────────────────────────────────────────────────────────────────────────
def _print_results(
    symbol:       str,
    tf_label:     str,
    results:      dict,
    bars:         int,
    actual_bars:  int,
    trials:       int,
) -> None:
    """
    Prints per-specialist metrics from the new nested results dict returned
    by MLSignalModel.train():

        {
            "buy":    { "xgb_cv_f1", "lgbm_cv_f1", "test_f1", "test_auc", ... },
            "sell":   { ... },
            "regime": { ... },
            "n_train": int,
            "n_test":  int,
            "n_features": int,
        }
    """
    buy    = results.get("buy",    {})
    sell   = results.get("sell",   {})
    regime = results.get("regime", {})
    n_train   = results.get("n_train",    0)
    n_test    = results.get("n_test",     0)
    n_feat    = results.get("n_features", 0)

    def _status_f1(f1: float) -> str:
        if f1 >= 0.60:
            return "✅"
        if f1 >= 0.50:
            return "🟡 above random, below target"
        return "⚠️  below 0.50 — check labels / data quality"

    def _status_auc(auc: float) -> str:
        if auc >= 0.65:
            return "✅"
        if auc >= 0.55:
            return "🟡 moderate"
        return "⚠️  near random (0.50)"

    buy_xgb_f1   = buy.get("xgb_cv_f1",   0.0)
    buy_lgbm_f1  = buy.get("lgbm_cv_f1",  0.0)
    buy_test_f1  = buy.get("test_f1",      0.0)
    buy_test_auc = buy.get("test_auc",     0.0)

    sell_xgb_f1   = sell.get("xgb_cv_f1",   0.0)
    sell_lgbm_f1  = sell.get("lgbm_cv_f1",  0.0)
    sell_test_f1  = sell.get("test_f1",      0.0)
    sell_test_auc = sell.get("test_auc",     0.0)

    reg_xgb_f1   = regime.get("xgb_cv_f1",   0.0)
    reg_lgbm_f1  = regime.get("lgbm_cv_f1",  0.0)
    reg_test_f1  = regime.get("test_f1",      0.0)
    reg_test_auc = regime.get("test_auc",     0.0)

    print(f"\n  ✅ {symbol} ({tf_label}) training complete:")
    print()
    print(f"     ┌─ BUY specialist ───────────────────────────────────")
    print(f"     │  XGB  CV F1  : {buy_xgb_f1:.4f}  {_status_f1(buy_xgb_f1)}")
    print(f"     │  LGBM CV F1  : {buy_lgbm_f1:.4f}  {_status_f1(buy_lgbm_f1)}")
    print(f"     │  Test F1     : {buy_test_f1:.4f}  {_status_f1(buy_test_f1)}")
    print(f"     │  Test AUC    : {buy_test_auc:.4f}  {_status_auc(buy_test_auc)}")
    print()
    print(f"     ├─ SELL specialist ──────────────────────────────────")
    print(f"     │  XGB  CV F1  : {sell_xgb_f1:.4f}  {_status_f1(sell_xgb_f1)}")
    print(f"     │  LGBM CV F1  : {sell_lgbm_f1:.4f}  {_status_f1(sell_lgbm_f1)}")
    print(f"     │  Test F1     : {sell_test_f1:.4f}  {_status_f1(sell_test_f1)}")
    print(f"     │  Test AUC    : {sell_test_auc:.4f}  {_status_auc(sell_test_auc)}")
    print()
    print(f"     ├─ REGIME specialist ────────────────────────────────")
    print(f"     │  XGB  CV F1  : {reg_xgb_f1:.4f}  {_status_f1(reg_xgb_f1)}")
    print(f"     │  LGBM CV F1  : {reg_lgbm_f1:.4f}  {_status_f1(reg_lgbm_f1)}")
    print(f"     │  Test F1     : {reg_test_f1:.4f}  {_status_f1(reg_test_f1)}")
    print(f"     │  Test AUC    : {reg_test_auc:.4f}  {_status_auc(reg_test_auc)}")
    print()
    print(f"     ├─ Dataset ──────────────────────────────────────────")
    print(f"     │  Train rows  : {n_train:,}  (80%)")
    print(f"     │  Test rows   : {n_test:,}  (20% held-out)")
    print(f"     │  Features    : {n_feat}")
    print(f"     │  Timeframe   : {tf_label}")
    print(f"     │  Bars req    : {bars:,}")
    print(f"     │  Bars recv   : {actual_bars:,}")
    print(f"     └─ Optuna      : {trials} trials × 3 specialists "
          f"= {trials * 3} total")

    # ── Overfit warning ────────────────────────────────────────────────────
    for spec_name, xgb_f1, lgbm_f1, test_f1 in (
        ("BUY",    buy_xgb_f1,  buy_lgbm_f1,  buy_test_f1),
        ("SELL",   sell_xgb_f1, sell_lgbm_f1, sell_test_f1),
        ("REGIME", reg_xgb_f1,  reg_lgbm_f1,  reg_test_f1),
    ):
        cv_mean = (xgb_f1 + lgbm_f1) / 2
        gap     = abs(test_f1 - cv_mean)
        if (spec_name != "REGIME" and (xgb_f1 > 0.95 or lgbm_f1 > 0.95)) or \
   (spec_name == "REGIME" and (xgb_f1 > 0.995 or lgbm_f1 > 0.995)):
            warn = (
                f"{spec_name} CV F1 > 0.95 — unusually high. "
                f"This may indicate label leakage or overfitting."
            )
            print(f"\n  ⚠️  WARNING: {warn}")
            logger.warning(warn)
        if gap > 0.10 and spec_name != "REGIME":
            # REGIME overfitting warning suppressed — high CV/test F1
            # on REGIME is expected (ADX label is highly learnable)
            warn = (
                f"{spec_name} CV/test F1 gap = {gap:.3f} — "
                f"possible overfit or distribution shift. "
                f"Consider retraining with more bars."
            )
            print(f"\n  ⚠️  WARNING: {warn}")
            logger.warning(warn)

    logger.info(
        "%s (%s): training complete — "
        "BUY F1=%.3f AUC=%.3f | SELL F1=%.3f AUC=%.3f | "
        "REGIME F1=%.3f AUC=%.3f | features=%d trials=%d bars=%d",
        symbol, tf_label,
        buy_test_f1,  buy_test_auc,
        sell_test_f1, sell_test_auc,
        reg_test_f1,  reg_test_auc,
        n_feat, trials, actual_bars,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Interactive M1 / M5 prompt
# ─────────────────────────────────────────────────────────────────────────────
def _ask_timeframe(bars: int) -> int:
    print("\n" + "=" * 60)
    print("  🤖  GODBOT — ML Model Retraining")
    print("=" * 60)
    print()
    print("  Select timeframe to train for:")
    print()
    print("    1 = M1  (1-minute scalping)")
    print(f"         {bars:,} bars ≈ {_approx_days(bars, 1)}")
    print("         RSI 7 | EMA 5/13/34 | MACD 5/13/4 | ATR 7")
    print("         TP ~6 pips | SL ~3 pips | Max spread 0.8 pips")
    print()
    print("    2 = M5  (5-minute scalping)  [recommended]")
    print(f"         {bars:,} bars ≈ {_approx_days(bars, 5)}")
    print("         RSI 9 | EMA 8/21/50 | MACD 8/21/5 | ATR 10")
    print("         TP ~12 pips | SL ~6 pips | Max spread 1.0 pips")
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
        print(f"     Signal score    : {p.get('signal_score',  '?')} / 10")
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

    bars = max(args.bars, getattr(CONFIG, "BARS_HISTORY", _TARGET_BARS))

    selected_tf = _ask_timeframe(bars)
    tf_label    = f"M{selected_tf}"

    # ── Build symbol list ──────────────────────────────────────────────────
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
        symbols_to_train = list(CONFIG.SYMBOLS)

    symbol_timeframes: dict[str, int] = {
        sym: selected_tf for sym in symbols_to_train
    }

    for sym in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"):
        symbol_timeframes.setdefault(sym, selected_tf)

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

            print(f"\n{'=' * 60}")
            print(f"  Training ML model: {symbol} ({sym_tf_label})")
            print(f"{'=' * 60}")
            logger.info("Starting training for %s (%s).", symbol, sym_tf_label)

            # ── Fetch raw OHLCV ────────────────────────────────────────────
            approx = _approx_days(bars, timeframe)
            print(
                f"  Fetching {bars:,} bars of {symbol} {sym_tf_label} "
                f"history (≈{approx})..."
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

            actual_bars = len(df_raw)
            if actual_bars < bars:
                print(
                    f"  ⚠️  MT5 returned {actual_bars:,} bars "
                    f"(requested {bars:,}) — broker history may be limited. "
                    f"Training will proceed if above MIN_TRAINING_BARS."
                )
                logger.warning(
                    "%s: MT5 returned %d bars, requested %d.",
                    symbol, actual_bars, bars,
                )

            print(f"  ✅ Got {actual_bars:,} bars")
            logger.info("%s: fetched %d bars.", symbol, actual_bars)

            # ── Compute indicators ─────────────────────────────────────────
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
                f"(this takes 15–50 minutes)..."
            )
            print(
                f"  Steps: Optuna tuning ({optuna_trials} trials × 3 "
                f"specialists) → 3-fold walk-forward CV → final fit → "
                f"temperature calibration → held-out test → save"
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
            _print_results(
                symbol=symbol,
                tf_label=sym_tf_label,
                results=results,
                bars=bars,
                actual_bars=actual_bars,
                trials=optuna_trials,
            )

            trained.append(symbol)

    finally:
        connector.disconnect()
        print("\n  MT5 disconnected.")
        logger.info("MT5 disconnected after retraining session.")

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  RETRAINING COMPLETE  ({tf_label})")
    print(f"{'=' * 60}")
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
            for specialist in ("buy", "sell", "regime"):
                print(f"    models/xgb_{specialist}_{sym}.pkl")
                print(f"    models/lgbm_{specialist}_{sym}.pkl")
                print(f"    models/scaler_{specialist}_{sym}.pkl")
                print(f"    models/temps_{specialist}_{sym}.pkl")
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
