# backtest/engine.py
"""
GODBOT Backtesting Engine
Replays historical M5 bars through the exact same gate pipeline
used in live trading and produces a full performance report.

Usage:
    from backtest.engine import BacktestEngine
    engine = BacktestEngine()
    report = engine.run(symbol="EURUSD", bars=10000)
    engine.export_report(report)
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore")

# ── Suppress noisy loggers during backtest bar-by-bar iteration ───────────────
logging.getLogger("SignalEngine").setLevel(logging.WARNING)
logging.getLogger("MLModel").setLevel(logging.WARNING)
logging.getLogger("Indicators").setLevel(logging.WARNING)
logging.getLogger("MT5Connector").setLevel(logging.WARNING)

# ── Path fix so backtest/ can import from project root ────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mt5_connector    import MT5Connector
from core.indicators       import IndicatorEngine
from signals.signal_engine import SignalEngine
from signals.ml_model      import MLSignalModel
from config.settings       import CONFIG
from monitoring.logger     import get_logger

logger = get_logger("Backtest")


# ────────────────────────────────────────────────────────────────────────────
#  Data structures
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    symbol:      str
    direction:   str
    entry_time:  datetime
    exit_time:   Optional[datetime]
    entry_price: float
    exit_price:  float
    sl:          float
    tp:          float
    volume:      float
    pnl_pips:    float
    pnl_usd:     float
    exit_reason: str
    gate2_score: int
    ml_conf:     float


@dataclass
class BacktestReport:
    symbol:           str
    timeframe:        int
    bars_tested:      int
    start_date:       str
    end_date:         str
    initial_balance:  float
    final_balance:    float
    total_trades:     int
    winning_trades:   int
    losing_trades:    int
    win_rate:         float
    total_pnl_usd:    float
    total_pnl_pips:   float
    avg_win_pips:     float
    avg_loss_pips:    float
    avg_rr_achieved:  float
    max_drawdown_usd: float
    max_drawdown_pct: float
    sharpe_ratio:     float
    profit_factor:    float
    expectancy_usd:   float
    trades:           list = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
#  Engine
# ────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Walk-forward backtest engine.

    Gate pipeline per bar:
      Gate 1 — Calendar/news  : skipped (no live RSS in backtest)
      Gate 2 — Technical score: SignalEngine confluence check
      Gate 3 — ML prediction  : direction + confidence filter
      Gates 4-6               : skipped for scalper (same as live)

    Position management:
      - Fixed 0.01 lots per trade
      - SL / TP set by SignalEngine ATR logic
      - High/low each bar checked for SL or TP hit
      - Max 1 open position at a time
      - EOD close at 23:45 UTC
    """

    PIP_VALUE_PER_LOT = 10.0
    LOT_SIZE          = 0.01
    PIP_VALUE         = PIP_VALUE_PER_LOT * LOT_SIZE   # $0.10 per pip

    WARMUP_BARS       = 100
    WINDOW_BARS       = 200    # bars fed to SignalEngine each step

    def __init__(
        self,
        ml_min_confidence: float = None,
        signal_score:      int   = None,
        style:             str   = "scalper",
    ):
        self.ml_min_conf  = ml_min_confidence or CONFIG.ML_MIN_CONFIDENCE
        self.signal_score = signal_score       or CONFIG.SCALPER_SIGNAL_SCORE
        self.style        = style

        self.connector  = MT5Connector()
        self.indicators = IndicatorEngine()
        self.signal_eng = SignalEngine(trading_style=style)
        self.ml_model   = MLSignalModel()

    # ────────────────────────────────────────────────────────────────────────
    #  Main entry point
    # ────────────────────────────────────────────────────────────────────────

    def run(
        self,
        symbol:          str   = "EURUSD",
        timeframe:       int   = 5,
        bars:            int   = 10000,
        initial_balance: float = 200.0,
    ) -> BacktestReport:

        logger.info(
            f"🔄 Backtest starting | {symbol} M{timeframe} | "
            f"{bars} bars | Balance: ${initial_balance:.2f}"
        )

        # ── Connect ───────────────────────────────────────────────────────────
        if not self.connector.connect():
            raise RuntimeError("Cannot connect to MT5")

        # ── Load ML model ─────────────────────────────────────────────────────
        if not self.ml_model.load(symbol):
            raise RuntimeError(
                "No saved ML model found. Run: python retrain_big.py"
            )

        # ── Fetch and compute indicators once upfront ─────────────────────────
        df_raw = self.connector.get_ohlcv(symbol, timeframe, bars=bars)
        if df_raw is None or df_raw.empty:
            raise RuntimeError(f"No data returned for {symbol}")

        df_full = self.indicators.compute_all(df_raw)
        if df_full is None or df_full.empty:
            raise RuntimeError("Indicator computation failed")

        n_bars = len(df_full)
        logger.info(f"   📊 {n_bars} usable bars after indicator warmup")
        logger.info(
            f"   📅 {str(df_full.index[0])[:16]}  →  "
            f"{str(df_full.index[-1])[:16]}"
        )

        start_date = str(df_full.index[0])
        end_date   = str(df_full.index[-1])

        # ── Walk-forward simulation ───────────────────────────────────────────
        trades:       list  = []
        equity:       float = initial_balance
        equity_curve: list  = [initial_balance]
        peak_equity:  float = initial_balance
        max_drawdown: float = 0.0
        open_trade:   Optional[BacktestTrade] = None

        ML_LABEL_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}

        # Progress counter — print a dot every 500 bars so user knows it is running
        for i in range(self.WARMUP_BARS, n_bars):

            if (i - self.WARMUP_BARS) % 500 == 0:
                pct = (i - self.WARMUP_BARS) / (n_bars - self.WARMUP_BARS) * 100
                print(
                    f"\r   ⏳ Progress: {pct:5.1f}%  |  "
                    f"Trades: {len(trades)}  |  "
                    f"Equity: ${equity:.2f}     ",
                    end="", flush=True,
                )

            bar       = df_full.iloc[i]
            bar_time  = df_full.index[i]
            bar_high  = float(bar["high"])
            bar_low   = float(bar["low"])
            bar_close = float(bar["close"])

            # ── Check open trade exit first ───────────────────────────────────
            if open_trade is not None:
                exit_price  = None
                exit_reason = None

                if open_trade.direction == "BUY":
                    if bar_low <= open_trade.sl:
                        exit_price  = open_trade.sl
                        exit_reason = "SL"
                    elif bar_high >= open_trade.tp:
                        exit_price  = open_trade.tp
                        exit_reason = "TP"
                else:  # SELL
                    if bar_high >= open_trade.sl:
                        exit_price  = open_trade.sl
                        exit_reason = "SL"
                    elif bar_low <= open_trade.tp:
                        exit_price  = open_trade.tp
                        exit_reason = "TP"

                # EOD close at 23:45 UTC
                if exit_price is None:
                    try:
                        if bar_time.hour == 23 and bar_time.minute >= 45:
                            exit_price  = bar_close
                            exit_reason = "EOD"
                    except AttributeError:
                        pass

                if exit_price is not None:
                    pips    = self._calc_pips(
                        open_trade.direction,
                        open_trade.entry_price,
                        exit_price,
                    )
                    pnl_usd = pips * self.PIP_VALUE

                    open_trade.exit_time   = bar_time
                    open_trade.exit_price  = exit_price
                    open_trade.pnl_pips    = round(pips,    2)
                    open_trade.pnl_usd     = round(pnl_usd, 4)
                    open_trade.exit_reason = exit_reason

                    equity += pnl_usd
                    equity_curve.append(equity)

                    if equity > peak_equity:
                        peak_equity = equity
                    dd = peak_equity - equity
                    if dd > max_drawdown:
                        max_drawdown = dd

                    trades.append(open_trade)
                    open_trade = None
                    continue   # do not look for a new signal on the exit bar

            # ── Already in a trade — skip signal logic ────────────────────────
            if open_trade is not None:
                continue

            # ── Gate 2: Technical signal (logging suppressed) ─────────────────
            window_df = df_full.iloc[max(0, i - self.WINDOW_BARS): i + 1].copy()

            # Temporarily silence SignalEngine INFO spam during the loop
            _se_log   = logging.getLogger("SignalEngine")
            _se_prev  = _se_log.level
            _se_log.setLevel(logging.WARNING)

            signal = self.signal_eng.evaluate(window_df, symbol)

            _se_log.setLevel(_se_prev)   # restore level after call

            if signal is None:
                continue

            # ── Gate 3: ML prediction ─────────────────────────────────────────
            _ml_log  = logging.getLogger("MLModel")
            _ml_prev = _ml_log.level
            _ml_log.setLevel(logging.WARNING)

            ml = self.ml_model.predict(window_df)

            _ml_log.setLevel(_ml_prev)

            ml_direction = ML_LABEL_MAP.get(ml["label"], "HOLD")

            if ml_direction != signal.signal.value:
                continue
            if ml["confidence"] < self.ml_min_conf:
                continue

            # ── Open trade ────────────────────────────────────────────────────
            direction  = signal.signal.value
            open_trade = BacktestTrade(
                symbol      = symbol,
                direction   = direction,
                entry_time  = bar_time,
                exit_time   = None,
                entry_price = bar_close,
                exit_price  = 0.0,
                sl          = signal.sl,
                tp          = signal.tp,
                volume      = self.LOT_SIZE,
                pnl_pips    = 0.0,
                pnl_usd     = 0.0,
                exit_reason = "",
                gate2_score = int(signal.strength * 8),
                ml_conf     = ml["confidence"],
            )

        print()   # newline after progress bar

        # ── Force-close any trade still open at end of data ───────────────────
        if open_trade is not None:
            last_close = float(df_full.iloc[-1]["close"])
            pips       = self._calc_pips(
                open_trade.direction,
                open_trade.entry_price,
                last_close,
            )
            pnl_usd = pips * self.PIP_VALUE
            open_trade.exit_time   = df_full.index[-1]
            open_trade.exit_price  = last_close
            open_trade.pnl_pips    = round(pips,    2)
            open_trade.pnl_usd     = round(pnl_usd, 4)
            open_trade.exit_reason = "END"
            equity += pnl_usd
            trades.append(open_trade)

        # ── Build and print report ────────────────────────────────────────────
        report = self._compute_report(
            symbol          = symbol,
            timeframe       = timeframe,
            bars_tested     = n_bars - self.WARMUP_BARS,
            start_date      = start_date,
            end_date        = end_date,
            initial_balance = initial_balance,
            final_balance   = equity,
            equity_curve    = equity_curve,
            max_drawdown    = max_drawdown,
            peak_equity     = peak_equity,
            trades          = trades,
        )

        self._print_report(report)
        return report

    # ────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_pips(direction: str, entry: float, exit_price: float) -> float:
        diff = exit_price - entry
        if direction == "SELL":
            diff = -diff
        return diff / 0.0001

    def _compute_report(
        self,
        symbol:          str,
        timeframe:       int,
        bars_tested:     int,
        start_date:      str,
        end_date:        str,
        initial_balance: float,
        final_balance:   float,
        equity_curve:    list,
        max_drawdown:    float,
        peak_equity:     float,
        trades:          list,
    ) -> BacktestReport:

        total     = len(trades)
        winners   = [t for t in trades if t.pnl_usd > 0]
        losers    = [t for t in trades if t.pnl_usd <= 0]
        win_count = len(winners)
        los_count = len(losers)
        win_rate  = (win_count / total * 100) if total > 0 else 0.0

        total_pnl_usd  = sum(t.pnl_usd  for t in trades)
        total_pnl_pips = sum(t.pnl_pips for t in trades)

        avg_win_pips  = float(np.mean([t.pnl_pips for t in winners])) if winners else 0.0
        avg_loss_pips = float(np.mean([t.pnl_pips for t in losers]))  if losers  else 0.0

        rr_list = []
        for t in trades:
            planned_sl = abs(t.entry_price - t.sl) / 0.0001
            if planned_sl > 0 and t.pnl_pips > 0:
                rr_list.append(t.pnl_pips / planned_sl)
        avg_rr = float(np.mean(rr_list)) if rr_list else 0.0

        max_dd_pct = (max_drawdown / peak_equity * 100) if peak_equity > 0 else 0.0

        if len(equity_curve) > 1:
            returns     = np.diff(equity_curve) / np.array(equity_curve[:-1])
            bars_per_yr = 288 * 252
            sharpe      = (
                float(np.mean(returns) / np.std(returns) * np.sqrt(bars_per_yr))
                if np.std(returns) > 0 else 0.0
            )
        else:
            sharpe = 0.0

        gross_profit  = sum(t.pnl_usd for t in winners)
        gross_loss    = abs(sum(t.pnl_usd for t in losers))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

        expectancy = (
            (win_rate / 100 * avg_win_pips  * self.PIP_VALUE) +
            ((1 - win_rate / 100) * avg_loss_pips * self.PIP_VALUE)
        )

        return BacktestReport(
            symbol           = symbol,
            timeframe        = timeframe,
            bars_tested      = bars_tested,
            start_date       = start_date,
            end_date         = end_date,
            initial_balance  = initial_balance,
            final_balance    = round(final_balance, 2),
            total_trades     = total,
            winning_trades   = win_count,
            losing_trades    = los_count,
            win_rate         = round(win_rate,     2),
            total_pnl_usd    = round(total_pnl_usd,  4),
            total_pnl_pips   = round(total_pnl_pips, 2),
            avg_win_pips     = round(avg_win_pips,   2),
            avg_loss_pips    = round(avg_loss_pips,  2),
            avg_rr_achieved  = round(avg_rr,         2),
            max_drawdown_usd = round(max_drawdown,   4),
            max_drawdown_pct = round(max_dd_pct,     2),
            sharpe_ratio     = round(sharpe,         3),
            profit_factor    = round(profit_factor,  3),
            expectancy_usd   = round(expectancy,     4),
            trades           = trades,
        )

    def _print_report(self, r: BacktestReport) -> None:
        pnl_icon = "📈" if r.total_pnl_usd >= 0 else "📉"
        print("\n" + "=" * 60)
        print(f"  📊 BACKTEST REPORT — {r.symbol} M{r.timeframe}")
        print("=" * 60)
        print(f"  Period         : {r.start_date[:10]}  →  {r.end_date[:10]}")
        print(f"  Bars tested    : {r.bars_tested:,}")
        print(f"  Initial balance: ${r.initial_balance:.2f}")
        print(f"  Final balance  : ${r.final_balance:.2f}")
        print(f"  Net P&L        : {pnl_icon}  ${r.total_pnl_usd:+.4f}  ({r.total_pnl_pips:+.1f} pips)")
        print("─" * 60)
        print(f"  Total trades   : {r.total_trades}")
        print(f"  Winners        : {r.winning_trades}  ✅")
        print(f"  Losers         : {r.losing_trades}   ❌")
        print(f"  Win rate       : {r.win_rate:.1f}%")
        print(f"  Avg win        : +{r.avg_win_pips:.1f} pips")
        print(f"  Avg loss       : {r.avg_loss_pips:.1f} pips")
        print(f"  Avg R:R        : {r.avg_rr_achieved:.2f}")
        print("─" * 60)
        print(f"  Max drawdown   : ${r.max_drawdown_usd:.4f}  ({r.max_drawdown_pct:.1f}%)")
        print(f"  Sharpe ratio   : {r.sharpe_ratio:.3f}")
        print(f"  Profit factor  : {r.profit_factor:.3f}")
        print(f"  Expectancy     : ${r.expectancy_usd:+.4f} per trade")
        print("=" * 60)

        if r.total_trades > 0:
            print("\n  Exit breakdown:")
            from collections import Counter
            reasons = Counter(t.exit_reason for t in r.trades)
            for reason, count in sorted(reasons.items()):
                pct = count / r.total_trades * 100
                print(f"    {reason:<12} {count:>4}  ({pct:.1f}%)")
        print()

    def export_report(
        self,
        report:   BacktestReport,
        filepath: str = None,
    ) -> str:
        if not filepath:
            ts       = datetime.now().strftime("%Y%m%d_%H%M")
            filepath = f"reports/backtest_{report.symbol}_M{report.timeframe}_{ts}.csv"

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        rows = [{
            "symbol":       t.symbol,
            "direction":    t.direction,
            "entry_time":   t.entry_time,
            "exit_time":    t.exit_time,
            "entry_price":  t.entry_price,
            "exit_price":   t.exit_price,
            "sl":           t.sl,
            "tp":           t.tp,
            "pnl_pips":     t.pnl_pips,
            "pnl_usd":      t.pnl_usd,
            "exit_reason":  t.exit_reason,
            "gate2_score":  t.gate2_score,
            "ml_conf":      t.ml_conf,
        } for t in report.trades]

        pd.DataFrame(rows).to_csv(filepath, index=False)
        logger.info(f"📁 Backtest exported → {filepath}")
        return filepath
