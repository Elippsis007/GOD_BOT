# monitoring/dashboard.py
import MetaTrader5 as mt5
import pandas as pd
import json
import os
from datetime import datetime
from typing import List
from config.settings import CONFIG
from monitoring.logger import get_logger

logger = get_logger("Dashboard")


class Dashboard:
    """
    Terminal-based performance dashboard.
    Displays account stats, open positions,
    recent signals and daily performance.
    Persists logs to disk — survives restarts.
    """

    def __init__(self, config=CONFIG):
        self.cfg          = config
        self.signals_log: List[dict] = []
        self.trades_log:  List[dict] = []
        self._scan_secs   = 60          # updated by main.py
        os.makedirs("reports", exist_ok=True)
        self._load_today()              # restore today's data

    # ── Scan interval (set by main.py after style known) ──
    def set_scan_secs(self, secs: int):
        self._scan_secs = secs

    # ── Persistence ───────────────────────────────────────
    def _load_today(self):
        """Loads today's signals and trades from disk."""
        today = datetime.now().strftime("%Y-%m-%d")

        for attr, path in [
            ("signals_log", "reports/signals_log.json"),
            ("trades_log",  "reports/trades_log.json"),
        ]:
            try:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        all_entries = json.load(f)
                    # Keep only today's entries
                    setattr(
                        self, attr,
                        [e for e in all_entries
                         if e.get("date") == today]
                    )
            except Exception as e:
                logger.debug(f"Load {attr} error: {e}")
                setattr(self, attr, [])

    def _save_logs(self):
        """Persists signals and trades to disk."""
        try:
            with open("reports/signals_log.json", "w") as f:
                json.dump(self.signals_log, f, indent=2)
            with open("reports/trades_log.json", "w") as f:
                json.dump(self.trades_log, f, indent=2)
        except Exception as e:
            logger.debug(f"Save logs error: {e}")

    # ── Master Display ────────────────────────────────────
    def display(self):
        """Renders the full dashboard in terminal."""
        self._clear_screen()
        self._header()
        self._account_panel()
        self._positions_panel()
        self._signals_panel()
        self._performance_panel()
        self._footer()

    # ── Header ────────────────────────────────────────────
    def _header(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("=" * 65)
        print(f"  🤖 GODBOT v3.0  |  {now}")
        print("=" * 65)

    # ── Account Panel ─────────────────────────────────────
    def _account_panel(self):
        info = mt5.account_info()
        if info is None:
            print("❌ Account info unavailable\n")
            return

        equity_diff = info.equity - info.balance
        equity_icon = "🟢" if equity_diff >= 0 else "🔴"

        print("\n📊 ACCOUNT OVERVIEW")
        print("-" * 40)
        print(f"  Account  : {info.login}")
        print(
            f"  Balance  : {info.balance:>12.2f} "
            f"{info.currency}"
        )
        print(
            f"  Equity   : {info.equity:>12.2f} "
            f"{info.currency}  "
            f"{equity_icon} ({equity_diff:+.2f})"
        )
        print(
            f"  Margin   : {info.margin:>12.2f} "
            f"{info.currency}"
        )
        print(
            f"  Free Mrgn: {info.margin_free:>12.2f} "
            f"{info.currency}"
        )
        print(f"  Leverage : 1:{info.leverage}")

    # ── Positions Panel ───────────────────────────────────
    def _positions_panel(self):
        positions = mt5.positions_get()
        count     = len(positions) if positions else 0
        print(f"\n📈 OPEN POSITIONS ({count})")
        print("-" * 65)

        if not positions:
            print("  No open positions")
            return

        print(
            f"  {'Symbol':<10} {'Type':<6} {'Vol':>6} "
            f"{'Open':>10} {'Current':>10} {'P&L':>10}"
        )
        print("  " + "-" * 58)

        total_pnl = 0
        for pos in positions:
            if pos.magic != self.cfg.MAGIC_NUMBER:
                continue

            tick      = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                continue
            current   = tick.bid if pos.type == 0 else tick.ask
            direction = "BUY" if pos.type == 0 else "SELL"
            pnl_icon  = "🟢" if pos.profit >= 0 else "🔴"
            total_pnl += pos.profit

            print(
                f"  {pos.symbol:<10} {direction:<6} "
                f"{pos.volume:>6.2f} "
                f"{pos.price_open:>10.5f} "
                f"{current:>10.5f} "
                f"{pnl_icon}{pos.profit:>9.2f}"
            )

        print("  " + "-" * 58)
        total_icon = "🟢" if total_pnl >= 0 else "🔴"
        print(
            f"  {'TOTAL P&L':<34}"
            f"{total_icon}{total_pnl:>9.2f}"
        )

    # ── Signals Panel ─────────────────────────────────────
    def _signals_panel(self):
        today = datetime.now().strftime("%Y-%m-%d")
        today_sigs = [
            s for s in self.signals_log
            if s.get("date") == today
        ]
        print(
            f"\n🎯 TODAY'S SIGNALS ({len(today_sigs)} total)"
        )
        print("-" * 65)

        if not today_sigs:
            print("  No signals generated yet today")
            return

        recent = today_sigs[-5:][::-1]
        for sig in recent:
            icon = "🟢" if sig["direction"] == "BUY" else "🔴"
            print(
                f"  {icon} {sig['symbol']:<8} "
                f"{sig['direction']:<5} | "
                f"Entry:{sig['entry']:<10} "
                f"SL:{sig['sl']:<10} "
                f"TP:{sig['tp']:<10} | "
                f"Conf:{sig['confidence']:.0%} | "
                f"{sig['time']}"
            )

    # ── Performance Panel ─────────────────────────────────
    def _performance_panel(self):
        print("\n📉 TODAY'S PERFORMANCE")
        print("-" * 40)

        today        = datetime.now().strftime("%Y-%m-%d")
        today_trades = [
            t for t in self.trades_log
            if t.get("date") == today
        ]

        if not today_trades:
            print("  No completed trades today")
            return

        wins     = [t for t in today_trades if t["pnl"] > 0]
        losses   = [t for t in today_trades if t["pnl"] <= 0]
        total    = sum(t["pnl"] for t in today_trades)
        win_rate = len(wins) / len(today_trades) * 100
        best     = max(t["pnl"] for t in today_trades)
        worst    = min(t["pnl"] for t in today_trades)

        print(f"  Trades   : {len(today_trades)}")
        print(f"  Wins     : {len(wins)} 🟢")
        print(f"  Losses   : {len(losses)} 🔴")
        print(f"  Win Rate : {win_rate:.1f}%")
        print(f"  Total P&L: {total:+.2f}")
        print(f"  Best     : {best:+.2f}")
        print(f"  Worst    : {worst:+.2f}")

    # ── Footer ────────────────────────────────────────────
    def _footer(self):
        print("\n" + "=" * 65)
        print(
            f"  ⏱️  Next scan in {self._scan_secs}s  |  "
            f"M=Menu  P=Pause  Q=Quit"
        )
        print("=" * 65)

    # ── Logging Methods ───────────────────────────────────
    def log_signal(
        self,
        symbol:     str,
        direction:  str,
        entry:      float,
        sl:         float,
        tp:         float,
        confidence: float
    ):
        self.signals_log.append({
            "symbol":     symbol,
            "direction":  direction,
            "entry":      round(entry, 5),
            "sl":         round(sl, 5),
            "tp":         round(tp, 5),
            "confidence": round(confidence, 4),
            "date":       datetime.now().strftime("%Y-%m-%d"),
            "time":       datetime.now().strftime("%H:%M:%S"),
        })
        self._save_logs()
        logger.info(
            f"📝 Signal logged: {symbol} {direction} "
            f"@ {entry} | Conf:{confidence:.0%}"
        )

    def log_trade(
        self,
        symbol:    str,
        direction: str,
        pnl:       float,
        ticket:    int
    ):
        self.trades_log.append({
            "symbol":    symbol,
            "direction": direction,
            "pnl":       round(pnl, 2),
            "ticket":    ticket,
            "date":      datetime.now().strftime("%Y-%m-%d"),
            "time":      datetime.now().strftime("%H:%M:%S"),
        })
        self._save_logs()
        icon = "✅" if pnl >= 0 else "❌"
        logger.info(
            f"📝 Trade logged: {symbol} {direction} "
            f"{icon} €{pnl:+.2f} | #{ticket}"
        )

    # ── Stats Helper (used by main._daily_summary) ────────
    def get_today_stats(self) -> dict:
        """
        Returns today's performance as a dict.
        Simplifies main.py _daily_summary.
        """
        today        = datetime.now().strftime("%Y-%m-%d")
        today_trades = [
            t for t in self.trades_log
            if t.get("date") == today
        ]
        today_sigs = [
            s for s in self.signals_log
            if s.get("date") == today
        ]

        wins    = [t for t in today_trades if t["pnl"] > 0]
        losses  = [t for t in today_trades if t["pnl"] <= 0]
        g_profit = sum(t["pnl"] for t in wins)   if wins   else 0.0
        g_loss   = sum(t["pnl"] for t in losses) if losses else 0.0

        return {
            "signals":      len(today_sigs),
            "trades":       len(today_trades),
            "winners":      len(wins),
            "losers":       len(losses),
            "gross_profit": round(g_profit, 2),
            "gross_loss":   round(g_loss,   2),
            "net_pnl":      round(g_profit + g_loss, 2),
            "win_rate":     (
                len(wins) / len(today_trades) * 100
                if today_trades else 0.0
            ),
            "best_trade":  max(
                (t["pnl"] for t in today_trades), default=0.0
            ),
            "worst_trade": min(
                (t["pnl"] for t in today_trades), default=0.0
            ),
        }

    # ── Export ────────────────────────────────────────────
    def export_report(
        self,
        filepath: str = "reports/daily_report.csv"
    ):
        """Exports today's trade log to CSV."""
        os.makedirs("reports", exist_ok=True)
        today        = datetime.now().strftime("%Y-%m-%d")
        today_trades = [
            t for t in self.trades_log
            if t.get("date") == today
        ]

        if not today_trades:
            logger.warning("No trades to export today")
            return

        df = pd.DataFrame(today_trades)
        df.to_csv(filepath, index=False)
        logger.info(f"📄 Report exported → {filepath}")

    def _clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")