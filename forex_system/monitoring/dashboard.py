# monitoring/dashboard.py
import MetaTrader5 as mt5
import pandas as pd
import json
import os
import time
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
        self.cfg              = config
        self.signals_log: List[dict] = []
        self.trades_log:  List[dict] = []
        self._scan_secs           = 60
        self._last_scan_time      = None
        self._scan_status         = "Waiting"
        self._display_initialized = False
        os.makedirs("reports", exist_ok=True)
        self._load_today()

    # ── Scan interval ─────────────────────────────────────────────────────────
    def set_scan_secs(self, secs: int) -> None:
        self._scan_secs = secs

    # ── Persistence ───────────────────────────────────────────────────────────
    def _load_today(self) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        for attr, path in [
            ("signals_log", "reports/signals_log.json"),
            ("trades_log",  "reports/trades_log.json"),
        ]:
            try:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        all_entries = json.load(f)
                    setattr(
                        self, attr,
                        [e for e in all_entries if e.get("date") == today],
                    )
            except Exception as e:
                logger.debug(f"Load {attr} error: {e}")
                setattr(self, attr, [])

    def _save_logs(self) -> None:
        try:
            with open("reports/signals_log.json", "w") as f:
                json.dump(self.signals_log, f, indent=2)
            with open("reports/trades_log.json", "w") as f:
                json.dump(self.trades_log, f, indent=2)
        except Exception as e:
            logger.debug(f"Save logs error: {e}")

    # ── Master Display ────────────────────────────────────────────────────────
    def display(self) -> None:
        self._clear_screen()
        self._header()
        self._account_panel()
        self._positions_panel()
        self._signals_panel()
        self._performance_panel()
        self._footer()

    # ── Header ────────────────────────────────────────────────────────────────
    def _header(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("=" * 65)
        print(f"  🤖 GODBOT v3.0  |  {now}")
        print("=" * 65)

    # ── Account Panel ─────────────────────────────────────────────────────────
    def _account_panel(self) -> None:
        """
        FIX: added single retry with 0.5 s sleep before giving up.
        The very first dashboard render fires immediately after MT5
        connect() returns — occasionally the IPC session needs a
        fraction of a second more to settle, causing the first
        account_info() call to return None even after a successful
        connect().  One retry is enough to bridge that gap without
        blocking the UI for a noticeable amount of time.
        """
        info = mt5.account_info()
        if info is None:
            time.sleep(0.5)
            info = mt5.account_info()

        if info is None:
            print("❌ Account info unavailable\n")
            return

        equity_diff = info.equity - info.balance
        equity_icon = "🟢" if equity_diff >= 0 else "🔴"

        print("\n📊 ACCOUNT OVERVIEW")
        print("-" * 40)
        print(f"  Account  : {info.login}")
        print(f"  Balance  : {info.balance:>12.2f} {info.currency}")
        print(
            f"  Equity   : {info.equity:>12.2f} {info.currency}  "
            f"{equity_icon} ({equity_diff:+.2f})"
        )
        print(f"  Margin   : {info.margin:>12.2f} {info.currency}")
        print(f"  Free Mrgn: {info.margin_free:>12.2f} {info.currency}")
        print(f"  Leverage : 1:{info.leverage}")

    # ── Positions Panel ───────────────────────────────────────────────────────
    def _positions_panel(self) -> None:
        all_positions = mt5.positions_get() or []
        bot_positions = [
            p for p in all_positions
            if p.magic == self.cfg.MAGIC_NUMBER
        ]

        print(f"\n📈 OPEN POSITIONS ({len(bot_positions)})")
        print("-" * 65)

        if not bot_positions:
            print("  No open positions")
            return

        print(
            f"  {'Symbol':<10} {'Type':<6} {'Vol':>6} "
            f"{'Open':>10} {'Current':>10} {'P&L':>10}"
        )
        print("  " + "-" * 58)

        total_pnl = 0.0
        for pos in bot_positions:
            tick = mt5.symbol_info_tick(pos.symbol)
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
        print(f"  {'TOTAL P&L':<34}{total_icon}{total_pnl:>9.2f}")

    # ── Signals Panel ─────────────────────────────────────────────────────────
    def _signals_panel(self) -> None:
        today      = datetime.now().strftime("%Y-%m-%d")
        today_sigs = [s for s in self.signals_log if s.get("date") == today]

        print(f"\n🎯 TODAY'S SIGNALS ({len(today_sigs)} total)")
        print("-" * 65)

        if not today_sigs:
            print("  No signals generated yet today")
            return

        recent = today_sigs[-5:][::-1]
        for sig in recent:
            icon = "🟢" if sig["direction"] == "BUY" else "🔴"
            print(
                f"  {icon} {sig['symbol']:<8} {sig['direction']:<5} | "
                f"Entry:{sig['entry']:<10} "
                f"SL:{sig['sl']:<10} "
                f"TP:{sig['tp']:<10} | "
                f"Conf:{sig['confidence']:.0%} | "
                f"{sig['time']}"
            )

    # ── Performance Panel ─────────────────────────────────────────────────────
    def _performance_panel(self) -> None:
        print("\n📉 TODAY'S PERFORMANCE")
        print("-" * 40)

        today        = datetime.now().strftime("%Y-%m-%d")
        today_trades = [t for t in self.trades_log if t.get("date") == today]

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

    # ── Footer ────────────────────────────────────────────────────────────────
    def _footer(self) -> None:
        scan_status = self._get_scan_status_text()
        last_scan   = self._get_last_scan_text()
        print("\n" + "=" * 65)
        print(
            f"  ⏱️  Next scan in {self._scan_secs}s  |  "
            f"{scan_status}  |  {last_scan}"
        )
        print("  M=Menu  P=Pause  Q=Quit")
        print("=" * 65)

    def _get_scan_status_text(self) -> str:
        status_map = {
            "Running":   "🔄 Scan Active",
            "Completed": "✅ Scan Complete",
            "Waiting":   "⏳ Waiting",
            "Paused":    "⏸  Paused",
            "Error":     "❌ Scan Error",
        }
        return status_map.get(self._scan_status, f"📡 {self._scan_status}")

    def _get_last_scan_text(self) -> str:
        if not self._last_scan_time:
            return "Last: Never"
        diff    = datetime.now() - self._last_scan_time
        seconds = int(diff.total_seconds())
        if seconds < 60:
            return f"Last: {seconds}s ago"
        return f"Last: {seconds // 60}m ago"

    def update_scan_status(self, status: str) -> None:
        self._scan_status = status
        if status == "Completed":
            self._last_scan_time = datetime.now()

    def force_refresh(self) -> None:
        self._clear_screen()
        self._display_initialized = False
        self.display()

    # ── Logging ───────────────────────────────────────────────────────────────
    def log_signal(
        self,
        symbol:     str,
        direction:  str,
        entry:      float,
        sl:         float,
        tp:         float,
        confidence: float,
    ) -> None:
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
        ticket:    int,
    ) -> None:
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

    # ── Stats ─────────────────────────────────────────────────────────────────
    def get_today_stats(self) -> dict:
        today        = datetime.now().strftime("%Y-%m-%d")
        today_trades = [t for t in self.trades_log if t.get("date") == today]
        today_sigs   = [s for s in self.signals_log if s.get("date") == today]

        wins     = [t for t in today_trades if t["pnl"] > 0]
        losses   = [t for t in today_trades if t["pnl"] <= 0]
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
            "best_trade":  max((t["pnl"] for t in today_trades), default=0.0),
            "worst_trade": min((t["pnl"] for t in today_trades), default=0.0),
        }

    # ── Export ────────────────────────────────────────────────────────────────
    def export_report(self, filepath: str = "reports/daily_report.csv") -> None:
        os.makedirs("reports", exist_ok=True)
        today        = datetime.now().strftime("%Y-%m-%d")
        today_trades = [t for t in self.trades_log if t.get("date") == today]

        if not today_trades:
            logger.warning("No trades to export today")
            return

        df = pd.DataFrame(today_trades)
        df.to_csv(filepath, index=False)
        logger.info(f"📄 Report exported → {filepath}")

    # ── Utilities ─────────────────────────────────────────────────────────────
    def _clear_screen(self) -> None:
        os.system("cls" if os.name == "nt" else "clear")
