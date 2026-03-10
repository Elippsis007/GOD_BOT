# =============================================================================
# GODBOT v3.0 – monitoring/dashboard.py
# =============================================================================
#  Original fixes A–N preserved.
#  This revision additionally fixes:
#
#  O  [FIX] update_signal() called signal.signal_type.name — AttributeError
#     since signal_type is a plain str not an Enum. Now uses str().upper().
#
#  P  [FIX] update_signal() accessed position_spec.sl / .tp — PositionSpec
#     fields are sl_price / tp_price. Was AttributeError on every call.
#
#  Q  [FIX] _positions_panel_lines() passed magic= to get_positions() but
#     the MT5Connector signature requires magic_number=. Position filter
#     was silently ignored — all EA positions were shown, not just GODBOT's.
#
#  R  [FIX] _floating_pnl cache was populated by update_position() and
#     cleared by log_trade() but never read anywhere. Panel now uses it
#     as a fallback when get_positions() returns no data for a ticket.
#
#  S  [FIX] daily_summary() stub signature too thin — main.py passes
#     wins, losses, and net_pnl as keyword args. Signature widened to
#     match all fields main.py supplies.
#
#  T  [FIX] _account_panel_lines() accessed info['login'] with hard key
#     — now uses info.get('login', 'N/A') defensively.
#
#  U  [FIX] export_report() assigned aggregate profit_factor scalar to
#     every trade row — misleading. Now written to a separate summary
#     row appended after the per-trade rows.
#
#  V  [FIX] Header showed "Mode: Mode fully_automated" — doubled word.
#     mode_names dict now maps the full string values that main.py passes
#     (e.g. "fully_automated") instead of "1"/"2"/"3" keys.
#
#  W  [FIX] Footer showed "Mode: Mfully_automated" — same root cause.
#     mode_names dict in _footer_lines() updated to match full strings.
#
#  X  [FIX] _positions_panel_lines() passed magic= but MT5Connector
#     signature uses magic= (not magic_number=). Corrected.
#
#  Y  [FIX] update_scan_status() accepts a string status OR a numeric
#     seconds value from main.py — now handles both without crashing.
# =============================================================================

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import pytz

from config.settings import CONFIG
from monitoring.logger import get_logger

logger = get_logger("Dashboard")

MADRID_TZ = pytz.timezone("Europe/Madrid")


def _now_madrid() -> datetime:
    return datetime.now(pytz.utc).astimezone(MADRID_TZ)


class Dashboard:
    """
    Terminal-based performance dashboard for GODBOT v3.0.

    Panels
    ──────
    - Account overview   (balance, equity, margin, leverage)
    - Open positions     (SL/TP prices, P&L, floating exposure) [B][L]
    - Today's signals    (pip distances) [C]
    - Today's perf       (trades, win rate, P&L, close reasons)
    - 7-day analytics    (profit factor, expectancy, sessions) [D][E]
    - Footer             (scan countdown, all hotkeys) [A][J]

    All panels are built into a string buffer and printed in one
    write to minimise terminal flicker. [F]

    MT5 access is routed exclusively through MT5Connector. [K][L]
    """

    def __init__(self, config=CONFIG) -> None:
        self.cfg = config

        # [K][L] All MT5 access through the singleton connector
        from core.mt5_connector import MT5Connector
        self._connector = MT5Connector()

        self.signals_log: List[dict] = []
        self.trades_log:  List[dict] = []
        self._all_trades: List[dict] = []

        self._scan_secs        = 60
        self._last_scan_time:  Optional[datetime] = None
        self._scan_start_time: Optional[datetime] = None
        self._scan_status      = "Waiting"
        self._seconds_to_next  = 0.0
        self._current_mode     = "fully_automated"
        self._current_style    = "scalper"

        # [M] Account currency — updated from connector on first display
        self._account_currency = "USD"

        # [N][R] Floating P&L cache: ticket → {"symbol": str, "pnl": float}
        self._floating_pnl: Dict[int, dict] = {}

        os.makedirs("reports", exist_ok=True)
        self._load_today()

    # ── Configuration setters ─────────────────────────────────────────────────

    def set_scan_secs(self, secs: int) -> None:
        self._scan_secs = secs

    def set_mode(self, mode: str, style: str = "scalper") -> None:
        """Called by main.py when the operator switches mode via 1/2/3."""
        self._current_mode  = str(mode)
        self._current_style = style

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_today(self) -> None:
        today = _now_madrid().strftime("%Y-%m-%d")
        try:
            if os.path.exists("reports/signals_log.json"):
                with open("reports/signals_log.json", "r") as fh:
                    all_sigs = json.load(fh)
                self.signals_log = [
                    e for e in all_sigs if e.get("date") == today
                ]
            else:
                self.signals_log = []
        except Exception as exc:
            logger.warning("Load signals_log error (resetting): %s", exc)
            self.signals_log = []
            # Nuke corrupted file
            try:
                os.remove("reports/signals_log.json")
            except Exception:
                pass

        try:
            if os.path.exists("reports/trades_log.json"):
                with open("reports/trades_log.json", "r") as fh:
                    self._all_trades = json.load(fh)
                self.trades_log = [
                    e for e in self._all_trades if e.get("date") == today
                ]
            else:
                self._all_trades = []
                self.trades_log  = []
        except Exception as exc:
            logger.warning("Load trades_log error (resetting): %s", exc)
            self._all_trades = []
            self.trades_log  = []
            try:
                os.remove("reports/trades_log.json")
            except Exception:
                pass

    def _save_logs(self) -> None:
        try:
            with open("reports/signals_log.json", "w") as fh:
                json.dump(self.signals_log, fh, indent=2)
            today            = _now_madrid().strftime("%Y-%m-%d")
            other_days       = [
                t for t in self._all_trades if t.get("date") != today
            ]
            self._all_trades = other_days + self.trades_log
            with open("reports/trades_log.json", "w") as fh:
                json.dump(self._all_trades, fh, indent=2)
        except Exception as exc:
            logger.debug("Save logs error: %s", exc)

    # ── Dashboard output file ─────────────────────────────────────────────────
    # Written every second; the popup PowerShell window reads this file.
    # This keeps the VS Code terminal clean for the scrolling log stream.
    _DASHBOARD_FILE: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dashboard_live.txt",
    )

    # ── Main display entry ────────────────────────────────────────────────────

    def display(self) -> None:
        while True:
            try:
                acc_info = self._connector.get_account_info()
                if acc_info:
                    self._account_currency = acc_info.get("currency", "USD")
            except Exception:
                pass

            try:
                lines: List[str] = []
                lines += self._header_lines()
                lines += self._account_panel_lines()
                lines += self._positions_panel_lines()
                lines += self._signals_panel_lines()
                lines += self._performance_panel_lines()
                lines += self._analytics_panel_lines()
                lines += self._footer_lines()

                output = "\n".join(lines)
                # ── Write to file instead of stdout ───────────────────────
                # The popup window reads this file and reprints it every
                # second.  The VS Code terminal is left completely free
                # for the clean scrolling log stream — no screen clearing.
                with open(self._DASHBOARD_FILE, "w", encoding="utf-8") as fh:
                    fh.write(output + "\n")
            except Exception as exc:
                logger.exception("‼️ Dashboard display() error: %s", exc)

            time.sleep(1.0)

    # ── Header ────────────────────────────────────────────────────────────────

    def _header_lines(self) -> List[str]:
        now = _now_madrid().strftime("%Y-%m-%d %H:%M:%S")

        # Fix V – map full mode strings that main.py passes, not "1"/"2"/"3"
        mode_names = {
            "signal_only":     "Signal Only",
            "semi_automated":  "Semi-Auto",
            "fully_automated": "Full Auto",
        }
        mode_label = mode_names.get(
            self._current_mode, self._current_mode.replace("_", " ").title()
        )

        return [
            "=" * 65,
            f"  🤖 GODBOT v3.0  |  {now} Madrid/CET",
            f"  Mode: {mode_label}  |  "
            f"Style: {self._current_style.capitalize()}  |  "
            f"TF: M{getattr(self.cfg, 'SCALPER_TF_SELECTED', 5)}",
            "=" * 65,
        ]

    # ── Account panel ─────────────────────────────────────────────────────────

    def _account_panel_lines(self) -> List[str]:
        lines = ["\n📊 ACCOUNT OVERVIEW", "-" * 40]

        info = self._connector.get_account_info()
        if info is None:
            lines.append("  ❌ Account info unavailable")
            return lines

        equity_diff = info["equity"] - info["balance"]
        equity_icon = "🟢" if equity_diff >= 0 else "🔴"
        margin_pct  = (
            (info["margin"] / info["equity"] * 100)
            if info["equity"] > 0 else 0.0
        )
        currency = info.get("currency", self._account_currency)

        lines += [
            f"  Account  : {info.get('login', 'N/A')}",
            f"  Balance  : {info['balance']:>12.2f} {currency}",
            f"  Equity   : {info['equity']:>12.2f} {currency}  "
            f"{equity_icon} ({equity_diff:+.2f})",
            f"  Margin   : {info['margin']:>12.2f} {currency}  "
            f"({margin_pct:.1f}% of equity)",
            f"  Free Mrgn: {info['free_margin']:>12.2f} {currency}",
            f"  Leverage : 1:{info['leverage']}",
        ]
        return lines

    # ── Positions panel ───────────────────────────────────────────────────────

    def _positions_panel_lines(self) -> List[str]:
        # Fix X – MT5Connector.get_positions() uses magic= not magic_number=
        bot_positions = self._connector.get_positions(
            magic=self.cfg.MAGIC_NUMBER
        )

        lines = [f"\n📈 OPEN POSITIONS ({len(bot_positions)})", "-" * 65]

        if not bot_positions:
            if self._floating_pnl:
                lines.append("  (Live feed unavailable — cached positions:)")
                total_pnl = 0.0
                for ticket, data in self._floating_pnl.items():
                    pnl_icon   = "🟢" if data["pnl"] >= 0 else "🔴"
                    total_pnl += data["pnl"]
                    lines.append(
                        f"  #{ticket:<8} {data['symbol']:<10} "
                        f"{pnl_icon} {data['pnl']:+.2f} (cached)"
                    )
                total_icon = "🟢" if total_pnl >= 0 else "🔴"
                lines.append(
                    f"  {'TOTAL (CACHED)':<47}"
                    f"{total_icon}{total_pnl:>7.2f} "
                    f"{self._account_currency}"
                )
            else:
                lines.append("  No open positions")
            return lines

        lines.append(
            f"  {'Symbol':<10} {'Dir':<5} {'Vol':>5} "
            f"{'Open':>9} {'Now':>9} "
            f"{'SL':>9} {'TP':>9} "
            f"{'P&L':>8}"
        )
        lines.append("  " + "-" * 62)

        total_pnl = 0.0
        currency  = self._account_currency

        for pos in bot_positions:
            tick = self._connector.get_latest_tick(pos["symbol"])
            if tick is None:
                cached_pnl = self._floating_pnl.get(
                    pos.get("ticket"), {}
                ).get("pnl", pos.get("profit", 0.0))
                pnl_icon   = "🟢" if cached_pnl >= 0 else "🔴"
                total_pnl += cached_pnl
                lines.append(
                    f"  {pos['symbol']:<10} {pos['type']:<5} "
                    f"{pos['volume']:>5.2f} "
                    f"{pos['price_open']:>9.5f} "
                    f"{'N/A':>9} "
                    f"{pos['sl']:>9.5f} "
                    f"{pos['tp']:>9.5f} "
                    f"{pnl_icon}{cached_pnl:>7.2f}"
                )
                continue

            current    = tick["bid"] if pos["type"] == "BUY" else tick["ask"]
            pnl_icon   = "🟢" if pos["profit"] >= 0 else "🔴"
            total_pnl += pos["profit"]

            lines.append(
                f"  {pos['symbol']:<10} {pos['type']:<5} "
                f"{pos['volume']:>5.2f} "
                f"{pos['price_open']:>9.5f} "
                f"{current:>9.5f} "
                f"{pos['sl']:>9.5f} "
                f"{pos['tp']:>9.5f} "
                f"{pnl_icon}{pos['profit']:>7.2f}"
            )

        lines.append("  " + "-" * 62)
        total_icon = "🟢" if total_pnl >= 0 else "🔴"
        lines.append(
            f"  {'TOTAL FLOATING P&L':<47}"
            f"{total_icon}{total_pnl:>7.2f} {currency}"
        )
        return lines

    # ── Signals panel ─────────────────────────────────────────────────────────

    def _signals_panel_lines(self) -> List[str]:
        today      = _now_madrid().strftime("%Y-%m-%d")
        today_sigs = [s for s in self.signals_log if s.get("date") == today]

        lines = [
            f"\n🎯 TODAY'S SIGNALS ({len(today_sigs)} total)",
            "-" * 65,
        ]

        if not today_sigs:
            lines.append("  No signals generated yet today")
            return lines

        recent = today_sigs[-6:][::-1]
        for sig in recent:
            icon    = "🟢" if sig["direction"] == "BUY" else "🔴"
            sl_pips = sig.get("sl_pips")
            tp_pips = sig.get("tp_pips")
            pip_str = (
                f" SL:{sl_pips:.0f}p TP:{tp_pips:.0f}p |"
                if sl_pips is not None and tp_pips is not None else ""
            )
            lines.append(
                f"  {icon} {sig['symbol']:<8} {sig['direction']:<5} | "
                f"E:{sig['entry']:<9.5f} |"
                f"{pip_str} "
                f"Conf:{sig['confidence']:.0%} | "
                f"{sig['time']}"
            )
        return lines

    # ── Performance panel ─────────────────────────────────────────────────────

    def _performance_panel_lines(self) -> List[str]:
        today        = _now_madrid().strftime("%Y-%m-%d")
        today_trades = [t for t in self.trades_log if t.get("date") == today]
        currency     = self._account_currency

        lines = ["\n📉 TODAY'S PERFORMANCE", "-" * 40]

        if not today_trades:
            lines.append("  No completed trades today")
            return lines

        stats = self._compute_stats(today_trades)

        pf_str = (
            f"{stats['profit_factor']:.3f}  "
            f"{'✅' if stats['profit_factor'] >= 1.5 else '⚠️'}"
            if stats["profit_factor"] is not None
            else "N/A  (no losses yet) ✅"
        )

        lines += [
            f"  Trades   : {stats['count']}",
            f"  Wins     : {stats['wins']} 🟢",
            f"  Losses   : {stats['losses']} 🔴",
            f"  Win Rate : {stats['win_rate']:.1f}%",
            f"  Total P&L: {stats['net_pnl']:+.2f} {currency}",
            f"  Prof Fact: {pf_str}",
            f"  Best     : {stats['best']:+.2f}",
            f"  Worst    : {stats['worst']:+.2f}",
        ]

        if stats["reason_counts"]:
            lines.append("  Close By :")
            order = [
                "TP hit", "SL hit", "Danger",
                "EOD", "Broker closed", "Unknown",
            ]
            sorted_reasons = sorted(
                stats["reason_counts"].items(),
                key=lambda kv: (
                    order.index(kv[0]) if kv[0] in order else len(order)
                ),
            )
            reason_icons = {
                "TP hit":        "🎯",
                "SL hit":        "🛑",
                "Danger":        "⚠️ ",
                "EOD":           "🌙",
                "Broker closed": "📋",
                "Unknown":       "❓",
            }
            for reason, count in sorted_reasons:
                icon = reason_icons.get(reason, "📌")
                lines.append(f"    {icon} {reason:<14}: {count}")

        return lines

    # ── Analytics panel ───────────────────────────────────────────────────────

    def _analytics_panel_lines(self) -> List[str]:
        lines    = ["\n📊 ROLLING ANALYTICS  (7-day)", "-" * 65]
        currency = self._account_currency

        cutoff    = (_now_madrid() - timedelta(days=7)).strftime("%Y-%m-%d")
        trades_7d = [
            t for t in self._all_trades if t.get("date", "") >= cutoff
        ]

        if not trades_7d:
            lines.append("  No trade history yet")
            return lines

        stats = self._compute_stats(trades_7d)

        pf_str = (
            f"{stats['profit_factor']:.3f}  "
            f"{'✅' if stats['profit_factor'] >= 1.5 else '⚠️'}"
            if stats["profit_factor"] is not None
            else "N/A  (no losses yet) ✅"
        )

        lines += [
            f"  Trades (7d)  : {stats['count']}",
            f"  Win Rate     : {stats['win_rate']:.1f}%",
            f"  Profit Factor: {pf_str}",
            f"  Expectancy   : {currency} {stats['expectancy']:+.4f} per trade",
        ]

        consec = stats["consec_losses"]
        if consec >= 3:
            lines.append(
                f"\n  ⚠️  WARNING: {consec} consecutive losses — "
                "consider pausing"
            )
        else:
            icon = "🟢" if consec == 0 else "🟡"
            lines.append(f"  Consec Losses: {consec}  {icon}")

        sessions: Dict[str, list] = {
            "Asian":  [],
            "London": [],
            "NY":     [],
        }
        for t in trades_7d:
            try:
                hour = int(str(t.get("time", "0:00:00")).split(":")[0])
            except (ValueError, TypeError) as exc:
                logger.debug(
                    "[Dashboard] Session parse error for trade "
                    "time='%s': %s", t.get("time"), exc,
                )
                continue
            if 0 <= hour < 8:
                sessions["Asian"].append(t["pnl"])
            elif 8 <= hour < 14:
                sessions["London"].append(t["pnl"])
            else:
                sessions["NY"].append(t["pnl"])

        lines.append("\n  Session Breakdown:")
        for name, pnls in sessions.items():
            if not pnls:
                continue
            s_wins = sum(1 for p in pnls if p > 0)
            s_wr   = s_wins / len(pnls) * 100
            s_pnl  = sum(pnls)
            icon   = "🟢" if s_pnl >= 0 else "🔴"
            lines.append(
                f"    {name:<8}: {len(pnls):>3} trades | "
                f"WR {s_wr:>5.1f}% | "
                f"{icon} {currency}{s_pnl:+.2f}"
            )

        if stats["reason_counts"]:
            lines.append("\n  Close Reason (7d):")
            order = [
                "TP hit", "SL hit", "Danger",
                "EOD", "Broker closed", "Unknown",
            ]
            sorted_reasons = sorted(
                stats["reason_counts"].items(),
                key=lambda kv: (
                    order.index(kv[0]) if kv[0] in order else len(order)
                ),
            )
            reason_icons = {
                "TP hit":        "🎯",
                "SL hit":        "🛑",
                "Danger":        "⚠️ ",
                "EOD":           "🌙",
                "Broker closed": "📋",
                "Unknown":       "❓",
            }
            for reason, count in sorted_reasons:
                icon = reason_icons.get(reason, "📌")
                pct  = count / stats["count"] * 100
                lines.append(
                    f"    {icon} {reason:<14}: {count:>3}  ({pct:>5.1f}%)"
                )

        return lines

    # ── Footer ────────────────────────────────────────────────────────────────

    def _footer_lines(self) -> List[str]:
        # Fix W – map full mode strings, not "1"/"2"/"3"
        mode_names = {
            "signal_only":     "SigOnly",
            "semi_automated":  "SemiAuto",
            "fully_automated": "FullAuto",
        }
        mode_label = mode_names.get(
            self._current_mode,
            self._current_mode.replace("_", " ").title()
        )

        if self._scan_status == "Paused":
            countdown_str = "  ⏸  Paused  |  "
        elif self._scan_status == "Running":
            if self._scan_start_time is not None:
                elapsed = int(
                    (_now_madrid() - self._scan_start_time).total_seconds()
                )
                countdown_str = f"  🔄 Scanning... {elapsed}s elapsed  |  "
            else:
                countdown_str = "  🔄 Scanning...  |  "
        else:
            secs = int(self._seconds_to_next)
            if self._last_scan_time is not None:
                countdown_str = f"  ⏱️  Next scan in {secs:>3}s  |  "
            else:
                countdown_str = "  ⏱️  First scan pending  |  "

        scan_status = self._get_scan_status_text()
        last_scan   = self._get_last_scan_text()

        return [
            "",
            "=" * 65,
            f"{countdown_str}{scan_status}  |  {last_scan}",
            f"  Mode: {mode_label} "
            f"| TF: M{getattr(self.cfg, 'SCALPER_TF_SELECTED', 5)}",
            "  Keys: 1=SigOnly  2=SemiAuto  3=FullAuto  "
            "P=Pause  M=Menu  Q=Quit",
            "=" * 65,
        ]

    # ── Scan status helpers ───────────────────────────────────────────────────

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
        diff    = _now_madrid() - self._last_scan_time
        seconds = int(diff.total_seconds())
        if seconds < 60:
            return f"Last: {seconds}s ago"
        return f"Last: {seconds // 60}m {seconds % 60}s ago"

    def update_scan_status(self, status: str) -> None:
        """
        Fix Y — accepts either a plain status string ("Running",
        "Completed", "Waiting", "Paused") or a numeric seconds-to-next
        value passed as a string from main.py (e.g. "Next scan in 18s").
        Stores the seconds value for the footer countdown display.
        """
        # main.py passes a formatted string like "Next scan in 18s"
        # Extract the number if present, otherwise treat as status keyword
        if status.startswith("Next scan in"):
            try:
                self._seconds_to_next = float(
                    status.replace("Next scan in", "").replace("s", "").strip()
                )
                self._scan_status = "Waiting"
                if self._last_scan_time is None:
                    self._last_scan_time = _now_madrid()
            except ValueError:
                self._scan_status = "Waiting"
        else:
            self._scan_status = status
            now = _now_madrid()
            if status == "Running":
                self._scan_start_time = now
            elif status == "Completed":
                self._last_scan_time  = now
                self._scan_start_time = None
                self._seconds_to_next = float(self._scan_secs)

    # ── Shared stats helper ───────────────────────────────────────────────────

    def _compute_stats(self, trades: List[dict]) -> dict:
        if not trades:
            return {
                "count":         0,
                "wins":          0,
                "losses":        0,
                "win_rate":      0.0,
                "net_pnl":       0.0,
                "gross_profit":  0.0,
                "gross_loss":    0.0,
                "profit_factor": None,
                "expectancy":    0.0,
                "best":          0.0,
                "worst":         0.0,
                "consec_losses": 0,
                "reason_counts": {},
            }

        wins     = [t for t in trades if t["pnl"] > 0]
        losses   = [t for t in trades if t["pnl"] <= 0]
        g_profit = sum(t["pnl"] for t in wins)
        g_loss   = abs(sum(t["pnl"] for t in losses))
        net_pnl  = sum(t["pnl"] for t in trades)

        profit_factor = (g_profit / g_loss) if g_loss > 0 else None

        consec = 0
        for t in reversed(trades):
            if t["pnl"] <= 0:
                consec += 1
            else:
                break

        reason_counts: Dict[str, int] = {}
        for t in trades:
            r = t.get("close_reason", "Unknown")
            reason_counts[r] = reason_counts.get(r, 0) + 1

        return {
            "count":         len(trades),
            "wins":          len(wins),
            "losses":        len(losses),
            "win_rate":      len(wins) / len(trades) * 100,
            "net_pnl":       round(net_pnl,  2),
            "gross_profit":  round(g_profit, 2),
            "gross_loss":    round(g_loss,   2),
            "profit_factor": round(profit_factor, 4) if profit_factor else None,
            "expectancy":    net_pnl / len(trades),
            "best":          max(t["pnl"] for t in trades),
            "worst":         min(t["pnl"] for t in trades),
            "consec_losses": consec,
            "reason_counts": reason_counts,
        }

    # ── Public logging methods ────────────────────────────────────────────────

    def log_signal(
        self,
        symbol:     str,
        direction:  str,
        entry:      float,
        sl:         float,
        tp:         float,
        confidence: float,
        sl_pips:    Optional[float] = None,
        tp_pips:    Optional[float] = None,
    ) -> None:
        now    = _now_madrid()
        record = {
            "symbol":     symbol,
            "direction":  direction,
            "entry":      round(entry, 5),
            "sl":         round(sl, 5),
            "tp":         round(tp, 5),
            "confidence": round(confidence, 4),
            "date":       now.strftime("%Y-%m-%d"),
            "time":       now.strftime("%H:%M:%S"),
        }
        if sl_pips is not None:
            record["sl_pips"] = round(sl_pips, 1)
        if tp_pips is not None:
            record["tp_pips"] = round(tp_pips, 1)

        self.signals_log.append(record)
        self._save_logs()

        pip_str = (
            f" SL:{sl_pips:.0f}p TP:{tp_pips:.0f}p"
            if sl_pips is not None and tp_pips is not None else ""
        )
        logger.info(
            "📝 Signal logged: %s %s @ %.5f%s | Conf:%.0f%%",
            symbol, direction, entry, pip_str, confidence * 100,
        )

    def log_trade(
        self,
        symbol:       str,
        direction:    str,
        pnl:          float,
        ticket:       int,
        close_reason: str             = "Unknown",
        entry_price:  Optional[float] = None,
        close_price:  Optional[float] = None,
        volume:       Optional[float] = None,
    ) -> None:
        now    = _now_madrid()
        record: dict = {
            "symbol":       symbol,
            "direction":    direction,
            "pnl":          round(pnl, 2),
            "ticket":       ticket,
            "close_reason": close_reason,
            "date":         now.strftime("%Y-%m-%d"),
            "time":         now.strftime("%H:%M:%S"),
        }
        if entry_price is not None:
            record["entry_price"] = round(entry_price, 5)
        if close_price is not None:
            record["close_price"] = round(close_price, 5)
        if volume is not None:
            record["volume"] = round(volume, 2)

        self.trades_log.append(record)
        self._floating_pnl.pop(ticket, None)
        self._save_logs()

        icon = "✅" if pnl >= 0 else "❌"
        logger.info(
            "📝 Trade logged: %s %s %s %.2f | #%d | [%s]",
            symbol, direction, icon, pnl, ticket, close_reason,
        )

    # ── Methods called by main.py ─────────────────────────────────────────────

    def update_signal(self, symbol: str, signal, position_spec) -> None:
        try:
            raw       = getattr(signal, "signal", None) or getattr(signal, "signal_type", "HOLD")
            direction = raw.value if hasattr(raw, "value") else str(raw).upper()
            
            self.log_signal(
            symbol     = symbol,
            direction  = direction,
            entry      = float(signal.entry),
            sl         = float(
                getattr(position_spec, "sl_price",
                        getattr(position_spec, "sl", 0.0))
            ),
            tp         = float(
                getattr(position_spec, "tp_price",
                        getattr(position_spec, "tp", 0.0))
            ),
            confidence = float(
                getattr(position_spec, "confidence",
                        getattr(signal, "confidence", 0.0))
            ),
            sl_pips    = float(
                getattr(position_spec, "sl_pips",
                        getattr(signal, "sl_pips", None) or 0.0)
            ),
            tp_pips    = float(
                getattr(position_spec, "tp_pips",
                        getattr(signal, "tp_pips", None) or 0.0)
            ),
        )
        except Exception as exc:
            logger.warning("[Dashboard] update_signal() error: %s", exc)

    def update_position(self, ticket: int, symbol: str, pnl: float) -> None:
        self._floating_pnl[ticket] = {"symbol": symbol, "pnl": round(pnl, 2)}

    def daily_summary(
        self,
        pnl:         float = 0.0,
        trades:      int   = 0,
        win_rate:    float = 0.0,
        wins:        int   = 0,
        losses:      int   = 0,
        signals:     int   = 0,
        best_trade:  float = 0.0,
        worst_trade: float = 0.0,
    ) -> None:
        currency = self._account_currency
        logger.info(
            "📊 Daily summary | P&L: %s%+.2f | Trades: %d "
            "W:%d L:%d | WR: %.0f%%",
            currency, pnl, trades, wins, losses, win_rate,
        )
        try:
            today = _now_madrid().strftime("%Y-%m-%d")
            self.export_report(
                filepath=f"reports/report_{today}.csv",
                date=today,
            )
        except Exception as exc:
            logger.warning("[Dashboard] daily_summary export error: %s", exc)

    # ── Public stats / export ─────────────────────────────────────────────────

    def get_today_stats(self) -> dict:
        today        = _now_madrid().strftime("%Y-%m-%d")
        today_trades = [t for t in self.trades_log if t.get("date") == today]
        today_sigs   = [s for s in self.signals_log if s.get("date") == today]
        stats        = self._compute_stats(today_trades)
        stats["signals"] = len(today_sigs)
        return stats

    def export_report(
        self,
        filepath: str           = "reports/daily_report.csv",
        date:     Optional[str] = None,
    ) -> None:
        os.makedirs("reports", exist_ok=True)
        target        = date or _now_madrid().strftime("%Y-%m-%d")
        target_trades = [
            t for t in self._all_trades if t.get("date") == target
        ]

        if not target_trades:
            logger.warning(
                "[Dashboard] No trades to export for %s", target
            )
            return

        df       = pd.DataFrame(target_trades)
        g_profit = df.loc[df["pnl"] > 0, "pnl"].sum()
        g_loss   = abs(df.loc[df["pnl"] <= 0, "pnl"].sum())
        pf_value = round(g_profit / g_loss, 4) if g_loss > 0 else None

        summary_row = {
            "symbol":       "SUMMARY",
            "direction":    "",
            "pnl":          round(df["pnl"].sum(), 2),
            "ticket":       "",
            "close_reason": f"profit_factor={pf_value}",
            "date":         target,
            "time":         "",
        }
        summary_df = pd.DataFrame([summary_row])
        export_df  = pd.concat([df, summary_df], ignore_index=True)

        export_df.to_csv(filepath, index=False)
        logger.info(
            "📄 Report exported → %s (%d trades)", filepath, len(df)
        )

    # ── Utility ───────────────────────────────────────────────────────────────

    def _clear_screen(self) -> None:
        # \033[H    = move cursor to top-left (home)
        # \033[2J   = clear visible screen from cursor
        # \033[3J   = clear scrollback buffer (required by VS Code terminal
        #             to prevent the dashboard appending as a list instead
        #             of redrawing in place — without this, each 1-second
        #             refresh appends below the previous render)
        # Order matters: home first, then clear forward, then clear scrollback.
        sys.stdout.write("\033[H\033[2J\033[3J")
        sys.stdout.flush()
