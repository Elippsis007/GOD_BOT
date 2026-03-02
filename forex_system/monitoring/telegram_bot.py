# monitoring/telegram_bot.py
import requests
import time
from datetime import datetime
from monitoring.logger import get_logger
# NEW — import token/chat_id directly (they're module-level, not in CONFIG)
from config.settings import CONFIG, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

logger = get_logger("TelegramBot")


class TelegramBot:
    """
    Sends formatted trade alerts to Telegram.
    Uses the Bot API directly via requests — no extra libraries needed.
    """

    BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self):
        self.token   = TELEGRAM_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            logger.warning("⚠️ Telegram not configured — alerts disabled")

    # ── Core Send ─────────────────────────────────────────
    def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a plain or Markdown-formatted message."""
        if not self.enabled:
            logger.warning("Telegram disabled — message not sent")
            return False
        try:
            url  = self.BASE_URL.format(token=self.token)
            data = {
                "chat_id":    self.chat_id,
                "text":       text,
                "parse_mode": parse_mode,
            }
            resp = requests.post(url, data=data, timeout=10)
            if resp.status_code == 200:
                logger.info("✅ Telegram message sent")
                return True
            else:
                logger.error(f"❌ Telegram error {resp.status_code}: {resp.text}")
                return False
        except requests.exceptions.Timeout:
            logger.error("❌ Telegram request timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Telegram send failed: {e}")
            return False

    # ── Formatted Signal Alert ────────────────────────────
    def send_signal_alert(self, signal, position=None) -> bool:
        """Send a formatted BUY/SELL signal alert."""
        if not CONFIG.SEND_SIGNALS:
            return False

        icon  = "🟢" if signal.signal.value == "BUY" else "🔴"
        arrow = "📈" if signal.signal.value == "BUY" else "📉"

        lines = [
            f"{icon} *GODBOT SIGNAL — {signal.signal.value}*",
            f"{'─' * 30}",
            f"💱 *Symbol:*     {signal.symbol}",
            f"{arrow} *Direction:*  {signal.signal.value}",
            f"🎯 *Confidence:* {signal.confidence:.0%}",
            f"",
            f"📍 *Entry:*      `{signal.entry:.5f}`",
            f"🛑 *Stop Loss:*  `{signal.sl:.5f}`",
            f"🎯 *Take Profit:*`{signal.tp:.5f}`",
        ]

        if position:
            lines += [
                f"",
                f"💼 *Size:*  {position.volume} lots",
                f"💸 *Risk:*  €{position.risk_usd:.2f}",
                f"📏 *R:R:*   1:{position.rr_ratio}",
            ]

        lines += [
            f"",
            f"📝 *Reasons:*",
        ]
        for r in signal.reasons[:5]:
            lines.append(f"  {r}")

        lines += [
            f"",
            f"🕐 `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
            f"{'─' * 30}",
            f"_GODBOT v1 — demo account_",
        ]

        return self.send_message("\n".join(lines))

    # ── Hold Alert ────────────────────────────────────────
    def send_hold_alert(self, symbol: str, direction: str,
                        entry: float, current: float, pnl_pips: float) -> bool:
        """Remind that a position is still valid."""
        if not CONFIG.SEND_HOLD:
            return False
        icon = "🟢" if direction == "BUY" else "🔴"
        pnl_icon = "📈" if pnl_pips >= 0 else "📉"
        msg = (
            f"{icon} *HOLD — {symbol} {direction}*\n"
            f"{'─' * 28}\n"
            f"📍 Entry:   `{entry:.5f}`\n"
            f"💱 Current: `{current:.5f}`\n"
            f"{pnl_icon} P&L:    `{pnl_pips:+.1f} pips`\n"
            f"🕐 `{datetime.now().strftime('%H:%M:%S')}`"
        )
        return self.send_message(msg)

    # ── News Warning ──────────────────────────────────────
    def send_news_warning(self, symbol: str, event: str,
                          minutes_away: int) -> bool:
        """Warn about upcoming high-impact news."""
        if not CONFIG.SEND_NEWS_WARNINGS:
            return False
        msg = (
            f"⚠️ *NEWS WARNING — {symbol}*\n"
            f"{'─' * 28}\n"
            f"📰 {event}\n"
            f"⏰ In ~{minutes_away} minutes\n"
            f"🛑 _Consider closing or hedging_"
        )
        return self.send_message(msg)

    # ── Exit Alert ────────────────────────────────────────
    def send_exit_alert(self, symbol: str, direction: str,
                        reason: str, pnl_pips: float) -> bool:
        """Alert that a position should be exited."""
        if not CONFIG.SEND_EXIT_SIGNALS:
            return False
        icon    = "✅" if pnl_pips >= 0 else "🔴"
        pnl_str = f"{pnl_pips:+.1f} pips"
        msg = (
            f"{icon} *EXIT SIGNAL — {symbol}*\n"
            f"{'─' * 28}\n"
            f"📊 Direction: {direction}\n"
            f"📝 Reason:    {reason}\n"
            f"💰 P&L:       `{pnl_str}`\n"
            f"🕐 `{datetime.now().strftime('%H:%M:%S')}`"
        )
        return self.send_message(msg)

    # ── System Status ─────────────────────────────────────
    def send_startup_message(self, balance: float) -> bool:
        """Send startup confirmation."""
        if not CONFIG.SEND_SYSTEM_ALERTS:
            return False
        msg = (
            f"🤖 *GODBOT STARTED*\n"
            f"{'─' * 28}\n"
            f"💰 Balance:  €{balance:,.2f}\n"
            f"📊 Symbols:  {', '.join(CONFIG.SYMBOLS)}\n"
            f"🎯 Min Score: {CONFIG.MIN_SIGNAL_SCORE}/8\n"
            f"⏰ Scanning every 60s\n"
            f"🕐 `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
        )
        return self.send_message(msg)

    def send_daily_summary(self, trades: int, wins: int,
                           pnl: float, balance: float) -> bool:
        """Send end-of-day performance summary."""
        if not CONFIG.SEND_SYSTEM_ALERTS:
            return False
        win_rate = (wins / trades * 100) if trades > 0 else 0
        icon = "✅" if pnl >= 0 else "🔴"
        msg = (
            f"{icon} *DAILY SUMMARY*\n"
            f"{'─' * 28}\n"
            f"📊 Trades:   {trades}\n"
            f"✅ Wins:     {wins} ({win_rate:.0f}%)\n"
            f"💰 P&L:      €{pnl:+.2f}\n"
            f"💼 Balance:  €{balance:,.2f}\n"
            f"🕐 `{datetime.now().strftime('%Y-%m-%d')}`"
        )
        return self.send_message(msg)