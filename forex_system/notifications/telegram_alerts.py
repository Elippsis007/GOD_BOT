# notifications/telegram_alerts.py
import requests
import threading
import pytz
from datetime import datetime
from monitoring.logger import get_logger
from config.settings   import CONFIG

logger = get_logger("TelegramAlerts")

# ── Credentials ───────────────────────────────────────────
TELEGRAM_TOKEN   = "8693437372:AAHQm4RwedYLkgKkYTlD5XZACHqMnvHVUBE"
TELEGRAM_CHAT_ID = "7688107635"


class TelegramAlerts:
    """
    Sends trading alerts to your phone via Telegram.

    Features:
    → Per-alert type ON/OFF controls
    → Quiet hours (no alerts 23:00-08:00 Spain)
    → Cooldown timers per alert type
    → Minimum confidence filter
    → Never spams — intelligent deduplication
    """

    BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

    def __init__(self):
        self.enabled   = True
        self.chat_id   = TELEGRAM_CHAT_ID
        self.timezone  = pytz.timezone("Europe/Madrid")
        self._cooldowns: dict = {}
        self._test_connection()

    # ── Connection Test ───────────────────────────────────
    def _test_connection(self):
        try:
            url      = f"{self.BASE_URL}/getMe"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data     = response.json()
                bot_name = data["result"]["username"]
                logger.info(
                    f"📱 Telegram connected: @{bot_name}"
                )
                return True
            else:
                logger.warning("⚠️ Telegram connection failed")
                self.enabled = False
                return False
        except Exception as e:
            logger.warning(f"⚠️ Telegram offline: {e}")
            self.enabled = False
            return False

    # ── Core Send ─────────────────────────────────────────
    def _send(self, message: str):
        """Sends message in background — never blocks trading."""
        if not self.enabled:
            return
        def _do_send():
            try:
                url  = f"{self.BASE_URL}/sendMessage"
                data = {
                    "chat_id":    self.chat_id,
                    "text":       message,
                    "parse_mode": "HTML"
                }
                requests.post(url, data=data, timeout=10)
            except Exception as e:
                logger.debug(f"Telegram send error: {e}")
        threading.Thread(
            target=_do_send, daemon=True
        ).start()

    # ── Master Gate ───────────────────────────────────────
    def _can_send(
        self,
        alert_type:  str,
        enabled_flag: bool,
        cooldown_secs: int,
        key:         str = ""
    ) -> bool:
        """
        Single check before every alert:
        1. Is Telegram enabled globally?
        2. Is this alert type turned ON?
        3. Are we in quiet hours?
        4. Has cooldown timer expired?
        5. Is confidence high enough? (signals only)
        """

        # Check 1 — Global enabled
        if not self.enabled:
            return False

        # Check 2 — Alert type enabled
        if not enabled_flag:
            logger.debug(
                f"Alert type {alert_type} is OFF in settings"
            )
            return False

        # Check 3 — Quiet hours
        if self._is_quiet_hours():
            logger.debug(
                f"Quiet hours active — "
                f"{alert_type} suppressed"
            )
            return False

        # Check 4 — Cooldown
        cooldown_key = f"{alert_type}_{key}"
        if self._in_cooldown(cooldown_key, cooldown_secs):
            logger.debug(
                f"Cooldown active for {alert_type}"
            )
            return False

        # All checks passed
        self._set_cooldown(cooldown_key)
        return True

    # ── Quiet Hours Check ─────────────────────────────────
    def _is_quiet_hours(self) -> bool:
        """
        Returns True if currently in quiet hours.
        Spain timezone: 23:00 - 08:00
        """
        if not CONFIG.TELEGRAM_QUIET_ON:
            return False

        hour  = datetime.now(self.timezone).hour
        start = CONFIG.TELEGRAM_QUIET_START  # 23
        end   = CONFIG.TELEGRAM_QUIET_END    # 8

        # Handles overnight window (23:00 - 08:00)
        if start > end:
            return hour >= start or hour < end
        return start <= hour < end

    # ── Cooldown Helpers ──────────────────────────────────
    def _in_cooldown(self, key: str, secs: int) -> bool:
        if key not in self._cooldowns:
            return False
        elapsed = (
            datetime.now() - self._cooldowns[key]
        ).total_seconds()
        return elapsed < secs

    def _set_cooldown(self, key: str):
        self._cooldowns[key] = datetime.now()

    # ── Time Helper ───────────────────────────────────────
    def _now(self) -> str:
        return datetime.now(self.timezone).strftime(
            "%H:%M — %d %b"
        )

    # ──────────────────────────────────────────────────────
    # ALERT METHODS
    # Each one checks its own settings flag first
    # ──────────────────────────────────────────────────────

    # ── BUY Signal ────────────────────────────────────────
    def buy_signal(
        self,
        symbol:     str,
        entry:      float,
        sl:         float,
        tp:         float,
        pips_sl:    float,
        pips_tp:    float,
        confidence: float,
        spread:     float,
        style:      str,
        reasons:    list
    ):
        # Confidence filter
        if confidence < CONFIG.ALERT_MIN_CONFIDENCE:
            logger.debug(
                f"BUY alert suppressed — "
                f"confidence {confidence:.0%} below "
                f"minimum {CONFIG.ALERT_MIN_CONFIDENCE:.0%}"
            )
            return

        if not self._can_send(
            alert_type   = "buy_signal",
            enabled_flag = CONFIG.TELEGRAM_SEND_SIGNALS,
            cooldown_secs= CONFIG.ALERT_COOLDOWN_SIGNAL,
            key          = f"{symbol}_{entry}"
        ):
            return

        reasons_text = "\n".join(
            f"  {r}" for r in reasons[:4]
        )
        style_label = (
            "SCALP" if style == "scalper" else "DAY"
        )
        msg = (
            f"🟢 <b>{style_label} BUY — {symbol}</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Entry  : <b>{entry}</b>\n"
            f"SL     : {sl} ({pips_sl:.1f} pips)\n"
            f"TP     : {tp} ({pips_tp:.1f} pips)\n"
            f"Spread : {spread:.1f} pips ✅\n"
            f"Conf   : {confidence:.0%}\n\n"
            f"Reasons:\n{reasons_text}\n\n"
            f"⚡ <b>Act within 60 seconds</b>"
        )
        self._send(msg)
        logger.info("📱 BUY alert → Telegram ✅")

    # ── SELL Signal ───────────────────────────────────────
    def sell_signal(
        self,
        symbol:     str,
        entry:      float,
        sl:         float,
        tp:         float,
        pips_sl:    float,
        pips_tp:    float,
        confidence: float,
        spread:     float,
        style:      str,
        reasons:    list
    ):
        # Confidence filter
        if confidence < CONFIG.ALERT_MIN_CONFIDENCE:
            logger.debug(
                f"SELL alert suppressed — "
                f"confidence {confidence:.0%} below minimum"
            )
            return

        if not self._can_send(
            alert_type   = "sell_signal",
            enabled_flag = CONFIG.TELEGRAM_SEND_SIGNALS,
            cooldown_secs= CONFIG.ALERT_COOLDOWN_SIGNAL,
            key          = f"{symbol}_{entry}"
        ):
            return

        reasons_text = "\n".join(
            f"  {r}" for r in reasons[:4]
        )
        style_label = (
            "SCALP" if style == "scalper" else "DAY"
        )
        msg = (
            f"🔴 <b>{style_label} SELL — {symbol}</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Entry  : <b>{entry}</b>\n"
            f"SL     : {sl} ({pips_sl:.1f} pips)\n"
            f"TP     : {tp} ({pips_tp:.1f} pips)\n"
            f"Spread : {spread:.1f} pips ✅\n"
            f"Conf   : {confidence:.0%}\n\n"
            f"Reasons:\n{reasons_text}\n\n"
            f"⚡ <b>Act within 60 seconds</b>"
        )
        self._send(msg)
        logger.info("📱 SELL alert → Telegram ✅")

    # ── Hold Alert ────────────────────────────────────────
    def hold_alert(
        self,
        symbol:    str,
        direction: str,
        ticket:    int,
        entry:     float,
        current:   float,
        pnl:       float,
        pips:      float,
        tp:        float,
        pips_left: float
    ):
        if not self._can_send(
            alert_type   = "hold",
            enabled_flag = CONFIG.TELEGRAM_SEND_HOLD,
            cooldown_secs= CONFIG.ALERT_COOLDOWN_HOLD,
            key          = str(ticket)
        ):
            return

        icon = "🟢" if pnl >= 0 else "🔴"
        msg  = (
            f"⏸️ <b>HOLD — {symbol} {direction}</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Opened  : {entry}\n"
            f"Current : {current}\n"
            f"P&L     : {icon} {pips:+.1f} pips "
            f"(€{pnl:+.2f})\n"
            f"Target  : {tp} "
            f"({pips_left:.1f} pips left)\n\n"
            f"✅ Signal still valid — hold position"
        )
        self._send(msg)
        logger.info("📱 HOLD alert → Telegram ✅")

    # ── Potential Exit ────────────────────────────────────
    def potential_exit(
        self,
        symbol:    str,
        direction: str,
        ticket:    int,
        entry:     float,
        current:   float,
        pnl:       float,
        pips:      float,
        reasons:   list
    ):
        if not self._can_send(
            alert_type   = "exit",
            enabled_flag = CONFIG.TELEGRAM_SEND_EXIT,
            cooldown_secs= CONFIG.ALERT_COOLDOWN_EXIT,
            key          = str(ticket)
        ):
            return

        reasons_text = "\n".join(
            f"  ⚠️ {r}" for r in reasons
        )
        msg = (
            f"🟡 <b>CONSIDER EXIT — {symbol}</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Direction : {direction}\n"
            f"Opened    : {entry}\n"
            f"Current   : {current}\n"
            f"Profit    : +{pips:.1f} pips "
            f"(€{pnl:+.2f})\n\n"
            f"Weakening:\n{reasons_text}\n\n"
            f"💡 A) Close now\n"
            f"   B) Move SL to breakeven\n"
            f"   C) Hold remaining pips"
        )
        self._send(msg)
        logger.info("📱 EXIT alert → Telegram ✅")

    # ── Danger Exit ───────────────────────────────────────
    def danger_exit(
        self,
        symbol:     str,
        direction:  str,
        ticket:     int,
        entry:      float,
        current:    float,
        pnl:        float,
        pips:       float,
        sl:         float,
        pips_to_sl: float,
        reasons:    list
    ):
        # Danger always bypasses quiet hours
        # Too critical to suppress
        if not self.enabled:
            return

        if self._in_cooldown(
            f"danger_{ticket}",
            CONFIG.ALERT_COOLDOWN_DANGER
        ):
            return

        self._set_cooldown(f"danger_{ticket}")

        if not CONFIG.TELEGRAM_SEND_DANGER:
            return

        reasons_text = "\n".join(
            f"  ❌ {r}" for r in reasons
        )
        msg = (
            f"🚨 <b>⚠️ DANGER — {symbol} {direction}</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Opened  : {entry}\n"
            f"Current : {current}\n"
            f"P&L     : 🔴 {pips:.1f} pips "
            f"(€{pnl:.2f})\n"
            f"SL      : {sl} "
            f"({pips_to_sl:.1f} pips away!)\n\n"
            f"Danger signals:\n{reasons_text}\n\n"
            f"⚡ <b>URGENT — Consider closing NOW</b>"
        )
        self._send(msg)
        logger.info("📱 DANGER alert → Telegram ✅")

    # ── News Warning ──────────────────────────────────────
    def news_warning(
        self,
        event:     str,
        currency:  str,
        time_str:  str,
        mins_away: float,
        symbols:   list
    ):
        # News bypasses quiet hours too
        # Important even if you are sleeping
        if not self.enabled:
            return

        if not CONFIG.TELEGRAM_SEND_NEWS:
            return

        if self._in_cooldown(
            f"news_{event}",
            CONFIG.ALERT_COOLDOWN_NEWS
        ):
            return

        self._set_cooldown(f"news_{event}")

        symbols_text = ", ".join(symbols)
        msg = (
            f"📰 <b>⚠️ NEWS WARNING</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Event    : <b>{event}</b>\n"
            f"Currency : {currency}\n"
            f"Time     : {time_str} your time\n"
            f"In       : {mins_away:.0f} minutes\n"
            f"Impact   : 🔴 HIGH\n\n"
            f"Affects  : {symbols_text}\n\n"
            f"⚡ Signals paused 30 mins before\n"
            f"   and 15 mins after\n"
            f"💡 Consider closing open trades"
        )
        self._send(msg)
        logger.info("📱 NEWS WARNING → Telegram ✅")

    # ── Trade Opened ──────────────────────────────────────
    def trade_opened(
        self,
        symbol:    str,
        direction: str,
        ticket:    int,
        entry:     float,
        sl:        float,
        tp:        float,
        volume:    float,
        risk:      float,
        style:     str,
        mode:      str
    ):
        if not self._can_send(
            alert_type   = "opened",
            enabled_flag = CONFIG.TELEGRAM_SEND_OPENED,
            cooldown_secs= 60,
            key          = str(ticket)
        ):
            return

        icon = "🟢" if direction == "BUY" else "🔴"
        msg  = (
            f"{icon} <b>TRADE OPENED — {symbol}</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Direction : <b>{direction}</b>\n"
            f"Entry     : {entry}\n"
            f"SL        : {sl}\n"
            f"TP        : {tp}\n"
            f"Volume    : {volume} lots\n"
            f"Risk      : €{risk:.2f}\n"
            f"Ticket    : #{ticket}\n"
            f"Style     : {style.title()}\n\n"
            f"✅ Trade placed successfully"
        )
        self._send(msg)
        logger.info("📱 TRADE OPENED → Telegram ✅")

    # ── Trade Closed ──────────────────────────────────────
    def trade_closed(
        self,
        symbol:    str,
        direction: str,
        ticket:    int,
        entry:     float,
        close:     float,
        pnl:       float,
        pips:      float,
        reason:    str
    ):
        if not self._can_send(
            alert_type   = "closed",
            enabled_flag = CONFIG.TELEGRAM_SEND_CLOSED,
            cooldown_secs= 60,
            key          = str(ticket)
        ):
            return

        icon   = "✅" if pnl >= 0 else "❌"
        emoji  = "🎉" if pnl > 0 else "💪"
        msg    = (
            f"{icon} <b>TRADE CLOSED — {symbol}</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Direction : {direction}\n"
            f"Opened    : {entry}\n"
            f"Closed    : {close}\n"
            f"Result    : <b>{pips:+.1f} pips "
            f"(€{pnl:+.2f})</b>\n"
            f"Reason    : {reason}\n"
            f"Ticket    : #{ticket}\n\n"
            f"{emoji} "
            f"{'Great trade!' if pnl > 0 else 'On to the next one!'}"
        )
        self._send(msg)
        logger.info("📱 TRADE CLOSED → Telegram ✅")

    # ── Daily Summary ─────────────────────────────────────
    def daily_summary(
        self,
        date:            str,
        style:           str,
        mode:            str,
        signals:         int,
        trades:          int,
        winners:         int,
        losers:          int,
        gross_profit:    float,
        gross_loss:      float,
        net_pnl:         float,
        win_rate:        float,
        balance:         float,
        best_trade:      float,
        worst_trade:     float,
        tomorrow_events: list
    ):
        # Daily summary always sends
        # regardless of quiet hours
        if not self.enabled:
            return
        if not CONFIG.TELEGRAM_SEND_SUMMARY:
            return

        events_text = (
            "\n".join(
                f"  🔴 {e['time']} — {e['event'][:30]}"
                for e in tomorrow_events[:3]
            )
            if tomorrow_events
            else "  ✅ No high impact events"
        )

        pnl_icon = "📈" if net_pnl >= 0 else "📉"
        msg      = (
            f"📊 <b>GODBOT DAILY SUMMARY</b>\n"
            f"📅 {date}\n\n"
            f"Style     : {style.title()}\n"
            f"Mode      : {mode}\n\n"
            f"Signals   : {signals}\n"
            f"Trades    : {trades}\n"
            f"Winners   : {winners} ✅\n"
            f"Losers    : {losers} ❌\n"
            f"Win Rate  : {win_rate:.0f}%\n\n"
            f"Profit    : €{gross_profit:+.2f}\n"
            f"Loss      : €{gross_loss:+.2f}\n"
            f"Net P&L   : {pnl_icon} "
            f"<b>€{net_pnl:+.2f}</b>\n\n"
            f"Best Trade : €{best_trade:+.2f}\n"
            f"Worst Trade: €{worst_trade:+.2f}\n"
            f"Balance    : €{balance:,.2f}\n\n"
            f"Tomorrow's Events:\n{events_text}"
        )
        self._send(msg)
        logger.info("📱 DAILY SUMMARY → Telegram ✅")

    # ── System Online ─────────────────────────────────────
    def system_online(
        self,
        balance: float,
        style:   str,
        mode:    str
    ):
        # Always sends — startup notification
        if not self.enabled:
            return
        msg = (
            f"🤖 <b>GODBOT IS ONLINE</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Balance : €{balance:,.2f}\n"
            f"Symbol  : EURUSD\n"
            f"Style   : {style.title()}\n"
            f"Mode    : {mode}\n\n"
            f"Alert Settings:\n"
            f"  Signals  : "
            f"{'✅' if CONFIG.TELEGRAM_SEND_SIGNALS else '❌'}\n"
            f"  Danger   : "
            f"{'✅' if CONFIG.TELEGRAM_SEND_DANGER else '❌'}\n"
            f"  News     : "
            f"{'✅' if CONFIG.TELEGRAM_SEND_NEWS else '❌'}\n"
            f"  Opened   : "
            f"{'✅' if CONFIG.TELEGRAM_SEND_OPENED else '❌'}\n"
            f"  Closed   : "
            f"{'✅' if CONFIG.TELEGRAM_SEND_CLOSED else '❌'}\n"
            f"  Hold     : "
            f"{'✅' if CONFIG.TELEGRAM_SEND_HOLD else '❌'}\n"
            f"  Summary  : "
            f"{'✅' if CONFIG.TELEGRAM_SEND_SUMMARY else '❌'}\n"
            f"  Quiet    : "
            f"{CONFIG.TELEGRAM_QUIET_START}:00 - "
            f"{CONFIG.TELEGRAM_QUIET_END}:00\n\n"
            f"📱 You will receive alerts here"
        )
        self._send(msg)
        logger.info("📱 SYSTEM ONLINE → Telegram ✅")

    # ── System Offline ────────────────────────────────────
    def system_offline(
        self,
        reason: str = "Manual shutdown"
    ):
        # Always sends — important notification
        if not self.enabled:
            return
        msg = (
            f"🔌 <b>GODBOT OFFLINE</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Reason: {reason}\n\n"
            f"All open positions remain in MT5\n"
            f"Check MT5 terminal directly"
        )
        self._send(msg)
        logger.info("📱 SYSTEM OFFLINE → Telegram ✅")

    # ── Toggle ────────────────────────────────────────────
    def toggle(self):
        self.enabled = not self.enabled
        status = "ON" if self.enabled else "OFF"
        logger.info(f"📱 Telegram: {status}")
        if self.enabled:
            self._send(
                f"📱 Telegram alerts turned ON\n"
                f"⏰ {self._now()}"
            )
        return self.enabled