# notifications/telegram_alerts.py
import requests
import threading
import pytz
from datetime import datetime
from monitoring.logger import get_logger
from config.settings   import CONFIG

logger = get_logger("TelegramAlerts")


def _escape(text: str) -> str:
    """Escape HTML special characters in user-supplied strings."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


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

    def __init__(self):
        self._token    = CONFIG.TELEGRAM_TOKEN
        self._base_url = f"https://api.telegram.org/bot{self._token}"
        self.enabled   = True
        self.chat_id   = CONFIG.TELEGRAM_CHAT_ID
        self.timezone  = pytz.timezone("Europe/Madrid")
        self._cooldowns: dict = {}

        if not self._token:
            logger.warning("⚠️ TELEGRAM_TOKEN is empty — alerts disabled")
            self.enabled = False
            return

        self._test_connection()

    # ── Connection Test ───────────────────────────────────────────────────────
    def _test_connection(self) -> bool:
        try:
            url      = f"{self._base_url}/getMe"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                bot_name = response.json()["result"]["username"]
                logger.info(f"📱 Telegram connected: @{bot_name}")
                return True
            else:
                logger.warning(
                    f"⚠️ Telegram connection failed — HTTP {response.status_code}"
                )
                self.enabled = False
                return False
        except Exception as e:
            logger.warning(f"⚠️ Telegram offline: {e}")
            self.enabled = False
            return False

    # ── Core Send ─────────────────────────────────────────────────────────────
    def _send(self, message: str) -> None:
        """Sends message in a background thread — never blocks trading."""
        if not self.enabled:
            return

        def _do_send() -> None:
            try:
                url  = f"{self._base_url}/sendMessage"
                data = {
                    "chat_id":    self.chat_id,
                    "text":       message,
                    "parse_mode": "HTML",
                }
                resp = requests.post(url, data=data, timeout=10)
                if resp.status_code == 429:
                    retry_after = resp.json().get(
                        "parameters", {}
                    ).get("retry_after", 30)
                    logger.warning(
                        f"Telegram rate-limited — retry after {retry_after}s"
                    )
                elif resp.status_code != 200:
                    logger.debug(
                        f"Telegram send failed: HTTP {resp.status_code} — "
                        f"{resp.text[:120]}"
                    )
            except Exception as e:
                logger.debug(f"Telegram send error: {e}")

        threading.Thread(target=_do_send, daemon=True).start()

    # ── Master Gate ───────────────────────────────────────────────────────────
    def _can_send(
        self,
        alert_type:    str,
        enabled_flag:  bool,
        cooldown_secs: int,
        key:           str = "",
    ) -> bool:
        if not self.enabled:
            return False
        if not enabled_flag:
            logger.debug(f"Alert type {alert_type} is OFF in settings")
            return False
        if self._is_quiet_hours():
            logger.debug(f"Quiet hours active — {alert_type} suppressed")
            return False
        cooldown_key = f"{alert_type}_{key}"
        if self._in_cooldown(cooldown_key, cooldown_secs):
            logger.debug(f"Cooldown active for {alert_type}")
            return False
        self._set_cooldown(cooldown_key)
        return True

    # ── Quiet Hours ───────────────────────────────────────────────────────────
    def _is_quiet_hours(self) -> bool:
        if not CONFIG.TELEGRAM_QUIET_ON:
            return False
        hour  = datetime.now(self.timezone).hour
        start = CONFIG.TELEGRAM_QUIET_START
        end   = CONFIG.TELEGRAM_QUIET_END
        if start > end:
            return hour >= start or hour < end
        return start <= hour < end

    # ── Cooldown Helpers ──────────────────────────────────────────────────────
    def _in_cooldown(self, key: str, secs: int) -> bool:
        if key not in self._cooldowns:
            return False
        return (datetime.now() - self._cooldowns[key]).total_seconds() < secs

    def _set_cooldown(self, key: str) -> None:
        self._cooldowns[key] = datetime.now()

    # ── Time Helper ───────────────────────────────────────────────────────────
    def _now(self) -> str:
        return datetime.now(self.timezone).strftime("%H:%M — %d %b")

    # ── BUY Signal ────────────────────────────────────────────────────────────
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
        reasons:    list,
    ) -> None:
        if confidence < CONFIG.ALERT_MIN_CONFIDENCE:
            return
        if not self._can_send(
            alert_type="buy_signal",
            enabled_flag=CONFIG.TELEGRAM_SEND_SIGNALS,
            cooldown_secs=CONFIG.ALERT_COOLDOWN_SIGNAL,
            key=f"{symbol}_{entry}",
        ):
            return

        reasons_text = "\n".join(f"  {_escape(r)}" for r in reasons[:4])
        style_label  = "SCALP" if style == "scalper" else "DAY"
        msg = (
            f"🟢 <b>{style_label} BUY — {_escape(symbol)}</b>\n"
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

    # ── SELL Signal ───────────────────────────────────────────────────────────
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
        reasons:    list,
    ) -> None:
        if confidence < CONFIG.ALERT_MIN_CONFIDENCE:
            return
        if not self._can_send(
            alert_type="sell_signal",
            enabled_flag=CONFIG.TELEGRAM_SEND_SIGNALS,
            cooldown_secs=CONFIG.ALERT_COOLDOWN_SIGNAL,
            key=f"{symbol}_{entry}",
        ):
            return

        reasons_text = "\n".join(f"  {_escape(r)}" for r in reasons[:4])
        style_label  = "SCALP" if style == "scalper" else "DAY"
        msg = (
            f"🔴 <b>{style_label} SELL — {_escape(symbol)}</b>\n"
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

    # ── Hold Alert ────────────────────────────────────────────────────────────
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
        pips_left: float,
    ) -> None:
        if not self._can_send(
            alert_type="hold",
            enabled_flag=CONFIG.TELEGRAM_SEND_HOLD,
            cooldown_secs=CONFIG.ALERT_COOLDOWN_HOLD,
            key=str(ticket),
        ):
            return

        icon = "🟢" if pnl >= 0 else "🔴"
        msg  = (
            f"⏸️ <b>HOLD — {_escape(symbol)} {direction}</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Opened  : {entry}\n"
            f"Current : {current}\n"
            f"P&L     : {icon} {pips:+.1f} pips (€{pnl:+.2f})\n"
            f"Target  : {tp} ({pips_left:.1f} pips left)\n\n"
            f"✅ Signal still valid — hold position"
        )
        self._send(msg)
        logger.info("📱 HOLD alert → Telegram ✅")

    # ── Potential Exit ────────────────────────────────────────────────────────
    def potential_exit(
        self,
        symbol:    str,
        direction: str,
        ticket:    int,
        entry:     float,
        current:   float,
        pnl:       float,
        tp:        float,
        reasons:   list,
    ) -> None:
        if not self._can_send(
            alert_type="exit",
            enabled_flag=CONFIG.TELEGRAM_SEND_EXIT,
            cooldown_secs=CONFIG.ALERT_COOLDOWN_EXIT,
            key=str(ticket),
        ):
            return

        reasons_text = "\n".join(f"  ⚠️ {_escape(r)}" for r in reasons)
        msg = (
            f"🟡 <b>CONSIDER EXIT — {_escape(symbol)}</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Direction : {direction}\n"
            f"Opened    : {entry}\n"
            f"Current   : {current}\n"
            f"TP Target : {tp}\n"
            f"P&L       : €{pnl:+.2f}\n\n"
            f"Near TP:\n{reasons_text}\n\n"
            f"💡 A) Close now\n"
            f"   B) Move SL to breakeven\n"
            f"   C) Hold remaining pips"
        )
        self._send(msg)
        logger.info("📱 EXIT alert → Telegram ✅")

    # ── Danger Exit ───────────────────────────────────────────────────────────
    def danger_exit(
        self,
        symbol:    str,
        direction: str,
        ticket:    int,
        entry:     float,
        current:   float,
        pnl:       float,
        sl:        float,
        reasons:   list,
    ) -> None:
        # Danger bypasses quiet hours — too critical to suppress
        if not self.enabled:
            return
        if self._in_cooldown(f"danger_{ticket}", CONFIG.ALERT_COOLDOWN_DANGER):
            return
        self._set_cooldown(f"danger_{ticket}")
        if not CONFIG.TELEGRAM_SEND_DANGER:
            return

        reasons_text = "\n".join(f"  ❌ {_escape(r)}" for r in reasons)
        msg = (
            f"🚨 <b>⚠️ DANGER — {_escape(symbol)} {direction}</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Opened  : {entry}\n"
            f"Current : {current}\n"
            f"P&L     : 🔴 €{pnl:.2f}\n"
            f"SL      : {sl}\n\n"
            f"Danger signals:\n{reasons_text}\n\n"
            f"⚡ <b>URGENT — Consider closing NOW</b>"
        )
        self._send(msg)
        logger.info("📱 DANGER alert → Telegram ✅")

    # ── Risk Warning ─────────────────────────────────────────────────────────
    def risk_warning(
        self,
        level:        str,
        message:      str,
        daily_pnl:    float = 0.0,
        daily_limit:  float = 0.0,
        pct_used:     float = 0.0,
        open_trades:  int   = 0,
        max_trades:   int   = 0,
    ) -> None:
        """
        Risk limit notifications — always bypass quiet hours.
        These are too important to suppress overnight.

        Levels and their Telegram messages:
          WARNING     — daily loss at 75% — amber alert
          URGENT      — daily loss at 90% — red alert
          LIMIT_HIT   — trading halted for today
          TRADES_NEAR — 2 of 3 slots used
          TRADES_FULL — all slots full, signal blocked
          LOW_MARGIN  — free margin too low
        """
        # Risk warnings always bypass quiet hours
        if not self.enabled:
            return

        # Cooldown — LIMIT_HIT/URGENT every 3 min, others every 10 min
        cooldown = 180 if level in ("LIMIT_HIT", "URGENT") else 600
        key      = f"risk_{level}"
        if self._in_cooldown(key, cooldown):
            return
        self._set_cooldown(key)

        # ── Build progress bar for loss-based alerts ──────────────────────────
        bar_line = ""
        if level in ("WARNING", "URGENT", "LIMIT_HIT") and daily_limit > 0:
            filled   = min(int(pct_used / 10), 10)
            bar      = "█" * filled + "░" * (10 - filled)
            bar_line = (
                f"\nLoss Used : [{bar}] {pct_used:.0f}%\n"
                f"Lost      : €{abs(daily_pnl):.2f} of €{daily_limit:.2f}\n"
                f"Remaining : €{max(daily_limit - abs(daily_pnl), 0):.2f}"
            )

        # ── Build slot line for trade-count alerts ────────────────────────────
        slot_line = ""
        if level in ("TRADES_NEAR", "TRADES_FULL") and max_trades > 0:
            slot_line = f"\nSlots Used: {open_trades} / {max_trades}"

        # ── Icon and title per level ──────────────────────────────────────────
        headers = {
            "WARNING":     "⚠️ RISK WARNING — Daily Loss 75%",
            "URGENT":      "⛔ URGENT — Daily Loss 90%",
            "LIMIT_HIT":   "🚫 DAILY LIMIT HIT — Trading Halted",
            "TRADES_NEAR": "⚠️ TRADE SLOTS NEAR LIMIT",
            "TRADES_FULL": "🚫 ALL TRADE SLOTS FULL",
            "LOW_MARGIN":  "⚠️ LOW MARGIN WARNING",
        }
        header = headers.get(level, f"⚠️ RISK ALERT — {level}")

        msg = (
            f"🛡 <b>{_escape(header)}</b>\n"
            f"⏰ {self._now()}\n"
            f"{bar_line}"
            f"{slot_line}\n\n"
            f"{_escape(message)}"
        )
        self._send(msg)
        logger.info(f"📱 RISK WARNING ({level}) → Telegram ✅")

    # ── News Warning ──────────────────────────────────────────────────────────
    def news_warning(
        self,
        event:     str,
        currency:  str,
        time_str:  str,
        mins_away: float,
        symbols:   list,
    ) -> None:
        if not self.enabled:
            return
        if not CONFIG.TELEGRAM_SEND_NEWS:
            return
        if self._in_cooldown(f"news_{event}", CONFIG.ALERT_COOLDOWN_NEWS):
            return
        self._set_cooldown(f"news_{event}")

        symbols_text = _escape(", ".join(symbols))
        msg = (
            f"📰 <b>⚠️ NEWS WARNING</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Event    : <b>{_escape(event)}</b>\n"
            f"Currency : {_escape(currency)}\n"
            f"Time     : {_escape(time_str)} your time\n"
            f"In       : {mins_away:.0f} minutes\n"
            f"Impact   : 🔴 HIGH\n\n"
            f"Affects  : {symbols_text}\n\n"
            f"⚡ Signals paused 30 mins before\n"
            f"   and 15 mins after\n"
            f"💡 Consider closing open trades"
        )
        self._send(msg)
        logger.info("📱 NEWS WARNING → Telegram ✅")

    # ── Trade Opened ──────────────────────────────────────────────────────────
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
        mode:      str,
    ) -> None:
        if not self._can_send(
            alert_type="opened",
            enabled_flag=CONFIG.TELEGRAM_SEND_OPENED,
            cooldown_secs=60,
            key=str(ticket),
        ):
            return

        icon = "🟢" if direction == "BUY" else "🔴"
        msg  = (
            f"{icon} <b>TRADE OPENED — {_escape(symbol)}</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Direction : <b>{direction}</b>\n"
            f"Entry     : {entry}\n"
            f"SL        : {sl}\n"
            f"TP        : {tp}\n"
            f"Volume    : {volume} lots\n"
            f"Risk      : €{risk:.2f}\n"
            f"Ticket    : #{ticket}\n"
            f"Style     : {_escape(style.title())}\n\n"
            f"✅ Trade placed successfully"
        )
        self._send(msg)
        logger.info("📱 TRADE OPENED → Telegram ✅")

    # ── Trade Closed ──────────────────────────────────────────────────────────
    def trade_closed(
        self,
        symbol:    str,
        direction: str,
        ticket:    int,
        entry:     float,
        close:     float,
        pnl:       float,
        pips:      float,
        reason:    str,
    ) -> None:
        if not self._can_send(
            alert_type="closed",
            enabled_flag=CONFIG.TELEGRAM_SEND_CLOSED,
            cooldown_secs=60,
            key=str(ticket),
        ):
            return

        icon  = "✅" if pnl >= 0 else "❌"
        emoji = "🎉" if pnl > 0 else "💪"
        msg   = (
            f"{icon} <b>TRADE CLOSED — {_escape(symbol)}</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Direction : {direction}\n"
            f"Opened    : {entry}\n"
            f"Closed    : {close}\n"
            f"Result    : <b>{pips:+.1f} pips (€{pnl:+.2f})</b>\n"
            f"Reason    : {_escape(reason)}\n"
            f"Ticket    : #{ticket}\n\n"
            f"{emoji} {'Great trade!' if pnl > 0 else 'On to the next one!'}"
        )
        self._send(msg)
        logger.info("📱 TRADE CLOSED → Telegram ✅")

    # ── Daily Summary ─────────────────────────────────────────────────────────
    def daily_summary(
        self,
        signals:      int,
        trades:       int,
        winners:      int,
        losers:       int,
        net_pnl:      float,
        win_rate:     float,
        best_trade:   float,
        worst_trade:  float,
        gross_profit: float,
        gross_loss:   float,
        balance:      float = 0.0,
        date:         str   = "",
    ) -> None:
        # Daily summary always sends regardless of quiet hours
        if not self.enabled:
            return
        if not CONFIG.TELEGRAM_SEND_SUMMARY:
            return

        today    = date or datetime.now(self.timezone).strftime("%d %b %Y")
        pnl_icon = "📈" if net_pnl >= 0 else "📉"
        msg      = (
            f"📊 <b>GODBOT DAILY SUMMARY</b>\n"
            f"📅 {_escape(today)}\n\n"
            f"Signals    : {signals}\n"
            f"Trades     : {trades}\n"
            f"Winners    : {winners} ✅\n"
            f"Losers     : {losers} ❌\n"
            f"Win Rate   : {win_rate:.0f}%\n\n"
            f"Gross P    : €{gross_profit:+.2f}\n"
            f"Gross L    : €{gross_loss:+.2f}\n"
            f"Net P&L    : {pnl_icon} <b>€{net_pnl:+.2f}</b>\n\n"
            f"Best Trade : €{best_trade:+.2f}\n"
            f"Worst Trade: €{worst_trade:+.2f}\n"
            f"Balance    : €{balance:,.2f}"
        )
        self._send(msg)
        logger.info("📱 DAILY SUMMARY → Telegram ✅")

    # ── System Online ─────────────────────────────────────────────────────────
    def system_online(
        self,
        balance: float,
        style:   str,
        mode:    str,
    ) -> None:
        if not self.enabled:
            return
        msg = (
            f"🤖 <b>GODBOT IS ONLINE</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Balance : €{balance:,.2f}\n"
            f"Symbol  : EURUSD\n"
            f"Style   : {_escape(style.title())}\n"
            f"Mode    : {_escape(mode)}\n\n"
            f"Alert Settings:\n"
            f"  Signals  : {'✅' if CONFIG.TELEGRAM_SEND_SIGNALS else '❌'}\n"
            f"  Danger   : {'✅' if CONFIG.TELEGRAM_SEND_DANGER  else '❌'}\n"
            f"  News     : {'✅' if CONFIG.TELEGRAM_SEND_NEWS    else '❌'}\n"
            f"  Opened   : {'✅' if CONFIG.TELEGRAM_SEND_OPENED  else '❌'}\n"
            f"  Closed   : {'✅' if CONFIG.TELEGRAM_SEND_CLOSED  else '❌'}\n"
            f"  Hold     : {'✅' if CONFIG.TELEGRAM_SEND_HOLD    else '❌'}\n"
            f"  Summary  : {'✅' if CONFIG.TELEGRAM_SEND_SUMMARY else '❌'}\n"
            f"  Quiet    : {CONFIG.TELEGRAM_QUIET_START}:00 — "
            f"{CONFIG.TELEGRAM_QUIET_END}:00\n\n"
            f"📱 You will receive alerts here"
        )
        self._send(msg)
        logger.info("📱 SYSTEM ONLINE → Telegram ✅")

    # ── System Offline ────────────────────────────────────────────────────────
    def system_offline(self, reason: str = "Manual shutdown") -> None:
        if not self.enabled:
            return
        msg = (
            f"🔌 <b>GODBOT OFFLINE</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Reason: {_escape(reason)}\n\n"
            f"All open positions remain in MT5\n"
            f"Check MT5 terminal directly"
        )
        self._send(msg)
        logger.info("📱 SYSTEM OFFLINE → Telegram ✅")

    # ── Toggle ────────────────────────────────────────────────────────────────
    def toggle(self) -> bool:
        self.enabled = not self.enabled
        status = "ON" if self.enabled else "OFF"
        logger.info(f"📱 Telegram: {status}")
        if self.enabled:
            self._send(f"📱 Telegram alerts turned ON\n⏰ {self._now()}")
        return self.enabled
