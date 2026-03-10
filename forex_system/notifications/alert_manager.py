# =============================================================================
# GODBOT v3.0 – notifications/alert_manager.py
# =============================================================================
#  Original fixes A–L preserved.
#  Additional fixes M–Q applied in previous revision.
#  This revision fixes:
#
#  R  [FIX] send() now calls telegram._send() directly (public send_message()
#     does not exist on TelegramAlerts — was AttributeError at runtime).
#
#  S  [FIX] trade_opened() _tg call now passes sl_pips and tp_pips which
#     were completely missing — was TypeError at runtime on every trade open.
#
#  T  [FIX] breakeven_moved() _tg call replaced with inline _send() since
#     TelegramAlerts has no breakeven_moved() method.
#
#  U  [FIX] signal_alert() reads signal.signal_type directly as a string
#     instead of .name — TradingSignal.signal_type is str not an Enum.
#
#  V  [FIX] buy_signal() / sell_signal() now actually use the fetched
#     currency variable in terminal output (was fetched then discarded).
#
#  W  [FIX] trade_closed() parameter renamed close → close_price to match
#     TelegramAlerts.trade_closed() and avoid shadowing built-in close().
#
#  X  [FIX] system_online() reads CONFIG.SYMBOLS and CONFIG.SCALPER_TF_SELECTED
#     for the terminal banner instead of hard-coded "EURUSD | M5".
# =============================================================================

from __future__ import annotations

import pytz
from datetime import datetime, timedelta
from typing import Optional

from notifications.sound_alerts    import SoundAlerts
from notifications.telegram_alerts import TelegramAlerts
from monitoring.logger             import get_logger

logger = get_logger("AlertManager")


# ── Module-level helpers ──────────────────────────────────────────────────────

def _get_config_cooldown(attr: str, default_secs: int) -> float:
    """
    [F][G] Read a cooldown value in SECONDS from CONFIG, convert to minutes.
    Returns a float (minutes) with a minimum floor of 0.5 min (30 s).
    """
    try:
        from config.settings import CONFIG
        secs = getattr(CONFIG, attr, default_secs)
    except Exception:
        secs = default_secs
    return max(0.5, secs / 60.0)


def _get_alert_min_confidence() -> float:
    """[A] Read ALERT_MIN_CONFIDENCE from CONFIG with safe fallback."""
    try:
        from config.settings import CONFIG
        return getattr(CONFIG, "ALERT_MIN_CONFIDENCE", 0.55)
    except Exception:
        return 0.55


def _pip_size_from_digits(
    digits: Optional[int],
    point:  Optional[float],
) -> float:
    """
    [B][M] Digit-aware pip size from symbol digits and point values.
    Accepts primitives so it works whether the data came from mt5 directly
    or from MT5Connector.get_symbol_info() dict.
    """
    if digits is None or point is None:
        return 0.0001
    if digits in (5, 3):
        return 10 * point
    return point


# ── AlertManager class ────────────────────────────────────────────────────────

class AlertManager:
    """
    Master alert coordinator for GODBOT v3.0.

    Every alert routes through here.
    Fires Terminal + Sound + Telegram simultaneously.
    Never duplicates the same alert within the cooldown window.

    Deduplication key:
      Entry price rounded to DEDUP_PRICE_DIGITS=3 so a 1-pip move between
      scan cycles does not bypass the cooldown.

    Confidence gate: [A]
      buy_signal() / sell_signal() / signal_alert() drop alerts below
      CONFIG.ALERT_MIN_CONFIDENCE (0.55) before any output is produced.

    Quiet hours: [Q]
      Telegram calls suppressed during CONFIG.TELEGRAM_QUIET_START/END.
      Terminal prints and sounds still fire so a watching operator sees them.
    """

    # [J] Corrected from 4 to 3 — groups within ~10 pips as documented
    DEDUP_PRICE_DIGITS = 3

    def __init__(self, dashboard=None) -> None:
        self.sound    = SoundAlerts()
        self.telegram = TelegramAlerts()
        self.timezone = pytz.timezone("Europe/Madrid")
        self._alerted: dict = {}

        # [C] Optional dashboard reference for log_signal() integration
        self._dashboard = dashboard

        # [M] MT5Connector for all MT5 data access
        from core.mt5_connector import MT5Connector
        self._connector = MT5Connector()

        # [N] Account currency cache — updated on first use
        self._currency = "USD"

    def set_dashboard(self, dashboard) -> None:
        """[C] Inject dashboard after construction if not passed to __init__."""
        self._dashboard = dashboard

    # ── Time helpers ──────────────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now(self.timezone).strftime("%H:%M — %d %b")

    def _now_aware(self) -> datetime:
        return datetime.now(pytz.utc).astimezone(self.timezone)

    # ── [Q] Quiet-hours gate ──────────────────────────────────────────────────

    def _is_quiet_hours(self) -> bool:
        """
        [Q] Returns True during CONFIG.TELEGRAM_QUIET_START/END window.
        Telegram calls are suppressed when True.
        Terminal prints and sound alerts are NOT suppressed.
        """
        try:
            from config.settings import CONFIG
            if not getattr(CONFIG, "TELEGRAM_QUIET_ON", True):
                return False
            q_start = getattr(CONFIG, "TELEGRAM_QUIET_START", 23)
            q_end   = getattr(CONFIG, "TELEGRAM_QUIET_END",   7)
            hour    = self._now_aware().hour
            if q_start > q_end:
                return hour >= q_start or hour < q_end
            return q_start <= hour < q_end
        except Exception:
            return False

    def _tg(self, fn, *args, **kwargs) -> None:
        """
        [Q] Wrapper for all Telegram calls.
        Silently skips the call during quiet hours.
        Catches all exceptions so a Telegram failure never propagates.
        Note: system_offline() and danger_exit() bypass this wrapper
        intentionally — they are always operationally critical.
        """
        if self._is_quiet_hours():
            logger.debug("[AM] Telegram suppressed — quiet hours active.")
            return
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            logger.warning("[AM] Telegram call failed: %s", exc)

    # ── [N] Currency helper ───────────────────────────────────────────────────

    def _get_currency(self) -> str:
        """[N] Read account currency from connector, cache result."""
        try:
            info = self._connector.get_account_info()
            if info:
                self._currency = info.get("currency", "USD")
        except Exception:
            pass
        return self._currency

    # ── [M] Pip size helper ───────────────────────────────────────────────────

    def _pip_size(self, symbol: str) -> float:
        """
        [M] Get pip size via MT5Connector.get_symbol_info() dict.
        Falls back to 0.0001 (standard 5-digit pair default) on failure.
        """
        try:
            info = self._connector.get_symbol_info(symbol)
            if info:
                return _pip_size_from_digits(
                    info.get("digits"), info.get("point")
                )
        except Exception:
            pass
        return 0.0001

    # ── [M] Current price helper ──────────────────────────────────────────────

    def _current_tick(self, symbol: str) -> Optional[dict]:
        """[M] Get latest tick via MT5Connector.get_latest_tick()."""
        try:
            return self._connector.get_latest_tick(symbol)
        except Exception:
            return None

    # ── Duplicate guard ───────────────────────────────────────────────────────

    def _already_alerted(self, key: str, minutes: float = 2.0) -> bool:
        """
        Returns True (suppress) if the same key fired within `minutes`.
        All datetimes are timezone-aware (Europe/Madrid) to prevent
        naive/aware comparison errors.
        """
        if key in self._alerted:
            age_mins = (
                self._now_aware() - self._alerted[key]
            ).total_seconds() / 60.0
            if age_mins < minutes:
                return True
        self._alerted[key] = self._now_aware()
        return False

    def _price_key(self, price: float) -> str:
        """[J] Round to DEDUP_PRICE_DIGITS before building dedup key."""
        return str(round(price, self.DEDUP_PRICE_DIGITS))

    # ── Dashboard signal logging helper ───────────────────────────────────────

    def _log_signal_to_dashboard(
        self,
        symbol:     str,
        direction:  str,
        entry:      float,
        sl:         float,
        tp:         float,
        confidence: float,
        sl_pips:    float,
        tp_pips:    float,
    ) -> None:
        """[C] Forward signal data to dashboard.log_signal() including pips."""
        if self._dashboard is None:
            return
        try:
            self._dashboard.log_signal(
                symbol     = symbol,
                direction  = direction,
                entry      = entry,
                sl         = sl,
                tp         = tp,
                confidence = confidence,
                sl_pips    = sl_pips,
                tp_pips    = tp_pips,
            )
        except Exception as exc:
            logger.debug("[AM] dashboard.log_signal() error: %s", exc)

    # ── [O] Plain-text send (called by main.py) ───────────────────────────────

    def send(self, message: str) -> None:
        """
        [O][R] Send a plain-text Telegram message.
        Called by main.py for startup, shutdown, fill confirmation,
        broker-closed detection, EOD close, and daily summary notifications.
        Respects quiet-hours suppression via _tg(). [Q]

        Fix R: TelegramAlerts has no send_message() method — routes to
        the internal _send() which is the correct delivery mechanism.
        """
        logger.info("[AM] send: %s", message)
        self._tg(self.telegram._send, message)

    # ── [P] signal_alert (called by main.py) ─────────────────────────────────

    def signal_alert(
        self,
        symbol:        str,
        signal,
        position_spec,
        sound:         bool = True,
        telegram:      bool = True,
    ) -> None:
        """
        [P] Called by main.py after signal generation and risk sizing.
        Routes to buy_signal() or sell_signal() using fields from both
        the TradingSignal and PositionSpec objects.

        Parameters
        ──────────
        signal        : TradingSignal with signal_type (str), entry, sl,
                        tp, sl_pips, tp_pips, confidence, atr, reasons
        position_spec : PositionSpec with volume, risk_usd, rr_ratio
        sound         : if False, sound alerts are suppressed
        telegram      : if False, Telegram alerts are suppressed
        """
        try:
            # Fix U – signal_type is a plain string ("BUY"/"SELL"/"HOLD"),
            # not an Enum. Calling .name on a string raises AttributeError.
            direction = str(signal.signal_type).upper()

            entry      = float(signal.entry)
            sl         = float(position_spec.sl)
            tp         = float(position_spec.tp)
            sl_pips    = float(
                getattr(position_spec, "sl_pips",
                        getattr(signal, "sl_pips", 0.0))
            )
            tp_pips    = float(
                getattr(position_spec, "tp_pips",
                        getattr(signal, "tp_pips", 0.0))
            )
            confidence = float(
                getattr(position_spec, "confidence",
                        getattr(signal, "confidence", 0.0))
            )
            atr     = float(getattr(signal, "atr",     0.0))
            reasons = list(getattr(signal,  "reasons", []))

            from config.settings import CONFIG
            style = getattr(CONFIG, "SCALPER_TF_SELECTED", "scalper")
            style = "scalper" if str(style) in ("1", "5") else "day_trader"

        except Exception as exc:
            logger.warning(
                "[AM] signal_alert() field extraction failed: %s", exc
            )
            return

        if direction == "BUY":
            self.buy_signal(
                symbol=symbol, entry=entry, sl=sl, tp=tp,
                confidence=confidence, reasons=reasons,
                style=style, atr=atr,
                _sound=sound, _telegram=telegram,
            )
        elif direction == "SELL":
            self.sell_signal(
                symbol=symbol, entry=entry, sl=sl, tp=tp,
                confidence=confidence, reasons=reasons,
                style=style, atr=atr,
                _sound=sound, _telegram=telegram,
            )

    # ── BUY Signal ────────────────────────────────────────────────────────────

    def buy_signal(
        self,
        symbol:     str,
        entry:      float,
        sl:         float,
        tp:         float,
        confidence: float,
        reasons:    list,
        style:      str,
        atr:        float,
        _sound:     bool = True,
        _telegram:  bool = True,
    ) -> None:
        # [A] Confidence gate
        min_conf = _get_alert_min_confidence()
        if confidence < min_conf:
            logger.debug(
                "[AM] BUY alert suppressed — conf=%.2f < threshold=%.2f",
                confidence, min_conf,
            )
            return

        key      = f"buy_{symbol}_{self._price_key(entry)}"
        cooldown = _get_config_cooldown("ALERT_COOLDOWN_SIGNAL", 120)
        if self._already_alerted(key, minutes=cooldown):
            return

        # [M] Pip size and spread via connector
        pip_size = self._pip_size(symbol)
        pips_sl  = abs(entry - sl) / pip_size if pip_size else 0.0
        pips_tp  = abs(tp - entry) / pip_size if pip_size else 0.0
        tick     = self._current_tick(symbol)
        spread   = tick["spread_pips"] if tick else 0.0
        # Fix V – currency was fetched but never used in terminal output
        currency = self._get_currency()

        style_label = "SCALP" if style == "scalper" else "DAY"
        print(f"\n🟢 {'═' * 50}")
        print(f"  {style_label} BUY SIGNAL — {symbol}")
        print(f"{'═' * 52}")
        print(f"  Time       : {self._now()}")
        print(f"  Entry Now  : {entry:.5f}")
        print(f"  Stop Loss  : {sl:.5f}  ({pips_sl:.1f} pips)")
        print(f"  Take Profit: {tp:.5f}  ({pips_tp:.1f} pips)")
        print(f"  Spread     : {spread:.1f} pips")
        print(f"  Confidence : {confidence:.0%}  (threshold {min_conf:.0%})")
        print(f"  ATR        : {atr:.5f}")
        print(f"  Account    : {currency}")
        print(f"  Reasons:")
        for r in reasons[:5]:
            print(f"    {r}")
        print(f"{'═' * 52}\n")

        if _sound:
            self.sound.buy_signal()
        if _telegram:
            self._tg(
                self.telegram.buy_signal,
                symbol, entry, sl, tp,
                pips_sl, pips_tp,
                confidence, spread,
                style, reasons,
            )

        self._log_signal_to_dashboard(
            symbol, "BUY", entry, sl, tp, confidence, pips_sl, pips_tp
        )
        logger.info(
            "🟢 BUY Alert fired | %s | Entry:%.5f | "
            "SL:%.1fp TP:%.1fp | Conf:%.0f%%",
            symbol, entry, pips_sl, pips_tp, confidence * 100,
        )

    # ── SELL Signal ───────────────────────────────────────────────────────────

    def sell_signal(
        self,
        symbol:     str,
        entry:      float,
        sl:         float,
        tp:         float,
        confidence: float,
        reasons:    list,
        style:      str,
        atr:        float,
        _sound:     bool = True,
        _telegram:  bool = True,
    ) -> None:
        # [A] Confidence gate
        min_conf = _get_alert_min_confidence()
        if confidence < min_conf:
            logger.debug(
                "[AM] SELL alert suppressed — conf=%.2f < threshold=%.2f",
                confidence, min_conf,
            )
            return

        key      = f"sell_{symbol}_{self._price_key(entry)}"
        cooldown = _get_config_cooldown("ALERT_COOLDOWN_SIGNAL", 120)
        if self._already_alerted(key, minutes=cooldown):
            return

        # [M] Pip size and spread via connector
        pip_size = self._pip_size(symbol)
        pips_sl  = abs(sl - entry) / pip_size if pip_size else 0.0
        pips_tp  = abs(entry - tp) / pip_size if pip_size else 0.0
        tick     = self._current_tick(symbol)
        spread   = tick["spread_pips"] if tick else 0.0
        # Fix V – currency was fetched but never used in terminal output
        currency = self._get_currency()

        style_label = "SCALP" if style == "scalper" else "DAY"
        print(f"\n🔴 {'═' * 50}")
        print(f"  {style_label} SELL SIGNAL — {symbol}")
        print(f"{'═' * 52}")
        print(f"  Time       : {self._now()}")
        print(f"  Entry Now  : {entry:.5f}")
        print(f"  Stop Loss  : {sl:.5f}  ({pips_sl:.1f} pips)")
        print(f"  Take Profit: {tp:.5f}  ({pips_tp:.1f} pips)")
        print(f"  Spread     : {spread:.1f} pips")
        print(f"  Confidence : {confidence:.0%}  (threshold {min_conf:.0%})")
        print(f"  ATR        : {atr:.5f}")
        print(f"  Account    : {currency}")
        print(f"  Reasons:")
        for r in reasons[:5]:
            print(f"    {r}")
        print(f"{'═' * 52}\n")

        if _sound:
            self.sound.sell_signal()
        if _telegram:
            self._tg(
                self.telegram.sell_signal,
                symbol, entry, sl, tp,
                pips_sl, pips_tp,
                confidence, spread,
                style, reasons,
            )

        self._log_signal_to_dashboard(
            symbol, "SELL", entry, sl, tp, confidence, pips_sl, pips_tp
        )
        logger.info(
            "🔴 SELL Alert fired | %s | Entry:%.5f | "
            "SL:%.1fp TP:%.1fp | Conf:%.0f%%",
            symbol, entry, pips_sl, pips_tp, confidence * 100,
        )

    # ── Breakeven Moved ───────────────────────────────────────────────────────

    def breakeven_moved(
        self,
        symbol:    str,
        direction: str,
        ticket:    int,
        new_sl:    float,
    ) -> None:
        """[D] Added — was completely missing."""
        key = f"be_{ticket}"
        if self._already_alerted(key, minutes=60):
            return

        icon = "🟢" if direction == "BUY" else "🔴"
        print(f"\n🔒 {'─' * 48}")
        print(f"  BREAKEVEN LOCKED — {symbol} {direction}")
        print(f"{'─' * 50}")
        print(f"  Time    : {self._now()}")
        print(f"  Ticket  : #{ticket}")
        print(f"  New SL  : {new_sl:.5f}  (at breakeven + buffer)")
        print(f"  {icon} Trade is now risk-free")
        print(f"{'─' * 50}\n")

        self.sound.hold_alert()

        # Fix T – TelegramAlerts has no breakeven_moved() method.
        # Build and send the message directly via _send() so the
        # notification still reaches the trader.
        msg = (
            f"🔒 <b>BREAKEVEN LOCKED — {symbol}</b>\n"
            f"⏰ {self._now()}\n\n"
            f"Direction : {direction}\n"
            f"New SL    : {new_sl:.5f}\n"
            f"Ticket    : #{ticket}\n\n"
            f"{icon} Trade is now risk-free"
        )
        self._tg(self.telegram._send, msg)

        logger.info(
            "🔒 Breakeven alert | %s %s #%d | SL→%.5f",
            symbol, direction, ticket, new_sl,
        )

    # ── Hold Alert ────────────────────────────────────────────────────────────

    def hold_alert(
        self,
        symbol:    str,
        direction: str,
        ticket:    int,
        entry:     float,
        current:   float,
        pnl:       float,
        tp:        float,
    ) -> None:
        key      = f"hold_{ticket}"
        cooldown = _get_config_cooldown("ALERT_COOLDOWN_HOLD", 120)
        if self._already_alerted(key, minutes=cooldown):
            return

        pip_size  = self._pip_size(symbol)
        pips      = (current - entry) / pip_size if pip_size else 0.0
        pips_left = abs(tp - current) / pip_size if pip_size else 0.0
        icon      = "🟢" if pnl >= 0 else "🔴"
        currency  = self._get_currency()

        print(f"\n⏸️  {'─' * 48}")
        print(f"  HOLD — {symbol} {direction} (#{ticket})")
        print(f"{'─' * 50}")
        print(f"  Opened  : {entry:.5f}")
        print(f"  Current : {current:.5f}")
        print(f"  P&L     : {icon} {pips:+.1f} pips  ({currency}{pnl:+.2f})")
        print(f"  Target  : {tp:.5f}  ({pips_left:.1f} pips left)")
        print(f"  ✅ Signal still valid — hold position")
        print(f"{'─' * 50}\n")

        self.sound.hold_alert()
        self._tg(
            self.telegram.hold_alert,
            symbol, direction, ticket,
            entry, current, pnl, pips, tp, pips_left,
        )

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
        key      = f"exit_{ticket}"
        cooldown = _get_config_cooldown("ALERT_COOLDOWN_EXIT", 120)
        if self._already_alerted(key, minutes=cooldown):
            return

        pip_size   = self._pip_size(symbol)
        pips       = abs(current - entry) / pip_size if pip_size else 0.0
        pips_to_tp = abs(tp - current)    / pip_size if pip_size else 0.0
        currency   = self._get_currency()

        print(f"\n🟡 {'═' * 50}")
        print(f"  CONSIDER EXIT — {symbol} {direction}")
        print(f"{'═' * 52}")
        print(f"  Time       : {self._now()}")
        print(f"  Opened     : {entry:.5f}")
        print(f"  Current    : {current:.5f}")
        print(f"  Profit     : +{pips:.1f} pips  ({currency}{pnl:+.2f})")
        print(f"  TP         : {tp:.5f}  ({pips_to_tp:.1f} pips away)")
        print(f"  Near TP:")
        for r in reasons:
            print(f"    ⚠️  {r}")
        print(f"  💡 Options:")
        print(f"     A) Close now  → lock profit")
        print(f"     B) Move SL up → protect gains")
        print(f"     C) Hold       → risk last pips")
        print(f"{'═' * 52}\n")

        self.sound.potential_exit()
        self._tg(
            self.telegram.potential_exit,
            symbol, direction, ticket,
            entry, current, pnl, tp, reasons,
        )

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
        key      = f"danger_{ticket}"
        cooldown = _get_config_cooldown("ALERT_COOLDOWN_DANGER", 30)
        if self._already_alerted(key, minutes=cooldown):
            return

        pip_size   = self._pip_size(symbol)
        pips       = (current - entry) / pip_size if pip_size else 0.0
        pips_to_sl = abs(current - sl)  / pip_size if pip_size else 0.0
        currency   = self._get_currency()

        print(f"\n🚨 {'═' * 50}")
        print(f"  ⚠️  DANGER — CONSIDER CLOSING {symbol} {direction}")
        print(f"{'═' * 52}")
        print(f"  Time       : {self._now()}")
        print(f"  Opened     : {entry:.5f}")
        print(f"  Current    : {current:.5f}")
        print(f"  P&L        : 🔴 {pips:.1f} pips  ({currency}{pnl:.2f})")
        print(f"  SL         : {sl:.5f}  ({pips_to_sl:.1f} pips away!)")
        print(f"  Danger signals:")
        for r in reasons:
            print(f"    ❌ {r}")
        print(f"\n  ⚡ URGENT — Consider closing NOW")
        print(f"{'═' * 52}\n")

        self.sound.danger_exit()
        # Danger bypasses _tg() wrapper — always sends regardless of
        # quiet hours, matching TelegramAlerts.danger_exit() behaviour.
        try:
            self.telegram.danger_exit(
                symbol, direction, ticket,
                entry, current, pnl,
                sl, reasons,
            )
        except Exception as exc:
            logger.warning("[AM] Telegram danger_exit error: %s", exc)

        logger.warning(
            "🚨 DANGER Exit | %s %s #%d | P&L:%s%.2f | "
            "SL:%.5f (%.1fp away)",
            symbol, direction, ticket, currency, pnl, sl, pips_to_sl,
        )

    # ── Risk Warning ──────────────────────────────────────────────────────────

    def risk_warning(
        self,
        level:       str,
        message:     str,
        daily_pnl:   float = 0.0,
        daily_limit: float = 0.0,
        pct_used:    float = 0.0,
        open_trades: int   = 0,
        max_trades:  int   = 0,
    ) -> None:
        cooldown = (
            _get_config_cooldown("ALERT_COOLDOWN_DANGER", 30)
            if level in ("LIMIT_HIT", "URGENT")
            else 10.0
        )
        key = f"risk_{level}"
        if self._already_alerted(key, minutes=cooldown):
            return

        currency = self._get_currency()
        icons    = {
            "WARNING":     "⚠️",
            "URGENT":      "⛔",
            "LIMIT_HIT":   "🚫",
            "TRADES_NEAR": "⚠️",
            "TRADES_FULL": "🚫",
            "LOW_MARGIN":  "⚠️",
        }
        icon = icons.get(level, "⚠️")

        print(f"\n{icon} {'═' * 50}")
        print(f"  RISK ALERT — {level.replace('_', ' ')}")
        print(f"{'═' * 52}")
        print(f"  Time : {self._now()}")

        if level in ("WARNING", "URGENT", "LIMIT_HIT"):
            bar_filled = int(min(pct_used, 100) / 10)
            bar        = "█" * bar_filled + "░" * (10 - bar_filled)
            print(
                f"  Daily Loss : {currency}{abs(daily_pnl):.2f} / "
                f"{currency}{daily_limit:.2f}"
            )
            print(f"  Used       : [{bar}] {pct_used:.0f}%")

        if level in ("TRADES_NEAR", "TRADES_FULL"):
            print(f"  Trades     : {open_trades} / {max_trades} slots used")

        print(f"  {message}")
        print(f"{'═' * 52}\n")

        if level in ("LIMIT_HIT", "URGENT", "TRADES_FULL"):
            self.sound.danger_exit()
        else:
            self.sound.news_warning()

        self._tg(
            self.telegram.risk_warning,
            level       = level,
            message     = message,
            daily_pnl   = daily_pnl,
            daily_limit = daily_limit,
            pct_used    = pct_used,
            open_trades = open_trades,
            max_trades  = max_trades,
        )
        logger.warning(
            "🛡 Risk Alert | Level:%s | DailyPnL:%s%+.2f | Used:%.0f%%",
            level, currency, daily_pnl, pct_used,
        )

    # ── News Warning ──────────────────────────────────────────────────────────

    def news_warning(
        self,
        event:     str,
        currency:  str,
        mins_away: float,
    ) -> None:
        key = f"news_{event}"
        if self._already_alerted(key, minutes=60):
            return

        local_time = self._now_aware()
        event_time = local_time + timedelta(minutes=int(mins_away))
        time_str   = event_time.strftime("%H:%M")

        print(f"\n📰 {'═' * 50}")
        print(f"  ⚠️  HIGH IMPACT NEWS IN {mins_away:.0f} MINUTES")
        print(f"{'═' * 52}")
        print(f"  Event    : {event}")
        print(f"  Currency : {currency}")
        print(f"  Time     : {time_str} Madrid/CET")
        print(f"  Impact   : 🔴 HIGH")
        print(f"  Affects  : EURUSD directly")
        print(f"  ⚡ Signals paused until 15 mins after event")
        print(f"  💡 Consider closing open trades before release")
        print(f"{'═' * 52}\n")

        self.sound.news_warning()
        self._tg(
            self.telegram.news_warning,
            event, currency, time_str,
            mins_away, ["EURUSD"],
        )

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
        icon     = "🟢" if direction == "BUY" else "🔴"
        pip_size = self._pip_size(symbol)
        pips_sl  = abs(entry - sl) / pip_size if pip_size else 0.0
        pips_tp  = abs(tp - entry) / pip_size if pip_size else 0.0
        currency = self._get_currency()

        print(f"\n{icon} {'─' * 48}")
        print(f"  ✅ TRADE OPENED — {symbol} {direction}")
        print(f"{'─' * 50}")
        print(f"  Ticket  : #{ticket}")
        print(f"  Entry   : {entry:.5f}")
        print(f"  SL      : {sl:.5f}  ({pips_sl:.1f} pips)")
        print(f"  TP      : {tp:.5f}  ({pips_tp:.1f} pips)")
        print(
            f"  RR      : 1:{pips_tp/pips_sl:.2f}"
            if pips_sl > 0 else "  RR      : N/A"
        )
        print(f"  Volume  : {volume} lots")
        print(f"  Risk    : {currency}{risk:.2f}")
        print(f"  Mode    : {mode}  |  Style: {style}")
        print(f"{'─' * 50}\n")

        self.sound.trade_opened()

        # Fix S – sl_pips and tp_pips were completely absent from this call.
        # TelegramAlerts.trade_opened() requires them — was TypeError at
        # runtime on every trade open.
        self._tg(
            self.telegram.trade_opened,
            symbol, direction, ticket,
            entry, sl, tp,
            pips_sl, pips_tp,
            volume, risk, style, mode,
        )

        logger.info(
            "✅ Trade opened | %s %s #%d | Entry:%.5f "
            "SL:%.1fp TP:%.1fp | Vol:%s Risk:%s%.2f | Style:%s Mode:%s",
            symbol, direction, ticket, entry,
            pips_sl, pips_tp, volume, currency, risk, style, mode,
        )

    # ── Trade Closed ──────────────────────────────────────────────────────────

    def trade_closed(
        self,
        symbol:      str,
        direction:   str,
        ticket:      int,
        entry:       float,
        close_price: float,   # Fix W – renamed from close to avoid shadowing
        pnl:         float,
        pips:        float,
        reason:      str = "Unknown",
    ) -> None:
        icon     = "✅" if pnl >= 0 else "❌"
        currency = self._get_currency()

        print(f"\n{icon} {'─' * 48}")
        print(f"  TRADE CLOSED — {symbol} {direction}")
        print(f"{'─' * 50}")
        print(f"  Ticket  : #{ticket}")
        print(f"  Opened  : {entry:.5f}")
        print(f"  Closed  : {close_price:.5f}")
        print(f"  Result  : {pips:+.1f} pips  ({currency}{pnl:+.2f})")
        print(f"  Reason  : {reason}")
        print(f"{'─' * 50}\n")

        if pnl >= 0:
            self.sound.trade_closed_profit()
        else:
            self.sound.trade_closed_loss()

        self._tg(
            self.telegram.trade_closed,
            symbol, direction, ticket,
            entry, close_price, pnl, pips, reason,
        )
        logger.info(
            "%s Trade closed | %s %s #%d | %+.1fp %s%+.2f | [%s]",
            icon, symbol, direction, ticket, pips, currency, pnl, reason,
        )

    # ── Daily Summary ─────────────────────────────────────────────────────────

    def daily_summary(
        self,
        trades:       int,
        winners:      int,
        losers:       int,
        net_pnl:      float,
        win_rate:     float,
        signals:      int,
        best_trade:   float,
        worst_trade:  float,
        gross_profit: float,
        gross_loss:   float,
    ) -> None:
        from datetime import date as _date
        today    = _date.today().strftime("%A %d %b %Y")
        currency = self._get_currency()

        balance = 0.0
        try:
            info    = self._connector.get_account_info()
            balance = info.get("balance", 0.0) if info else 0.0
        except Exception:
            pass

        g_loss_abs = abs(gross_loss)
        pf_str     = (
            f"{gross_profit / g_loss_abs:.3f}"
            if g_loss_abs > 0 else "N/A"
        )
        pnl_icon = "📈" if net_pnl >= 0 else "📉"

        print(f"\n📊 {'═' * 50}")
        print(f"  DAILY SUMMARY — {today}")
        print(f"{'═' * 52}")
        print(f"  Signals     : {signals}")
        print(f"  Trades      : {trades}")
        print(f"  Winners     : {winners} ✅")
        print(f"  Losers      : {losers} ❌")
        print(f"  Win Rate    : {win_rate:.1f}%")
        print(f"  Gross P     : {currency}{gross_profit:+.2f}")
        print(f"  Gross L     : {currency}{gross_loss:+.2f}")
        print(f"  Profit Fact : {pf_str}")
        print(f"  Net P&L     : {pnl_icon} {currency}{net_pnl:+.2f}")
        print(f"  Best Trade  : {currency}{best_trade:+.2f}")
        print(f"  Worst Trade : {currency}{worst_trade:+.2f}")
        print(f"  Balance     : {currency}{balance:,.2f}")
        print(f"{'═' * 52}\n")

        if net_pnl >= 0:
            self.sound.trade_closed_profit()
        else:
            self.sound.trade_closed_loss()

        self._tg(
            self.telegram.daily_summary,
            signals      = signals,
            trades       = trades,
            winners      = winners,
            losers       = losers,
            net_pnl      = net_pnl,
            win_rate     = win_rate,
            best_trade   = best_trade,
            worst_trade  = worst_trade,
            gross_profit = gross_profit,
            gross_loss   = gross_loss,
            balance      = balance,
            date         = today,
        )
        logger.info(
            "📊 Daily Summary | Trades:%d | W:%d L:%d | "
            "Net:%s%+.2f | WR:%.1f%%",
            trades, winners, losers, currency, net_pnl, win_rate,
        )

    # ── System Online ─────────────────────────────────────────────────────────

    def system_online(
        self,
        balance: float,
        style:   str,
        mode:    str,
    ) -> None:
        mode_names = {"1": "Signal Only", "2": "Semi-Auto", "3": "Full Auto"}
        mode_label = mode_names.get(str(mode), f"Mode {mode}")
        currency   = self._get_currency()

        # Fix X – Read symbols and timeframe from CONFIG instead of
        # hard-coding "EURUSD | M5" which would be wrong if config changes.
        try:
            from config.settings import CONFIG
            symbols_str = ", ".join(CONFIG.SYMBOLS) if CONFIG.SYMBOLS else "—"
            tf_str      = str(
                getattr(CONFIG, "SCALPER_TF_SELECTED", "M5")
            )
        except Exception:
            symbols_str = "EURUSD"
            tf_str      = "M5"

        print(f"\n{'═' * 52}")
        print(f"  🤖 GODBOT v3.0 — SYSTEM ONLINE")
        print(f"{'═' * 52}")
        print(f"  Time    : {self._now()}")
        print(f"  Balance : {currency}{balance:,.2f}")
        print(f"  Style   : {style.capitalize()}")
        print(f"  Mode    : {mode_label}")
        print(f"  Symbols : {symbols_str}  |  TF: {tf_str}")
        print(f"  ✅ All systems go — scanning markets")
        print(f"{'═' * 52}\n")

        self.sound.system_ready()
        self._tg(self.telegram.system_online, balance, style, mode)
        logger.info(
            "🤖 System online | Balance:%s%s | Style:%s | Mode:%s",
            currency, f"{balance:,.2f}", style, mode_label,
        )

    # ── System Offline ────────────────────────────────────────────────────────

    def system_offline(self, reason: str = "Manual shutdown") -> None:
        print(f"\n{'═' * 52}")
        print(f"  🔴 GODBOT v3.0 — SYSTEM OFFLINE")
        print(f"{'═' * 52}")
        print(f"  Time   : {self._now()}")
        print(f"  Reason : {reason}")
        print(f"  ✅ Shutdown complete")
        print(f"{'═' * 52}\n")

        # [Q] system_offline always sends regardless of quiet hours —
        # a shutdown notification is always operationally important.
        try:
            self.telegram.system_offline(reason)
        except Exception as exc:
            logger.warning("[AM] Telegram system_offline error: %s", exc)

        logger.info("🔴 System offline | Reason: %s", reason)
