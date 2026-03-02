# notifications/alert_manager.py
import MetaTrader5 as mt5
import pytz
from datetime import datetime
from typing import Optional
from notifications.sound_alerts   import SoundAlerts
from notifications.telegram_alerts import TelegramAlerts
from monitoring.logger             import get_logger

logger = get_logger("AlertManager")


class AlertManager:
    """
    Master alert coordinator.
    Every alert in the system goes through here.
    Fires Terminal + Sound + Telegram simultaneously.
    Never duplicates the same alert twice.
    """

    def __init__(self):
        self.sound    = SoundAlerts()
        self.telegram = TelegramAlerts()
        self.timezone = pytz.timezone("Europe/Madrid")
        self._alerted: dict = {}  # Tracks recent alerts

    # ── Time Helper ───────────────────────────────────────
    def _now(self) -> str:
        return datetime.now(self.timezone).strftime(
            "%H:%M — %d %b"
        )

    # ── Duplicate Guard ───────────────────────────────────
    def _already_alerted(
        self,
        key:     str,
        minutes: int = 5
    ) -> bool:
        """Prevents same alert firing repeatedly."""
        if key in self._alerted:
            age = (
                datetime.now() - self._alerted[key]
            ).seconds / 60
            if age < minutes:
                return True
        self._alerted[key] = datetime.now()
        return False

    # ── BUY Signal ────────────────────────────────────────
    def buy_signal(
        self,
        symbol:     str,
        entry:      float,
        sl:         float,
        tp:         float,
        confidence: float,
        reasons:    list,
        style:      str,
        atr:        float
    ):
        key = f"buy_{symbol}_{entry}"
        if self._already_alerted(key):
            return

        sym_info = mt5.symbol_info(symbol)
        point    = sym_info.point if sym_info else 0.00001
        pips_sl  = abs(entry - sl)  / point / 10
        pips_tp  = abs(tp - entry)  / point / 10
        tick     = mt5.symbol_info_tick(symbol)
        spread   = (
            (tick.ask - tick.bid) / point / 10
            if tick else 0
        )

        # ── Terminal ──────────────────────────────────────
        print(f"\n🟢 {'═'*50}")
        print(
            f"  {'SCALP' if style == 'scalper' else 'DAY'}"
            f" BUY SIGNAL — {symbol}"
        )
        print(f"{'═'*52}")
        print(f"  Time       : {self._now()}")
        print(f"  Entry Now  : {entry}")
        print(f"  Stop Loss  : {sl} ({pips_sl:.1f} pips)")
        print(f"  Take Profit: {tp} ({pips_tp:.1f} pips)")
        print(f"  Spread     : {spread:.1f} pips ✅")
        print(f"  Confidence : {confidence:.0%}")
        print(f"  Reasons:")
        for r in reasons[:4]:
            print(f"    {r}")
        print(f"{'═'*52}\n")

        # ── Sound ─────────────────────────────────────────
        self.sound.buy_signal()

        # ── Telegram ──────────────────────────────────────
        self.telegram.buy_signal(
            symbol, entry, sl, tp,
            pips_sl, pips_tp,
            confidence, spread,
            style, reasons
        )

        logger.info(
            f"🟢 BUY Alert fired | {symbol} | "
            f"Entry:{entry} | Conf:{confidence:.0%}"
        )

    # ── SELL Signal ───────────────────────────────────────
    def sell_signal(
        self,
        symbol:     str,
        entry:      float,
        sl:         float,
        tp:         float,
        confidence: float,
        reasons:    list,
        style:      str,
        atr:        float
    ):
        key = f"sell_{symbol}_{entry}"
        if self._already_alerted(key):
            return

        sym_info = mt5.symbol_info(symbol)
        point    = sym_info.point if sym_info else 0.00001
        pips_sl  = abs(sl - entry) / point / 10
        pips_tp  = abs(entry - tp) / point / 10
        tick     = mt5.symbol_info_tick(symbol)
        spread   = (
            (tick.ask - tick.bid) / point / 10
            if tick else 0
        )

        # ── Terminal ──────────────────────────────────────
        print(f"\n🔴 {'═'*50}")
        print(
            f"  {'SCALP' if style == 'scalper' else 'DAY'}"
            f" SELL SIGNAL — {symbol}"
        )
        print(f"{'═'*52}")
        print(f"  Time       : {self._now()}")
        print(f"  Entry Now  : {entry}")
        print(f"  Stop Loss  : {sl} ({pips_sl:.1f} pips)")
        print(f"  Take Profit: {tp} ({pips_tp:.1f} pips)")
        print(f"  Spread     : {spread:.1f} pips ✅")
        print(f"  Confidence : {confidence:.0%}")
        print(f"  Reasons:")
        for r in reasons[:4]:
            print(f"    {r}")
        print(f"{'═'*52}\n")

        # ── Sound ─────────────────────────────────────────
        self.sound.sell_signal()

        # ── Telegram ──────────────────────────────────────
        self.telegram.sell_signal(
            symbol, entry, sl, tp,
            pips_sl, pips_tp,
            confidence, spread,
            style, reasons
        )

        logger.info(
            f"🔴 SELL Alert fired | {symbol} | "
            f"Entry:{entry} | Conf:{confidence:.0%}"
        )

    # ── Hold Alert ────────────────────────────────────────
    def hold_alert(
        self,
        symbol:    str,
        direction: str,
        ticket:    int,
        entry:     float,
        current:   float,
        pnl:       float,
        tp:        float
    ):
        key = f"hold_{ticket}"
        if self._already_alerted(key, minutes=15):
            return

        sym_info  = mt5.symbol_info(symbol)
        point     = sym_info.point if sym_info else 0.00001
        pips      = (current - entry) / point / 10
        pips_left = abs(tp - current) / point / 10
        icon      = "🟢" if pnl >= 0 else "🔴"

        # ── Terminal ──────────────────────────────────────
        print(f"\n⏸️  {'─'*48}")
        print(f"  HOLD — {symbol} {direction} (#{ticket})")
        print(f"{'─'*50}")
        print(f"  Opened  : {entry}")
        print(f"  Current : {current}")
        print(
            f"  P&L     : {icon} {pips:+.1f} pips "
            f"(€{pnl:+.2f})"
        )
        print(f"  Target  : {tp} ({pips_left:.1f} pips left)")
        print(f"  ✅ Signal still valid — hold position")
        print(f"{'─'*50}\n")

        # ── Sound ─────────────────────────────────────────
        self.sound.hold_alert()

        # ── Telegram ──────────────────────────────────────
        self.telegram.hold_alert(
            symbol, direction, ticket,
            entry, current, pnl, pips, tp, pips_left
        )

    # ── Potential Exit ────────────────────────────────────
    def potential_exit(
        self,
        symbol:    str,
        direction: str,
        ticket:    int,
        entry:     float,
        current:   float,
        pnl:       float,
        reasons:   list
    ):
        key = f"exit_{ticket}"
        if self._already_alerted(key, minutes=10):
            return

        sym_info = mt5.symbol_info(symbol)
        point    = sym_info.point if sym_info else 0.00001
        pips     = abs(current - entry) / point / 10

        # ── Terminal ──────────────────────────────────────
        print(f"\n🟡 {'═'*50}")
        print(f"  CONSIDER EXIT — {symbol} {direction}")
        print(f"{'═'*52}")
        print(f"  Time    : {self._now()}")
        print(f"  Opened  : {entry}")
        print(f"  Current : {current}")
        print(f"  Profit  : +{pips:.1f} pips (€{pnl:+.2f})")
        print(f"  Weakening signals:")
        for r in reasons:
            print(f"    ⚠️  {r}")
        print(f"  💡 Options:")
        print(f"     A) Close now  → lock profit")
        print(f"     B) Move SL up → protect gains")
        print(f"     C) Hold       → risk last pips")
        print(f"{'═'*52}\n")

        # ── Sound ─────────────────────────────────────────
        self.sound.potential_exit()

        # ── Telegram ──────────────────────────────────────
        self.telegram.potential_exit(
            symbol, direction, ticket,
            entry, current, pnl, pips, reasons
        )

    # ── Danger Exit ───────────────────────────────────────
    def danger_exit(
        self,
        symbol:    str,
        direction: str,
        ticket:    int,
        entry:     float,
        current:   float,
        pnl:       float,
        sl:        float,
        reasons:   list
    ):
        key = f"danger_{ticket}"
        if self._already_alerted(key, minutes=3):
            return

        sym_info   = mt5.symbol_info(symbol)
        point      = sym_info.point if sym_info else 0.00001
        pips       = (current - entry) / point / 10
        pips_to_sl = abs(current - sl) / point / 10

        # ── Terminal ──────────────────────────────────────
        print(f"\n🚨 {'═'*50}")
        print(
            f"  ⚠️  DANGER — CONSIDER CLOSING "
            f"{symbol} {direction}"
        )
        print(f"{'═'*52}")
        print(f"  Time       : {self._now()}")
        print(f"  Opened     : {entry}")
        print(f"  Current    : {current}")
        print(
            f"  P&L        : 🔴 {pips:.1f} pips "
            f"(€{pnl:.2f})"
        )
        print(
            f"  SL         : {sl} "
            f"({pips_to_sl:.1f} pips away!)"
        )
        print(f"  Danger signals:")
        for r in reasons:
            print(f"    ❌ {r}")
        print(
            f"\n  ⚡ URGENT — Consider closing NOW"
        )
        print(f"{'═'*52}\n")

        # ── Sound ─────────────────────────────────────────
        self.sound.danger_exit()

        # ── Telegram ──────────────────────────────────────
        self.telegram.danger_exit(
            symbol, direction, ticket,
            entry, current, pnl, pips,
            sl, pips_to_sl, reasons
        )

    # ── News Warning ──────────────────────────────────────
    def news_warning(
        self,
        event:     str,
        currency:  str,
        mins_away: float
    ):
        key = f"news_{event}"
        if self._already_alerted(key, minutes=60):
            return

        local_time = datetime.now(self.timezone)
        event_time = local_time.replace(
            minute=local_time.minute +
            int(mins_away)
        )
        time_str   = event_time.strftime("%H:%M")

        # ── Terminal ──────────────────────────────────────
        print(f"\n📰 {'═'*50}")
        print(f"  ⚠️  HIGH IMPACT NEWS IN {mins_away:.0f} MINUTES")
        print(f"{'═'*52}")
        print(f"  Event    : {event}")
        print(f"  Currency : {currency}")
        print(f"  Time     : {time_str} your time")
        print(f"  Impact   : 🔴 HIGH")
        print(f"  Affects  : EURUSD directly")
        print(f"  ⚡ Signals paused until 15 mins after")
        print(f"  💡 Consider closing open trades")
        print(f"{'═'*52}\n")

        # ── Sound ─────────────────────────────────────────
        self.sound.news_warning()

        # ── Telegram ──────────────────────────────────────
        self.telegram.news_warning(
            event, currency, time_str,
            mins_away, ["EURUSD"]
        )

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
        icon = "🟢" if direction == "BUY" else "🔴"

        # ── Terminal ──────────────────────────────────────
        print(f"\n{icon} {'─'*48}")
        print(f"  ✅ TRADE OPENED — {symbol} {direction}")
        print(f"{'─'*50}")
        print(f"  Ticket  : #{ticket}")
        print(f"  Entry   : {entry}")
        print(f"  SL      : {sl}")
        print(f"  TP      : {tp}")
        print(f"  Volume  : {volume} lots")
        print(f"  Risk    : €{risk:.2f}")
        print(f"{'─'*50}\n")

        # ── Sound ─────────────────────────────────────────
        self.sound.trade_opened()

        # ── Telegram ──────────────────────────────────────
        self.telegram.trade_opened(
            symbol, direction, ticket,
            entry, sl, tp, volume,
            risk, style, mode
        )

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
        # ── Terminal ──────────────────────────────────────
        icon   = "✅" if pnl >= 0 else "❌"
        print(f"\n{icon} {'─'*48}")
        print(f"  TRADE CLOSED — {symbol} {direction}")
        print(f"{'─'*50}")
        print(f"  Ticket  : #{ticket}")
        print(f"  Opened  : {entry}")
        print(f"  Closed  : {close}")
        print(
            f"  Result  : {pips:+.1f} pips "
            f"(€{pnl:+.2f})"
        )
        print(f"  Reason  : {reason}")
        print(f"{'─'*50}\n")

        # ── Sound ─────────────────────────────────────────
        if pnl >= 0:
            self.sound.trade_closed_profit()
        else:
            self.sound.trade_closed_loss()

        # ── Telegram ──────────────────────────────────────
        self.telegram.trade_closed(
            symbol, direction, ticket,
            entry, close, pnl, pips, reason
        )

    # ── System Online ─────────────────────────────────────
    def system_online(
        self,
        balance: float,
        style:   str,
        mode:    str
    ):
        self.sound.system_ready()
        self.telegram.system_online(balance, style, mode)

    # ── System Offline ────────────────────────────────────
    def system_offline(self, reason: str = "Manual shutdown"):
        self.telegram.system_offline(reason)