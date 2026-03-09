# notifications/alert_manager.py
import MetaTrader5 as mt5
import pytz
from datetime import datetime, timedelta
from typing import Optional
from notifications.sound_alerts    import SoundAlerts
from notifications.telegram_alerts import TelegramAlerts
from monitoring.logger              import get_logger

logger = get_logger("AlertManager")


class AlertManager:
    """
    Master alert coordinator.
    Every alert in the system goes through here.
    Fires Terminal + Sound + Telegram simultaneously.
    Never duplicates the same alert twice.

    FIX — Duplicate alert deduplication key:
        buy_signal() and sell_signal() previously keyed on the raw
        entry price (e.g. "buy_EURUSD_1.08423").  A 1-pip move between
        two calls for the same signal produced a different key and
        bypassed the cooldown, firing duplicate alerts.

        Fix:
          - Entry price is rounded to DEDUP_PRICE_DIGITS decimal places
            before inclusion in the key so small price movements within
            the same pip bucket are treated as the same alert.
          - DEDUP_PRICE_DIGITS = 3 groups prices within a 0.001 bucket
            (~10 pips for major FX pairs) which is comfortably within
            any reasonable cooldown window.
          - Direction is preserved in the key so a genuine signal
            reversal (BUY → SELL at the same rounded price) still fires.

    FIX — Mixed naive/aware datetimes in _already_alerted():
        The original stored naive datetime.now() in self._alerted but
        the rest of the class used timezone-aware datetimes via
        self.timezone.  On machines where the local timezone is not
        Madrid this caused silent comparison errors.  All datetime
        usage in _already_alerted() is now consistently timezone-aware
        (Europe/Madrid).
    """

    # ── Deduplication precision ───────────────────────────────────────────────
    # Entry price is rounded to this many decimal places when building
    # the dedup key.  3 d.p. = 0.001 bucket (~10 pips for major pairs).
    # Increase to 4 for tighter dedup (1-pip bucket) if preferred.
    DEDUP_PRICE_DIGITS = 3

    def __init__(self):
        self.sound    = SoundAlerts()
        self.telegram = TelegramAlerts()
        self.timezone = pytz.timezone("Europe/Madrid")
        self._alerted: dict = {}

    # ── Time Helper ───────────────────────────────────────────────────────────
    def _now(self) -> str:
        return datetime.now(self.timezone).strftime("%H:%M — %d %b")

    def _now_aware(self) -> datetime:
        """
        FIX — Returns the current time as a timezone-aware datetime
        (Europe/Madrid) for consistent use inside _already_alerted().
        Replaces bare datetime.now() which returned a naive datetime
        and caused silent comparison failures on non-Madrid systems.
        """
        return datetime.now(pytz.utc).astimezone(self.timezone)

    # ── Duplicate Guard ───────────────────────────────────────────────────────
    def _already_alerted(self, key: str, minutes: int = 5) -> bool:
        """
        Prevents the same alert firing repeatedly within cooldown window.

        FIX — Now uses timezone-aware datetimes consistently so the age
        calculation is correct regardless of the host machine's local
        timezone setting.
        """
        if key in self._alerted:
            age = (self._now_aware() - self._alerted[key]).total_seconds() / 60
            if age < minutes:
                return True
        # Record or refresh the timestamp for this key
        self._alerted[key] = self._now_aware()
        return False

    # ── Dedup Key Helper ──────────────────────────────────────────────────────
    def _price_key(self, price: float) -> str:
        """
        FIX — Round price to DEDUP_PRICE_DIGITS decimal places so minor
        tick-level movements between successive calls for the same signal
        do not produce a different dedup key and bypass the cooldown.

        Example with DEDUP_PRICE_DIGITS = 3:
            1.08423  →  "1.084"
            1.08431  →  "1.084"   (same key — cooldown applies)
            1.08501  →  "1.085"   (different key — genuine new level)
        """
        return str(round(price, self.DEDUP_PRICE_DIGITS))

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
    ):
        # FIX: key uses rounded price so 1-pip movements don't bypass cooldown
        key = f"buy_{symbol}_{self._price_key(entry)}"
        if self._already_alerted(key):
            return

        sym_info = mt5.symbol_info(symbol)
        point    = sym_info.point if sym_info else 0.00001
        pips_sl  = abs(entry - sl) / point / 10
        pips_tp  = abs(tp - entry) / point / 10
        tick     = mt5.symbol_info_tick(symbol)
        spread   = (tick.ask - tick.bid) / point / 10 if tick else 0

        print(f"\n🟢 {'═'*50}")
        print(f"  {'SCALP' if style == 'scalper' else 'DAY'} BUY SIGNAL — {symbol}")
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

        self.sound.buy_signal()
        self.telegram.buy_signal(
            symbol, entry, sl, tp,
            pips_sl, pips_tp,
            confidence, spread,
            style, reasons,
        )
        logger.info(
            f"🟢 BUY Alert fired | {symbol} | "
            f"Entry:{entry} | Conf:{confidence:.0%}"
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
    ):
        # FIX: key uses rounded price so 1-pip movements don't bypass cooldown
        key = f"sell_{symbol}_{self._price_key(entry)}"
        if self._already_alerted(key):
            return

        sym_info = mt5.symbol_info(symbol)
        point    = sym_info.point if sym_info else 0.00001
        pips_sl  = abs(sl - entry) / point / 10
        pips_tp  = abs(entry - tp) / point / 10
        tick     = mt5.symbol_info_tick(symbol)
        spread   = (tick.ask - tick.bid) / point / 10 if tick else 0

        print(f"\n🔴 {'═'*50}")
        print(f"  {'SCALP' if style == 'scalper' else 'DAY'} SELL SIGNAL — {symbol}")
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

        self.sound.sell_signal()
        self.telegram.sell_signal(
            symbol, entry, sl, tp,
            pips_sl, pips_tp,
            confidence, spread,
            style, reasons,
        )
        logger.info(
            f"🔴 SELL Alert fired | {symbol} | "
            f"Entry:{entry} | Conf:{confidence:.0%}"
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
    ):
        key = f"hold_{ticket}"
        if self._already_alerted(key, minutes=15):
            return

        sym_info  = mt5.symbol_info(symbol)
        point     = sym_info.point if sym_info else 0.00001
        pips      = (current - entry) / point / 10
        pips_left = abs(tp - current) / point / 10
        icon      = "🟢" if pnl >= 0 else "🔴"

        print(f"\n⏸️  {'─'*48}")
        print(f"  HOLD — {symbol} {direction} (#{ticket})")
        print(f"{'─'*50}")
        print(f"  Opened  : {entry}")
        print(f"  Current : {current}")
        print(f"  P&L     : {icon} {pips:+.1f} pips (€{pnl:+.2f})")
        print(f"  Target  : {tp} ({pips_left:.1f} pips left)")
        print(f"  ✅ Signal still valid — hold position")
        print(f"{'─'*50}\n")

        self.sound.hold_alert()
        self.telegram.hold_alert(
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
    ):
        key = f"exit_{ticket}"
        if self._already_alerted(key, minutes=10):
            return

        sym_info   = mt5.symbol_info(symbol)
        point      = sym_info.point if sym_info else 0.00001
        pips       = abs(current - entry) / point / 10
        pips_to_tp = abs(tp - current) / point / 10

        print(f"\n🟡 {'═'*50}")
        print(f"  CONSIDER EXIT — {symbol} {direction}")
        print(f"{'═'*52}")
        print(f"  Time       : {self._now()}")
        print(f"  Opened     : {entry}")
        print(f"  Current    : {current}")
        print(f"  Profit     : +{pips:.1f} pips (€{pnl:+.2f})")
        print(f"  TP         : {tp} ({pips_to_tp:.1f} pips away)")
        print(f"  Near TP:")
        for r in reasons:
            print(f"    ⚠️  {r}")
        print(f"  💡 Options:")
        print(f"     A) Close now  → lock profit")
        print(f"     B) Move SL up → protect gains")
        print(f"     C) Hold       → risk last pips")
        print(f"{'═'*52}\n")

        self.sound.potential_exit()
        self.telegram.potential_exit(
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
    ):
        key = f"danger_{ticket}"
        if self._already_alerted(key, minutes=3):
            return

        sym_info   = mt5.symbol_info(symbol)
        point      = sym_info.point if sym_info else 0.00001
        pips       = (current - entry) / point / 10
        pips_to_sl = abs(current - sl) / point / 10

        print(f"\n🚨 {'═'*50}")
        print(f"  ⚠️  DANGER — CONSIDER CLOSING {symbol} {direction}")
        print(f"{'═'*52}")
        print(f"  Time       : {self._now()}")
        print(f"  Opened     : {entry}")
        print(f"  Current    : {current}")
        print(f"  P&L        : 🔴 {pips:.1f} pips (€{pnl:.2f})")
        print(f"  SL         : {sl} ({pips_to_sl:.1f} pips away!)")
        print(f"  Danger signals:")
        for r in reasons:
            print(f"    ❌ {r}")
        print(f"\n  ⚡ URGENT — Consider closing NOW")
        print(f"{'═'*52}\n")

        self.sound.danger_exit()
        self.telegram.danger_exit(
            symbol, direction, ticket,
            entry, current, pnl, pips,
            sl, pips_to_sl, reasons,
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
    ):
        """
        Fired by RiskManager when any limit is approached or hit.

        Levels:
          WARNING     — daily loss at 75%
          URGENT      — daily loss at 90%
          LIMIT_HIT   — daily loss limit reached, trading halted
          TRADES_NEAR — 2 of 3 trade slots used
          TRADES_FULL — all trade slots full, signal blocked
          LOW_MARGIN  — free margin below 200% safety threshold
        """
        cooldown = 3 if level in ("LIMIT_HIT", "URGENT") else 10
        key      = f"risk_{level}"
        if self._already_alerted(key, minutes=cooldown):
            return

        icons = {
            "WARNING":     "⚠️",
            "URGENT":      "⛔",
            "LIMIT_HIT":   "🚫",
            "TRADES_NEAR": "⚠️",
            "TRADES_FULL": "🚫",
            "LOW_MARGIN":  "⚠️",
        }
        icon = icons.get(level, "⚠️")

        print(f"\n{icon} {'═'*50}")
        print(f"  RISK ALERT — {level.replace('_', ' ')}")
        print(f"{'═'*52}")
        print(f"  Time : {self._now()}")

        if level in ("WARNING", "URGENT", "LIMIT_HIT"):
            bar_filled = int(pct_used / 10)
            bar        = "█" * bar_filled + "░" * (10 - bar_filled)
            print(f"  Daily Loss : €{abs(daily_pnl):.2f} / €{daily_limit:.2f}")
            print(f"  Used       : [{bar}] {pct_used:.0f}%")

        if level in ("TRADES_NEAR", "TRADES_FULL"):
            print(f"  Trades     : {open_trades} / {max_trades} slots used")

        print(f"  {message}")
        print(f"{'═'*52}\n")

        if level in ("LIMIT_HIT", "URGENT", "TRADES_FULL"):
            self.sound.danger_exit()
        else:
            self.sound.news_warning()

        self.telegram.risk_warning(
            level=level,
            message=message,
            daily_pnl=daily_pnl,
            daily_limit=daily_limit,
            pct_used=pct_used,
            open_trades=open_trades,
            max_trades=max_trades,
        )

        logger.warning(
            f"🛡 Risk Alert | Level:{level} | "
            f"DailyPnL:€{daily_pnl:+.2f} | "
            f"Used:{pct_used:.0f}%"
        )

    # ── News Warning ──────────────────────────────────────────────────────────
    def news_warning(
        self,
        event:     str,
        currency:  str,
        mins_away: float,
    ):
        key = f"news_{event}"
        if self._already_alerted(key, minutes=60):
            return

        local_time = datetime.now(self.timezone)
        event_time = local_time + timedelta(minutes=int(mins_away))
        time_str   = event_time.strftime("%H:%M")

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

        self.sound.news_warning()
        self.telegram.news_warning(
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
    ):
        icon = "🟢" if direction == "BUY" else "🔴"

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

        self.sound.trade_opened()
        self.telegram.trade_opened(
            symbol, direction, ticket,
            entry, sl, tp, volume,
            risk, style, mode,
        )

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
    ):
        icon = "✅" if pnl >= 0 else "❌"

        print(f"\n{icon} {'─'*48}")
        print(f"  TRADE CLOSED — {symbol} {direction}")
        print(f"{'─'*50}")
        print(f"  Ticket  : #{ticket}")
        print(f"  Opened  : {entry}")
        print(f"  Closed  : {close}")
        print(f"  Result  : {pips:+.1f} pips (€{pnl:+.2f})")
        print(f"  Reason  : {reason}")
        print(f"{'─'*50}\n")

        if pnl >= 0:
            self.sound.trade_closed_profit()
        else:
            self.sound.trade_closed_loss()

        self.telegram.trade_closed(
            symbol, direction, ticket,
            entry, close, pnl, pips, reason,
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
    ):
        """
        Called from main.py _daily_summary() at 23:55 each night.
        Generates date, balance, style and mode internally so
        main.py does not need to pass them.
        """
        from datetime import date as _date
        import MetaTrader5 as _mt5

        today   = _date.today().strftime("%A %d %b %Y")
        info    = _mt5.account_info()
        balance = info.balance if info else 0.0

        pnl_icon = "📈" if net_pnl >= 0 else "📉"

        print(f"\n📊 {'═'*50}")
        print(f"  DAILY SUMMARY — {today}")
        print(f"{'═'*52}")
        print(f"  Signals    : {signals}")
        print(f"  Trades     : {trades}")
        print(f"  Winners    : {winners} ✅")
        print(f"  Losers     : {losers} ❌")
        print(f"  Win Rate   : {win_rate:.1f}%")
        print(f"  Gross P    : €{gross_profit:+.2f}")
        print(f"  Gross L    : €{gross_loss:+.2f}")
        print(f"  Net P&L    : {pnl_icon} €{net_pnl:+.2f}")
        print(f"  Best Trade : €{best_trade:+.2f}")
        print(f"  Worst Trade: €{worst_trade:+.2f}")
        print(f"  Balance    : €{balance:,.2f}")
        print(f"{'═'*52}\n")

        if net_pnl >= 0:
            self.sound.trade_closed_profit()
        else:
            self.sound.trade_closed_loss()

        self.telegram.daily_summary(
            signals=signals,
            trades=trades,
            winners=winners,
            losers=losers,
            net_pnl=net_pnl,
            win_rate=win_rate,
            best_trade=best_trade,
            worst_trade=worst_trade,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            balance=balance,
            date=today,
        )

        logger.info(
            f"📊 Daily Summary | Trades:{trades} | "
            f"W:{winners} L:{losers} | "
            f"Net:€{net_pnl:+.2f} | "
            f"WinRate:{win_rate:.1f}%"
        )

    # ── System Online ─────────────────────────────────────────────────────────
    def system_online(
        self,
        balance: float,
        style:   str,
        mode:    str,
    ):
        self.sound.system_ready()
        self.telegram.system_online(balance, style, mode)

    # ── System Offline ────────────────────────────────────────────────────────
    def system_offline(self, reason: str = "Manual shutdown"):
        self.telegram.system_offline(reason)