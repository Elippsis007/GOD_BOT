# risk/risk_manager.py
import MetaTrader5 as mt5
from dataclasses import dataclass
from datetime import date
from typing import Optional
from config.settings import CONFIG
from monitoring.logger import get_logger

logger = get_logger("RiskManager")


@dataclass
class PositionSpec:
    symbol:    str
    direction: str       # "BUY" or "SELL"
    volume:    float
    entry:     float
    sl:        float
    tp:        float
    risk_usd:  float
    rr_ratio:  float


class RiskManager:
    """
    Kelly-inspired position sizing with hard circuit breakers.
    Respects: max daily loss, max open trades, account exposure.
    Daily P&L resets automatically at midnight each trading day.
    Sends Telegram + sound alerts when limits are approached or hit.
    """

    def __init__(self, config=CONFIG):
        self.cfg          = config
        self._daily_pnl   = 0.0
        self._daily_reset = date.today()

        # ── Alert state flags ─────────────────────────────────────────────────
        # Prevent the same warning from firing repeatedly every scan cycle.
        # Flags reset when _daily_pnl resets at midnight.
        self._warned_daily_75    = False   # fired when daily loss hits 75%
        self._warned_daily_90    = False   # fired when daily loss hits 90%
        self._warned_trades_near = False   # fired when 2 of 3 slots used
        self._warned_trades_full = False   # fired when all slots full

        # AlertManager injected after construction to avoid circular imports
        self._alerts = None

    def set_alerts(self, alert_manager) -> None:
        """
        Called from main.py after AlertManager is created:
            self.risk_mgr.set_alerts(self.alerts)
        Kept as a setter to avoid circular import between
        risk_manager → alert_manager → risk_manager.
        """
        self._alerts = alert_manager

    # ── Daily Reset ───────────────────────────────────────────────────────────
    def _check_daily_reset(self) -> None:
        today = date.today()
        if today != self._daily_reset:
            logger.info(
                f"🔄 New trading day — resetting daily P&L "
                f"(was ${self._daily_pnl:+.2f})"
            )
            self._daily_pnl      = 0.0
            self._daily_reset    = today
            # Reset all warning flags for the new day
            self._warned_daily_75    = False
            self._warned_daily_90    = False
            self._warned_trades_near = False
            self._warned_trades_full = False

    # ── Core Sizing ───────────────────────────────────────────────────────────
    def calculate_position(
        self,
        symbol:    str,
        direction: str,
        entry:     float,
        sl:        float,
        tp:        float,
    ) -> Optional[PositionSpec]:

        if not self._pre_trade_checks(symbol):
            return None

        account = self._get_account()
        if not account:
            return None

        balance  = account["balance"]
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            logger.error(f"Symbol info unavailable: {symbol}")
            return None

        # Risk in account currency
        risk_usd    = balance * self.cfg.RISK_PER_TRADE
        sl_distance = abs(entry - sl)

        if sl_distance == 0:
            logger.error("SL distance is zero — aborting")
            return None

        # Pip value calculation
        pip_value  = sym_info.trade_contract_size * sym_info.point
        sl_pips    = sl_distance / sym_info.point
        volume_raw = risk_usd / (sl_pips * pip_value)

        # Normalise to allowed lot steps
        volume = self._normalize_volume(volume_raw, sym_info)

        # Validate reward:risk
        tp_distance = abs(tp - entry)
        rr = round(tp_distance / sl_distance, 2)
        if rr < 1.5:
            logger.warning(f"R:R={rr} too low — minimum 1.5 required")
            return None

        spec = PositionSpec(
            symbol    = symbol,
            direction = direction,
            volume    = volume,
            entry     = round(entry, sym_info.digits),
            sl        = round(sl,    sym_info.digits),
            tp        = round(tp,    sym_info.digits),
            risk_usd  = round(risk_usd, 2),
            rr_ratio  = rr,
        )
        logger.info(
            f"📐 Position | {symbol} {direction} | "
            f"Vol={volume} | Risk=${risk_usd:.2f} | R:R={rr}"
        )
        return spec

    # ── Pre-Trade Circuit Breakers ────────────────────────────────────────────
    def _pre_trade_checks(self, symbol: str) -> bool:
        self._check_daily_reset()

        account = self._get_account()
        if not account:
            return False

        balance        = account["balance"]
        daily_limit    = balance * self.cfg.MAX_DAILY_LOSS
        loss_pct       = (-self._daily_pnl / daily_limit * 100) if daily_limit > 0 else 0

        # ── 1. Daily loss limit warnings ──────────────────────────────────────
        if self._daily_pnl < 0:

            # 90% warning — urgent
            if loss_pct >= 90 and not self._warned_daily_90:
                self._warned_daily_90 = True
                msg = (
                    f"⛔ URGENT: Daily loss at {loss_pct:.0f}% of limit\n"
                    f"Lost: €{abs(self._daily_pnl):.2f} of €{daily_limit:.2f} max\n"
                    f"Only €{daily_limit - abs(self._daily_pnl):.2f} remaining "
                    f"before trading halts for today"
                )
                logger.warning(msg)
                if self._alerts:
                    self._alerts.risk_warning(
                        level="URGENT",
                        message=msg,
                        daily_pnl=self._daily_pnl,
                        daily_limit=daily_limit,
                        pct_used=loss_pct,
                    )

            # 75% warning — caution
            elif loss_pct >= 75 and not self._warned_daily_75:
                self._warned_daily_75 = True
                msg = (
                    f"⚠️ WARNING: Daily loss at {loss_pct:.0f}% of limit\n"
                    f"Lost: €{abs(self._daily_pnl):.2f} of €{daily_limit:.2f} max\n"
                    f"€{daily_limit - abs(self._daily_pnl):.2f} remaining "
                    f"before trading halts for today"
                )
                logger.warning(msg)
                if self._alerts:
                    self._alerts.risk_warning(
                        level="WARNING",
                        message=msg,
                        daily_pnl=self._daily_pnl,
                        daily_limit=daily_limit,
                        pct_used=loss_pct,
                    )

        # ── Hard limit — stop trading ─────────────────────────────────────────
        if self._daily_pnl <= -daily_limit:
            msg = (
                f"🚫 DAILY LOSS LIMIT HIT — trading halted for today\n"
                f"Total loss: €{abs(self._daily_pnl):.2f} "
                f"(limit was €{daily_limit:.2f})\n"
                f"Trading resumes tomorrow at midnight"
            )
            logger.warning(msg)
            if self._alerts:
                self._alerts.risk_warning(
                    level="LIMIT_HIT",
                    message=msg,
                    daily_pnl=self._daily_pnl,
                    daily_limit=daily_limit,
                    pct_used=100.0,
                )
            return False

        # ── 2. Max concurrent trades warnings ────────────────────────────────
        positions = mt5.positions_get()
        open_count = len(positions) if positions else 0
        max_trades = self.cfg.MAX_OPEN_TRADES

        # Near limit — 2 of 3 slots used
        if open_count >= max_trades - 1 and not self._warned_trades_near:
            self._warned_trades_near = True
            msg = (
                f"⚠️ TRADES NEAR LIMIT: {open_count}/{max_trades} slots used\n"
                f"Only 1 trade slot remaining"
            )
            logger.warning(msg)
            if self._alerts:
                self._alerts.risk_warning(
                    level="TRADES_NEAR",
                    message=msg,
                    open_trades=open_count,
                    max_trades=max_trades,
                )

        # All slots full — block and alert
        if open_count >= max_trades:
            if not self._warned_trades_full:
                self._warned_trades_full = True
                msg = (
                    f"🚫 MAX TRADES REACHED: {open_count}/{max_trades} slots full\n"
                    f"New signals blocked until a position closes"
                )
                logger.warning(msg)
                if self._alerts:
                    self._alerts.risk_warning(
                        level="TRADES_FULL",
                        message=msg,
                        open_trades=open_count,
                        max_trades=max_trades,
                    )
            return False

        # Reset trades_full flag when a slot frees up
        if open_count < max_trades:
            self._warned_trades_full = False
        if open_count < max_trades - 1:
            self._warned_trades_near = False

        # ── 3. Duplicate symbol check ─────────────────────────────────────────
        open_symbols = [p.symbol for p in (positions or [])]
        if symbol in open_symbols:
            logger.warning(f"🚫 Already have open position on {symbol}")
            return False

        # ── 4. Margin check (require 200% free margin) ────────────────────────
        if account["free_margin"] < account["margin"] * 2:
            msg = (
                f"⚠️ LOW MARGIN WARNING\n"
                f"Free margin: €{account['free_margin']:.2f} — "
                f"below 200% safety threshold"
            )
            logger.warning(msg)
            if self._alerts:
                self._alerts.risk_warning(
                    level="LOW_MARGIN",
                    message=msg,
                )
            return False

        return True

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _normalize_volume(self, volume: float, sym_info) -> float:
        step   = sym_info.volume_step
        volume = round(volume / step) * step
        volume = max(sym_info.volume_min, min(volume, sym_info.volume_max))
        return round(volume, 2)

    def _get_account(self) -> Optional[dict]:
        info = mt5.account_info()
        if info is None:
            return None
        return {
            "balance":     info.balance,
            "equity":      info.equity,
            "margin":      info.margin or 1,
            "free_margin": info.margin_free,
        }

    def update_daily_pnl(self, pnl: float) -> None:
        """
        Called by main.py after every trade closes to keep the
        daily P&L counter accurate for the circuit breaker check.
        """
        self._check_daily_reset()
        self._daily_pnl += pnl
        logger.info(
            f"📊 Daily P&L updated: ${self._daily_pnl:+.2f} "
            f"(limit: -{self.cfg.MAX_DAILY_LOSS * 100:.0f}% of balance)"
        )
