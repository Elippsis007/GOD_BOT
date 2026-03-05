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
    direction: str        # "BUY" or "SELL"
    volume:    float
    entry:     float
    sl:        float
    tp:        float
    risk_usd:  float
    rr_ratio:  float


class RiskManager:
    """
    Equity-proportional position sizing with hard circuit breakers.

    Upgrades vs previous version:
      - Uses live EQUITY (not balance) for all sizing calculations so
        position size shrinks automatically during a drawdown.
      - Hard absolute cap: single trade risk never exceeds
        MAX_LOSS_PER_TRADE regardless of equity level.
      - Kelly Criterion estimate logged for reference (not used for
        sizing directly — pure Kelly is too aggressive for retail).
      - Consecutive loss counter: after N losses in a row the risk
        percentage is halved automatically (anti-martingale protection).
      - Daily drawdown circuit breaker checks equity drop from the
        session-open equity snapshot, not just closed P&L, so
        floating losses also count toward the daily limit.
      - All existing warnings (75%, 90%, LIMIT_HIT, TRADES_NEAR,
        TRADES_FULL, LOW_MARGIN) retained and improved.
    """

    # ── Anti-martingale: halve risk after this many consecutive losses ────────
    CONSECUTIVE_LOSS_LIMIT = 3

    def __init__(self, config=CONFIG):
        self.cfg               = config
        self._daily_pnl        = 0.0
        self._daily_reset      = date.today()
        self._session_equity   = None   # equity snapshot at session open
        self._consecutive_loss = 0      # anti-martingale counter

        # ── Alert state flags ─────────────────────────────────────────────────
        self._warned_daily_75    = False
        self._warned_daily_90    = False
        self._warned_trades_near = False
        self._warned_trades_full = False

        # AlertManager injected after construction to avoid circular imports
        self._alerts = None

    def set_alerts(self, alert_manager) -> None:
        self._alerts = alert_manager

    # ── Session Equity Snapshot ───────────────────────────────────────────────
    def snapshot_session_equity(self) -> None:
        """
        Call once at bot startup (from main.py) to record the
        equity at the start of the session.  The daily drawdown
        circuit breaker measures the DROP from this snapshot so
        floating losses also contribute to the daily limit.
        """
        account = self._get_account()
        if account:
            self._session_equity = account["equity"]
            logger.info(
                f"📸 Session equity snapshot: €{self._session_equity:.2f}"
            )

    # ── Daily Reset ───────────────────────────────────────────────────────────
    def _check_daily_reset(self) -> None:
        today = date.today()
        if today != self._daily_reset:
            logger.info(
                f"🔄 New trading day — resetting daily P&L "
                f"(was €{self._daily_pnl:+.2f})"
            )
            self._daily_pnl        = 0.0
            self._daily_reset      = today
            self._consecutive_loss = 0
            self._warned_daily_75    = False
            self._warned_daily_90    = False
            self._warned_trades_near = False
            self._warned_trades_full = False
            # Re-snapshot equity for the new day
            self.snapshot_session_equity()

    # ── Core Sizing ───────────────────────────────────────────────────────────
    def calculate_position(
        self,
        symbol:     str,
        direction:  str,
        entry:      float,
        sl:         float,
        tp:         float,
        win_rate:   float = 0.50,   # passed from ML model confidence
    ) -> Optional[PositionSpec]:

        if not self._pre_trade_checks(symbol):
            return None

        account = self._get_account()
        if not account:
            return None

        # ── Use EQUITY not balance ────────────────────────────────────────────
        equity = account["equity"]

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            logger.error(f"Symbol info unavailable: {symbol}")
            return None

        # ── Base risk: 1% of equity ───────────────────────────────────────────
        base_risk_pct = self.cfg.RISK_PER_TRADE   # e.g. 0.01

        # ── Anti-martingale: halve risk after CONSECUTIVE_LOSS_LIMIT losses ───
        if self._consecutive_loss >= self.CONSECUTIVE_LOSS_LIMIT:
            base_risk_pct = base_risk_pct / 2.0
            logger.warning(
                f"⚠️  Anti-martingale active — risk halved to "
                f"{base_risk_pct*100:.2f}% after "
                f"{self._consecutive_loss} consecutive losses"
            )

        risk_usd = equity * base_risk_pct

        # ── Hard absolute cap (e.g. $50 max per trade) ────────────────────────
        risk_usd = min(risk_usd, self.cfg.MAX_LOSS_PER_TRADE)

        # ── Kelly Criterion (logged only — for reference) ─────────────────────
        rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 1.0
        kelly_pct = win_rate - ((1 - win_rate) / rr)
        kelly_usd = equity * max(kelly_pct * 0.25, 0)   # quarter-Kelly
        logger.debug(
            f"📐 Kelly: win_rate={win_rate:.0%} RR={rr:.2f} → "
            f"Kelly={kelly_pct*100:.1f}% | "
            f"Quarter-Kelly=€{kelly_usd:.2f} | "
            f"Using equity-1%=€{risk_usd:.2f}"
        )

        sl_distance = abs(entry - sl)
        if sl_distance == 0:
            logger.error("SL distance is zero — aborting")
            return None

        # ── Pip value & volume calculation ────────────────────────────────────
        pip_value  = sym_info.trade_contract_size * sym_info.point
        sl_pips    = sl_distance / sym_info.point
        volume_raw = risk_usd / (sl_pips * pip_value)
        volume     = self._normalize_volume(volume_raw, sym_info)

        # ── Minimum lot guard — recalculate actual risk at min lot ────────────
        actual_risk = volume * sl_pips * pip_value
        if actual_risk > self.cfg.MAX_LOSS_PER_TRADE * 1.5:
            logger.warning(
                f"⛔ Actual risk €{actual_risk:.2f} exceeds hard cap "
                f"even at minimum lot — skipping trade"
            )
            return None

        # ── Validate reward:risk ──────────────────────────────────────────────
        tp_distance = abs(tp - entry)
        rr_actual   = round(tp_distance / sl_distance, 2)
        if rr_actual < 1.5:
            logger.warning(f"R:R={rr_actual} too low — minimum 1.5 required")
            return None

        spec = PositionSpec(
            symbol    = symbol,
            direction = direction,
            volume    = volume,
            entry     = round(entry,  sym_info.digits),
            sl        = round(sl,     sym_info.digits),
            tp        = round(tp,     sym_info.digits),
            risk_usd  = round(actual_risk, 2),
            rr_ratio  = rr_actual,
        )

        logger.info(
            f"📝 Position | {symbol} {direction} | "
            f"Vol={volume} | Risk=€{actual_risk:.2f} "
            f"({actual_risk/equity*100:.2f}% equity) | "
            f"R:R={rr_actual} | "
            f"Equity=€{equity:.2f}"
        )
        return spec

    # ── Trade Result Feedback ─────────────────────────────────────────────────
    def record_trade_result(self, pnl: float) -> None:
        """
        Call after every trade closes.  Updates daily P&L and the
        consecutive loss counter used by the anti-martingale guard.
        """
        self._check_daily_reset()
        self._daily_pnl += pnl

        if pnl < 0:
            self._consecutive_loss += 1
            logger.warning(
                f"📉 Loss recorded — consecutive losses: "
                f"{self._consecutive_loss}"
            )
        else:
            if self._consecutive_loss > 0:
                logger.info(
                    f"📈 Win — consecutive loss streak reset "
                    f"(was {self._consecutive_loss})"
                )
            self._consecutive_loss = 0

        logger.info(
            f"📊 Daily P&L: €{self._daily_pnl:+.2f} | "
            f"Consecutive losses: {self._consecutive_loss}"
        )

    # kept for backwards compatibility with existing main.py calls
    def update_daily_pnl(self, pnl: float) -> None:
        self.record_trade_result(pnl)

    # ── Pre-Trade Circuit Breakers ────────────────────────────────────────────
    def _pre_trade_checks(self, symbol: str) -> bool:
        self._check_daily_reset()

        account = self._get_account()
        if not account:
            return False

        equity = account["equity"]

        # ── 1. Equity drawdown from session open (catches floating losses) ────
        if self._session_equity and self._session_equity > 0:
            equity_drop     = self._session_equity - equity
            equity_drop_pct = equity_drop / self._session_equity * 100
            daily_limit_pct = self.cfg.MAX_DAILY_LOSS * 100   # e.g. 3.0

            if equity_drop_pct >= daily_limit_pct:
                msg = (
                    f"🚫 DAILY EQUITY DRAWDOWN LIMIT HIT\n"
                    f"Session open: €{self._session_equity:.2f}\n"
                    f"Current equity: €{equity:.2f}\n"
                    f"Drop: €{equity_drop:.2f} ({equity_drop_pct:.1f}%) "
                    f"≥ limit {daily_limit_pct:.0f}%\n"
                    f"Trading halted for today"
                )
                logger.warning(msg)
                if self._alerts:
                    self._alerts.risk_warning(
                        level="LIMIT_HIT",
                        message=msg,
                        daily_pnl=-equity_drop,
                        daily_limit=self._session_equity * self.cfg.MAX_DAILY_LOSS,
                        pct_used=equity_drop_pct,
                    )
                return False

            # 90% of daily limit warning
            if equity_drop_pct >= daily_limit_pct * 0.9 and \
               not self._warned_daily_90:
                self._warned_daily_90 = True
                msg = (
                    f"⛔ URGENT: Equity down {equity_drop_pct:.1f}% "
                    f"today (limit {daily_limit_pct:.0f}%)\n"
                    f"€{equity_drop:.2f} lost — only "
                    f"€{self._session_equity*self.cfg.MAX_DAILY_LOSS - equity_drop:.2f} "
                    f"remaining before halt"
                )
                logger.warning(msg)
                if self._alerts:
                    self._alerts.risk_warning(
                        level="URGENT",
                        message=msg,
                        daily_pnl=-equity_drop,
                        daily_limit=self._session_equity * self.cfg.MAX_DAILY_LOSS,
                        pct_used=equity_drop_pct,
                    )

            # 75% of daily limit warning
            elif equity_drop_pct >= daily_limit_pct * 0.75 and \
                 not self._warned_daily_75:
                self._warned_daily_75 = True
                msg = (
                    f"⚠️  WARNING: Equity down {equity_drop_pct:.1f}% "
                    f"today (limit {daily_limit_pct:.0f}%)"
                )
                logger.warning(msg)
                if self._alerts:
                    self._alerts.risk_warning(
                        level="WARNING",
                        message=msg,
                        daily_pnl=-equity_drop,
                        daily_limit=self._session_equity * self.cfg.MAX_DAILY_LOSS,
                        pct_used=equity_drop_pct,
                    )

        # ── 2. Max concurrent trades ──────────────────────────────────────────
        positions  = mt5.positions_get()
        open_count = len(positions) if positions else 0
        max_trades = self.cfg.MAX_OPEN_TRADES

        if open_count >= max_trades - 1 and not self._warned_trades_near:
            self._warned_trades_near = True
            msg = (
                f"⚠️  TRADES NEAR LIMIT: {open_count}/{max_trades} "
                f"slots used — 1 remaining"
            )
            logger.warning(msg)
            if self._alerts:
                self._alerts.risk_warning(
                    level="TRADES_NEAR",
                    message=msg,
                    open_trades=open_count,
                    max_trades=max_trades,
                )

        if open_count >= max_trades:
            if not self._warned_trades_full:
                self._warned_trades_full = True
                msg = (
                    f"🚫 MAX TRADES: {open_count}/{max_trades} — "
                    f"new signals blocked"
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

        if open_count < max_trades:
            self._warned_trades_full = False
        if open_count < max_trades - 1:
            self._warned_trades_near = False

        # ── 3. Duplicate symbol check ─────────────────────────────────────────
        open_symbols = [p.symbol for p in (positions or [])]
        if symbol in open_symbols:
            logger.warning(f"🚫 Already have open position on {symbol}")
            return False

        # ── 4. Margin check (200% free margin) ───────────────────────────────
        if account["free_margin"] < account["margin"] * 2:
            msg = (
                f"⚠️  LOW MARGIN: Free €{account['free_margin']:.2f} "
                f"below 200% safety threshold"
            )
            logger.warning(msg)
            if self._alerts:
                self._alerts.risk_warning(level="LOW_MARGIN", message=msg)
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
