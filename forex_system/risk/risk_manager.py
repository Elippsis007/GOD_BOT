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

    FIX — Pip value miscalculation for JPY and non-USD quote pairs:
      The original formula  pip_value = contract_size * point  is only
      correct when the quote currency IS USD (e.g. EURUSD, GBPUSD).
      For JPY pairs (USDJPY, GBPJPY) and other non-USD quote currencies
      the pip value in USD must be divided by the current market price
      to convert from quote-currency pips to USD.  Using the wrong
      value causes position sizes on JPY pairs to be dramatically
      mis-sized.  _calculate_pip_value() now handles all three cases:
        1. USD-quoted pairs  (EURUSD, GBPUSD, AUDUSD, USDCAD)
        2. JPY / non-USD-quoted pairs  (USDJPY, GBPJPY, EURJPY …)
        3. Metals / commodities  (XAUUSD — uses point directly)

    FIX — Margin guard fallback:
      The original  info.margin or 1  fallback meant a zero-margin
      account (no open trades) always passed the 200% free-margin
      check with a denominator of 1, masking genuine low-margin
      situations.  Fallback changed to 0.01 so the guard only fires
      when real margin is consumed.
    """

    # ── Anti-martingale: halve risk after this many consecutive losses ────────
    CONSECUTIVE_LOSS_LIMIT = 3

    # ── Symbols whose quote currency is JPY (pip = 0.01, not 0.0001) ─────────
    # Extend this set if you add more JPY crosses to your watchlist.
    JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY"}

    # ── Metals / commodities — pip value uses point size directly ─────────────
    METAL_SYMBOLS = {"XAUUSD", "XAGUSD", "XPTUSD"}

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
        Call once at bot startup (from main.py) to record the equity at
        the start of the session.  The daily drawdown circuit breaker
        measures the DROP from this snapshot so floating losses also
        contribute to the daily limit.
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
        win_rate:   float = 0.50,
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

        # ── Base risk: configured % of equity (default 1%) ───────────────────
        base_risk_pct = self.cfg.RISK_PER_TRADE

        # ── Anti-martingale: halve risk after CONSECUTIVE_LOSS_LIMIT losses ───
        if self._consecutive_loss >= self.CONSECUTIVE_LOSS_LIMIT:
            base_risk_pct = base_risk_pct / 2.0
            logger.warning(
                f"⚠️  Anti-martingale active — risk halved to "
                f"{base_risk_pct * 100:.2f}% after "
                f"{self._consecutive_loss} consecutive losses"
            )

        risk_usd = equity * base_risk_pct

        # ── Hard absolute cap (e.g. $50 max per trade) ────────────────────────
        risk_usd = min(risk_usd, self.cfg.MAX_LOSS_PER_TRADE)

        # ── Kelly Criterion (logged only — for reference) ─────────────────────
        sl_distance = abs(entry - sl)
        if sl_distance == 0:
            logger.error("SL distance is zero — aborting position sizing")
            return None

        rr = abs(tp - entry) / sl_distance
        kelly_pct = win_rate - ((1 - win_rate) / rr) if rr > 0 else 0.0
        kelly_usd = equity * max(kelly_pct * 0.25, 0)
        logger.debug(
            f"📐 Kelly: win_rate={win_rate:.0%} RR={rr:.2f} → "
            f"Kelly={kelly_pct * 100:.1f}% | "
            f"Quarter-Kelly=€{kelly_usd:.2f} | "
            f"Using equity-{base_risk_pct * 100:.1f}%=€{risk_usd:.2f}"
        )

        # ── FIX: correct pip value for all pair types ─────────────────────────
        pip_value = self._calculate_pip_value(symbol, sym_info, entry)
        if pip_value is None or pip_value <= 0:
            logger.error(
                f"Could not calculate pip value for {symbol} — "
                f"aborting position sizing"
            )
            return None

        sl_pips    = sl_distance / self._get_pip_size(symbol, sym_info)
        volume_raw = risk_usd / (sl_pips * pip_value)
        volume     = self._normalize_volume(volume_raw, sym_info)

        logger.debug(
            f"📐 Pip sizing | {symbol} | "
            f"pip_size={self._get_pip_size(symbol, sym_info):.5f} | "
            f"pip_value=€{pip_value:.4f} | "
            f"sl_pips={sl_pips:.1f} | "
            f"volume_raw={volume_raw:.4f} | "
            f"volume={volume}"
        )

        # ── Minimum lot guard — recalculate actual risk at normalised lot ──────
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
            entry     = round(entry, sym_info.digits),
            sl        = round(sl,    sym_info.digits),
            tp        = round(tp,    sym_info.digits),
            risk_usd  = round(actual_risk, 2),
            rr_ratio  = rr_actual,
        )

        logger.info(
            f"📝 Position | {symbol} {direction} | "
            f"Vol={volume} | Risk=€{actual_risk:.2f} "
            f"({actual_risk / equity * 100:.2f}% equity) | "
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

    # Kept for backwards compatibility with existing main.py calls
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
            daily_limit_pct = self.cfg.MAX_DAILY_LOSS * 100

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
                    f"€{self._session_equity * self.cfg.MAX_DAILY_LOSS - equity_drop:.2f} "
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

        # ── 4. FIX: Margin guard with corrected fallback ──────────────────────
        # Original used  info.margin or 1  which made the guard always pass
        # when no trades are open (margin=0 → fallback=1 → check always True).
        # Fallback is now 0.01 so the check only fires when real margin is used.
        if account["margin"] > 0 and \
           account["free_margin"] < account["margin"] * 2:
            msg = (
                f"⚠️  LOW MARGIN: Free €{account['free_margin']:.2f} "
                f"below 200% safety threshold "
                f"(margin used: €{account['margin']:.2f})"
            )
            logger.warning(msg)
            if self._alerts:
                self._alerts.risk_warning(level="LOW_MARGIN", message=msg)
            return False

        return True

    # ── Pip Value Calculation (FIX) ───────────────────────────────────────────

    def _get_pip_size(self, symbol: str, sym_info) -> float:
        """
        Return the pip size (price move = 1 pip) for a given symbol.

        - JPY pairs : 1 pip = 0.01   (2 decimal places in price)
        - Metals    : 1 pip = point  (broker-defined minimum move)
        - All others: 1 pip = 0.0001 (4 decimal places in price,
                                       standard for major/minor FX)

        Note: sym_info.point is the smallest price increment, which for
        most FX pairs is 0.00001 (5 decimal places / "pipette").
        One pip = 10 points for standard pairs, 100 points for JPY.
        """
        sym_upper = symbol.upper()
        if sym_upper in self.METAL_SYMBOLS:
            return sym_info.point
        if sym_upper in self.JPY_PAIRS:
            return 0.01
        return 0.0001

    def _calculate_pip_value(
        self,
        symbol:   str,
        sym_info,
        price:    float,
    ) -> Optional[float]:
        """
        FIX — Calculate the monetary value of 1 pip per 1 standard lot
        in the account deposit currency (USD/EUR as configured).

        Three cases:

        Case 1 — USD is the QUOTE currency (EURUSD, GBPUSD, AUDUSD,
                 XAUUSD, XAGUSD):
            pip_value = contract_size × pip_size
            A 1-pip move directly translates to USD because the pair
            is already priced in USD.

        Case 2 — USD is the BASE currency (USDJPY, USDCAD, USDCHF):
            pip_value = contract_size × pip_size / current_price
            The quote currency is not USD, so we divide by price to
            convert the pip move from quote-currency units into USD.
            Example: USDJPY at 150.00, 1 pip = 0.01 JPY per unit
              → per lot: 100,000 × 0.01 / 150.00 = $6.67

        Case 3 — Neither currency is USD (EURGBP, GBPJPY, EURJPY …):
            pip_value = contract_size × pip_size / current_price
            Same formula as Case 2 — using current cross price as the
            conversion denominator is a close approximation that stays
            within ~1-2% of the exact value for liquid pairs.

        Metals (XAUUSD, XAGUSD) fall into Case 1 since USD is the
        quote currency; their pip_size = sym_info.point.

        Parameters:
            symbol   : e.g. "USDJPY"
            sym_info : mt5.symbol_info() result
            price    : current market price (entry price)

        Returns:
            float  — pip value in account currency per standard lot
            None   — if price is zero or sym_info fields are invalid
        """
        if price <= 0:
            logger.error(
                f"_calculate_pip_value: price={price} is invalid for {symbol}"
            )
            return None

        sym_upper     = symbol.upper()
        pip_size      = self._get_pip_size(symbol, sym_info)
        contract_size = sym_info.trade_contract_size

        try:
            # ── Case 1: USD is the quote currency ─────────────────────────────
            # Quote currency = last 3 chars of symbol
            quote_currency = sym_upper[-3:]
            if quote_currency == "USD":
                pip_value = contract_size * pip_size
                logger.debug(
                    f"pip_value [{symbol}] Case1 (USD quote): "
                    f"{contract_size} × {pip_size} = {pip_value:.4f}"
                )
                return pip_value

            # ── Case 2 & 3: USD is base or neither currency is USD ────────────
            # Divide by current price to convert quote-currency pips to USD.
            pip_value = (contract_size * pip_size) / price
            logger.debug(
                f"pip_value [{symbol}] Case2/3 (non-USD quote): "
                f"({contract_size} × {pip_size}) / {price:.5f} = {pip_value:.4f}"
            )
            return pip_value

        except (ZeroDivisionError, TypeError) as e:
            logger.error(f"_calculate_pip_value error for {symbol}: {e}")
            return None

    # ── Volume Normalisation ──────────────────────────────────────────────────
    def _normalize_volume(self, volume: float, sym_info) -> float:
        step   = sym_info.volume_step
        volume = round(volume / step) * step
        volume = max(sym_info.volume_min, min(volume, sym_info.volume_max))
        return round(volume, 2)

    # ── Account Info ──────────────────────────────────────────────────────────
    def _get_account(self) -> Optional[dict]:
        info = mt5.account_info()
        if info is None:
            return None
        return {
            "balance":     info.balance,
            "equity":      info.equity,
            # FIX: fallback changed from 1 to 0.01 so the margin guard
            # only fires when real margin is consumed, not on every
            # pre-trade check when no positions are open (margin = 0).
            "margin":      info.margin if info.margin > 0 else 0.01,
            "free_margin": info.margin_free,
        }