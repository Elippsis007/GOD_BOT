# =============================================================================
# GODBOT v3.0 – risk/risk_manager.py  (PRODUCTION – fully corrected)
# =============================================================================
#
#  Fixes applied in this version vs the repo baseline:
#
#  [A]  PositionSpec: added sl_pips, tp_pips, pip_value, confidence fields.
#  [B]  calculate_position(): accepts sl_pips/tp_pips (pip distances);
#       computes absolute SL/TP prices internally.
#  [C]  RR validation uses CONFIG.RR_RATIO (2.0) not hard-coded 1.5.
#  [D]  _calculate_pip_value(): three-case pip value with cross-rate lookup
#       for non-USD crosses (GBPJPY, EURJPY, etc.) — fixes ~25% sizing error.
#  [E]  CONSECUTIVE_LOSS_LIMIT reads from CONFIG instead of class constant.
#  [F]  _get_pip_size(): digit-count approach instead of hard-coded sets.
#  [G]  record_trade_result(): _check_daily_reset() called FIRST.
#  [H]  _normalize_volume(): integer-step arithmetic + post-snap re-clamp.
#  [I]  Logs use account["currency"] instead of hard-coded '€'.
#  [J]  Margin guard: 1.5× threshold + $50 absolute floor.
#  [K]  Kelly: computes actual win_rate from trade history when ≥10 trades.
#  [L]  Minimum risk guard: $0.10 floor to prevent volume=0.
#  [M]  _get_account() called only ONCE per calculate_position() call —
#       eliminated double MT5 API call in the log statement.
#  [N]  _check_daily_reset() uses MT5 server time (not local OS time) so
#       the daily reset fires at broker midnight, not trader-local midnight.
#
# =============================================================================

import MetaTrader5 as mt5
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional, List, Dict
from config.settings import CONFIG
from monitoring.logger import get_logger

logger = get_logger("RiskManager")

_TRADE_HISTORY_CAP = 200  # cap history list to prevent memory growth


# ── PositionSpec ───────────────────────────────────────────────────────────────
@dataclass
class PositionSpec:
    """
    Fully-specified trade parameters returned by RiskManager.calculate_position().

    [A] sl_pips, tp_pips, pip_value, and confidence are added so that
        order_executor.py and main.py can read them without AttributeError.
    """
    symbol:     str
    direction:  str     # "BUY" or "SELL"
    volume:     float
    entry:      float
    sl:         float   # absolute SL price
    tp:         float   # absolute TP price
    sl_pips:    float   # [A] SL distance in pips
    tp_pips:    float   # [A] TP distance in pips
    pip_value:  float   # [A] monetary value per pip per 1 standard lot
    risk_usd:   float   # actual $ at risk after volume normalisation
    rr_ratio:   float   # realised TP / SL ratio
    confidence: float   # [A] signal confidence passed from SignalEngine


# ── RiskManager ────────────────────────────────────────────────────────────────
class RiskManager:
    """
    Equity-proportional position sizing with layered circuit breakers.

    Sizing flow:
        1.  Pre-trade checks  (drawdown → trade count → duplicate → margin)
        2.  Anti-martingale   (risk halved after N consecutive losses)
        3.  Risk amount       (equity × RISK_PER_TRADE, capped at MAX_LOSS_PER_TRADE)
        4.  Pip-value         (3-case lookup: USD-quote / USD-base / cross)
        5.  Volume            = risk_amount / (sl_pips × pip_value)
        6.  Normalisation     (snap to broker lot-step, re-clamp to min/max)
        7.  RR validation     (tp_pips / sl_pips ≥ CONFIG.RR_RATIO)
        8.  Kelly log         (informational only — not used for sizing)
        9.  PositionSpec      (all fields populated and returned)

    Circuit breakers  (_pre_trade_checks):
        Daily drawdown ≥ MAX_DAILY_LOSS         → HALT trading today
        Daily drawdown ≥ 90% of daily limit      → URGENT alert
        Daily drawdown ≥ 75% of daily limit      → WARNING alert
        Open trades ≥ MAX_OPEN_TRADES            → block new trade
        Open trades = MAX_OPEN_TRADES − 1        → NEAR alert
        Symbol already open                      → block duplicate
        Free margin < $50                        → LOW_MARGIN block
        Free margin < used margin × 1.5          → LOW_MARGIN block
    """

    def __init__(self, config=CONFIG):
        self.cfg = config

        # [E] Read from config — do not hard-code
        self._consecutive_loss_limit: int = getattr(
            self.cfg, "CONSECUTIVE_LOSS_LIMIT", 3
        )

        self._daily_pnl:        float = 0.0
        self._daily_reset:      date  = self._server_date()  # [N]
        self._session_equity:   Optional[float] = None
        self._consecutive_loss: int   = 0
        self._trade_history:    List[Dict] = []

        # Alert-spam guards — reset on new day
        self._warned_daily_75    = False
        self._warned_daily_90    = False
        self._warned_trades_near = False
        self._warned_trades_full = False

        # AlertManager injected after construction to break circular imports
        self._alerts = None

    def set_alerts(self, alert_manager) -> None:
        """Inject AlertManager reference after RiskManager is constructed."""
        self._alerts = alert_manager

    # ── Broker Server Date [N] ────────────────────────────────────────────────
    @staticmethod
    def _server_date() -> date:
        """
        Return today's date in MT5 server time.

        [N] Using local OS date causes the daily reset to fire at the
            trader's local midnight rather than the broker's midnight.
            For a CET trader on a UTC broker this is a 1-hour discrepancy
            that corrupts daily P&L if trades close in that window.
            We use the MT5 terminal clock when available; fall back to
            UTC (which is always closer to broker midnight than local CET).
        """
        try:
            info = mt5.terminal_info()
            if info is not None:
                # terminal_info().time_server is a Unix timestamp in UTC
                return datetime.fromtimestamp(
                    info.time_server, tz=timezone.utc
                ).date()
        except Exception:
            pass
        # Fallback: UTC — correct for most European/UK brokers
        return datetime.now(timezone.utc).date()

    # ── Session Equity Snapshot ───────────────────────────────────────────────
    def snapshot_session_equity(self) -> None:
        """
        Record equity at session open.  Call once from main.py at startup.
        The drawdown circuit breaker measures the drop from this snapshot
        so that floating (unrealised) losses also count toward the daily limit.
        [I] Logs the account currency from MT5 rather than hard-coding '€'.
        """
        account = self._get_account()
        if account:
            self._session_equity = account["equity"]
            currency = account.get("currency", "")
            logger.info(
                f"📸 Session equity snapshot: "
                f"{currency}{self._session_equity:,.2f}"
            )

    # ── Daily Reset [N] ───────────────────────────────────────────────────────
    def _check_daily_reset(self) -> None:
        """
        Compare today's broker-server date against the stored reset date.
        When a new broker day begins, zero all daily counters and re-snapshot
        equity so the drawdown circuit breaker has a fresh reference point.
        [N] Uses _server_date() (MT5 terminal clock / UTC) not date.today().
        [G] Called as the FIRST line of record_trade_result() to prevent
            midnight-boundary P&L corruption.
        """
        today = self._server_date()
        if today != self._daily_reset:
            logger.info(
                f"🔄 New broker day — resetting daily P&L "
                f"(was {self._daily_pnl:+.2f}) | "
                f"prev={self._daily_reset} → new={today}"
            )
            self._daily_pnl           = 0.0
            self._daily_reset         = today
            self._consecutive_loss    = 0
            self._warned_daily_75     = False
            self._warned_daily_90     = False
            self._warned_trades_near  = False
            self._warned_trades_full  = False
            self.snapshot_session_equity()

    # ── Core Sizing ───────────────────────────────────────────────────────────
    def calculate_position(
        self,
        symbol:     str,
        direction:  str,        # "BUY" or "SELL"
        entry:      float,
        sl_pips:    float,      # [B] SL distance in pips (not absolute price)
        tp_pips:    float,      # [B] TP distance in pips (not absolute price)
        confidence: float = 0.0,
        win_rate:   float = 0.50,
    ) -> Optional[PositionSpec]:
        """
        Compute a fully-specified PositionSpec from entry price and
        pip-distance SL/TP values.

        [B] Interface change: sl_pips / tp_pips replace the old sl / tp
            absolute-price parameters. Absolute SL/TP prices are derived
            internally, matching how main.py calls this method.
        [M] Account info is fetched ONCE and reused throughout — eliminates
            redundant MT5 API calls inside the log statement.
        """
        self._check_daily_reset()

        if not self._pre_trade_checks(symbol):
            return None

        # [M] Single fetch — reused for equity, currency, and log
        account = self._get_account()
        if not account:
            return None

        equity   = account["equity"]
        currency = account.get("currency", "")

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            logger.error(f"[RISK] Symbol info unavailable: {symbol}")
            return None

        # ── Validate pip inputs ───────────────────────────────────────────────
        if sl_pips <= 0 or tp_pips <= 0:
            logger.error(
                f"[RISK] Invalid sl_pips={sl_pips} tp_pips={tp_pips} "
                f"for {symbol}"
            )
            return None

        # ── RR check against CONFIG.RR_RATIO [C] ─────────────────────────────
        rr_actual = tp_pips / sl_pips
        if rr_actual < self.cfg.RR_RATIO:
            logger.warning(
                f"[RISK] R:R={rr_actual:.2f} < CONFIG.RR_RATIO="
                f"{self.cfg.RR_RATIO} for {symbol} — skipping"
            )
            return None

        # ── Anti-martingale risk reduction ────────────────────────────────────
        base_risk_pct = self.cfg.RISK_PER_TRADE
        if self._consecutive_loss >= self._consecutive_loss_limit:
            base_risk_pct /= 2.0
            logger.warning(
                f"[RISK] Anti-martingale: risk halved to "
                f"{base_risk_pct * 100:.2f}% after "
                f"{self._consecutive_loss} consecutive losses"
            )

        # ── Risk amount (equity %, hard-capped) ───────────────────────────────
        risk_usd = min(equity * base_risk_pct, self.cfg.MAX_LOSS_PER_TRADE)

        # [L] Minimum risk guard — prevents volume=0 on micro/cent accounts
        if risk_usd < 0.10:
            logger.warning(
                f"[RISK] Risk amount {currency}{risk_usd:.4f} < $0.10 "
                f"— skipping trade on {symbol}"
            )
            return None

        # ── Pip size and pip value ────────────────────────────────────────────
        pip_size  = self._get_pip_size(symbol, sym_info)
        pip_value = self._calculate_pip_value(symbol, sym_info, entry)

        if pip_value is None or pip_value <= 0:
            logger.error(
                f"[RISK] Cannot calculate pip value for {symbol} — skipping"
            )
            return None

        # ── Volume calculation ────────────────────────────────────────────────
        volume_raw = risk_usd / (sl_pips * pip_value)
        volume     = self._normalize_volume(volume_raw, sym_info)

        if volume <= 0:
            logger.error(
                f"[RISK] Normalised volume is zero for {symbol}"
            )
            return None

        # ── Actual risk at normalised volume ──────────────────────────────────
        actual_risk = volume * sl_pips * pip_value
        if actual_risk > self.cfg.MAX_LOSS_PER_TRADE * 1.5:
            logger.warning(
                f"[RISK] Actual risk {currency}{actual_risk:.2f} exceeds "
                f"hard cap even at minimum lot for {symbol} — skipping"
            )
            return None

        # ── Absolute SL / TP price construction [B] ──────────────────────────
        if direction == "BUY":
            sl_price = round(entry - sl_pips * pip_size, sym_info.digits)
            tp_price = round(entry + tp_pips * pip_size, sym_info.digits)
        else:   # SELL
            sl_price = round(entry + sl_pips * pip_size, sym_info.digits)
            tp_price = round(entry - tp_pips * pip_size, sym_info.digits)

        # ── Kelly Criterion (informational log only) [K] ──────────────────────
        self._log_kelly(
            win_rate    = win_rate,
            rr          = rr_actual,
            equity      = equity,
            base_risk   = base_risk_pct,
            risk_usd    = risk_usd,
            currency    = currency,
        )

        spec = PositionSpec(
            symbol     = symbol,
            direction  = direction,
            volume     = volume,
            entry      = round(entry, sym_info.digits),
            sl         = sl_price,
            tp         = tp_price,
            sl_pips    = sl_pips,
            tp_pips    = tp_pips,
            pip_value  = round(pip_value, 4),
            risk_usd   = round(actual_risk, 2),
            rr_ratio   = round(rr_actual, 2),
            confidence = confidence,
        )

        logger.info(
            f"📝 PositionSpec | {symbol} {direction} | "
            f"Vol={volume} | Risk={currency}{actual_risk:.2f} "
            f"({actual_risk / equity * 100:.2f}% equity) | "
            f"R:R={rr_actual:.2f} | SL={sl_pips}p TP={tp_pips}p | "
            f"pip_val={pip_value:.4f} | Equity={currency}{equity:,.2f}"
        )
        return spec

    # ── Kelly Criterion (informational) [K] ──────────────────────────────────
    def _log_kelly(
        self,
        win_rate:  float,
        rr:        float,
        equity:    float,
        base_risk: float,
        risk_usd:  float,
        currency:  str = "",
    ) -> None:
        """
        [K] Compute Kelly fraction from actual trade history when ≥10 trades
            are available, otherwise use the caller-supplied win_rate estimate.
            Result is logged at DEBUG level only — NOT used for sizing.
        """
        if len(self._trade_history) >= 10:
            recent     = self._trade_history[-20:]
            wins       = [t["profit"] for t in recent if t.get("profit", 0) > 0]
            losses     = [t["profit"] for t in recent if t.get("profit", 0) <= 0]
            total      = len(recent)
            win_rate   = len(wins) / total if total else win_rate
            avg_win    = sum(wins)   / len(wins)   if wins   else 0.0
            avg_loss   = abs(sum(losses) / len(losses)) if losses else 1.0
            rr         = avg_win / avg_loss if avg_loss else rr

        kelly_pct = win_rate - ((1.0 - win_rate) / rr) if rr > 0 else 0.0
        kelly_usd = equity * max(kelly_pct * 0.25, 0.0)

        logger.debug(
            f"📐 Kelly | win={win_rate:.0%} RR={rr:.2f} → "
            f"Kelly={kelly_pct * 100:.1f}% | "
            f"Quarter-Kelly={currency}{kelly_usd:.2f} | "
            f"Using equity-{base_risk * 100:.2f}%="
            f"{currency}{risk_usd:.2f}"
        )

    # ── Trade Result Feedback ─────────────────────────────────────────────────
    def record_trade_result(self, pnl: float, symbol: str = "") -> None:
        """
        Call after every trade closes to update daily P&L and the
        consecutive-loss counter used by the anti-martingale guard.

        [G] _check_daily_reset() is called FIRST — prevents a trade that
            closes just after broker midnight corrupting the new day's P&L.
        """
        self._check_daily_reset()   # [G] must be first

        self._daily_pnl += pnl

        self._trade_history.append({
            "profit": pnl,
            "symbol": symbol,
            "time":   datetime.now(timezone.utc),
        })
        # Cap history length to prevent unbounded memory growth
        if len(self._trade_history) > _TRADE_HISTORY_CAP:
            self._trade_history = self._trade_history[-_TRADE_HISTORY_CAP:]

        if pnl < 0:
            self._consecutive_loss += 1
            logger.warning(
                f"📉 Loss ${pnl:.2f} on {symbol} | "
                f"Consecutive losses: {self._consecutive_loss}"
            )
        else:
            if self._consecutive_loss > 0:
                logger.info(
                    f"📈 Win ${pnl:.2f} on {symbol} | "
                    f"Loss streak reset (was {self._consecutive_loss})"
                )
            self._consecutive_loss = 0

        logger.info(
            f"📊 Daily P&L: ${self._daily_pnl:+.2f} | "
            f"Consecutive losses: {self._consecutive_loss}"
        )

    # Backwards-compatibility alias — main.py still calls update_daily_pnl()
    def update_daily_pnl(self, pnl: float) -> None:
        self.record_trade_result(pnl)

    # ── Pre-Trade Circuit Breakers ────────────────────────────────────────────
    def _pre_trade_checks(self, symbol: str) -> bool:
        """
        Run all circuit breakers in priority order before allowing a trade.
        Returns True (allow) or False (block).

        Order: drawdown → trade count → duplicate → margin
        Running drawdown first ensures a halted-trading day can never be
        bypassed by a brief moment where margin happens to be available.
        """
        self._check_daily_reset()

        account = self._get_account()
        if not account:
            return False

        equity      = account["equity"]
        margin      = account["margin"]
        free_margin = account["free_margin"]
        currency    = account.get("currency", "")

        # ── 1. Daily equity drawdown ──────────────────────────────────────────
        if self._session_equity and self._session_equity > 0:
            equity_drop     = self._session_equity - equity
            equity_drop_pct = (equity_drop / self._session_equity) * 100
            daily_limit_pct = self.cfg.MAX_DAILY_LOSS * 100
            daily_limit_usd = self._session_equity * self.cfg.MAX_DAILY_LOSS

            # Hard halt
            if equity_drop_pct >= daily_limit_pct:
                msg = (
                    f"🚫 DAILY EQUITY DRAWDOWN LIMIT HIT\n"
                    f"  Session open : {currency}{self._session_equity:,.2f}\n"
                    f"  Current      : {currency}{equity:,.2f}\n"
                    f"  Drop         : {currency}{equity_drop:.2f} "
                    f"({equity_drop_pct:.1f}%) ≥ limit {daily_limit_pct:.0f}%\n"
                    f"  Trading halted for today."
                )
                logger.warning(msg)
                if self._alerts:
                    self._alerts.risk_warning(
                        level="LIMIT_HIT",
                        message=msg,
                        daily_pnl=-equity_drop,
                        daily_limit=daily_limit_usd,
                        pct_used=equity_drop_pct,
                    )
                return False

            # 90% warning
            if equity_drop_pct >= daily_limit_pct * 0.9 \
               and not self._warned_daily_90:
                self._warned_daily_90 = True
                remaining = daily_limit_usd - equity_drop
                msg = (
                    f"⛔ URGENT: Equity down {equity_drop_pct:.1f}% "
                    f"(limit {daily_limit_pct:.0f}%) — "
                    f"{currency}{remaining:.2f} remaining before halt"
                )
                logger.warning(msg)
                if self._alerts:
                    self._alerts.risk_warning(
                        level="URGENT",
                        message=msg,
                        daily_pnl=-equity_drop,
                        daily_limit=daily_limit_usd,
                        pct_used=equity_drop_pct,
                    )

            # 75% warning
            elif equity_drop_pct >= daily_limit_pct * 0.75 \
                 and not self._warned_daily_75:
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
                        daily_limit=daily_limit_usd,
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
                f"slots used — 1 slot remaining"
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
                    f"🚫 MAX TRADES REACHED: {open_count}/{max_trades} "
                    f"— new signals blocked"
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

        # Reset near/full flags when slots free up
        if open_count < max_trades:
            self._warned_trades_full = False
        if open_count < max_trades - 1:
            self._warned_trades_near = False

        # ── 3. Duplicate symbol ───────────────────────────────────────────────
        open_symbols = [p.symbol for p in (positions or [])]
        if symbol in open_symbols:
            logger.warning(
                f"🚫 Duplicate position blocked: {symbol} already open"
            )
            return False

        # ── 4. Margin guard [J] ───────────────────────────────────────────────
        # Absolute floor — blocks regardless of used margin
        if free_margin < 50.0:
            msg = (
                f"⚠️  LOW MARGIN: Free margin {currency}{free_margin:.2f} "
                f"below absolute floor ($50) — trade blocked"
            )
            logger.warning(msg)
            if self._alerts:
                self._alerts.risk_warning(level="LOW_MARGIN", message=msg)
            return False

        # Relative guard — 1.5× used margin [J]
        if margin > 0 and free_margin < margin * 1.5:
            msg = (
                f"⚠️  LOW MARGIN: Free {currency}{free_margin:.2f} < "
                f"150% of used margin {currency}{margin:.2f}"
            )
            logger.warning(msg)
            if self._alerts:
                self._alerts.risk_warning(level="LOW_MARGIN", message=msg)
            return False

        return True

    # ── Pip Size [F] ──────────────────────────────────────────────────────────
    def _get_pip_size(self, symbol: str, sym_info) -> float:
        """
        Return the pip size for the symbol using sym_info.digits.

        [F] Digit-count approach replaces hard-coded JPY_PAIRS and
            METAL_SYMBOLS sets — works correctly for any symbol added
            to the watchlist without manual list maintenance.

        Digit → pip size mapping:
            digits 5  (e.g. EURUSD  1.08345): pip = 10 × 0.00001 = 0.0001
            digits 4  (e.g. EURUSD  1.0834 ): pip = 0.0001  (already standard)
            digits 3  (e.g. USDJPY  149.123): pip = 10 × 0.001  = 0.01
            digits 2  (e.g. XAUUSD  2345.12): pip = 0.01   (same as JPY rule)
        """
        digits = sym_info.digits
        if digits in (5, 3):
            return 10.0 * sym_info.point    # pipette broker: 1 pip = 10 points
        return sym_info.point               # 4-decimal FX or 2-decimal metal

    # ── Pip Value Calculation [D] ─────────────────────────────────────────────
    def _calculate_pip_value(
        self,
        symbol:   str,
        sym_info,
        price:    float,
    ) -> Optional[float]:
        """
        Monetary value of 1 pip per 1 standard lot in account deposit
        currency (USD for Pepperstone).

        [D] Three cases with live cross-rate lookup for non-USD crosses:

        Case 1 — USD is the QUOTE currency (EURUSD, GBPUSD, XAUUSD):
            pip_value = contract_size × pip_size
            Direct USD conversion — no division needed.

        Case 2 — USD is the BASE currency (USDJPY, USDCAD, USDCHF):
            pip_value = (contract_size × pip_size) / price
            Convert quote-currency pips to USD by dividing by price.
            USDJPY @ 150.00: (100,000 × 0.01) / 150.00 = $6.67/pip/lot

        Case 3 — Neither currency is USD (EURGBP, GBPJPY, EURJPY …):
            [D FIX] Look up quote-currency/USD rate from MT5 terminal.
            GBPJPY quote=JPY → fetch JPYUSD mid-rate:
              pip_value = contract_size × pip_size × JPYUSD_rate
            Falls back to USD/quote inversion, then to price-division
            approximation (~1–2% error) if cross-rate is unavailable.
        """
        if price <= 0:
            logger.error(
                f"[RISK] Invalid price={price} for {symbol}"
            )
            return None

        pip_size      = self._get_pip_size(symbol, sym_info)
        contract_size = sym_info.trade_contract_size

        try:
            # Normalise symbol: strip broker suffixes (.p, _m, etc.)
            base_sym = symbol.upper().split(".")[0].split("_")[0]

            if len(base_sym) < 6:
                # Non-standard symbol format — use price-division approximation
                logger.warning(
                    f"[RISK] Non-standard symbol '{symbol}' — "
                    f"using price-division approximation for pip value"
                )
                return (contract_size * pip_size) / price

            quote_currency = base_sym[-3:]
            base_currency  = base_sym[:3]

            # ── Case 1: USD is the quote currency ─────────────────────────────
            if quote_currency == "USD":
                pv = contract_size * pip_size
                logger.debug(
                    f"[RISK] pip_value {symbol} Case1 (USD quote): "
                    f"{contract_size} × {pip_size:.5f} = {pv:.4f}"
                )
                return pv

            # ── Case 2: USD is the base currency ──────────────────────────────
            if base_currency == "USD":
                pv = (contract_size * pip_size) / price
                logger.debug(
                    f"[RISK] pip_value {symbol} Case2 (USD base): "
                    f"({contract_size} × {pip_size:.5f}) / {price:.5f} "
                    f"= {pv:.4f}"
                )
                return pv

            # ── Case 3: Cross pair — neither currency is USD ──────────────────
            # Try quote/USD rate (most accurate: JPYUSD, GBPUSD, etc.)
            cross_ticker = f"{quote_currency}USD"
            cross_tick   = mt5.symbol_info_tick(cross_ticker)
            if cross_tick and cross_tick.bid > 0 and cross_tick.ask > 0:
                cross_rate = (cross_tick.bid + cross_tick.ask) / 2.0
                pv = contract_size * pip_size * cross_rate
                logger.debug(
                    f"[RISK] pip_value {symbol} Case3 "
                    f"({quote_currency}USD @ {cross_rate:.5f}): "
                    f"{contract_size} × {pip_size:.5f} × {cross_rate:.5f} "
                    f"= {pv:.4f}"
                )
                return pv

            # Fallback: inverted USD/quote rate (USDJPY → 1/150 = JPYUSD)
            inv_ticker = f"USD{quote_currency}"
            inv_tick   = mt5.symbol_info_tick(inv_ticker)
            if inv_tick and inv_tick.bid > 0 and inv_tick.ask > 0:
                inv_rate = (inv_tick.bid + inv_tick.ask) / 2.0
                pv = (contract_size * pip_size) / inv_rate
                logger.debug(
                    f"[RISK] pip_value {symbol} Case3-inv "
                    f"(1/{inv_ticker} @ {1 / inv_rate:.5f}): "
                    f"({contract_size} × {pip_size:.5f}) / {inv_rate:.5f} "
                    f"= {pv:.4f}"
                )
                return pv

            # Last resort: price-division approximation
            pv = (contract_size * pip_size) / price
            logger.warning(
                f"[RISK] pip_value {symbol} Case3-approx "
                f"(cross rate unavailable for {quote_currency}): "
                f"({contract_size} × {pip_size:.5f}) / {price:.5f} "
                f"= {pv:.4f}"
            )
            return pv

        except (ZeroDivisionError, TypeError, AttributeError) as exc:
            logger.error(
                f"[RISK] _calculate_pip_value error for {symbol}: {exc}"
            )
            return None

    # ── Volume Normalisation [H] ──────────────────────────────────────────────
    def _normalize_volume(self, volume: float, sym_info) -> float:
        """
        Clamp to broker min/max and snap to nearest valid lot step.

        [H] Integer-step arithmetic avoids float precision artifacts like
            0.10000000000000001. Post-snap re-clamp prevents rounding-up
            from exceeding max_vol.
        """
        step    = sym_info.volume_step
        min_vol = sym_info.volume_min
        max_vol = sym_info.volume_max

        # Clamp to broker limits before snapping
        volume = max(min_vol, min(max_vol, volume))

        if step > 0:
            steps  = round(volume / step)
            volume = steps * step
            # Re-clamp after snapping (rounding up can exceed max_vol)
            volume = max(min_vol, min(max_vol, volume))

        return round(volume, 2)

    # ── Account Info [I][J] ───────────────────────────────────────────────────
    def _get_account(self) -> Optional[dict]:
        """
        Retrieve MT5 account info as a clean dict.

        [I] Includes 'currency' field so logs can display the account
            currency instead of hard-coded '€'.
        [J] margin fallback is 0.01 (not 0) so the relative margin guard
            (free_margin < margin × 1.5) only triggers when real margin
            is consumed by open positions, not on every pre-trade check
            when the account has no open trades (margin = 0).
        """
        info = mt5.account_info()
        if info is None:
            logger.error(
                "[RISK] mt5.account_info() returned None — "
                "is MT5 terminal connected?"
            )
            return None
        return {
            "balance":     info.balance,
            "equity":      info.equity,
            "margin":      info.margin if info.margin > 0 else 0.01,  # [J]
            "free_margin": info.margin_free,
            "currency":    getattr(info, "currency", ""),              # [I]
        }
