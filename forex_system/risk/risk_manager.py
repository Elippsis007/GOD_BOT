# Position sizing & SL/TP logic
# risk/risk_manager.py
import MetaTrader5 as mt5
from dataclasses import dataclass
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
    """

    def __init__(self, config=CONFIG):
        self.cfg = config
        self._daily_pnl    = 0.0
        self._daily_reset  = None

    # ── Core Sizing ───────────────────────────────────────
    def calculate_position(
        self,
        symbol: str,
        direction: str,
        entry: float,
        sl: float,
        tp: float
    ) -> Optional[PositionSpec]:

        # Safety checks
        if not self._pre_trade_checks(symbol):
            return None

        account = self._get_account()
        if not account:
            return None

        balance    = account["balance"]
        sym_info   = mt5.symbol_info(symbol)
        if sym_info is None:
            logger.error(f"Symbol info unavailable: {symbol}")
            return None

        # Risk in account currency
        risk_usd   = balance * self.cfg.RISK_PER_TRADE
        sl_distance = abs(entry - sl)

        if sl_distance == 0:
            logger.error("SL distance is zero — aborting")
            return None

        # Pip value calculation
        pip_value  = sym_info.trade_contract_size * sym_info.point
        sl_pips    = sl_distance / sym_info.point
        volume_raw = risk_usd / (sl_pips * pip_value)

        # Normalize to allowed lot steps
        volume = self._normalize_volume(volume_raw, sym_info)

        # Validate reward:risk
        tp_distance = abs(tp - entry)
        rr = round(tp_distance / sl_distance, 2)
        if rr < 1.5:
            logger.warning(f"R:R={rr} too low — minimum 1.5 required")
            return None

        spec = PositionSpec(
            symbol=symbol,
            direction=direction,
            volume=volume,
            entry=round(entry, sym_info.digits),
            sl=round(sl, sym_info.digits),
            tp=round(tp, sym_info.digits),
            risk_usd=round(risk_usd, 2),
            rr_ratio=rr
        )
        logger.info(
            f"📐 Position | {symbol} {direction} | "
            f"Vol={volume} | Risk=${risk_usd:.2f} | R:R={rr}"
        )
        return spec

    # ── Pre-Trade Circuit Breakers ────────────────────────
    def _pre_trade_checks(self, symbol: str) -> bool:
        account = self._get_account()
        if not account:
            return False

        # 1. Daily loss limit
        if self._daily_pnl <= -(account["balance"] * self.cfg.MAX_DAILY_LOSS):
            logger.warning("🚫 Daily loss limit reached — no new trades")
            return False

        # 2. Max concurrent trades
        positions = mt5.positions_get()
        if positions and len(positions) >= self.cfg.MAX_OPEN_TRADES:
            logger.warning(f"🚫 Max {self.cfg.MAX_OPEN_TRADES} trades open")
            return False

        # 3. Duplicate symbol check
        open_symbols = [p.symbol for p in (positions or [])]
        if symbol in open_symbols:
            logger.warning(f"🚫 Already have open position on {symbol}")
            return False

        # 4. Margin check (require 200% free margin)
        if account["free_margin"] < account["margin"] * 2:
            logger.warning("🚫 Insufficient free margin")
            return False

        return True

    def _normalize_volume(self, volume: float, sym_info) -> float:
        step = sym_info.volume_step
        volume = round(volume / step) * step
        volume = max(sym_info.volume_min, min(volume, sym_info.volume_max))
        return round(volume, 2)

    def _get_account(self) -> Optional[dict]:
        info = mt5.account_info()
        if info is None:
            return None
        return {"balance": info.balance, "equity": info.equity,
                "margin": info.margin or 1, "free_margin": info.margin_free}

    def update_daily_pnl(self, pnl: float):
        self._daily_pnl += pnl
        logger.info(f"📊 Daily P&L updated: ${self._daily_pnl:.2f}")
