# Order send & trade management
# execution/order_executor.py
import MetaTrader5 as mt5
import time
from typing import Optional
from risk.risk_manager import PositionSpec
from config.settings import CONFIG
from monitoring.logger import get_logger

logger = get_logger("OrderExecutor")

RETCODE_MESSAGES = {
    mt5.TRADE_RETCODE_DONE:      "✅ Order executed",
    mt5.TRADE_RETCODE_REJECT:    "❌ Rejected",
    mt5.TRADE_RETCODE_CANCEL:    "❌ Cancelled",
    mt5.TRADE_RETCODE_PLACED:    "⏳ Order placed",
    mt5.TRADE_RETCODE_REQUOTE:   "🔄 Requote",
    mt5.TRADE_RETCODE_NO_MONEY:  "💸 Insufficient funds",
    mt5.TRADE_RETCODE_INVALID:   "❌ Invalid request",
}

class OrderExecutor:

    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds

    def __init__(self, config=CONFIG):
        self.cfg = config

    # ── Market Order ──────────────────────────────────────
    def send_market_order(
        self, spec: PositionSpec
    ) -> Optional[dict]:

        order_type = (
            mt5.ORDER_TYPE_BUY
            if spec.direction == "BUY"
            else mt5.ORDER_TYPE_SELL
        )
        price = (
            mt5.symbol_info_tick(spec.symbol).ask
            if spec.direction == "BUY"
            else mt5.symbol_info_tick(spec.symbol).bid
        )

        request = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      spec.symbol,
            "volume":      spec.volume,
            "type":        order_type,
            "price":       price,
            "sl":          spec.sl,
            "tp":          spec.tp,
            "deviation":   self.cfg.SLIPPAGE,
            "magic":       self.cfg.MAGIC_NUMBER,
            "comment":     self.cfg.COMMENT,
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        return self._execute_with_retry(request, spec)

    # ── Trailing Stop ─────────────────────────────────────
    def modify_trailing_stop(
        self,
        ticket:       int,
        symbol:       str,
        trail_points: int = 50
    ) -> bool:
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        pos = position[0]

        sym_info    = mt5.symbol_info(symbol)
        tick        = mt5.symbol_info_tick(symbol)
        point       = sym_info.point
        trail_price = trail_points * point

        if pos.type == mt5.ORDER_TYPE_BUY:
            new_sl = tick.bid - trail_price
            if new_sl <= pos.sl:
                return True   # no update needed
        else:
            new_sl = tick.ask + trail_price
            if new_sl >= pos.sl:
                return True

        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "ticket":   ticket,
            "sl":       round(new_sl, sym_info.digits),
            "tp":       pos.tp,
        }
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE

    # ── Close Position ────────────────────────────────────
    def close_position(self, ticket: int) -> bool:
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"Position {ticket} not found")
            return False

        pos   = positions[0]
        tick  = mt5.symbol_info_tick(pos.symbol)
        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        close_type = (
            mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )

        request = {
            "action":   mt5.TRADE_ACTION_DEAL,
            "symbol":   pos.symbol,
            "volume":   pos.volume,
            "type":     close_type,
            "position": ticket,
            "price":    price,
            "deviation": self.cfg.SLIPPAGE,
            "magic":    self.cfg.MAGIC_NUMBER,
            "comment":  f"Close {ticket}",
        }
        result = mt5.order_send(request)
        success = result.retcode == mt5.TRADE_RETCODE_DONE
        if success:
            logger.info(f"✅ Closed position {ticket}")
        return success

    # ── Retry Logic ───────────────────────────────────────
    def _execute_with_retry(self, request: dict, spec: PositionSpec) -> Optional[dict]:
        for attempt in range(1, self.MAX_RETRIES + 1):
            result = mt5.order_send(request)
            msg    = RETCODE_MESSAGES.get(result.retcode, f"Code {result.retcode}")
            logger.info(f"  Attempt {attempt}: {msg}")

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"🚀 Trade executed | {spec.symbol} {spec.direction} | "
                    f"Vol={spec.volume} | Ticket={result.order}"
                )
                return {
                    "ticket":    result.order,
                    "symbol":    spec.symbol,
                    "direction": spec.direction,
                    "volume":    spec.volume,
                    "price":     result.price,
                    "sl":        spec.sl,
                    "tp":        spec.tp,
                }

            elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                # Re-fetch price on requote
                tick = mt5.symbol_info_tick(spec.symbol)
                request["price"] = (
                    tick.ask if spec.direction == "BUY" else tick.bid
                )
                time.sleep(self.RETRY_DELAY)

            else:
                logger.error(f"Order failed: {msg}")
                break

        return None
