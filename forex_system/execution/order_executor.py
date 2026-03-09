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


def _get_filling_mode(symbol: str) -> int:
    """Return the first filling mode the broker actually supports for this symbol.

    Priority order: FOK → IOC → RETURN (most brokers accept at least one).
    Falls back to ORDER_FILLING_FOK if symbol info is unavailable.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.warning(f"symbol_info({symbol}) returned None — defaulting to FOK filling")
        return mt5.ORDER_FILLING_FOK

    filling_flags = info.filling_mode
    if filling_flags & mt5.ORDER_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    if filling_flags & mt5.ORDER_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


class OrderExecutor:

    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    def __init__(self, config=CONFIG):
        self.cfg = config

    # ── Market Order ──────────────────────────────────────────────────────────
    def send_market_order(self, spec: PositionSpec) -> Optional[dict]:
        tick = mt5.symbol_info_tick(spec.symbol)
        if tick is None:
            logger.error(f"No tick data for {spec.symbol} — cannot place order")
            return None

        order_type = (
            mt5.ORDER_TYPE_BUY if spec.direction == "BUY" else mt5.ORDER_TYPE_SELL
        )
        price = tick.ask if spec.direction == "BUY" else tick.bid

        request = {
            "action":        mt5.TRADE_ACTION_DEAL,
            "symbol":        spec.symbol,
            "volume":        spec.volume,
            "type":          order_type,
            "price":         price,
            "sl":            spec.sl,
            "tp":            spec.tp,
            "deviation":     self.cfg.SLIPPAGE,
            "magic":         self.cfg.MAGIC_NUMBER,
            "comment":       self.cfg.COMMENT,
            "type_time":     mt5.ORDER_TIME_GTC,
            "type_filling":  _get_filling_mode(spec.symbol),
        }

        return self._execute_with_retry(request, spec)

    # ── Trailing Stop ─────────────────────────────────────────────────────────
    def modify_trailing_stop(
        self,
        ticket:       int,
        symbol:       str,
        trail_points: int = 50,
    ) -> bool:
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.warning(f"Trailing stop: position {ticket} not found")
            return False
        pos = position[0]

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            logger.warning(f"Trailing stop: symbol_info({symbol}) returned None")
            return False

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.warning(f"Trailing stop: no tick for {symbol}")
            return False

        point       = sym_info.point
        trail_price = trail_points * point

        if pos.type == mt5.ORDER_TYPE_BUY:
            new_sl = tick.bid - trail_price
            if new_sl <= pos.sl:
                return True
        else:
            new_sl = tick.ask + trail_price
            if new_sl >= pos.sl:
                return True

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "ticket": ticket,
            "sl":     round(new_sl, sym_info.digits),
            "tp":     pos.tp,
        }
        result = mt5.order_send(request)
        success = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if not success:
            retcode = result.retcode if result else "None"
            logger.warning(f"Trailing stop modify failed for {ticket}: retcode={retcode}")
        return success

    # ── Close Position ────────────────────────────────────────────────────────
    def close_position(self, ticket: int) -> bool:
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"Position {ticket} not found")
            return False

        pos = positions[0]

        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            logger.error(f"No tick for {pos.symbol} — cannot close position {ticket}")
            return False

        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        close_type = (
            mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )

        request = {
            "action":        mt5.TRADE_ACTION_DEAL,
            "symbol":        pos.symbol,
            "volume":        pos.volume,
            "type":          close_type,
            "position":      ticket,
            "price":         price,
            "deviation":     self.cfg.SLIPPAGE,
            "magic":         self.cfg.MAGIC_NUMBER,
            "comment":       f"Close {ticket}",
            "type_filling":  _get_filling_mode(pos.symbol),
        }

        result = mt5.order_send(request)
        success = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if success:
            logger.info(f"✅ Closed position {ticket} on {pos.symbol}")
        else:
            retcode = result.retcode if result else "None"
            msg = RETCODE_MESSAGES.get(retcode, f"Code {retcode}")
            logger.error(f"❌ Failed to close {ticket}: {msg}")
        return success

    # ── Retry Logic ───────────────────────────────────────────────────────────
    def _execute_with_retry(self, request: dict, spec: PositionSpec) -> Optional[dict]:
        for attempt in range(1, self.MAX_RETRIES + 1):
            result = mt5.order_send(request)

            if result is None:
                logger.error(f"Attempt {attempt}: order_send returned None (terminal disconnected?)")
                time.sleep(self.RETRY_DELAY)
                continue

            msg = RETCODE_MESSAGES.get(result.retcode, f"Code {result.retcode}")
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
                tick = mt5.symbol_info_tick(spec.symbol)
                if tick is None:
                    logger.error(f"Requote: no tick for {spec.symbol} — aborting")
                    break
                request["price"] = tick.ask if spec.direction == "BUY" else tick.bid
                request["type_filling"] = _get_filling_mode(spec.symbol)
                time.sleep(self.RETRY_DELAY)

            else:
                logger.error(f"Order failed after attempt {attempt}: {msg}")
                break

        return None