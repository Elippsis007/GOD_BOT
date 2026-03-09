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
    """Return the first filling mode the broker supports for this symbol.
    Priority: FOK → IOC → RETURN.
    Falls back to FOK if symbol info is unavailable.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.warning(
            f"symbol_info({symbol}) returned None — defaulting to FOK filling"
        )
        return mt5.ORDER_FILLING_FOK

    filling_flags = info.filling_mode
    if filling_flags & mt5.ORDER_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    if filling_flags & mt5.ORDER_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def _normalise_volume(symbol: str, volume: float) -> Optional[float]:
    """
    Clamp and round volume to broker's min / max / step constraints.

    Returns None if symbol info is unavailable or volume is zero after
    normalisation — caller should abort the order in that case.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error(f"_normalise_volume: symbol_info({symbol}) returned None")
        return None

    vol_min  = info.volume_min
    vol_max  = info.volume_max
    vol_step = info.volume_step

    # Clamp to broker limits
    volume = max(vol_min, min(vol_max, volume))

    # Round to nearest valid step
    if vol_step > 0:
        volume = round(round(volume / vol_step) * vol_step, 8)

    # Final sanity check
    if volume < vol_min:
        logger.error(
            f"_normalise_volume: {symbol} volume {volume:.4f} is below "
            f"broker minimum {vol_min} after normalisation — aborting"
        )
        return None

    return round(volume, 2)


def _validate_sl_tp(
    symbol:    str,
    direction: str,
    price:     float,
    sl:        float,
    tp:        float,
) -> bool:
    """
    Pre-flight SL/TP validation before sending to MT5.

    Checks:
      1. SL and TP are not zero or None.
      2. SL is on the correct side of price (BUY: sl < price, SELL: sl > price).
      3. TP is on the correct side of price (BUY: tp > price, SELL: tp < price).
      4. SL and TP are at least 1 point away from price (broker stop level).

    Returns True if valid, False if the order should be rejected.
    """
    if not sl or not tp:
        logger.error(
            f"_validate_sl_tp: {symbol} SL={sl} TP={tp} — "
            "zero or None SL/TP rejected"
        )
        return False

    sym_info = mt5.symbol_info(symbol)
    min_dist = (sym_info.point * max(sym_info.trade_stops_level, 1)
                if sym_info else 0.00001)

    if direction == "BUY":
        if sl >= price:
            logger.error(
                f"_validate_sl_tp: BUY {symbol} SL={sl:.5f} is at or above "
                f"entry={price:.5f} — rejected"
            )
            return False
        if tp <= price:
            logger.error(
                f"_validate_sl_tp: BUY {symbol} TP={tp:.5f} is at or below "
                f"entry={price:.5f} — rejected"
            )
            return False
        if (price - sl) < min_dist:
            logger.warning(
                f"_validate_sl_tp: BUY {symbol} SL distance "
                f"{(price - sl):.5f} < broker min {min_dist:.5f} — "
                "order may be rejected by broker"
            )
    else:  # SELL
        if sl <= price:
            logger.error(
                f"_validate_sl_tp: SELL {symbol} SL={sl:.5f} is at or below "
                f"entry={price:.5f} — rejected"
            )
            return False
        if tp >= price:
            logger.error(
                f"_validate_sl_tp: SELL {symbol} TP={tp:.5f} is at or above "
                f"entry={price:.5f} — rejected"
            )
            return False
        if (sl - price) < min_dist:
            logger.warning(
                f"_validate_sl_tp: SELL {symbol} SL distance "
                f"{(sl - price):.5f} < broker min {min_dist:.5f} — "
                "order may be rejected by broker"
            )

    return True


class OrderExecutor:

    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    def __init__(self, config=CONFIG):
        self.cfg = config

    # ── Market Order ──────────────────────────────────────────────────────────
    def send_market_order(self, spec: PositionSpec) -> Optional[dict]:
        tick = mt5.symbol_info_tick(spec.symbol)
        if tick is None:
            logger.error(
                f"No tick data for {spec.symbol} — cannot place order"
            )
            return None

        sym_info = mt5.symbol_info(spec.symbol)
        if sym_info is None:
            logger.error(
                f"No symbol info for {spec.symbol} — cannot place order"
            )
            return None

        order_type = (
            mt5.ORDER_TYPE_BUY
            if spec.direction == "BUY"
            else mt5.ORDER_TYPE_SELL
        )
        price = tick.ask if spec.direction == "BUY" else tick.bid

        # ── Fix 2: Normalise SL/TP to broker's digit precision ────────────
        digits = sym_info.digits
        sl     = round(spec.sl, digits)
        tp     = round(spec.tp, digits)

        # ── Fix 1: Validate SL/TP before sending ─────────────────────────
        if not _validate_sl_tp(spec.symbol, spec.direction, price, sl, tp):
            logger.error(
                f"Order aborted — invalid SL/TP for {spec.symbol} "
                f"{spec.direction} | SL={sl} TP={tp} Entry≈{price}"
            )
            return None

        # ── Fix 3: Normalise volume to broker constraints ─────────────────
        volume = _normalise_volume(spec.symbol, spec.volume)
        if volume is None:
            logger.error(
                f"Order aborted — volume normalisation failed for "
                f"{spec.symbol} vol={spec.volume}"
            )
            return None

        if volume != spec.volume:
            logger.info(
                f"Volume adjusted: {spec.volume:.4f} → {volume:.4f} "
                f"(broker constraints for {spec.symbol})"
            )

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       spec.symbol,
            "volume":       volume,
            "type":         order_type,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "deviation":    self.cfg.SLIPPAGE,
            "magic":        self.cfg.MAGIC_NUMBER,
            "comment":      self.cfg.COMMENT,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": _get_filling_mode(spec.symbol),
        }

        logger.info(
            f"📤 Sending order | {spec.symbol} {spec.direction} | "
            f"Vol={volume} | Entry≈{price:.{digits}f} | "
            f"SL={sl:.{digits}f} | TP={tp:.{digits}f}"
        )

        # Update spec with normalised values for return dict consistency
        spec_sl = sl
        spec_tp = tp

        return self._execute_with_retry(request, spec, spec_sl, spec_tp, volume)

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
            logger.warning(
                f"Trailing stop: symbol_info({symbol}) returned None"
            )
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
                return True   # existing SL is already better — no update needed
        else:
            new_sl = tick.ask + trail_price
            if new_sl >= pos.sl:
                return True   # existing SL is already better — no update needed

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "ticket": ticket,
            "sl":     round(new_sl, sym_info.digits),
            "tp":     pos.tp,
        }
        result  = mt5.order_send(request)
        success = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if not success:
            retcode = result.retcode if result else "None"
            logger.warning(
                f"Trailing stop modify failed for {ticket}: retcode={retcode}"
            )
        return success

    # ── Close Position ────────────────────────────────────────────────────────
    def close_position(self, ticket: int) -> Optional[dict]:
        """
        Close an open position by ticket number.

        Returns a dict with close details on success, None on failure.
        Return type changed from bool to Optional[dict] for consistency
        with send_market_order() — callers can still do `if result:`.
        """
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"Position {ticket} not found")
            return None

        pos = positions[0]

        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            logger.error(
                f"No tick for {pos.symbol} — cannot close position {ticket}"
            )
            return None

        sym_info = mt5.symbol_info(pos.symbol)
        digits   = sym_info.digits if sym_info else 5

        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        close_type = (
            mt5.ORDER_TYPE_SELL
            if pos.type == mt5.ORDER_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       pos.volume,
            "type":         close_type,
            "position":     ticket,
            "price":        price,
            "deviation":    self.cfg.SLIPPAGE,
            "magic":        self.cfg.MAGIC_NUMBER,
            "comment":      f"Close {ticket}",
            "type_filling": _get_filling_mode(pos.symbol),
        }

        result  = mt5.order_send(request)
        success = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

        if success:
            logger.info(
                f"✅ Closed position {ticket} on {pos.symbol} "
                f"@ {price:.{digits}f}"
            )
            return {
                "ticket":    ticket,
                "symbol":    pos.symbol,
                "price":     price,
                "volume":    pos.volume,
            }
        else:
            retcode = result.retcode if result else "None"
            msg     = RETCODE_MESSAGES.get(retcode, f"Code {retcode}")
            logger.error(f"❌ Failed to close {ticket}: {msg}")
            return None

    # ── Retry Logic ───────────────────────────────────────────────────────────
    def _execute_with_retry(
        self,
        request:    dict,
        spec:       PositionSpec,
        sl:         float,
        tp:         float,
        volume:     float,
    ) -> Optional[dict]:
        for attempt in range(1, self.MAX_RETRIES + 1):
            result = mt5.order_send(request)

            if result is None:
                logger.error(
                    f"Attempt {attempt}: order_send returned None "
                    "(terminal disconnected?)"
                )
                time.sleep(self.RETRY_DELAY)
                continue

            msg = RETCODE_MESSAGES.get(result.retcode, f"Code {result.retcode}")
            logger.info(f"  Attempt {attempt}: {msg}")

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"🚀 Trade executed | {spec.symbol} {spec.direction} | "
                    f"Vol={volume} | Ticket={result.order} | "
                    f"Price={result.price} | SL={sl} | TP={tp}"
                )
                return {
                    "ticket":    result.order,
                    "symbol":    spec.symbol,
                    "direction": spec.direction,
                    "volume":    volume,
                    "price":     result.price,
                    "sl":        sl,
                    "tp":        tp,
                }

            elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                tick = mt5.symbol_info_tick(spec.symbol)
                if tick is None:
                    logger.error(
                        f"Requote: no tick for {spec.symbol} — aborting"
                    )
                    break
                # Refresh price only — SL/TP and volume stay validated
                request["price"] = (
                    tick.ask if spec.direction == "BUY" else tick.bid
                )
                request["type_filling"] = _get_filling_mode(spec.symbol)
                logger.info(
                    f"  Requote handled — new price "
                    f"{request['price']:.5f}, retrying…"
                )
                time.sleep(self.RETRY_DELAY)

            else:
                logger.error(
                    f"Order failed after attempt {attempt}: {msg} "
                    f"(retcode={result.retcode})"
                )
                break

        logger.error(
            f"❌ Order aborted after {self.MAX_RETRIES} attempts — "
            f"{spec.symbol} {spec.direction}"
        )
        return None
