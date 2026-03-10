# =============================================================================
# GODBOT v3.0 – execution/order_executor.py  (PRODUCTION – fully corrected)
# =============================================================================
#
#  Fixes applied vs the repo baseline:
#
#  [A]  send_market_order() and _execute_with_retry() logs now include
#       sl_pips, tp_pips, pip_value, and confidence from the updated
#       PositionSpec. Return dict also includes these fields.
#
#  [B]  _validate_sl_tp() fallback min_dist uses _estimate_point(symbol)
#       instead of hard-coded 0.00001 — fixes JPY and metals being 100×
#       too small when sym_info is unavailable.
#
#  [C]  modify_trailing_stop() now accepts trail_pips (pip distance) and
#       converts to price distance internally using pip_size from digits.
#       pip_to_points=True (default) — set False for raw-points legacy.
#
#  [D]  close_position() refreshes tick price on every filling-mode attempt,
#       not just once before the loop.
#
#  [E]  _execute_with_retry() uses explicit retcode + comment logging.
#       No more mt5.last_error() calls (library error ≠ order error).
#
#  [F]  _normalise_volume() integer-step arithmetic + post-snap re-clamp.
#
#  [G]  Added modify_sl_tp() — dedicated SL/TP update for breakeven moves
#       and partial-TP scaling. Replaces misuse of modify_trailing_stop().
#
#  [H]  RETCODE_MESSAGES and unrecoverable retcodes built lazily via
#       functions, not at module import time, to prevent AttributeError
#       before MT5 is initialised.
#
#  [I]  Added get_open_position() helper — returns live position as a
#       clean dict so main.py's monitor does not need to import mt5 directly.
#
#  [J]  _execute_with_retry() only hard-aborts on genuinely unrecoverable
#       codes (INVALID, NO_MONEY, TRADE_DISABLED). Transient codes
#       (TIMEOUT, PRICE_CHANGED, PRICE_OFF) are retried.
#
#  [K]  Filling mode priority changed to RETURN → IOC → FOK.
#       Pepperstone ECN uses RETURN natively — FOK-first wastes one retry.
#
#  [L]  RETRY_DELAY reduced from 1.0 s to 0.5 s for M5/M1 scalping speed.
#
#  [M]  Price refreshed unconditionally on every retry attempt, not only
#       on REQUOTE, preventing repeated PRICE_CHANGED rejections.
#
# =============================================================================

import MetaTrader5 as mt5
import time
from typing import Optional
from risk.risk_manager import PositionSpec
from config.settings import CONFIG
from monitoring.logger import get_logger

logger = get_logger("OrderExecutor")


# ── Lazy retcode message map [H] ──────────────────────────────────────────────
def _build_retcode_messages() -> dict:
    """
    [H] Build the retcode → human-readable message dict at call time,
    not at module import time. MT5 constants are unavailable before
    mt5.initialize() is called on some platforms (AttributeError).
    Returns an empty dict if MT5 is not yet initialised.
    """
    try:
        return {
            mt5.TRADE_RETCODE_DONE:              "✅ Order executed",
            mt5.TRADE_RETCODE_REJECT:            "❌ Rejected",
            mt5.TRADE_RETCODE_CANCEL:            "❌ Cancelled",
            mt5.TRADE_RETCODE_PLACED:            "⏳ Order placed",
            mt5.TRADE_RETCODE_REQUOTE:           "🔄 Requote",
            mt5.TRADE_RETCODE_NO_MONEY:          "💸 Insufficient funds",
            mt5.TRADE_RETCODE_INVALID:           "❌ Invalid request",
            mt5.TRADE_RETCODE_INVALID_FILL:      "❌ Unsupported filling mode",
            mt5.TRADE_RETCODE_TIMEOUT:           "⏱️ Timeout",
            mt5.TRADE_RETCODE_PRICE_CHANGED:     "💱 Price changed",
            mt5.TRADE_RETCODE_PRICE_OFF:         "💱 Off quotes",
            mt5.TRADE_RETCODE_CONNECTION:        "🔌 No connection",
            mt5.TRADE_RETCODE_TOO_MANY_REQUESTS: "🚦 Too many requests",
        }
    except AttributeError:
        return {}   # MT5 not yet initialised — caller gets generic "Code N" label


# ── Unrecoverable retcodes [J] ────────────────────────────────────────────────
def _unrecoverable_retcodes() -> set:
    """
    [J] Only these three codes are genuinely unrecoverable at order time.
    All others (TIMEOUT, PRICE_CHANGED, PRICE_OFF, CONNECTION) are
    transient and worth retrying.
    Returns an empty set if MT5 is not yet initialised.
    """
    try:
        return {
            mt5.TRADE_RETCODE_INVALID,
            mt5.TRADE_RETCODE_NO_MONEY,
            mt5.TRADE_RETCODE_TRADE_DISABLED,
        }
    except AttributeError:
        return set()


# ── Symbol point estimator [B] ────────────────────────────────────────────────
def _estimate_point(symbol: str) -> float:
    """
    [B] Best-guess point size based on the symbol name.
    Used as a fallback in _validate_sl_tp() when sym_info is None.
    Without this, JPY pairs get 0.00001 (100× too small) and metals
    get 0.00001 (10× too small), making the stop-level warning useless.
    """
    sym = symbol.upper()
    if "JPY" in sym:
        return 0.001    # 3-digit JPY pairs (e.g. USDJPY 149.123)
    if any(m in sym for m in ("XAU", "XAG", "GOLD", "SILVER")):
        return 0.01     # 2-digit metals (e.g. XAUUSD 2345.12)
    return 0.00001      # 5-digit standard FX (e.g. EURUSD 1.08345)


# ── Filling mode selector [K] ─────────────────────────────────────────────────
def _get_filling_mode(symbol: str) -> int:
    """
    Return the broker's preferred filling mode for this symbol.

    [K] Priority changed to RETURN → IOC → FOK.
        Pepperstone ECN accounts use RETURN natively. Sending FOK first
        produces an INVALID_FILL rejection, wastes one retry attempt
        (0.5 s), then falls through to RETURN anyway. On an M5 scalper
        that is unacceptable latency for no gain.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.warning(
            f"[OE] symbol_info({symbol}) returned None — "
            f"defaulting to RETURN filling"
        )
        return mt5.ORDER_FILLING_RETURN

    flags = info.filling_mode
    logger.debug(
        f"[OE] _get_filling_mode [{symbol}] flags={flags} | "
        f"RETURN={bool(flags & mt5.ORDER_FILLING_RETURN)} "
        f"IOC={bool(flags & mt5.ORDER_FILLING_IOC)} "
        f"FOK={bool(flags & mt5.ORDER_FILLING_FOK)}"
    )

    if flags & mt5.ORDER_FILLING_RETURN:
        return mt5.ORDER_FILLING_RETURN
    if flags & mt5.ORDER_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_FOK


# ── Volume normalisation [F] ──────────────────────────────────────────────────
def _normalise_volume(symbol: str, volume: float) -> Optional[float]:
    """
    Clamp and snap volume to broker min / max / step constraints.

    [F] Integer-step arithmetic prevents float artifacts like
        0.10000000000000001. Post-snap re-clamp prevents rounding
        up past vol_max. Returns None if sym_info unavailable or
        volume rounds below broker minimum.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error(
            f"[OE] _normalise_volume: symbol_info({symbol}) returned None"
        )
        return None

    vol_min  = info.volume_min
    vol_max  = info.volume_max
    vol_step = info.volume_step

    # Clamp before snapping
    volume = max(vol_min, min(vol_max, volume))

    # [F] Integer-step snap
    if vol_step > 0:
        steps  = round(volume / vol_step)
        volume = steps * vol_step
        # Re-clamp — rounding up could exceed vol_max
        volume = max(vol_min, min(vol_max, volume))

    volume = round(volume, 2)

    if volume < vol_min:
        logger.error(
            f"[OE] _normalise_volume: {symbol} volume {volume:.4f} "
            f"below broker min {vol_min} after normalisation — aborting"
        )
        return None

    return volume


# ── SL/TP validator [B] ───────────────────────────────────────────────────────
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
      1. SL and TP are non-zero and non-None.
      2. SL is on the correct side of the entry price.
      3. TP is on the correct side of the entry price.
      4. SL / TP distances respect the broker minimum stop level.

    [B] Fallback min_dist uses _estimate_point(symbol) so JPY pairs
        (point ≈ 0.001) and metals (point ≈ 0.01) get appropriate
        minimums when sym_info is unavailable, not a hard-coded 0.00001
        which is 100× too small for JPY.
    """
    if not sl or not tp:
        logger.error(
            f"[OE] _validate_sl_tp: {symbol} SL={sl} TP={tp} — "
            "zero or None rejected"
        )
        return False

    sym_info = mt5.symbol_info(symbol)
    if sym_info:
        stop_level = max(sym_info.trade_stops_level, 1)
        min_dist   = sym_info.point * stop_level
    else:
        # [B] Symbol-name-based fallback — sensible for JPY and metals
        min_dist = _estimate_point(symbol) * 10

    if direction == "BUY":
        if sl >= price:
            logger.error(
                f"[OE] _validate_sl_tp: BUY {symbol} "
                f"SL={sl:.5f} ≥ entry={price:.5f} — rejected"
            )
            return False
        if tp <= price:
            logger.error(
                f"[OE] _validate_sl_tp: BUY {symbol} "
                f"TP={tp:.5f} ≤ entry={price:.5f} — rejected"
            )
            return False
        if (price - sl) < min_dist:
            logger.warning(
                f"[OE] _validate_sl_tp: BUY {symbol} "
                f"SL dist {price - sl:.5f} < broker min {min_dist:.5f} — "
                "order may be rejected"
            )
    else:  # SELL
        if sl <= price:
            logger.error(
                f"[OE] _validate_sl_tp: SELL {symbol} "
                f"SL={sl:.5f} ≤ entry={price:.5f} — rejected"
            )
            return False
        if tp >= price:
            logger.error(
                f"[OE] _validate_sl_tp: SELL {symbol} "
                f"TP={tp:.5f} ≥ entry={price:.5f} — rejected"
            )
            return False
        if (sl - price) < min_dist:
            logger.warning(
                f"[OE] _validate_sl_tp: SELL {symbol} "
                f"SL dist {sl - price:.5f} < broker min {min_dist:.5f} — "
                "order may be rejected"
            )

    return True


# ── OrderExecutor ─────────────────────────────────────────────────────────────
class OrderExecutor:
    """
    Handles all MT5 order submission, modification, and closure.

    Execution guarantees:
      - Filling mode priority: RETURN → IOC → FOK (Pepperstone ECN safe)
      - Price refreshed on every retry attempt
      - Transient errors (TIMEOUT, PRICE_CHANGED) retried; fatal errors abort
      - Volume snapped to broker lot step with integer arithmetic
      - SL/TP validated before sending; direction and distance both checked
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 0.5   # [L] 0.5 s — fast enough for M5, still within M1 budget

    def __init__(self, config=CONFIG):
        self.cfg = config

    # ── Market Order ──────────────────────────────────────────────────────────
    def send_market_order(self, spec: PositionSpec) -> Optional[dict]:
        """
        Place a market order from a fully-populated PositionSpec.

        [A] Confirmation log includes sl_pips, tp_pips, pip_value, and
            confidence so the operator can verify pip distances and risk
            without reverse-engineering from absolute prices.
        """
        tick = mt5.symbol_info_tick(spec.symbol)
        if tick is None:
            logger.error(
                f"[OE] No tick data for {spec.symbol} — cannot place order"
            )
            return None

        sym_info = mt5.symbol_info(spec.symbol)
        if sym_info is None:
            logger.error(
                f"[OE] No symbol info for {spec.symbol} — cannot place order"
            )
            return None

        order_type = (
            mt5.ORDER_TYPE_BUY
            if spec.direction == "BUY"
            else mt5.ORDER_TYPE_SELL
        )
        price  = tick.ask if spec.direction == "BUY" else tick.bid
        digits = sym_info.digits
        sl     = round(spec.sl, digits)
        tp     = round(spec.tp, digits)

        if not _validate_sl_tp(spec.symbol, spec.direction, price, sl, tp):
            logger.error(
                f"[OE] Order aborted — SL/TP validation failed | "
                f"{spec.symbol} {spec.direction} | "
                f"SL={sl} TP={tp} Entry≈{price}"
            )
            return None

        volume = _normalise_volume(spec.symbol, spec.volume)
        if volume is None:
            logger.error(
                f"[OE] Order aborted — volume normalisation failed | "
                f"{spec.symbol} vol={spec.volume}"
            )
            return None

        if volume != spec.volume:
            logger.info(
                f"[OE] Volume adjusted: {spec.volume:.4f} → {volume:.4f} "
                f"({spec.symbol} broker constraints)"
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

        # [A] Full confirmation log — pip distances visible at a glance
        logger.info(
            f"📤 Sending order | {spec.symbol} {spec.direction} | "
            f"Vol={volume} | Entry≈{price:.{digits}f} | "
            f"SL={sl:.{digits}f} ({spec.sl_pips:.1f}p) | "
            f"TP={tp:.{digits}f} ({spec.tp_pips:.1f}p) | "
            f"PipVal=${spec.pip_value:.4f} | "
            f"Risk=${spec.risk_usd:.2f} | "
            f"RR={spec.rr_ratio:.2f} | "
            f"Conf={spec.confidence:.0%}"
        )
        logger.debug(
            f"[OE] Request dump: "
            f"action={request['action']} symbol={request['symbol']} "
            f"volume={request['volume']} type={request['type']} "
            f"price={request['price']} sl={request['sl']} tp={request['tp']} "
            f"deviation={request['deviation']} "
            f"filling={request['type_filling']} "
            f"magic={request['magic']} comment={request['comment']}"
        )

        return self._execute_with_retry(request, spec, sl, tp, volume)

    # ── Modify SL and TP [G] ──────────────────────────────────────────────────
    def modify_sl_tp(
        self,
        ticket: int,
        new_sl: float,
        new_tp: float,
    ) -> bool:
        """
        [G] Update SL and/or TP on an existing position without any
        trailing-distance logic — used for breakeven moves, partial TP
        scaling, and manual adjustments.

        Pass new_sl=0.0 to keep existing SL.
        Pass new_tp=0.0 to keep existing TP.
        """
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"[OE] modify_sl_tp: ticket {ticket} not found")
            return False

        pos      = positions[0]
        sym_info = mt5.symbol_info(pos.symbol)
        digits   = sym_info.digits if sym_info else 5

        final_sl = round(new_sl, digits) if new_sl else pos.sl
        final_tp = round(new_tp, digits) if new_tp else pos.tp

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "ticket": ticket,
            "sl":     final_sl,
            "tp":     final_tp,
        }

        result  = mt5.order_send(request)
        success = (
            result is not None and
            result.retcode == mt5.TRADE_RETCODE_DONE
        )

        if success:
            logger.info(
                f"[OE] ✅ modify_sl_tp {ticket} ({pos.symbol}) | "
                f"SL={final_sl:.{digits}f} TP={final_tp:.{digits}f}"
            )
        else:
            retcode = result.retcode if result else "None"
            comment = result.comment if result else "N/A"
            logger.warning(
                f"[OE] modify_sl_tp failed for {ticket}: "
                f"retcode={retcode} comment='{comment}'"
            )

        return success

    # ── Trailing Stop [C] ─────────────────────────────────────────────────────
    def modify_trailing_stop(
        self,
        ticket:        int,
        symbol:        str,
        trail_pips:    float = 8.0,
        pip_to_points: bool  = True,
    ) -> bool:
        """
        Update the SL of an open position to trail the current price
        by trail_pips pips.

        [C] FIX: The repo used trail_points × point (raw MT5 points =
            pipettes). For a standard 5-digit pair, point = 0.00001,
            so trail_points=50 gives 50 × 0.00001 = 0.0005 = 0.5 pips —
            10× too tight for an 8-pip trailing stop target.

            pip_to_points=True (default): interprets trail_pips as PIPS
            and converts to price distance using the correct pip_size.

            pip_to_points=False: treats trail_pips as raw MT5 points
            (legacy mode — for any caller that already does the conversion).

        Only moves the SL if the new level is better (tighter) than the
        existing SL — never widens the stop.
        """
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(
                f"[OE] modify_trailing_stop: ticket {ticket} not found"
            )
            return False
        pos = positions[0]

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            logger.warning(
                f"[OE] modify_trailing_stop: "
                f"symbol_info({symbol}) returned None"
            )
            return False

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.warning(
                f"[OE] modify_trailing_stop: no tick for {symbol}"
            )
            return False

        # [C] Convert pips → price distance using digit-count pip_size
        if pip_to_points:
            digits     = sym_info.digits
            pip_size   = 10.0 * sym_info.point if digits in (5, 3) else sym_info.point
            trail_dist = trail_pips * pip_size
        else:
            trail_dist = trail_pips * sym_info.point   # legacy raw-points path

        digits = sym_info.digits

        if pos.type == mt5.ORDER_TYPE_BUY:
            new_sl = tick.bid - trail_dist
            # Only tighten — never move SL backwards on a long
            if new_sl <= pos.sl:
                return True
        else:  # SELL
            new_sl = tick.ask + trail_dist
            # Only tighten — never move SL backwards on a short
            if new_sl >= pos.sl:
                return True

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "ticket": ticket,
            "sl":     round(new_sl, digits),
            "tp":     pos.tp,
        }

        result  = mt5.order_send(request)
        success = (
            result is not None and
            result.retcode == mt5.TRADE_RETCODE_DONE
        )

        if success:
            logger.debug(
                f"[OE] ✅ Trailing stop {ticket} ({symbol}) updated | "
                f"new_SL={round(new_sl, digits):.{digits}f} | "
                f"trail={trail_pips:.1f}pip"
            )
        else:
            retcode = result.retcode if result else "None"
            logger.warning(
                f"[OE] Trailing stop modify failed {ticket}: "
                f"retcode={retcode}"
            )

        return success

    # ── Close Position [D] ────────────────────────────────────────────────────
    def close_position(self, ticket: int) -> Optional[dict]:
        """
        Close an open position by ticket, cycling all three filling modes.

        [D] Tick price is refreshed at the start of every filling-mode
            attempt. On a busy M5 scalper session a retry takes 0.2–0.5 s
            and the prior price fetch will be stale, causing PRICE_CHANGED
            rejections on every attempt. Refreshing per-attempt fixes this.
        """
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"[OE] close_position: ticket {ticket} not found")
            return None

        pos      = positions[0]
        sym_info = mt5.symbol_info(pos.symbol)
        digits   = sym_info.digits if sym_info else 5

        close_type = (
            mt5.ORDER_TYPE_SELL
            if pos.type == mt5.ORDER_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )

        filling_modes = [
            mt5.ORDER_FILLING_RETURN,
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_FOK,
        ]

        for filling in filling_modes:
            # [D] Fresh tick on every attempt
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                logger.error(
                    f"[OE] close_position: no tick for {pos.symbol} "
                    f"on filling={filling} attempt"
                )
                continue

            price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       pos.symbol,
                "volume":       pos.volume,
                "type":         close_type,
                "position":     ticket,
                "price":        price,
                "deviation":    self.cfg.SLIPPAGE,
                "magic":        self.cfg.MAGIC_NUMBER,
                "comment":      f"GODBOT_v3 close #{ticket}",
                "type_filling": filling,
            }

            result  = mt5.order_send(request)
            success = (
                result is not None and
                result.retcode == mt5.TRADE_RETCODE_DONE
            )

            if success:
                logger.info(
                    f"[OE] ✅ Closed #{ticket} {pos.symbol} "
                    f"@ {price:.{digits}f} | "
                    f"Vol={pos.volume} | filling={filling}"
                )
                return {
                    "ticket": ticket,
                    "symbol": pos.symbol,
                    "price":  price,
                    "volume": pos.volume,
                }

            retcode = result.retcode if result else "None"
            comment = result.comment if result else "N/A"
            logger.debug(
                f"[OE] close filling={filling} failed: "
                f"retcode={retcode} comment='{comment}'"
            )
            time.sleep(0.2)

        logger.error(
            f"[OE] ❌ Failed to close #{ticket} — all filling modes exhausted"
        )
        return None

    # ── Get Open Position [I] ─────────────────────────────────────────────────
    def get_open_position(self, ticket: int) -> Optional[dict]:
        """
        [I] Return live position data as a clean dict.

        Allows main.py's _monitor_positions() to check SL proximity,
        live P&L, and hold duration without importing mt5 directly,
        keeping the execution layer as the single MT5 access point.

        Returns None if the position is no longer open (TP/SL hit,
        manually closed, or broker-closed).
        """
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return None

        pos      = positions[0]
        sym_info = mt5.symbol_info(pos.symbol)
        digits   = sym_info.digits if sym_info else 5

        tick  = mt5.symbol_info_tick(pos.symbol)
        price = (
            (tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask)
            if tick else pos.price_open
        )

        return {
            "ticket":     pos.ticket,
            "symbol":     pos.symbol,
            "direction":  "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
            "volume":     pos.volume,
            "open_price": pos.price_open,
            "sl":         pos.sl,
            "tp":         pos.tp,
            "current":    price,
            "profit":     pos.profit,
            "digits":     digits,
            "open_time":  pos.time,
        }

    # ── Retry Logic [E][J][M] ─────────────────────────────────────────────────
    def _execute_with_retry(
        self,
        request: dict,
        spec:    PositionSpec,
        sl:      float,
        tp:      float,
        volume:  float,
    ) -> Optional[dict]:
        """
        Try all three filling modes across MAX_RETRIES attempts.

        [E]  Explicit retcode + comment logging — no more mt5.last_error()
             which returns library-level errors, not order-level errors.
        [J]  Only INVALID, NO_MONEY, TRADE_DISABLED abort immediately.
             Transient codes (TIMEOUT, PRICE_CHANGED, PRICE_OFF) are retried.
        [M]  Price refreshed unconditionally on every attempt, not only
             on REQUOTE, preventing repeated PRICE_CHANGED rejections.
        """
        filling_modes = [
            mt5.ORDER_FILLING_RETURN,
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_FOK,
        ]
        unrecoverable = _unrecoverable_retcodes()
        retcode_msgs  = _build_retcode_messages()

        for attempt in range(1, self.MAX_RETRIES + 1):

            # [K] Cycle filling modes across attempts
            filling = filling_modes[(attempt - 1) % len(filling_modes)]
            request["type_filling"] = filling

            # [M] Always refresh price — stale price causes PRICE_CHANGED loop
            tick = mt5.symbol_info_tick(spec.symbol)
            if tick is None:
                logger.error(
                    f"[OE] Attempt {attempt}: no tick for "
                    f"{spec.symbol} — aborting"
                )
                break

            request["price"] = (
                tick.ask if spec.direction == "BUY" else tick.bid
            )

            logger.debug(
                f"[OE] Attempt {attempt}/{self.MAX_RETRIES} | "
                f"filling={filling} | price={request['price']:.5f}"
            )

            result = mt5.order_send(request)

            if result is None:
                logger.error(
                    f"[OE] Attempt {attempt}: order_send() returned None — "
                    "terminal may be disconnected"
                )
                time.sleep(self.RETRY_DELAY)
                continue

            # [E] Clean diagnostic log — retcode + broker comment
            msg = retcode_msgs.get(result.retcode, f"Code {result.retcode}")
            logger.info(
                f"[OE] Attempt {attempt} [filling={filling}]: "
                f"{msg} (retcode={result.retcode}) "
                f"comment='{result.comment}'"
            )

            # ── Success ───────────────────────────────────────────────────────
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"🚀 Trade executed | "
                    f"{spec.symbol} {spec.direction} | "
                    f"Ticket={result.order} | "
                    f"Vol={volume} | Price={result.price:.5f} | "
                    f"SL={sl:.5f} ({spec.sl_pips:.1f}p) | "
                    f"TP={tp:.5f} ({spec.tp_pips:.1f}p) | "
                    f"Risk=${spec.risk_usd:.2f} | "
                    f"Conf={spec.confidence:.0%} | "
                    f"filling={filling}"
                )
                # [A] Return dict includes pip distances and risk for callers
                return {
                    "ticket":     result.order,
                    "symbol":     spec.symbol,
                    "direction":  spec.direction,
                    "volume":     volume,
                    "price":      result.price,
                    "sl":         sl,
                    "tp":         tp,
                    "sl_pips":    spec.sl_pips,
                    "tp_pips":    spec.tp_pips,
                    "risk_usd":   spec.risk_usd,
                    "confidence": spec.confidence,
                }

            # ── Requote — price already refreshed above, fall through ─────────
            elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                logger.info(
                    f"[OE] Requote on attempt {attempt} — "
                    f"new price={request['price']:.5f}, retrying…"
                )
                time.sleep(self.RETRY_DELAY)
                continue

            # ── Bad filling mode — try next mode ──────────────────────────────
            elif result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
                logger.warning(
                    f"[OE] Filling={filling} rejected — "
                    f"trying next mode on attempt {attempt + 1}"
                )
                time.sleep(self.RETRY_DELAY)
                continue

            # ── [J] Unrecoverable — abort immediately ─────────────────────────
            elif result.retcode in unrecoverable:
                logger.error(
                    f"[OE] Unrecoverable on attempt {attempt}: "
                    f"{msg} (retcode={result.retcode}) — aborting"
                )
                break

            # ── [J] Transient — retry (TIMEOUT, PRICE_CHANGED, etc.) ──────────
            else:
                logger.warning(
                    f"[OE] Transient error on attempt {attempt}: "
                    f"{msg} (retcode={result.retcode}) — retrying"
                )
                time.sleep(self.RETRY_DELAY)
                continue

        logger.error(
            f"[OE] ❌ Order aborted after {self.MAX_RETRIES} attempts | "
            f"{spec.symbol} {spec.direction}"
        )
        return None
