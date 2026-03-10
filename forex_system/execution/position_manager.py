# =============================================================================
# GODBOT v3.0 – execution/position_manager.py
# =============================================================================
# Fixes / improvements applied in this version:
#
#  [A]  _modify_sl() called mt5.order_send() directly instead of using
#       executor.modify_sl_tp() which was added to order_executor.py
#       in this session. Unified: _modify_sl() now delegates to
#       executor.modify_sl_tp() so all MT5 order interaction goes
#       through one place and benefits from the retry/logging already
#       there.
#
#  [B]  pip_size calculation used point * 10 unconditionally. This is
#       correct for 5-digit brokers (e.g. EURUSD = 1.08345) but wrong
#       for 4-digit brokers (point already = 0.0001) and for JPY pairs
#       (point = 0.001, pip = 0.01). Fixed: pip_size now uses
#       sym_info.digits to determine the correct pip size, matching
#       the same logic used in risk_manager.py and order_executor.py.
#
#  [C]  Max hold time used datetime.utcnow() which is naive (no tzinfo).
#       tracked["opened_at"] is set in main.py using datetime.utcnow()
#       too, so the subtraction worked — but mixing naive and aware
#       datetimes is fragile. Standardised both sides to UTC-aware
#       datetime.now(timezone.utc) so the comparison is always safe.
#
#  [D]  _force_close() tried to get P&L from result.get("profit", 0.0)
#       but close_position() in order_executor.py returns a dict with
#       keys: ticket, symbol, price, volume — there is NO "profit" key.
#       Real P&L must be fetched from MT5 deal history after the close.
#       Added _get_deal_pnl(ticket) which queries mt5.history_deals_get()
#       to retrieve the actual closed P&L for accurate trade logging and
#       risk manager feedback.
#
#  [E]  on_new_signal() iterated self.open_positions while potentially
#       modifying it inside _force_close() (which calls
#       self.open_positions.pop()). Fixed: snapshot the items to a list
#       before iterating (already done for manage() but not here).
#
#  [F]  manage() read MAX_HOLD_CANDLES from the module-level constant
#       (hard-coded 12) instead of from CONFIG.get_scalper_profile()
#       which may have a different max_hold_candles value depending on
#       whether M1 or M5 is selected. Fixed: reads from profile with
#       fallback to the module constant.
#
#  [G]  _force_close() called dashboard.log_trade() without passing
#       entry_price, close_price, or volume — those optional fields
#       were added to log_trade() in our dashboard.py update and are
#       now populated here so the exported CSV has complete trade data.
#
#  [H]  Breakeven buffer was hard-coded to BREAKEVEN_BUFFER_PIPS = 1.0
#       module constant. For M1 scalping a 1-pip buffer eats 17% of a
#       6-pip SL. Changed to read from CONFIG.get_scalper_profile()
#       with the module constant as fallback, so M1 and M5 can have
#       different buffers.
#
#  [I]  Added trailing_stop management to manage() — after breakeven
#       is moved, subsequent calls apply a trailing stop using
#       executor.modify_trailing_stop() with the profile's trailing_stop
#       value. Previously trailing stops were handled separately in
#       main.py but were not connected to the be_moved flag, meaning
#       the trail started from entry rather than from the breakeven SL.
#
#  [J]  Added _get_live_position() helper that fetches the current MT5
#       position state for a ticket, used to verify the position still
#       exists before attempting any modification, preventing spurious
#       error logs when a position was closed by the broker between
#       manage() cycles.
#
# =============================================================================

"""
PositionManager — active exit management for open trades.

Responsibilities
────────────────
  • Breakeven move    : when floating profit ≥ SL distance, move SL to
                        entry + buffer pips (locks in breakeven).
  • Trailing stop     : after breakeven is locked, trail SL using the
                        profile's trailing_stop pip value. [I]
  • Max hold time     : force-close any scalper trade still open after
                        MAX_HOLD_CANDLES (reads from active profile). [F]
  • Counter-signal    : when SignalEngine fires the opposite direction on
                        a symbol with an open position, close immediately.

Called from main.py
────────────────────
  • manage()          — called every 15 seconds by _manage_positions()
  • on_new_signal()   — called from _process_symbol() after all gates pass
"""

import MetaTrader5 as mt5
from datetime  import datetime, timezone, timedelta
from typing    import Optional
from monitoring.logger import get_logger

logger = get_logger("PositionManager")

# ── Module-level constants (used as fallbacks if profile load fails) ──────────
MAX_HOLD_CANDLES      = 12      # [F] fallback — overridden by profile
BREAKEVEN_BUFFER_PIPS = 1.0     # [H] fallback — overridden by profile

# Minutes per candle for each supported timeframe
CANDLE_MINUTES = {1: 1, 5: 5, 15: 15, 60: 60, 240: 240}


class PositionManager:

    def __init__(
        self,
        executor,
        alerts,
        dashboard,
        risk_mgr,
        open_positions: dict,
        style:          str,
        mode:           str,
    ):
        self.executor       = executor
        self.alerts         = alerts
        self.dashboard      = dashboard
        self.risk_mgr       = risk_mgr
        # Shared reference to ForexSystem._open_positions — mutations here
        # are immediately visible in main.py
        self.open_positions = open_positions
        self.style          = style
        self.mode           = mode

    # ── Public: called every 15 s from _manage_positions() ───────────────────
    def manage(self, style: str) -> None:
        """
        Iterate all tracked positions and apply exit rules:
          1. Breakeven move (once, when profit ≥ SL distance)
          2. Trailing stop  (after breakeven, every cycle)   [I]
          3. Max hold time  (force-close stale scalp trades) [F]
        """
        if not self.open_positions:
            return

        # Load active profile values
        profile         = self._get_profile()
        be_buffer_pips  = profile.get("breakeven_buffer", BREAKEVEN_BUFFER_PIPS)  # [H]
        trailing_pips   = profile.get("trailing_stop",   5.0)                     # [I]
        max_hold_candles = profile.get("max_hold_candles", MAX_HOLD_CANDLES)       # [F]
        tf_selected     = self._get_tf()
        tf_minutes      = CANDLE_MINUTES.get(tf_selected, 5)
        max_minutes     = max_hold_candles * tf_minutes

        # Snapshot keys — dict may shrink during iteration
        for ticket in list(self.open_positions.keys()):
            tracked = self.open_positions.get(ticket)
            if tracked is None:
                continue

            symbol    = tracked["symbol"]
            direction = tracked["direction"]
            entry     = tracked["entry"]
            sl        = tracked.get("sl", 0.0)
            be_moved  = tracked.get("be_moved", False)
            opened_at = tracked.get("opened_at")

            # [J] Verify position still exists before any modification
            live_pos = self._get_live_position(ticket)
            if live_pos is None:
                logger.debug(
                    f"[PM] Ticket #{ticket} no longer open — "
                    "removing from tracking"
                )
                self.open_positions.pop(ticket, None)
                continue

            sym_info = mt5.symbol_info(symbol)
            tick     = mt5.symbol_info_tick(symbol)
            if sym_info is None or tick is None:
                continue

            # [B] Correct pip_size using digit count
            pip_size = self._get_pip_size(sym_info)
            current  = tick.bid if direction == "BUY" else tick.ask

            # ── 1. Breakeven move ─────────────────────────────────────────
            if not be_moved and sl:
                sl_dist_pips = abs(entry - sl) / pip_size
                profit_pips  = (
                    (current - entry) / pip_size
                    if direction == "BUY"
                    else (entry - current) / pip_size
                )

                if profit_pips >= sl_dist_pips:
                    # [H] Use profile-defined buffer, not hard-coded constant
                    be_sl = (
                        entry + be_buffer_pips * pip_size
                        if direction == "BUY"
                        else entry - be_buffer_pips * pip_size
                    )
                    be_sl = round(be_sl, sym_info.digits)

                    # [A] Use executor.modify_sl_tp() instead of direct mt5 call
                    success = self._modify_sl(
                        ticket, symbol, be_sl, live_pos["tp"]
                    )
                    if success:
                        tracked["sl"]       = be_sl
                        tracked["be_moved"] = True
                        logger.info(
                            f"🔒 Breakeven | {symbol} {direction} #{ticket} | "
                            f"profit={profit_pips:.1f}p ≥ "
                            f"SL dist={sl_dist_pips:.1f}p | "
                            f"SL → {be_sl:.{sym_info.digits}f} "
                            f"(+{be_buffer_pips}p buffer)"
                        )
                        try:
                            self.alerts.breakeven_moved(
                                symbol    = symbol,
                                direction = direction,
                                ticket    = ticket,
                                new_sl    = be_sl,
                            )
                        except AttributeError:
                            pass

            # ── 2. Trailing stop (after breakeven is locked) ──────────────
            # [I] Trail with profile value; only after BE is confirmed
            if be_moved:
                try:
                    self.executor.modify_trailing_stop(
                        ticket     = ticket,
                        symbol     = symbol,
                        trail_pips = trailing_pips,
                        pip_to_points = True,
                    )
                    # Sync tracked SL from live MT5 position after trail
                    updated = self._get_live_position(ticket)
                    if updated and updated["sl"] != tracked.get("sl"):
                        tracked["sl"] = updated["sl"]
                except Exception as exc:
                    logger.debug(
                        f"[PM] Trailing stop error #{ticket}: {exc}"
                    )

            # ── 3. Max hold time force-close ──────────────────────────────
            if style == "scalper" and opened_at is not None:
                # [C] Use UTC-aware datetime for safe subtraction
                now_utc    = datetime.now(timezone.utc)
                opened_utc = (
                    opened_at.replace(tzinfo=timezone.utc)
                    if opened_at.tzinfo is None
                    else opened_at
                )
                age_minutes = (now_utc - opened_utc).total_seconds() / 60

                if age_minutes >= max_minutes:
                    logger.info(
                        f"⏰ Max hold time | {symbol} {direction} #{ticket} | "
                        f"age={age_minutes:.0f}min ≥ "
                        f"limit={max_minutes}min — force-closing"
                    )
                    self._force_close(
                        ticket, tracked, reason="Max hold time"
                    )

    # ── Public: called from _process_symbol() after all gates pass ────────────
    def on_new_signal(self, symbol: str, new_direction: str, signal) -> None:
        """
        Close any existing position on this symbol that is in the
        opposite direction before the new signal is processed.

        [E] Snapshot items to list before iterating to prevent
            RuntimeError from dict size change during _force_close().
        """
        for ticket, tracked in list(self.open_positions.items()):  # [E]
            if tracked["symbol"] != symbol:
                continue
            existing_dir = tracked["direction"]
            if existing_dir != new_direction:
                logger.info(
                    f"🔄 Counter-signal | {symbol} | "
                    f"existing={existing_dir} new={new_direction} "
                    f"#{ticket} — closing existing position"
                )
                self._force_close(
                    ticket, tracked,
                    reason=f"Counter-signal ({new_direction})",
                )

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _modify_sl(
        self,
        ticket: int,
        symbol: str,
        new_sl: float,
        tp:     float,
    ) -> bool:
        """
        [A] Delegate to executor.modify_sl_tp() instead of calling
        mt5.order_send() directly, so all MT5 order interaction goes
        through one place with consistent logging and error handling.
        """
        try:
            return self.executor.modify_sl_tp(
                ticket = ticket,
                new_sl = new_sl,
                new_tp = tp,
                symbol = symbol,
            )
        except Exception as exc:
            logger.warning(
                f"[PM] _modify_sl failed for #{ticket}: {exc}"
            )
            return False

    def _force_close(
        self, ticket: int, tracked: dict, reason: str
    ) -> None:
        """
        Close a position at market and record the result.

        [D] P&L is fetched from MT5 deal history after close rather than
            from the close_position() return dict (which has no profit key).
        [G] dashboard.log_trade() now receives entry_price, close_price,
            and volume for complete CSV export data.
        """
        symbol    = tracked["symbol"]
        direction = tracked["direction"]
        entry     = tracked["entry"]

        result = self.executor.close_position(ticket)
        if not result:
            logger.warning(
                f"[PM] Force-close failed for #{ticket} "
                f"({symbol} {direction})"
            )
            return

        # Remove from tracking immediately after confirmed close
        self.open_positions.pop(ticket, None)

        close_px = result.get("price", entry)
        volume   = result.get("volume", tracked.get("volume", 0.0))

        # [D] Fetch actual P&L from MT5 deal history
        pnl = self._get_deal_pnl(ticket)

        # Fallback: estimate pnl from pip move if deal history unavailable
        if pnl == 0.0:
            sym_info = mt5.symbol_info(symbol)
            if sym_info:
                pip_size = self._get_pip_size(sym_info)
                pip_move = (
                    (close_px - entry) / pip_size
                    if direction == "BUY"
                    else (entry - close_px) / pip_size
                )
                # Rough estimate: $10/pip per lot (USD account)
                pnl = round(pip_move * 10.0 * volume, 2)

        sym_info = mt5.symbol_info(symbol)
        pip_size = self._get_pip_size(sym_info) if sym_info else 0.0001
        pips     = (
            (close_px - entry) / pip_size
            if direction == "BUY"
            else (entry - close_px) / pip_size
        )

        logger.info(
            f"✅ Force-closed | {symbol} {direction} #{ticket} | "
            f"reason={reason} | "
            f"entry={entry:.5f} close={close_px:.5f} | "
            f"pips={pips:+.1f} | P&L=${pnl:+.2f}"
        )

        # Alerts
        try:
            self.alerts.trade_closed(
                symbol    = symbol,
                direction = direction,
                ticket    = ticket,
                entry     = entry,
                close     = close_px,
                pnl       = pnl,
                pips      = round(pips, 1),
                reason    = reason,
            )
        except Exception as exc:
            logger.warning(f"[PM] Alert failed on force-close: {exc}")

        # [G] Dashboard log with full trade data
        try:
            self.dashboard.log_trade(
                symbol       = symbol,
                direction    = direction,
                pnl          = pnl,
                ticket       = ticket,
                close_reason = reason,
                entry_price  = entry,
                close_price  = close_px,
                volume       = volume,
            )
        except Exception as exc:
            logger.warning(f"[PM] Dashboard log failed on force-close: {exc}")

        # Risk manager feedback
        try:
            self.risk_mgr.record_trade_result(pnl, symbol=symbol)
        except Exception as exc:
            logger.warning(
                f"[PM] Risk manager record failed on force-close: {exc}"
            )

    def _get_deal_pnl(self, ticket: int) -> float:
        """
        [D] Fetch actual closed P&L from MT5 deal history for a given
        position ticket. MT5 stores deals (fills) separately from orders;
        the closing deal has the real profit including swap and commission.

        Returns 0.0 if history is unavailable or the deal is not found.
        """
        try:
            # Search last 24 hours to find the closing deal
            from_time = datetime.now(timezone.utc) - timedelta(hours=24)
            to_time   = datetime.now(timezone.utc) + timedelta(minutes=1)

            deals = mt5.history_deals_get(from_time, to_time)
            if deals is None or len(deals) == 0:
                return 0.0

            # Find the deal that closes this position (position_id matches ticket)
            close_deals = [
                d for d in deals
                if d.position_id == ticket and
                d.entry == mt5.DEAL_ENTRY_OUT
            ]

            if not close_deals:
                return 0.0

            # Sum profit from all closing deals for this position
            total_pnl = sum(d.profit for d in close_deals)
            logger.debug(
                f"[PM] Deal history P&L for #{ticket}: "
                f"${total_pnl:+.2f} "
                f"({len(close_deals)} closing deal(s))"
            )
            return round(total_pnl, 2)

        except Exception as exc:
            logger.debug(f"[PM] _get_deal_pnl error for #{ticket}: {exc}")
            return 0.0

    def _get_live_position(self, ticket: int) -> Optional[dict]:
        """
        [J] Return a lightweight dict of the live MT5 position state
        for a given ticket, or None if the position is no longer open.
        Used to verify existence and sync SL/TP before modifications.
        """
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return None
            pos = positions[0]
            return {
                "ticket":     pos.ticket,
                "symbol":     pos.symbol,
                "sl":         pos.sl,
                "tp":         pos.tp,
                "volume":     pos.volume,
                "profit":     pos.profit,
                "open_price": pos.price_open,
                "open_time":  pos.time,
            }
        except Exception as exc:
            logger.debug(
                f"[PM] _get_live_position error for #{ticket}: {exc}"
            )
            return None

    @staticmethod
    def _get_pip_size(sym_info) -> float:
        """
        [B] Return correct pip size using sym_info.digits:
            digits 5 or 3 → 5-digit broker  → pip = 10 × point
            digits 4 or 2 → 4-digit / metals → pip = point

        Consistent with risk_manager.py and order_executor.py.
        """
        if sym_info.digits in (5, 3):
            return 10 * sym_info.point
        return sym_info.point

    @staticmethod
    def _get_profile() -> dict:
        """
        Load the active scalper profile from CONFIG with a safe fallback.
        [F][H][I] Used to read max_hold_candles, breakeven_buffer,
        and trailing_stop from the correct M1/M5 profile.
        """
        try:
            from config.settings import CONFIG
            return CONFIG.get_scalper_profile()
        except Exception:
            return {
                "max_hold_candles": MAX_HOLD_CANDLES,
                "breakeven_buffer": BREAKEVEN_BUFFER_PIPS,
                "trailing_stop":    5.0,
            }

    @staticmethod
    def _get_tf() -> int:
        """Return the currently selected scalper timeframe integer (1 or 5)."""
        try:
            from config.settings import CONFIG
            return getattr(CONFIG, "SCALPER_TF_SELECTED", 5)
        except Exception:
            return 5
