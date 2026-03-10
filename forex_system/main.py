# =============================================================================
# GODBOT v3.0 – main.py  (PRODUCTION – fully corrected 2026-03-10)
# =============================================================================
#
#  AF [FIX] send_market_order() → single PositionSpec object.
#  AG [FIX] signal.signal_type → signal.signal.value everywhere.
#  AH [FIX] record_trade_result(pnl=float, symbol=str) — no ticket= kwarg.
#  AI [FIX] modify_trailing_stop() — symbol= added.
#  AJ [FIX] close_position() success check via result.get("ticket").
#  AK [FIX] PositionSpec fields .sl / .tp (not .sl_price / .tp_price).
#  AL [FIX] ml_model.predict(ind) — one argument only.
#  AM [FIX] risk.calculate_position(direction=) not signal_type=.
#  AN [FIX] _get_pip_size_safe() helper — no sym_info arg needed.
#  AO [NEW] _in_session() — full forex market hours (Sun 22:00–Fri 22:00 UTC).
#  AP [NEW] Pre-execution spread re-check before send_market_order().
#  AQ [NEW] news_gate minutes_after=30.
#  AR [FIX] _daily_summary() reads risk._daily_pnl / _trade_history.
#  AS [FIX] _monitor_positions() indentation — all code inside for-loop.
#  AT [FIX] _scan_symbols() uses logger.exception — full traceback visible.
#  AU [FIX] _process_symbol() has breadcrumb debug logs at every gate so
#           silent failures are impossible.
#  AV [FIX] start() wraps _main_loop with logger.exception for full crash
#           traceback.
#
# =============================================================================

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Suppress HuggingFace / transformers startup noise
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_VERBOSITY",         "error")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN",  "1")
os.environ.setdefault("HF_HUB_VERBOSITY",               "warning")
os.environ.setdefault("TOKENIZERS_PARALLELISM",          "false")

# ---------------------------------------------------------------------------
# Portable Numba cache
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(_REPO_ROOT / ".numba_cache"))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore

from config.settings              import CONFIG
from core.mt5_connector           import MT5Connector
from core.data_handler            import DataHandler
from indicators.indicators_engine import IndicatorEngine
from signals.signal_engine        import SignalEngine, SignalType
from signals.ml_model             import MLSignalModel
from risk.risk_manager            import RiskManager
from execution.order_executor     import OrderExecutor
from monitoring.dashboard         import Dashboard
from notifications.alert_manager  import AlertManager
from research.calendar_scanner    import CalendarScanner
from research.sentiment_analyzer  import SentimentAnalyzer
from research.intermarket         import IntermarketAnalyzer
from research.cot_reader          import COTReader

logger = logging.getLogger("GODBOT.main")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROFILE_PATH       = _REPO_ROOT / "data" / "trader_profile.json"
_SEMI_AUTO_TIMEOUT = 30


# =============================================================================
# TraderProfile / ProfileManager
# =============================================================================
@dataclass
class TraderProfile:
    style:      str = "scalper"
    mode:       str = "semi_automated"
    name:       str = "Trader"
    scalper_tf: str = "M5"


class ProfileManager:
    def load(self) -> TraderProfile:
        if PROFILE_PATH.exists():
            try:
                data  = json.loads(PROFILE_PATH.read_text())
                clean = {k: v for k, v in data.items()
                         if not k.startswith("_") and k != "version"}
                return TraderProfile(**clean)
            except Exception as exc:
                logger.warning("Could not load profile (%s); using defaults.", exc)
        return TraderProfile()

    def save(self, profile: TraderProfile) -> None:
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROFILE_PATH.write_text(
            json.dumps(profile.__dict__, indent=2), encoding="utf-8"
        )
        logger.info("Profile saved to %s", PROFILE_PATH)


# =============================================================================
# StartupMenu
# =============================================================================
class StartupMenu:
    _STYLES = {"1": "scalper",        "2": "day_trader"}
    _TFS    = {"1": "M1",             "2": "M5"}
    _MODES  = {"1": "signal_only",
               "2": "semi_automated",
               "3": "fully_automated"}

    def __init__(self) -> None:
        self._pm = ProfileManager()

    def run(self) -> TraderProfile:
        profile = self._pm.load()
        print("\n" + "=" * 60)
        print("  GODBOT v3.0 – Startup Configuration")
        print("=" * 60)

        print("\nTrading style:\n  1) Scalper\n  2) Day Trader")
        choice = input(f"  Select [default={profile.style}]: ").strip()
        profile.style = self._STYLES.get(choice, profile.style)

        if profile.style == "scalper":
            print("\nScalper timeframe:\n  1) M1\n  2) M5")
            choice = input(f"  Select [default={profile.scalper_tf}]: ").strip()
            profile.scalper_tf = self._TFS.get(choice, profile.scalper_tf)

        print("\nOperation mode:")
        print("  1) Signal Only\n  2) Semi-Automated\n  3) Fully Automated")
        choice = input(f"  Select [default={profile.mode}]: ").strip()
        profile.mode = self._MODES.get(choice, profile.mode)

        name = input(f"\nYour name [default={profile.name}]: ").strip()
        if name:
            profile.name = name

        if input("\nSave this profile? [y/N]: ").strip().lower() == "y":
            self._pm.save(profile)

        return profile


# =============================================================================
# ForexSystem
# =============================================================================
class ForexSystem:
    """Core trading engine."""

    def __init__(self, profile: TraderProfile) -> None:
        self.profile   = profile
        self._running  = False
        self._paused   = False
        self._sound_on = True
        self._tg_on    = True

        if profile.style == "scalper":
            tf_int_map = {"M1": 1, "M5": 5}
            CONFIG.SCALPER_TF_SELECTED = tf_int_map.get(profile.scalper_tf, 5)
            logger.info("Scalper TF set to %s", profile.scalper_tf)

        CONFIG.get_scalper_profile()

        self.connector     = MT5Connector()
        self.data          = DataHandler()
        self.indicators    = IndicatorEngine()
        self.signal_engine = SignalEngine()
        self.ml_model      = MLSignalModel()
        self.risk          = RiskManager()
        self.executor      = OrderExecutor()
        self.dashboard     = Dashboard()
        self.alerts        = AlertManager()
        self.calendar      = CalendarScanner()
        self.sentiment     = SentimentAnalyzer()
        self.intermarket   = IntermarketAnalyzer()
        self.cot           = COTReader()

        self.dashboard.set_mode(profile.mode)
        self.dashboard.set_scan_secs(
            CONFIG.get_scalper_profile().get("scan_secs", 20)
        )

        self._open_tickets:        dict[int, str] = {}
        self._last_scan:           float = 0.0
        self._last_trail:          float = 0.0
        self._last_summary:        float = 0.0
        self._session_open_equity: Optional[float] = None
        self._negative_warned:     set   = set()   # tickets that received going-negative alert
        self._breakeven_set:       set   = set()   # tickets where breakeven opportunity alerted

        logger.info("ForexSystem initialised for %s (%s / %s)",
                    profile.name, profile.style, profile.mode)

    # ------------------------------------------------------------------ props

    @property
    def _scalper_profile(self) -> dict:
        return CONFIG.get_scalper_profile()

    @property
    def _max_spread(self) -> float:
        return float(self._scalper_profile.get("max_spread", CONFIG.SCALPER_MAX_SPREAD))

    @property
    def _trailing_pips(self) -> float:
        return float(self._scalper_profile.get("trailing_stop", CONFIG.TRAILING_STOP_PIPS))

    @property
    def _ml_threshold(self) -> float:
        return float(self._scalper_profile.get("ml_threshold", CONFIG.ML_MIN_CONFIDENCE))

    @property
    def _scan_secs(self) -> float:
        return float(self._scalper_profile.get("scan_secs", 20.0))

    # ------------------------------------------------------------------ session

    def _in_session(self) -> bool:
        """
        Full forex market hours: Sunday 22:00 UTC → Friday 22:00 UTC.
        Blocks only genuine weekend closure.
        """
        now_utc  = datetime.now(timezone.utc)
        weekday  = now_utc.weekday()   # 0=Mon … 6=Sun
        hour_utc = now_utc.hour

        if weekday == 5:                          # Saturday — always closed
            return False
        if weekday == 6 and hour_utc < 22:        # Sunday before open
            return False
        if weekday == 4 and hour_utc >= 22:       # Friday after close
            return False

        if getattr(CONFIG, "ENABLE_QUIET_HOURS", False):
            q_start, q_end = getattr(CONFIG, "QUIET_HOURS_UTC", (22, 7))
            if q_start > q_end:
                if hour_utc >= q_start or hour_utc < q_end:
                    return False
            elif q_start <= hour_utc < q_end:
                return False

        return True

    # ------------------------------------------------------------------ helpers

    def _open_symbols(self) -> set[str]:
        return set(self._open_tickets.values())

    @staticmethod
    def _signal_direction(signal) -> str:
        raw = getattr(signal, "signal", None)
        if raw is None:
            return "HOLD"
        if isinstance(raw, SignalType):
            return raw.value
        return str(raw).upper()

    @staticmethod
    def _fetch_closed_pnl(ticket: int) -> float:
        if mt5 is None:
            return 0.0
        try:
            deals = mt5.history_deals_get(position=ticket)
            if deals:
                return sum(d.profit for d in deals)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _fetch_closed_deal_info(ticket: int, symbol: str) -> tuple:
        """
        Return (entry_price, close_price, direction, pips) from MT5 deal history.
        Used by trade_closed() to give a full picture on Telegram.
        """
        entry_px = close_px = 0.0
        direction = "?"
        pips      = 0.0
        if mt5 is None:
            return entry_px, close_px, direction, pips
        try:
            deals = mt5.history_deals_get(position=ticket)
            if deals:
                sorted_deals = sorted(deals, key=lambda d: d.time)
                entry_deal   = sorted_deals[0]
                close_deal   = sorted_deals[-1]
                entry_px     = float(entry_deal.price)
                close_px     = float(close_deal.price)
                direction    = "BUY" if entry_deal.type == 0 else "SELL"
                pip_size     = ForexSystem._get_pip_size_safe(symbol)
                if pip_size and entry_px and close_px:
                    pips = (
                        (close_px - entry_px) / pip_size if direction == "BUY"
                        else (entry_px - close_px) / pip_size
                    )
        except Exception:
            pass
        return entry_px, close_px, direction, round(pips, 1)

    @staticmethod
    def _get_pip_size_safe(symbol: str) -> float:
        if mt5 is None:
            return 0.0001
        try:
            info = mt5.symbol_info(symbol)
            if info is not None:
                return 10.0 * info.point if info.digits in (5, 3) else info.point
        except Exception:
            pass
        return 0.0001

    # ------------------------------------------------------------------ start

    def start(self) -> None:
        logger.info("Connecting to MT5…")
        if not self.connector.connect():
            logger.critical("MT5 connection failed – aborting.")
            return

        self._running = True
        logger.info("GODBOT v3.0 started. Mode: %s", self.profile.mode)
        self.alerts.send(f"🤖 GODBOT v3.0 started – {self.profile.mode}")

        acc = self.risk._get_account()
        if acc:
            self._session_open_equity = acc["equity"]
            self.risk._session_equity = acc["equity"]
            logger.info("Session equity set: %.2f", self._session_open_equity)
        else:
            logger.warning("Could not read account equity on startup.")

        self.risk.set_alerts(self.alerts)

        logger.info("Starting KB-Listener thread…")
        threading.Thread(
            target=self._keyboard_listener, daemon=True, name="KB-Listener"
        ).start()

        logger.info("Starting Dashboard thread…")
        threading.Thread(
            target=self.dashboard.display, daemon=True, name="Dashboard"
        ).start()

        logger.info("Entering main loop…")
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt – shutting down.")
        except Exception as exc:
            logger.exception("‼️ FATAL exception in main loop: %s", exc)
            self.alerts.send(f"🚨 GODBOT crash: {exc}")
        finally:
            self._shutdown()

    # ------------------------------------------------------------------ loop

    def _main_loop(self) -> None:
        scan_interval    = self._scan_secs
        trail_interval   = 15.0
        summary_interval = 3600.0

        while self._running:
            now = time.monotonic()

            if self._paused:
                time.sleep(1.0)
                continue

            secs_left = max(0.0, scan_interval - (now - self._last_scan))
            self.dashboard.update_scan_status(f"Next scan in {secs_left:.0f}s")

            if now - self._last_scan >= scan_interval:
                self._last_scan = now
                if self._in_session():
                    self._scan_symbols()
                else:
                    logger.info("⏸ Outside session window – scan skipped (weekday=%d hour=%d UTC).",
                datetime.now(timezone.utc).weekday(),
                datetime.now(timezone.utc).hour)

            if now - self._last_trail >= trail_interval:
                self._last_trail = now
                self._update_trailing_stops()

            if now - self._last_summary >= summary_interval:
                self._last_summary = now
                self._daily_summary()

            # ── FIX: wrap these two so an exception here cannot propagate
            # to start()'s outer handler and trigger a clean _shutdown()
            # with no visible cause (all stdout was being wiped by the
            # dashboard's os.system("cls") every second).
            try:
                self._monitor_positions()
            except Exception as exc:
                logger.exception("‼️ _monitor_positions() crashed: %s", exc)

            try:
                self._eod_check()
            except Exception as exc:
                logger.exception("‼️ _eod_check() crashed: %s", exc)

            time.sleep(1.0)

    # ------------------------------------------------------------------ scan

    def _scan_symbols(self) -> None:
        for symbol in CONFIG.SYMBOLS:
            try:
                self._process_symbol(symbol)
            except Exception as exc:
                # logger.exception prints the FULL traceback — nothing hidden
                logger.exception("‼️ EXCEPTION in _process_symbol(%s): %s", symbol, exc)

    def _process_symbol(self, symbol: str) -> None:  # noqa: C901

        # ── 1. Spread gate ────────────────────────────────────────────────
        logger.debug("[%s] ▶ Step 1: spread check", symbol)
        tick = self.data.get_tick_data(symbol)
        if tick is None:
            logger.debug("[%s] ✗ No tick data.", symbol)
            return
        spread_pips = tick.get("spread_pips", 999.0)
        if spread_pips > self._max_spread:
            logger.debug("[%s] ✗ Spread %.2f > max %.2f.", symbol, spread_pips, self._max_spread)
            return

        # ── 2. Duplicate-position gate ────────────────────────────────────
        logger.debug("[%s] ▶ Step 2: duplicate check", symbol)
        if symbol in self._open_symbols():
            logger.debug("[%s] ✗ Already open.", symbol)
            return

        # ── 3. News gate ──────────────────────────────────────────────────
        logger.debug("[%s] ▶ Step 3: news gate", symbol)
        news_check = self.calendar.is_safe_to_trade(symbol, minutes_after=30)
        if not news_check.get("safe", True):
            logger.info("[%s] ✗ News window: %s", symbol, news_check.get("reason", ""))
            return

        # ── 4. OHLCV data ─────────────────────────────────────────────────
        logger.debug("[%s] ▶ Step 4: OHLCV fetch", symbol)
        tf = self._scalper_profile.get("tf_primary", CONFIG.PRIMARY_TF)
        df = self.data.get_data(symbol, tf)
        if df is None or len(df) < 50:
            logger.debug("[%s] ✗ Insufficient bars.", symbol)
            return

        # ── 5. Indicators ─────────────────────────────────────────────────
        logger.debug("[%s] ▶ Step 5: indicators", symbol)
        tf_confirm = self._scalper_profile.get("tf_confirm", 15)
        htf_df     = self.data.get_data(symbol, tf_confirm)
        ind        = self.indicators.compute_all(df, htf_df=htf_df)
        if ind is None:
            logger.debug("[%s] ✗ Indicators returned None.", symbol)
            return

        # ── 6. Signal engine ──────────────────────────────────────────────
        logger.debug("[%s] ▶ Step 6: signal engine", symbol)
        signal = self.signal_engine.evaluate(ind, symbol)
        if signal is None:
            # Show a quick snapshot so the operator can see the market state
            try:
                rsi   = float(ind.get("rsi",   0))
                adx   = float(ind.get("adx",   0))
                ema_f = float(ind.get("ema_fast", 0))
                ema_s = float(ind.get("ema_slow", 0))
                trend = "↑ Bull" if ema_f > ema_s else "↓ Bear"
                logger.debug(
                    "[%s] ✗ No signal — RSI=%.1f  ADX=%.1f  EMA=%s  "
                    "(min score=%d, market quiet or no edge)",
                    symbol, rsi, adx, trend,
                    getattr(CONFIG, "SCALPER_SIGNAL_SCORE", 3),
                )
            except Exception:
                logger.debug("[%s] ✗ No signal (None).", symbol)
            return

        direction = self._signal_direction(signal)
        if direction == "HOLD":
            logger.debug("[%s] ✗ HOLD – market ranging.", symbol)
            return

        logger.info("[%s] ✔ Signal direction: %s", symbol, direction)

        # ── 7. ML veto ────────────────────────────────────────────────────
        logger.debug("[%s] ▶ Step 7: ML predict", symbol)
        try:
            ml = self.ml_model.predict(ind)
        except Exception as exc:
            logger.exception("[%s] ‼️ ml_model.predict() CRASHED: %s", symbol, exc)
            return

        ml_label = ml.get("label", "HOLD")
        ml_conf  = float(ml.get("confidence", 0.0))
        logger.info("[%s] ML → label=%s conf=%.3f threshold=%.3f",
                    symbol, ml_label, ml_conf, self._ml_threshold)

        if ml_label == "HOLD":
            logger.info("[%s] ✗ ML HOLD veto.", symbol)
            return
        if ml_conf < self._ml_threshold:
            logger.info("[%s] ✗ ML confidence %.3f < threshold %.3f – veto.",
                        symbol, ml_conf, self._ml_threshold)
            return
        if direction == "BUY" and ml_label != "BUY":
            logger.info("[%s] ✗ ML label %s ≠ signal BUY – veto.", symbol, ml_label)
            return
        if direction == "SELL" and ml_label != "SELL":
            logger.info("[%s] ✗ ML label %s ≠ signal SELL – veto.", symbol, ml_label)
            return

        logger.info("[%s] ✔ ML passed: label=%s conf=%.3f", symbol, ml_label, ml_conf)

        # ── 8. Day-trader filters ─────────────────────────────────────────
        if self.profile.style == "day_trader":
            logger.debug("[%s] ▶ Step 8: day-trader filters", symbol)
            if not self._day_trader_filters(symbol, direction):
                return

        # ── 9. Risk sizing ────────────────────────────────────────────────
        logger.debug("[%s] ▶ Step 9: risk sizing", symbol)
        try:
            position_spec = self.risk.calculate_position(
                symbol     = symbol,
                direction  = direction,
                entry      = float(signal.entry),
                sl_pips    = float(signal.sl_pips),
                tp_pips    = float(signal.tp_pips),
                confidence = ml_conf,
            )
        except Exception as exc:
            logger.exception("[%s] ‼️ risk.calculate_position() CRASHED: %s", symbol, exc)
            return

        if position_spec is None:
            logger.info("[%s] ✗ Risk manager blocked trade.", symbol)
            return

        logger.info("[%s] ✔ Position spec: vol=%.2f SL=%.1fp TP=%.1fp risk=%.2f",
                    symbol, position_spec.volume,
                    position_spec.sl_pips, position_spec.tp_pips,
                    position_spec.risk_usd)

        # ── 10. Alerts + dashboard ────────────────────────────────────────
        logger.debug("[%s] ▶ Step 10: signal alert", symbol)
        try:
            self.alerts.signal_alert(
                symbol        = symbol,
                signal        = signal,
                position_spec = position_spec,
                sound         = self._sound_on,
                telegram      = self._tg_on,
            )
        except Exception as exc:
            logger.exception("[%s] ‼️ alerts.signal_alert() CRASHED: %s", symbol, exc)

        try:
            self.dashboard.update_signal(symbol, signal, position_spec)
        except Exception as exc:
            logger.exception("[%s] ‼️ dashboard.update_signal() CRASHED: %s", symbol, exc)

        # ── 11. Execution gate ────────────────────────────────────────────
        if self.profile.mode == "signal_only":
            logger.info("[%s] Signal-only mode – no order sent. (%s entry=%.5f)",
                        symbol, direction, signal.entry)
            return

        if self.profile.mode == "semi_automated":
            logger.debug("[%s] ▶ Step 11: semi-auto confirm", symbol)
            if not self._semi_auto_confirm(symbol, direction, signal):
                return

        # ── 12. Pre-execution spread re-check ─────────────────────────────
        logger.debug("[%s] ▶ Step 12: pre-exec spread", symbol)
        pre_exec_tick = self.data.get_tick_data(symbol)
        if pre_exec_tick is None:
            logger.warning("[%s] ✗ No tick before execution – abort.", symbol)
            return
        pre_exec_spread = pre_exec_tick.get("spread_pips", 999.0)
        if pre_exec_spread > self._max_spread:
            logger.info("[%s] ✗ Spread widened to %.2f before exec (max %.2f) – abort.",
                        symbol, pre_exec_spread, self._max_spread)
            return

        # ── 13. Send order ────────────────────────────────────────────────
        logger.info("[%s] ▶ Step 13: sending market order…", symbol)
        try:
            result = self.executor.send_market_order(spec=position_spec)
        except Exception as exc:
            logger.exception("[%s] ‼️ send_market_order() CRASHED: %s", symbol, exc)
            return

        if result and result.get("ticket"):
            ticket = result["ticket"]
            self._open_tickets[ticket] = symbol
            logger.info(
                "✅ Order filled: %s %s | ticket=%d | vol=%.2f | "
                "SL=%.5f (%.1fp) TP=%.5f (%.1fp) | risk=%.2f | R:R=%.2f",
                direction, symbol, ticket,
                position_spec.volume,
                position_spec.sl,  position_spec.sl_pips,
                position_spec.tp,  position_spec.tp_pips,
                position_spec.risk_usd, position_spec.rr_ratio,
            )
            try:
                self.alerts.trade_opened(
                    symbol    = symbol,
                    direction = direction,
                    ticket    = ticket,
                    entry     = float(signal.entry),
                    sl        = float(position_spec.sl),
                    tp        = float(position_spec.tp),
                    volume    = float(position_spec.volume),
                    risk      = float(position_spec.risk_usd),
                    style     = self.profile.style,
                    mode      = self.profile.mode,
                )
            except Exception as exc:
                logger.exception("[%s] ‼️ alerts.trade_opened() CRASHED: %s", symbol, exc)
        else:
            logger.warning("[%s] ✗ Order send returned no ticket. Result: %s",
                           symbol, result)

    # ------------------------------------------------------------------ semi-auto

    def _semi_auto_confirm(self, symbol: str, direction: str, signal) -> bool:
        confirmed = threading.Event()
        rejected  = threading.Event()

        def _prompt() -> None:
            ans = input(
                f"\n[SEMI-AUTO] {direction} {symbol} | "
                f"SL={signal.sl_pips:.1f}p TP={signal.tp_pips:.1f}p | "
                f"Confirm? [y/N] (timeout {_SEMI_AUTO_TIMEOUT}s): "
            ).strip().lower()
            if ans == "y":
                confirmed.set()
            else:
                rejected.set()

        t = threading.Thread(target=_prompt, daemon=True)
        t.start()
        t.join(timeout=_SEMI_AUTO_TIMEOUT)

        if confirmed.is_set():
            return True
        if not rejected.is_set():
            logger.info("Semi-auto timeout for %s – skipping.", symbol)
        return False

    # ------------------------------------------------------------------ day-trader

    def _day_trader_filters(self, symbol: str, direction: str) -> bool:
        try:
            sent_result = self.sentiment.get_sentiment(symbol)
            if sent_result is not None:
                sent_score = sent_result.get("score", 0.0)
                if direction == "BUY" and sent_score < 0:
                    logger.debug("[%s] ✗ Sentiment bearish (%.2f) vs BUY.", symbol, sent_score)
                    return False
                if direction == "SELL" and sent_score > 0:
                    logger.debug("[%s] ✗ Sentiment bullish (%.2f) vs SELL.", symbol, sent_score)
                    return False
        except Exception as exc:
            logger.exception("[%s] ‼️ sentiment.get_sentiment() CRASHED: %s", symbol, exc)

        try:
            im_result = self.intermarket.get_intermarket_signal(symbol)
            if im_result is not None:
                im_bias = im_result.get("bias", "Neutral")
                if im_bias == "Bullish" and direction == "SELL":
                    logger.debug("[%s] ✗ Intermarket Bullish vs SELL.", symbol)
                    return False
                if im_bias == "Bearish" and direction == "BUY":
                    logger.debug("[%s] ✗ Intermarket Bearish vs BUY.", symbol)
                    return False
        except Exception as exc:
            logger.exception("[%s] ‼️ intermarket.get_intermarket_signal() CRASHED: %s", symbol, exc)

        try:
            cot_result = self.cot.get_cot_signal(symbol)
            if cot_result is not None:
                cot_bias = cot_result.get("signal", "neutral").lower()
                if cot_bias == "bullish" and direction == "SELL":
                    logger.debug("[%s] ✗ COT bullish vs SELL.", symbol)
                    return False
                if cot_bias == "bearish" and direction == "BUY":
                    logger.debug("[%s] ✗ COT bearish vs BUY.", symbol)
                    return False
        except Exception as exc:
            logger.exception("[%s] ‼️ cot.get_cot_signal() CRASHED: %s", symbol, exc)

        return True

    # ------------------------------------------------------------------ monitor

    def _monitor_positions(self) -> None:
        closed: list[int] = []

        for ticket, symbol in list(self._open_tickets.items()):
            pos = self.executor.get_open_position(ticket)

            if pos is None:
                logger.info("Position %d (%s) closed by broker/TP/SL.", ticket, symbol)
                pnl = self._fetch_closed_pnl(ticket)
                entry_px, close_px, dir_hist, pips_hist = \
                    self._fetch_closed_deal_info(ticket, symbol)
                try:
                    self.alerts.trade_closed(
                        symbol      = symbol,
                        direction   = dir_hist,
                        ticket      = ticket,
                        entry       = entry_px,
                        close_price = close_px,
                        pnl         = pnl,
                        pips        = pips_hist,
                        reason      = "broker_closed",
                    )
                except Exception as exc:
                    logger.exception("‼️ alerts.trade_closed() CRASHED: %s", exc)
                self._negative_warned.discard(ticket)
                self._breakeven_set.discard(ticket)
                self.risk.record_trade_result(pnl=pnl, symbol=symbol)
                self.dashboard.log_trade(
                    symbol       = symbol,
                    direction    = dir_hist,
                    pnl          = pnl,
                    ticket       = ticket,
                    close_reason = "broker_closed",
                )
                closed.append(ticket)
                continue

            # pos is guaranteed non-None beyond this point
            pnl       = pos.get("profit",    0.0)
            entry     = pos.get("entry",     0.0)
            current   = pos.get("current",   0.0)
            sl        = pos.get("sl",        0.0)
            tp        = pos.get("tp",        0.0)
            direction = pos.get("direction", "?")
            self.dashboard.update_position(ticket, symbol, pnl)

            pip = self._get_pip_size_safe(symbol)

            if tp and current and pip:
                pips_to_tp = abs(tp - current) / pip
                pips_to_sl = abs(current - sl) / pip if sl else 999.0

                if pips_to_tp <= 2.0:
                    try:
                        self.alerts.potential_exit(
                            symbol    = symbol,
                            direction = direction,
                            ticket    = ticket,
                            entry     = entry,
                            current   = current,
                            pnl       = pnl,
                            tp        = tp,
                            reasons   = [
                                f"Price within {pips_to_tp:.1f} pips of TP target",
                                "Consider locking profit or moving SL to breakeven",
                            ],
                        )
                    except Exception as exc:
                        logger.exception("‼️ alerts.potential_exit() CRASHED: %s", exc)

                elif pips_to_sl <= 2.0:
                    try:
                        self.alerts.danger_exit(
                            symbol    = symbol,
                            direction = direction,
                            ticket    = ticket,
                            entry     = entry,
                            current   = current,
                            pnl       = pnl,
                            sl        = sl,
                            reasons   = [
                                f"Price within {pips_to_sl:.1f} pips of SL",
                                "Trade approaching stop loss — consider closing manually",
                            ],
                        )
                    except Exception as exc:
                        logger.exception("‼️ alerts.danger_exit() CRASHED: %s", exc)

                elif pnl > 0:
                    try:
                        self.alerts.hold_alert(
                            symbol    = symbol,
                            direction = direction,
                            ticket    = ticket,
                            entry     = entry,
                            current   = current,
                            pnl       = pnl,
                            tp        = tp,
                        )
                    except Exception as exc:
                        logger.exception("‼️ alerts.hold_alert() CRASHED: %s", exc)

                # ── Going-negative early warning (fires ONCE per ticket) ───
                if pnl < -1.0 and ticket not in self._negative_warned:
                    self._negative_warned.add(ticket)
                    neg_pips = abs(entry - current) / pip if entry else 0.0
                    try:
                        self.alerts.send(
                            f"⚠️ GOING NEGATIVE — {symbol} {direction}\n"
                            f"Ticket  : #{ticket}\n"
                            f"Entry   : {entry:.5f}  ▸  Now: {current:.5f}\n"
                            f"P&L     : {pnl:+.2f}  ({neg_pips:.1f} pips against)\n"
                            f"SL      : {sl:.5f}  ({pips_to_sl:.1f} pips to stop)\n"
                            f"⚡ Consider cutting early to preserve capital"
                        )
                    except Exception as exc:
                        logger.exception("‼️ going-negative alert CRASHED: %s", exc)

                # ── Breakeven opportunity (fires ONCE when in +SL_PIPS profit) ──
                if ticket not in self._breakeven_set and entry:
                    be_threshold = float(getattr(CONFIG, "SCALPER_SL_PIPS", 6.0))
                    profit_pips  = (
                        (current - entry) / pip if direction == "BUY"
                        else (entry - current) / pip
                    )
                    if profit_pips >= be_threshold:
                        self._breakeven_set.add(ticket)
                        be_price = (
                            entry + pip if direction == "BUY"
                            else entry - pip
                        )
                        try:
                            self.alerts.breakeven_moved(
                                symbol    = symbol,
                                direction = direction,
                                ticket    = ticket,
                                new_sl    = be_price,
                            )
                        except Exception as exc:
                            logger.exception("‼️ breakeven alert CRASHED: %s", exc)

            acc = self.risk._get_account()
            if acc and self._session_open_equity:
                drawdown = (
                    (self._session_open_equity - acc["equity"])
                    / self._session_open_equity
                )
                if drawdown >= CONFIG.MAX_DAILY_LOSS:
                    logger.warning(
                        "Drawdown %.1f%% ≥ limit %.1f%% – danger exit %s.",
                        drawdown * 100, CONFIG.MAX_DAILY_LOSS * 100, symbol,
                    )
                    self._close_position(ticket, symbol, reason="drawdown_limit")
                    closed.append(ticket)

        for t in closed:
            self._open_tickets.pop(t, None)

    # ------------------------------------------------------------------ close

    def _close_position(self, ticket: int, symbol: str, reason: str = "") -> None:
        try:
            result = self.executor.close_position(ticket)
        except Exception as exc:
            logger.exception("‼️ close_position(%d) CRASHED: %s", ticket, exc)
            return

        if result and result.get("ticket"):
            pnl = self._fetch_closed_pnl(ticket)
            self.risk.record_trade_result(pnl=pnl, symbol=symbol)
            self.dashboard.log_trade(
                symbol       = symbol,
                direction    = "?",
                pnl          = pnl,
                ticket       = ticket,
                close_reason = reason,
            )
            logger.info("Closed %d (%s) reason=%s pnl=%.2f", ticket, symbol, reason, pnl)
            entry_px, close_px, dir_hist, pips_hist = \
                self._fetch_closed_deal_info(ticket, symbol)
            try:
                self.alerts.trade_closed(
                    symbol      = symbol,
                    direction   = dir_hist,
                    ticket      = ticket,
                    entry       = entry_px,
                    close_price = close_px,
                    pnl         = pnl,
                    pips        = pips_hist,
                    reason      = reason,
                )
            except Exception as exc:
                logger.exception("‼️ alerts.trade_closed() CRASHED: %s", exc)
            self._negative_warned.discard(ticket)
            self._breakeven_set.discard(ticket)
        else:
            logger.error("Failed to close ticket %d (%s). Result: %s", ticket, symbol, result)

    # ------------------------------------------------------------------ trailing

    def _update_trailing_stops(self) -> None:
        trail_pips = self._trailing_pips
        for ticket, symbol in list(self._open_tickets.items()):
            try:
                pos = self.executor.get_open_position(ticket)
                if pos is None:
                    continue

                tp            = pos.get("tp",      0.0)
                current_price = pos.get("current", 0.0)
                pip           = self._get_pip_size_safe(symbol)

                if tp and current_price and pip:
                    dist_to_tp = abs(tp - current_price) / pip
                    if dist_to_tp < 2.0:
                        logger.debug("%s ticket=%d within 2p of TP – trail skipped.",
                                     symbol, ticket)
                        continue

                self.executor.modify_trailing_stop(
                    ticket     = ticket,
                    symbol     = symbol,
                    trail_pips = trail_pips,
                )
            except Exception as exc:
                logger.exception("‼️ trailing stop update CRASHED for %d (%s): %s",
                                 ticket, symbol, exc)

    # ------------------------------------------------------------------ EOD

    def _eod_check(self) -> None:
        if not getattr(CONFIG, "CLOSE_TRADES_EOD", True):
            return
        now_utc  = datetime.now(timezone.utc)
        eod_hour = getattr(CONFIG, "EOD_CLOSE_HOUR_UTC", 21)
        if now_utc.hour == eod_hour and now_utc.minute >= 55:
            if not self._open_tickets:
                return
            logger.info("EOD close at %02d:%02d UTC – closing %d position(s).",
                        now_utc.hour, now_utc.minute, len(self._open_tickets))
            for ticket, symbol in list(self._open_tickets.items()):
                self._close_position(ticket, symbol, reason="EOD")
            self._open_tickets.clear()

    # ------------------------------------------------------------------ summary

    def _daily_summary(self) -> None:
        pnl     = getattr(self.risk, "_daily_pnl",     0.0)
        history = getattr(self.risk, "_trade_history", [])

        today        = datetime.now(timezone.utc).date()
        today_trades = [
            t for t in history
            if hasattr(t.get("time"), "date") and t["time"].date() == today
        ]

        trades = len(today_trades)
        wins   = sum(1 for t in today_trades if t.get("profit", 0.0) > 0)
        losses = trades - wins
        wr     = (wins / trades * 100) if trades else 0.0

        msg = (f"📊 Daily Summary | P&L: {pnl:+.2f} | "
               f"Trades: {trades} | W:{wins} L:{losses} | WR:{wr:.0f}%")
        logger.info(msg)
        try:
            self.alerts.telegram._send(msg)
        except Exception:
            pass
        self.dashboard.daily_summary(
            pnl=pnl, trades=trades, wins=wins, losses=losses, win_rate=wr
        )

    # ------------------------------------------------------------------ keyboard

    def _keyboard_listener(self) -> None:
        print("\nControls: [q]uit  [p]ause/resume  [s]ound  [t]elegram  [m]enu")
        while self._running:
            try:
                key = input().strip().lower()
                logger.info("⌨️  Key pressed: %r", key)
            except EOFError:
                logger.warning("⌨️  EOFError in keyboard listener — ignoring, loop continues.")
                time.sleep(1.0)
                continue
            except Exception as exc:
                logger.exception("⌨️  keyboard listener exception: %s", exc)
                time.sleep(1.0)
                continue

            if key == "q":
                logger.info("Quit requested via keyboard.")
                self._running = False
            elif key == "p":
                self._paused = not self._paused
                state = "PAUSED" if self._paused else "RESUMED"
                logger.info("Bot %s.", state)
                print(f"  ── Bot {state} ──")
            elif key == "s":
                self._sound_on = not self._sound_on
                print(f"  Sound: {'ON' if self._sound_on else 'OFF'}")
            elif key == "t":
                self._tg_on = not self._tg_on
                print(f"  Telegram: {'ON' if self._tg_on else 'OFF'}")
            elif key == "m":
                self._runtime_menu()

    def _runtime_menu(self) -> None:
        print("\n── Runtime Menu ──────────────────────────")
        print("  1) Show open positions")
        print("  2) Show daily P&L")
        print("  3) Close all positions")
        print("  4) Back")
        choice = input("  Select: ").strip()
        if choice == "1":
            for t, s in self._open_tickets.items():
                pos = self.executor.get_open_position(t)
                pnl = pos.get("profit", "?") if pos else "closed"
                print(f"    ticket={t}  {s}  pnl={pnl}")
        elif choice == "2":
            self._daily_summary()
        elif choice == "3":
            for ticket, symbol in list(self._open_tickets.items()):
                self._close_position(ticket, symbol, reason="manual")
            self._open_tickets.clear()

    # ------------------------------------------------------------------ shutdown

    def _shutdown(self) -> None:
        logger.info("Shutting down GODBOT v3.0…")
        self._running = False
        for ticket, symbol in list(self._open_tickets.items()):
            self._close_position(ticket, symbol, reason="shutdown")
        self._open_tickets.clear()
        self._daily_summary()
        self.alerts.send("🛑 GODBOT v3.0 stopped.")
        try:
            self.connector.disconnect()
            logger.info("MT5 disconnected.")
        except Exception as exc:
            logger.warning("MT5 disconnect error: %s", exc)


# =============================================================================
# Entry point
# =============================================================================
def _setup_logging() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    _log_fmt  = "%(asctime)s | %(levelname)-8s | %(name)-26s | %(message)s"
    _log_date = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level   = logging.DEBUG,
        format  = _log_fmt,
        datefmt = _log_date,
        stream  = sys.stdout,
    )

    # ── FIX: File handler so logs persist even when the dashboard
    # clears the screen (os.system("cls") was wiping all stdout output
    # every second — making exceptions invisible in the terminal).
    # Every log line now also goes to logs/godbot_YYYYMMDD_HHMMSS.log.
    log_dir = _REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"godbot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _fh = logging.FileHandler(str(log_file), encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter(_log_fmt, datefmt=_log_date))
    logging.getLogger().addHandler(_fh)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    logger.info("📝 Logging to: %s", log_file)

    # ── Auto-open a second terminal window showing the DASHBOARD ─────────
    # Layout after this change:
    #   VS Code terminal   = clean scrolling log (scan steps, signals, etc.)
    #   Popup window       = dashboard (account, positions, performance)
    #
    # The dashboard writes to dashboard_live.txt every second.
    # The popup runs a PowerShell loop: cls → print file → sleep 1s.
    # cls works correctly in a real PowerShell console (it's only
    # dangerous when called via Python's os.system() because of the
    # subprocess console handle inheritance — here it's inside the
    # already-spawned PowerShell process so it's perfectly safe).
    if sys.platform == "win32":
        try:
            dash_file = str(_REPO_ROOT / "dashboard_live.txt").replace("'", "''")
            ps_cmd    = (
                f"$host.UI.RawUI.WindowTitle = 'GODBOT Dashboard'; "
                f"while ($true) {{ "
                f"  cls; "
                f"  if (Test-Path '{dash_file}') {{ "
                f"    Get-Content '{dash_file}' -Encoding UTF8; "
                f"  }} else {{ "
                f"    Write-Host 'Waiting for GODBOT dashboard...'; "
                f"  }}; "
                f"  Start-Sleep -Milliseconds 950 "
                f"}}"
            )
            subprocess.Popen(
                ["powershell", "-NoExit", "-Command", ps_cmd],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
            logger.info("📺 Dashboard window spawned.")
        except Exception as exc:
            logger.warning("Could not open dashboard window: %s", exc)


if __name__ == "__main__":
    _setup_logging()
    menu    = StartupMenu()
    profile = menu.run()

    print(f"\n  Starting GODBOT v3.0 as {profile.name!r} "
          f"({profile.style} / {profile.mode} / {profile.scalper_tf})\n")

    system = ForexSystem(profile)
    system.start()
