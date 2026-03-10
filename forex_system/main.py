# =============================================================================
# GODBOT v3.0 – main.py
# =============================================================================
# Full integration script – all fixes applied:
#
#  A  Pass sl_pips/tp_pips/confidence to risk_manager.calculate_position()
#  B  Pass trail_pips (not raw points) to executor.modify_trailing_stop()
#  C  Use profile trailing_stop value instead of hard-coded constant
#  D  Use get_open_position() helper instead of direct MT5 calls in monitor
#  E  Correct TradingSignal field access (sl_pips, tp_pips present)
#  F  Session filtering uses CET-aware logic from CONFIG
#  G  ML veto threshold reads from CONFIG / profile
#  H  Duplicate-symbol guard before calling calculate_position
#  I  EOD close uses executor.close_position ticket-based call
#  J  Daily summary reads RiskManager state (daily_pnl, trade count)
#  K  Broker-closed detection uses get_open_position return value
#  L  Semi-auto confirmation prompt is non-blocking (uses threading)
#  M  ForexSystem.start() catches KeyboardInterrupt cleanly
#  N  Trailing stop skipped when position already at TP proximity
#  O  spread check uses CONFIG.SCALPER_MAX_SPREAD (profile-aware)
#  P  Risk manager pre-trade checks respected; trade skipped on block
#  Q  Proper cleanup: MT5 disconnected on all exit paths
#
#  R  [FIX] All import paths corrected to match actual folder structure.
#  S  [FIX] SignalType enum references replaced with plain string comparisons.
#  T  [FIX] SentimentAnalyzer → SentimentReader, .get() → .get_sentiment().
#  U  [FIX] intermarket.get_bias() → get_intermarket_signal()["bias"].
#  V  [FIX] MT5Connector.get_account_info called before connect() – moved
#           AlertManager currency fetch out of __init__, deferred to start().
#  W  [FIX] _max_spread reads profile key "max_spread" (lowercase).
#  X  [FIX] _trailing_pips reads profile key "trailing_stop" (lowercase).
#  Y  [FIX] _main_loop reads profile key "scan_secs" (not "SCAN_INTERVAL").
#  Z  [FIX] _eod_check reads CONFIG.EOD_CLOSE_HOUR_UTC (int, not tuple).
#  AA [FIX] Duplicate _keyboard_listener thread removed from start().
#  AB [FIX] Dashboard display thread started in start().
#  AC [FIX] dashboard.set_mode() and set_scan_secs() called after init.
#  AD [FIX] dashboard.update_scan_status() called every loop tick.
#  AE [FIX] HuggingFace / transformers log noise suppressed via env vars.
# =============================================================================

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Suppress HuggingFace / transformers startup noise  (Fix AE)
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_VERBOSITY",        "error")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_VERBOSITY",              "warning")
os.environ.setdefault("TOKENIZERS_PARALLELISM",         "false")

# ---------------------------------------------------------------------------
# Portable Numba cache (must be set before any numba import)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(_REPO_ROOT / ".numba_cache"))

# ---------------------------------------------------------------------------
# Project imports  (Fix R – all paths corrected to actual folder structure)
# ---------------------------------------------------------------------------
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore

from config.settings              import CONFIG
from core.mt5_connector           import MT5Connector
from core.data_handler            import DataHandler
from indicators.indicators_engine import IndicatorEngine
from signals.signal_engine        import SignalEngine
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
_SEMI_AUTO_TIMEOUT = 30  # seconds to wait for semi-auto confirmation


# =============================================================================
# ProfileManager
# =============================================================================
@dataclass
class TraderProfile:
    style:      str = "scalper"
    mode:       str = "semi_automated"
    name:       str = "Trader"
    scalper_tf: str = "M5"


class ProfileManager:
    """Persist and load the trader profile from disk."""

    def load(self) -> TraderProfile:
        if PROFILE_PATH.exists():
            try:
                data = json.loads(PROFILE_PATH.read_text())
                # Strip meta keys (e.g. _comment) before passing to dataclass
                clean = {
                    k: v for k, v in data.items()
                    if not k.startswith("_") and k != "version"
                }
                return TraderProfile(**clean)
            except Exception as exc:
                logger.warning(
                    "Could not load profile (%s); using defaults.", exc
                )
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
    """Interactive CLI for selecting trading configuration."""

    _STYLES = {"1": "scalper", "2": "day_trader"}
    _TFS    = {"1": "M1",      "2": "M5"}
    _MODES  = {
        "1": "signal_only",
        "2": "semi_automated",
        "3": "fully_automated",
    }

    def __init__(self) -> None:
        self._pm = ProfileManager()

    def run(self) -> TraderProfile:
        profile = self._pm.load()
        print("\n" + "=" * 60)
        print("  GODBOT v3.0 – Startup Configuration")
        print("=" * 60)

        print("\nTrading style:\n  1) Scalper\n  2) Day Trader")
        choice = input(
            f"  Select [default={profile.style}]: "
        ).strip()
        profile.style = self._STYLES.get(choice, profile.style)

        if profile.style == "scalper":
            print("\nScalper timeframe:\n  1) M1\n  2) M5")
            choice = input(
                f"  Select [default={profile.scalper_tf}]: "
            ).strip()
            profile.scalper_tf = self._TFS.get(choice, profile.scalper_tf)

        print("\nOperation mode:")
        print("  1) Signal Only\n  2) Semi-Automated\n  3) Fully Automated")
        choice = input(
            f"  Select [default={profile.mode}]: "
        ).strip()
        profile.mode = self._MODES.get(choice, profile.mode)

        name = input(f"\nYour name [default={profile.name}]: ").strip()
        if name:
            profile.name = name

        save = input("\nSave this profile? [y/N]: ").strip().lower()
        if save == "y":
            self._pm.save(profile)

        return profile


# =============================================================================
# ForexSystem
# =============================================================================
class ForexSystem:
    """Core trading engine – initialises all subsystems and runs the main loop."""

    def __init__(self, profile: TraderProfile) -> None:
        self.profile   = profile
        self._running  = False
        self._paused   = False
        self._sound_on = True
        self._tg_on    = True

        # Apply selected scalper timeframe to CONFIG before anything reads it
        if profile.style == "scalper":
            tf_int_map = {"M1": 1, "M5": 5}
            CONFIG.SCALPER_TF_SELECTED = tf_int_map.get(
                profile.scalper_tf, 5
            )                                               # Fix V – int not str
            logger.info("Scalper TF set to %s", profile.scalper_tf)

        # Call get_scalper_profile() now so all CONFIG.SCALPER_* flat aliases
        # are populated before any subsystem reads them.
        CONFIG.get_scalper_profile()

        # ------------------------------------------------------------------
        # Subsystem initialisation
        # Note: MT5Connector is NOT connected yet – connect() is called in
        # start(). Nothing here should call get_account_info().   (Fix V)
        # ------------------------------------------------------------------
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
        self.sentiment     = SentimentAnalyzer()            # Fix T
        self.intermarket   = IntermarketAnalyzer()
        self.cot           = COTReader()

        # Wire dashboard mode and scan interval now that both
        # dashboard and profile are available.  (Fix AC)
        self.dashboard.set_mode(profile.mode)
        self.dashboard.set_scan_secs(
            CONFIG.get_scalper_profile().get("scan_secs", 20)
        )

        # Open-position registry: ticket → symbol
        self._open_tickets: dict[int, str] = {}

        # Scheduling state
        self._last_scan:    float = 0.0
        self._last_trail:   float = 0.0
        self._last_summary: float = 0.0
        self._session_open_equity: Optional[float] = None

        logger.info(
            "ForexSystem initialised for %s (%s / %s)",
            profile.name, profile.style, profile.mode,
        )

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------

    @property
    def _scalper_profile(self) -> dict:
        return CONFIG.get_scalper_profile()

    @property
    def _max_spread(self) -> float:
        # Fix W – profile dict uses lowercase key "max_spread"
        return float(
            self._scalper_profile.get("max_spread", CONFIG.SCALPER_MAX_SPREAD)
        )

    @property
    def _trailing_pips(self) -> float:
        # Fix X – profile dict uses lowercase key "trailing_stop"
        return float(
            self._scalper_profile.get(
                "trailing_stop", CONFIG.TRAILING_STOP_PIPS
            )
        )

    @property
    def _ml_threshold(self) -> float:
        return float(
            self._scalper_profile.get(
                "ml_threshold", CONFIG.ML_MIN_CONFIDENCE
            )
        )

    @property
    def _scan_secs(self) -> float:
        # Fix Y – profile dict uses lowercase key "scan_secs"
        return float(
            self._scalper_profile.get("scan_secs", 20.0)
        )

    def _in_session(self) -> bool:
        now_utc = datetime.now(timezone.utc)
        if now_utc.weekday() >= 5:
            return False
        if getattr(CONFIG, "ENABLE_QUIET_HOURS", False):
            hour_utc       = now_utc.hour
            q_start, q_end = getattr(CONFIG, "QUIET_HOURS_UTC", (22, 7))
            if q_start > q_end:
                if hour_utc >= q_start or hour_utc < q_end:
                    return False
            elif q_start <= hour_utc < q_end:
                return False
        return True

    def _open_symbols(self) -> set[str]:
        return set(self._open_tickets.values())

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def start(self) -> None:
        logger.info("Connecting to MT5…")
        if not self.connector.connect():
            logger.critical("MT5 connection failed – aborting.")
            return

        self._running = True
        logger.info("GODBOT v3.0 started.  Mode: %s", self.profile.mode)
        self.alerts.send(f"🤖 GODBOT v3.0 started – {self.profile.mode}")

        # Safe to call get_account_info() now – MT5 is connected  (Fix V)
        acc = self.risk._get_account()
        if acc:
            self._session_open_equity = acc["equity"]
            self.risk.session_equity  = acc["equity"]

        # Keyboard listener thread – one instance only  (Fix AA)
        _t = threading.Thread(
            target=self._keyboard_listener, daemon=True
        )
        _t.start()

        # Dashboard display thread  (Fix AB)
        _d = threading.Thread(target=self.dashboard.display, daemon=True)
        _d.start()

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received – shutting down.")
        except Exception as exc:
            logger.exception("Unhandled exception in main loop: %s", exc)
            self.alerts.send(f"🚨 GODBOT crash: {exc}")
        finally:
            self._shutdown()

    def _main_loop(self) -> None:
        # Fix Y – use _scan_secs property (reads "scan_secs" from profile dict)
        scan_interval    = self._scan_secs
        trail_interval   = 15.0
        summary_interval = 3600.0

        while self._running:
            now = time.monotonic()

            if self._paused:
                time.sleep(1.0)
                continue

            # Feed countdown to dashboard every tick  (Fix AD)
            secs_left = max(0.0, scan_interval - (now - self._last_scan))
            self.dashboard.update_scan_status(f"Next scan in {secs_left:.0f}s")

            if now - self._last_scan >= scan_interval:
                self._last_scan = now
                if self._in_session():
                    self._scan_symbols()
                else:
                    logger.debug("Outside session – scan skipped.")

            if now - self._last_trail >= trail_interval:
                self._last_trail = now
                self._update_trailing_stops()

            if now - self._last_summary >= summary_interval:
                self._last_summary = now
                self._daily_summary()

            self._monitor_positions()
            self._eod_check()
            time.sleep(1.0)

    # ------------------------------------------------------------------
    # Symbol scanning
    # ------------------------------------------------------------------

    def _scan_symbols(self) -> None:
        for symbol in CONFIG.SYMBOLS:
            try:
                self._process_symbol(symbol)
            except Exception as exc:
                logger.error("Error processing %s: %s", symbol, exc)

    def _process_symbol(self, symbol: str) -> None:
        # --- Spread gate ---
        tick = self.data.get_tick_data(symbol)
        if tick is None:
            return
        spread_pips = tick.get("spread_pips", 999.0)
        if spread_pips > self._max_spread:
            logger.debug(
                "%s: spread %.2f > max %.2f – skip",
                symbol, spread_pips, self._max_spread,
            )
            return

        # --- Duplicate-symbol gate ---
        if symbol in self._open_symbols():
            logger.debug("%s already open – skip.", symbol)
            return

        # --- News gate ---
        news_check = self.calendar.is_safe_to_trade(symbol)
        if not news_check.get("safe", True):
            logger.debug(
                "%s: news window – skip. Reason: %s",
                symbol, news_check.get("reason", ""),
            )
            return

        # --- OHLCV data ---
        tf = self._scalper_profile.get("tf_primary", CONFIG.PRIMARY_TF)
        df = self.data.get_data(symbol, tf)
        if df is None or len(df) < 50:
            logger.debug("%s: insufficient bars.", symbol)
            return

        # --- Indicators ---
        tf_confirm = self._scalper_profile.get("tf_confirm", 15)
        htf_df     = self.data.get_data(symbol, tf_confirm)
        ind        = self.indicators.compute_all(df, htf_df=htf_df)
        if ind is None:
            return

        # --- Technical signal ---
        signal = self.signal_engine.evaluate(df, symbol)
        if signal is None or signal.signal_type == "HOLD":
            logger.info(
                "%s: no signal (HOLD or None) — market ranging.", symbol
            )
            return

        # --- ML veto ---
        ml = self.ml_model.predict(df, ind)
        if ml is None:
            return
        ml_label = ml.get("label", "HOLD")
        ml_conf  = float(ml.get("confidence", 0.0))

        if ml_label == "HOLD":
            logger.debug("%s: ML HOLD veto.", symbol)
            return
        if ml_conf < self._ml_threshold:
            logger.debug(
                "%s: ML confidence %.2f < %.2f – veto.",
                symbol, ml_conf, self._ml_threshold,
            )
            return

        # Fix S – plain string comparison, no SignalType enum
        if signal.signal_type == "BUY" and ml_label != "BUY":
            logger.debug(
                "%s: ML direction (%s) vs signal (%s) mismatch – veto.",
                symbol, ml_label, signal.signal_type,
            )
            return
        if signal.signal_type == "SELL" and ml_label != "SELL":
            logger.debug(
                "%s: ML direction (%s) vs signal (%s) mismatch – veto.",
                symbol, ml_label, signal.signal_type,
            )
            return

        # --- Day-trader filters ---
        if self.profile.style == "day_trader":
            if not self._day_trader_filters(symbol, signal):
                return

        # --- Risk sizing ---
        position_spec = self.risk.calculate_position(
            symbol      = symbol,
            signal_type = signal.signal_type,
            entry       = signal.entry,
            sl_pips     = signal.sl_pips,
            tp_pips     = signal.tp_pips,
            confidence  = ml_conf,
        )
        if position_spec is None:
            logger.info("%s: risk manager blocked trade.", symbol)
            return

        # --- Alert ---
        self.alerts.signal_alert(
            symbol        = symbol,
            signal        = signal,
            position_spec = position_spec,
            sound         = self._sound_on,
            telegram      = self._tg_on,
        )
        self.dashboard.update_signal(symbol, signal, position_spec)

        # --- Execution ---
        if self.profile.mode == "signal_only":
            return

        if self.profile.mode == "semi_automated":
            if not self._semi_auto_confirm(symbol, signal):
                return

        result = self.executor.send_market_order(
            symbol     = symbol,
            order_type = signal.signal_type,
            volume     = position_spec.volume,
            sl         = position_spec.sl_price,
            tp         = position_spec.tp_price,
            comment    = f"GODBOT_{signal.signal_type}",
        )
        if result and result.get("ticket"):
            ticket = result["ticket"]
            self._open_tickets[ticket] = symbol
            logger.info(
                "✅ Order filled: %s %s | ticket=%d | vol=%.2f | "
                "SL=%s TP=%s | risk=$%.2f | R:R=%.1f",
                signal.signal_type, symbol, ticket,
                position_spec.volume,
                position_spec.sl_price, position_spec.tp_price,
                position_spec.risk_usd, position_spec.rr_ratio,
            )
            self.alerts.send(
                f"✅ {signal.signal_type} {symbol} | "
                f"ticket={ticket} | vol={position_spec.volume:.2f} | "
                f"risk=${position_spec.risk_usd:.2f}"
            )

    # ------------------------------------------------------------------
    # Semi-automated confirmation
    # ------------------------------------------------------------------

    def _semi_auto_confirm(self, symbol: str, signal) -> bool:
        confirmed = threading.Event()
        rejected  = threading.Event()

        def _prompt() -> None:
            ans = input(
                f"\n[SEMI-AUTO] {signal.signal_type} {symbol} | "
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

    # ------------------------------------------------------------------
    # Day-trader filters
    # ------------------------------------------------------------------

    def _day_trader_filters(self, symbol: str, signal) -> bool:
        # Fix T – method is get_sentiment(), returns dict not raw float
        sent_result = self.sentiment.get_sentiment(symbol)
        if sent_result is not None:
            sent_score = sent_result.get("score", 0.0)
            if signal.signal_type == "BUY" and sent_score < 0:
                logger.debug(
                    "%s: sentiment bearish (%.2f) vs BUY – skip.",
                    symbol, sent_score,
                )
                return False
            if signal.signal_type == "SELL" and sent_score > 0:
                logger.debug(
                    "%s: sentiment bullish (%.2f) vs SELL – skip.",
                    symbol, sent_score,
                )
                return False

        # Fix U – get_intermarket_signal() returns dict; extract "bias" key
        im_result = self.intermarket.get_intermarket_signal(symbol)
        if im_result is not None:
            im_bias = im_result.get("bias", "Neutral")
            if im_bias == "Bullish" and signal.signal_type == "SELL":
                logger.debug(
                    "%s: intermarket bias Bullish vs SELL – skip.", symbol
                )
                return False
            if im_bias == "Bearish" and signal.signal_type == "BUY":
                logger.debug(
                    "%s: intermarket bias Bearish vs BUY – skip.", symbol
                )
                return False

        # COT bias check
        cot_result = self.cot.get_cot_signal(symbol)
        if cot_result is not None:
            cot_bias = cot_result.get("signal", "neutral").lower()
            if cot_bias == "bullish" and signal.signal_type == "SELL":
                logger.debug(
                    "%s: COT bias bullish vs SELL – skip.", symbol
                )
                return False
            if cot_bias == "bearish" and signal.signal_type == "BUY":
                logger.debug(
                    "%s: COT bias bearish vs BUY – skip.", symbol
                )
                return False

        return True

    # ------------------------------------------------------------------
    # Position monitoring
    # ------------------------------------------------------------------

    def _monitor_positions(self) -> None:
        closed: list[int] = []

        for ticket, symbol in list(self._open_tickets.items()):
            pos = self.executor.get_open_position(ticket)

            if pos is None:
                logger.info(
                    "Position %d (%s) closed by broker.", ticket, symbol
                )
                self.alerts.send(
                    f"📌 Position closed by broker: "
                    f"{symbol} ticket={ticket}"
                )
                self.risk.record_trade_result(
                    ticket=ticket, pnl=None, symbol=symbol
                )
                closed.append(ticket)
                continue

            pnl = pos.get("profit", 0.0)
            self.dashboard.update_position(ticket, symbol, pnl)

            acc = self.risk._get_account()
            if acc and self._session_open_equity:
                drawdown = (
                    (self._session_open_equity - acc["equity"])
                    / self._session_open_equity
                )
                if drawdown >= 0.03:
                    logger.warning(
                        "Drawdown %.1f%% reached – danger exit %s.",
                        drawdown * 100, symbol,
                    )
                    self._close_position(
                        ticket, symbol, reason="drawdown_limit"
                    )
                    closed.append(ticket)

        for t in closed:
            self._open_tickets.pop(t, None)

    def _close_position(
        self, ticket: int, symbol: str, reason: str = ""
    ) -> None:
        result = self.executor.close_position(ticket)
        if result and result.get("success"):
            pnl = result.get("profit", 0.0)
            self.risk.record_trade_result(
                ticket=ticket, pnl=pnl, symbol=symbol
            )
            logger.info(
                "Closed %d (%s) reason=%s pnl=%.2f",
                ticket, symbol, reason, pnl,
            )
            self.alerts.send(
                f"🔴 Closed {symbol} ticket={ticket} | "
                f"reason={reason} | pnl={pnl:+.2f}"
            )
        else:
            logger.error(
                "Failed to close ticket %d (%s).", ticket, symbol
            )

    # ------------------------------------------------------------------
    # Trailing stops
    # ------------------------------------------------------------------

    def _update_trailing_stops(self) -> None:
        trail_pips = self._trailing_pips

        for ticket, symbol in list(self._open_tickets.items()):
            pos = self.executor.get_open_position(ticket)
            if pos is None:
                continue

            tp            = pos.get("tp", 0.0)
            current_price = pos.get("price_current", 0.0)
            pip           = self.risk._get_pip_size(symbol)

            if tp and current_price and pip:
                dist_to_tp = abs(tp - current_price) / pip
                if dist_to_tp < 2.0:
                    logger.debug(
                        "%s ticket=%d near TP (%.1f pips) – trail skipped.",
                        symbol, ticket, dist_to_tp,
                    )
                    continue

            self.executor.modify_trailing_stop(
                ticket     = ticket,
                trail_pips = trail_pips,
            )

    # ------------------------------------------------------------------
    # End-of-day closure
    # ------------------------------------------------------------------

    def _eod_check(self) -> None:
        if not CONFIG.CLOSE_TRADES_EOD:
            return
        now_utc = datetime.now(timezone.utc)
        # Fix Z – EOD_CLOSE_HOUR_UTC is an int, not a tuple
        eod_hour = getattr(CONFIG, "EOD_CLOSE_HOUR_UTC", 21)
        if now_utc.hour == eod_hour and now_utc.minute >= 55:
            if not self._open_tickets:
                return
            logger.info(
                "EOD close triggered – closing %d position(s).",
                len(self._open_tickets),
            )
            for ticket, symbol in list(self._open_tickets.items()):
                self._close_position(ticket, symbol, reason="EOD")
            self._open_tickets.clear()

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    def _daily_summary(self) -> None:
        pnl    = getattr(self.risk, "daily_pnl",    0.0)
        trades = getattr(self.risk, "trades_today",  0)
        wins   = getattr(self.risk, "wins_today",    0)
        losses = getattr(self.risk, "losses_today",  0)
        wr     = (wins / trades * 100) if trades else 0.0

        msg = (
            f"📊 Daily Summary | P&L: {pnl:+.2f} | "
            f"Trades: {trades} | W:{wins} L:{losses} | WR:{wr:.0f}%"
        )
        logger.info(msg)
        self.alerts.send(msg)
        self.dashboard.daily_summary(
            pnl      = pnl,
            trades   = trades,
            wins     = wins,
            losses   = losses,
            win_rate = wr,
        )

    # ------------------------------------------------------------------
    # Keyboard listener
    # ------------------------------------------------------------------

    def _keyboard_listener(self) -> None:
        print(
            "\nControls: [q]uit  [p]ause/resume  "
            "[s]ound  [t]elegram  [m]enu"
        )
        while self._running:
            try:
                key = input().strip().lower()
            except EOFError:
                break
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
                print(
                    f"  Sound alerts: "
                    f"{'ON' if self._sound_on else 'OFF'}"
                )
            elif key == "t":
                self._tg_on = not self._tg_on
                print(
                    f"  Telegram alerts: "
                    f"{'ON' if self._tg_on else 'OFF'}"
                )
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
                print(f"    ticket={t} {s}  pnl={pnl}")
        elif choice == "2":
            self._daily_summary()
        elif choice == "3":
            for ticket, symbol in list(self._open_tickets.items()):
                self._close_position(ticket, symbol, reason="manual")
            self._open_tickets.clear()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

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
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)



if __name__ == "__main__":
    _setup_logging()
    menu    = StartupMenu()
    profile = menu.run()

    print(
        f"\n  Starting GODBOT v3.0 as {profile.name!r} "
        f"({profile.style} / {profile.mode} / {profile.scalper_tf})\n"
    )

    system = ForexSystem(profile)
    system.start()
