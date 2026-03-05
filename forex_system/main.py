# main.py – GODBOT v3.0
import os
import tempfile

# ── Portable Numba cache ──────────────────────────────────────────────────────
os.environ["NUMBA_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), ".numba_cache")

import time
import json
import sys
import threading
import schedule
import traceback

import pytz
import MetaTrader5 as mt5
from datetime import datetime
from typing import Optional

from core.mt5_connector       import MT5Connector, MT5_TIMEFRAME_MAP
from core.indicators          import IndicatorEngine
from signals.signal_engine    import SignalEngine
from signals.ml_model         import MLSignalModel
from risk.risk_manager        import RiskManager
from execution.order_executor  import OrderExecutor
from monitoring.logger        import get_logger
from monitoring.dashboard     import Dashboard
from notifications.alert_manager import AlertManager
from research.calendar_scanner   import CalendarScanner
from research.sentiment_analyzer import SentimentAnalyzer
from research.intermarket        import IntermarketAnalyzer
from research.cot_reader         import COTReader
from config.settings import (
    CONFIG,
    PROFILE_FILE,
    SCALPER_SCAN_SECS,    DAYTRADER_SCAN_SECS,
    SCALPER_TF_PRIMARY,   SCALPER_TF_CONFIRM,
    DAYTRADER_TF_PRIMARY, DAYTRADER_TF_CONFIRM,
    SCALPER_MAX_SPREAD,   DAYTRADER_MAX_SPREAD,
    SCALPER_SIGNAL_SCORE, DAYTRADER_SIGNAL_SCORE,
    SCALPER_SL_PIPS,      DAYTRADER_SL_PIPS,
    SCALPER_TP_PIPS,      DAYTRADER_TP_PIPS,
    LONDON_OPEN_CET,      NY_CLOSE_CET,
    OVERLAP_START,        OVERLAP_END,
    TRADER_NAME,          TRADER_TIMEZONE,
)

logger   = get_logger("Main")
TIMEZONE = pytz.timezone(TRADER_TIMEZONE)

ML_LABEL_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}


# ────────────────────────────────────────────────────────────────────────────
#  Profile persistence
# ────────────────────────────────────────────────────────────────────────────
class ProfileManager:
    DEFAULT = {"style": "scalper", "mode": "1", "name": TRADER_NAME}

    def load(self) -> dict:
        try:
            os.makedirs("data", exist_ok=True)
            if os.path.exists(PROFILE_FILE):
                with open(PROFILE_FILE) as f:
                    return json.load(f)
        except Exception:
            pass
        return self.DEFAULT.copy()

    def save(self, profile: dict) -> None:
        try:
            os.makedirs("data", exist_ok=True)
            with open(PROFILE_FILE, "w") as f:
                json.dump(profile, f, indent=2)
        except Exception as e:
            logger.error(f"Profile save error: {e}")


# ────────────────────────────────────────────────────────────────────────────
#  Interactive startup menu
# ────────────────────────────────────────────────────────────────────────────
class StartupMenu:
    STYLES = {"1": "scalper", "2": "daytrader"}
    MODES  = {"1": "Signal Only", "2": "Semi Automated", "3": "Fully Automated"}

    def __init__(self):
        self.profile_mgr = ProfileManager()

    def run(self) -> dict:
        self._print_banner()
        profile     = self.profile_mgr.load()
        has_profile = os.path.exists(PROFILE_FILE)

        if has_profile:
            self._print_saved_profile(profile)
            if self._ask("Use saved profile? (Y/N): ", ["y", "n"]) == "y":
                logger.info(
                    f"✅ Loaded profile: {profile['style'].title()} | Mode {profile['mode']}"
                )
                return profile

        profile = self._select_settings(profile)
        if self._ask("Save as profile? (Y/N): ", ["y", "n"]) == "y":
            self.profile_mgr.save(profile)
            print("  ✅ Profile saved!\n")
        return profile

    def _print_banner(self) -> None:
        print("\n" + "=" * 60)
        print("  🤖  GODBOT v3.0  —  Intelligent Forex Trading System")
        print("=" * 60 + "\n")

    def _print_saved_profile(self, profile: dict) -> None:
        print("  📂  Saved Profile Found")
        print(f"      Style : {profile.get('style', '?').title()}")
        print(f"      Mode  : {profile.get('mode', '?')} — "
              f"{self.MODES.get(profile.get('mode', '1'), '?')}")
        print(f"      Name  : {profile.get('name', '?')}\n")

    def _select_settings(self, profile: dict) -> dict:
        print("  Select Trading Style:")
        print("    1 = Scalper    (short-term, tight spreads)")
        print("    2 = Day Trader (swing positions)\n")
        style_key = self._ask("Style (1 or 2): ", ["1", "2"])
        profile["style"] = self.STYLES[style_key]

        print("\n  Select Operation Mode:")
        for k, v in self.MODES.items():
            print(f"    {k} = {v}")
        profile["mode"] = self._ask("\nMode (1 / 2 / 3): ", ["1", "2", "3"])

        name = input(f"  Trader name [{profile.get('name', TRADER_NAME)}]: ").strip()
        if name:
            profile["name"] = name
        elif "name" not in profile:
            profile["name"] = TRADER_NAME
        print()
        return profile

    def _ask(self, prompt: str, choices: list) -> str:
        while True:
            answer = input(f"  {prompt}").strip().lower()
            if answer in choices:
                return answer
            print(f"  ⚠️  Please enter: {' or '.join(choices)}")


# ────────────────────────────────────────────────────────────────────────────
#  Core system
# ────────────────────────────────────────────────────────────────────────────
class ForexSystem:
    MODE_NAMES = {"1": "Signal Only", "2": "Semi Automated", "3": "Fully Automated"}

    def __init__(self, profile: dict):
        self.profile  = profile
        self.style    = profile["style"]
        self.mode     = profile["mode"]
        self._running = False
        self._paused  = False
        self._last_candle_time: dict = {}

        self.connector   = MT5Connector()
        self.indicators  = IndicatorEngine()
        self.signal_eng  = SignalEngine()
        self.ml_model    = MLSignalModel()
        self.risk_mgr    = RiskManager()
        self.executor    = OrderExecutor()
        self.dashboard   = Dashboard()
        self.alerts      = AlertManager()
        self.calendar    = CalendarScanner()
        self.sentiment   = SentimentAnalyzer()
        self.intermarket = IntermarketAnalyzer()
        self.cot         = COTReader()

        # Wire risk manager to alert system
        self.risk_mgr.set_alerts(self.alerts)

        self._start_keyboard_listener()

    # ── startup ───────────────────────────────────────────────────────────────
    def start(self) -> None:
        if not self.connector.connect():
            raise RuntimeError("Cannot connect to MT5 — is the terminal open?")

        self.risk_mgr.snapshot_session_equity()

        account = self.connector.get_account_info()
        self._print_launch_summary(account)

        if self.cot.should_update():
            logger.info("📥 Downloading COT data…")
            self.cot.download_cot_data()

        self.calendar.print_todays_events()
        self._load_ml_models()

        self.alerts.system_online(
            account.get("balance", 0),
            self.style,
            self.MODE_NAMES[self.mode],
        )
        print("\n  ⌨️  Keyboard: M=Menu  P=Pause  S=Sound  T=Telegram  Q=Quit\n")
        self._running = True
        self._run_loop()

    def _print_launch_summary(self, account: dict) -> None:
        print("\n" + "─" * 60)
        print(f"  👤  Trader  : {self.profile.get('name', TRADER_NAME)}")
        print(f"  💼  Style   : {self.style.title()}")
        print(f"  🎮  Mode    : {self.mode} — {self.MODE_NAMES[self.mode]}")
        print(f"  💰  Balance : {account.get('balance', 0):,.2f} {account.get('currency', '')}")
        print(f"  📈  Equity  : {account.get('equity',  0):,.2f} {account.get('currency', '')}")
        print(f"  🏦  Broker  : {account.get('company', 'Unknown')}")
        print(f"  📋  Watchlist: {', '.join(CONFIG.WATCHLIST)}")
        print("─" * 60 + "\n")

    def _load_ml_models(self) -> None:
        tf = SCALPER_TF_PRIMARY if self.style == "scalper" else DAYTRADER_TF_PRIMARY
        for symbol in CONFIG.WATCHLIST:
            logger.info(f"🤖 Loading / training ML model for {symbol}…")
            loaded = self.ml_model.load(symbol)
            if not loaded:
                logger.info(f"   No saved model found for {symbol} — training from scratch…")
                df_raw = self.connector.get_ohlcv(symbol, tf, bars=5000)
                if df_raw is not None and not df_raw.empty:
                    df = self.indicators.compute_all(df_raw)
                    if df is not None and not df.empty:
                        self.ml_model.train(df, symbol)
                        logger.info(f"   ✅ ML model trained and saved for {symbol}.")
                    else:
                        logger.warning(f"   ⚠️  Indicators returned empty data for {symbol}.")
                else:
                    logger.warning(f"   ⚠️  No OHLCV data for {symbol} — skipping.")
            else:
                logger.info(f"   ✅ ML model loaded from disk for {symbol}.")

    # ── main loop ─────────────────────────────────────────────────────────────
    def _run_loop(self) -> None:
        scan_secs = SCALPER_SCAN_SECS if self.style == "scalper" else DAYTRADER_SCAN_SECS
        self.dashboard.set_scan_secs(scan_secs)

        schedule.every(scan_secs).seconds.do(self._scan_markets)
        schedule.every(30).seconds.do(self._monitor_positions)
        schedule.every(60).seconds.do(self._update_trailing_stops)
        schedule.every().day.at("23:45").do(self._close_all_day_trades)
        schedule.every().day.at("23:55").do(self._daily_summary)

        logger.info(
            f"⚙️ Scan: {scan_secs}s | Mode: {self.MODE_NAMES[self.mode]} | "
            f"Style: {self.style.title()} | Watchlist: {', '.join(CONFIG.WATCHLIST)}"
        )

        while self._running:
            try:
                if not self._paused:
                    schedule.run_pending()
                    self.dashboard.display()
                time.sleep(1)
            except KeyboardInterrupt:
                self.stop()
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                logger.debug(traceback.format_exc())
                time.sleep(5)

    # ── session helpers ───────────────────────────────────────────────────────
    def _is_trading_session(self) -> bool:
        now = datetime.now(TIMEZONE)
        if now.weekday() in (5, 6):
            logger.debug("Weekend — market closed, skipping scan.")
            return False
        return True

    def _is_overlap_session(self) -> bool:
        hour = datetime.now(TIMEZONE).hour
        return OVERLAP_START <= hour < OVERLAP_END

    # ── scanning ──────────────────────────────────────────────────────────────
    def _scan_markets(self) -> None:
        if not self._is_trading_session():
            return
        for symbol in CONFIG.WATCHLIST:
            self._process_symbol(symbol)

    # ── per-symbol processing (replaces _process_eurusd) ─────────────────────
    def _process_symbol(self, symbol: str) -> None:
        tf = SCALPER_TF_PRIMARY if self.style == "scalper" else DAYTRADER_TF_PRIMARY

        # ── Spread gate ───────────────────────────────────────────────────────
        tick     = mt5.symbol_info_tick(symbol)
        sym_info = mt5.symbol_info(symbol)
        if tick and sym_info:
            spread = (tick.ask - tick.bid) / sym_info.point / 10
            max_sp = SCALPER_MAX_SPREAD if self.style == "scalper" else DAYTRADER_MAX_SPREAD
            if spread > max_sp:
                logger.debug(f"[{symbol}] Spread too wide: {spread:.1f} pips (max {max_sp})")
                return

        # ── Gate 1 — Calendar ─────────────────────────────────────────────────
        safety = self.calendar.is_safe_to_trade(
            symbol, minutes_before=30, minutes_after=15
        )
        if not safety["safe"]:
            evt = safety["events"][0] if safety["events"] else {}
            logger.debug(f"[{symbol}] News block: {evt.get('event', '?')}")
            return

        # ── Candle guard ──────────────────────────────────────────────────────
        tf_mapped = MT5_TIMEFRAME_MAP.get(tf, tf)
        rates     = mt5.copy_rates_from_pos(symbol, tf_mapped, 0, 1)
        if rates is not None and len(rates) > 0:
            candle_time = rates[0]["time"]
            if self._last_candle_time.get(symbol) == candle_time:
                logger.debug(f"[{symbol}] Same candle — skipping heavy computation.")
                return
            self._last_candle_time[symbol] = candle_time
            logger.debug(
                f"[{symbol}] New candle @ "
                f"{datetime.utcfromtimestamp(candle_time).strftime('%H:%M:%S')} UTC"
            )

        # ── Gate 2 — Technical ────────────────────────────────────────────────
        df_raw = self.connector.get_ohlcv(symbol, tf)
        if df_raw is None or df_raw.empty:
            logger.debug(f"[{symbol}] ⛔ Gate 2 BLOCKED — no OHLCV data returned")
            return
        df = self.indicators.compute_all(df_raw)
        if df is None or df.empty:
            logger.debug(f"[{symbol}] ⛔ Gate 2 BLOCKED — indicators returned empty DataFrame")
            return

        signal = self.signal_eng.evaluate(df, symbol)
        if not signal:
            logger.debug(f"[{symbol}] ⛔ Gate 2 BLOCKED — signal score too low")
            return

        logger.debug(
            f"[{symbol}] ✅ Gate 2 PASSED — {signal.signal.value} | "
            f"Strength={signal.strength:.2f} | Conf={signal.confidence:.0%}"
        )

        # ── Gate 3 — ML ───────────────────────────────────────────────────────
        ml           = self.ml_model.predict(df, symbol)
        ml_direction = ML_LABEL_MAP.get(ml["label"], str(ml["label"]))

        logger.debug(
            f"[{symbol}] 🤖 ML prediction: {ml_direction} | "
            f"Confidence: {ml['confidence']:.0%} | "
            f"Required: {CONFIG.ML_MIN_CONFIDENCE:.0%} | "
            f"Signal wants: {signal.signal.value} | "
            f"HOLD={ml['probabilities']['HOLD']:.0%} "
            f"BUY={ml['probabilities']['BUY']:.0%} "
            f"SELL={ml['probabilities']['SELL']:.0%}"
        )

        if ml_direction != signal.signal.value:
            logger.debug(
                f"[{symbol}] ⛔ Gate 3 BLOCKED — ML disagrees: "
                f"ML={ml_direction} vs Signal={signal.signal.value}"
            )
            return
        if ml["confidence"] < CONFIG.ML_MIN_CONFIDENCE:
            logger.debug(
                f"[{symbol}] ⛔ Gate 3 BLOCKED — ML confidence too low: "
                f"{ml['confidence']:.0%} < {CONFIG.ML_MIN_CONFIDENCE:.0%}"
            )
            return

        logger.debug(
            f"[{symbol}] ✅ Gate 3 PASSED — ML={ml_direction} Conf={ml['confidence']:.0%}"
        )

        # ── Gate 4 — Sentiment (Day Trader only) ──────────────────────────────
        if self.style == "scalper":
            logger.debug(f"[{symbol}] ⏭️  Gate 4 SKIPPED — Scalper mode")
        else:
            sent = self.sentiment.get_symbol_sentiment(symbol)
            logger.debug(
                f"[{symbol}] 📰 Sentiment: {sent['label']} | "
                f"Score: {sent['score']:+.3f} | Conf: {sent['confidence']:.0%}"
            )
            if signal.signal.value == "BUY" and sent["score"] < -0.1:
                logger.debug(f"[{symbol}] ⛔ Gate 4 BLOCKED — Sentiment bearish")
                return
            if signal.signal.value == "SELL" and sent["score"] > 0.1:
                logger.debug(f"[{symbol}] ⛔ Gate 4 BLOCKED — Sentiment bullish")
                return
            logger.debug(f"[{symbol}] ✅ Gate 4 PASSED — Sentiment {sent['label']}")

        # ── Gate 5 — Intermarket (Day Trader only) ────────────────────────────
        if self.style == "scalper":
            logger.debug(f"[{symbol}] ⏭️  Gate 5 SKIPPED — Scalper mode")
        else:
            inter = self.intermarket.get_intermarket_signal(symbol)
            logger.debug(f"[{symbol}] 🌐 Intermarket score: {inter['score']:+.3f}")
            if signal.signal.value == "BUY" and inter["score"] < -0.1:
                logger.debug(f"[{symbol}] ⛔ Gate 5 BLOCKED — Intermarket bearish")
                return
            if signal.signal.value == "SELL" and inter["score"] > 0.1:
                logger.debug(f"[{symbol}] ⛔ Gate 5 BLOCKED — Intermarket bullish")
                return
            logger.debug(f"[{symbol}] ✅ Gate 5 PASSED — Intermarket {inter['score']:+.3f}")

        # ── Gate 6 — COT (Day Trader only) ────────────────────────────────────
        if self.style == "scalper":
            logger.debug(f"[{symbol}] ⏭️  Gate 6 SKIPPED — Scalper mode")
        else:
            cot = self.cot.get_cot_signal(symbol)
            logger.debug(f"[{symbol}] 📊 COT bias: {cot['bias']}")
            if signal.signal.value == "BUY" and cot["bias"] == "Bearish":
                logger.debug(f"[{symbol}] ⛔ Gate 6 BLOCKED — COT Bearish vs BUY")
                return
            if signal.signal.value == "SELL" and cot["bias"] == "Bullish":
                logger.debug(f"[{symbol}] ⛔ Gate 6 BLOCKED — COT Bullish vs SELL")
                return
            logger.debug(f"[{symbol}] ✅ Gate 6 PASSED — COT bias={cot['bias']}")

        # ── All gates passed — fire alert ─────────────────────────────────────
        logger.info(
            f"🚀 ALL GATES PASSED — firing {signal.signal.value} alert on {symbol}"
        )

        direction = signal.signal.value
        common = dict(
            symbol=symbol, entry=signal.entry, sl=signal.sl, tp=signal.tp,
            confidence=ml["confidence"], reasons=signal.reasons,
            style=self.style, atr=signal.atr,
        )
        if direction == "BUY":
            self.alerts.buy_signal(**common)
        else:
            self.alerts.sell_signal(**common)

        self.dashboard.log_signal(
            symbol=symbol, direction=direction,
            entry=signal.entry, sl=signal.sl, tp=signal.tp,
            confidence=ml["confidence"],
        )

        # ── Mode 1 — signal only ──────────────────────────────────────────────
        if self.mode == "1":
            return

        # ── Mode 2 — semi: ask user ───────────────────────────────────────────
        if self.mode == "2":
            print(
                f"\n  ⚡ {direction} {symbol} @ {signal.entry:.5f}"
                f"  SL {signal.sl:.5f}  TP {signal.tp:.5f}"
            )
            if input("  Execute trade? (Y/N): ").strip().lower() != "y":
                return

        # ── Modes 2 & 3 — place order ─────────────────────────────────────────
        spec = self.risk_mgr.calculate_position(
            symbol    = symbol,
            direction = direction,
            entry     = signal.entry,
            sl        = signal.sl,
            tp        = signal.tp,
            win_rate  = ml["confidence"],
        )
        if spec is None:
            logger.warning(f"[{symbol}] ⚠️ Position sizing rejected — skipping trade")
            return

        result = self.executor.send_market_order(spec)
        if result:
            self.alerts.trade_opened(
                symbol    = symbol,
                direction = direction,
                ticket    = result["ticket"],
                entry     = result["price"],
                sl        = spec.sl,
                tp        = spec.tp,
                volume    = spec.volume,
                risk      = spec.risk_usd,
                style     = self.style,
                mode      = self.MODE_NAMES[self.mode],
            )
            logger.info(
                f"✅ Order placed: {direction} {symbol} "
                f"lot={spec.volume:.2f} risk=€{spec.risk_usd:.2f}"
            )
        else:
            logger.warning(f"❌ Order failed: {symbol} {direction}")

    # ── position monitoring ───────────────────────────────────────────────────
    def _monitor_positions(self) -> None:
        positions = mt5.positions_get(magic=CONFIG.MAGIC_NUMBER)
        if not positions:
            return

        for pos in positions:
            symbol    = pos.symbol
            ticket    = pos.ticket
            direction = "BUY" if pos.type == 0 else "SELL"
            entry     = pos.price_open
            pnl       = pos.profit
            sl        = pos.sl
            tp        = pos.tp

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(f"No tick for {symbol} — skipping {ticket}")
                continue

            current  = tick.bid if direction == "BUY" else tick.ask
            sym_info = mt5.symbol_info(symbol)
            point    = sym_info.point if sym_info else 0.00001

            pips_to_sl = abs(current - sl) / point / 10 if sl else 999

            # ── Danger check ──────────────────────────────────────────────────
            danger_reasons: list = []
            if pips_to_sl <= 5:
                danger_reasons.append(f"SL very close ({pips_to_sl:.1f} pips)")
            if pnl < -CONFIG.MAX_LOSS_PER_TRADE:
                danger_reasons.append(f"Max loss breached (€{pnl:.2f})")

            if danger_reasons:
                self.alerts.danger_exit(
                    symbol=symbol, direction=direction, ticket=ticket,
                    entry=entry, current=current, pnl=pnl, sl=sl,
                    reasons=danger_reasons,
                )
                if self.mode == "3":
                    result = self.executor.close_position(ticket)
                    if result:
                        pips = (current - entry) / point / 10
                        if direction == "SELL":
                            pips = -pips
                        self.alerts.trade_closed(
                            symbol=symbol, direction=direction, ticket=ticket,
                            entry=entry, close=current, pnl=pnl,
                            pips=round(pips, 1), reason="Danger exit",
                        )
                        self.dashboard.log_trade(
                            symbol=symbol, direction=direction,
                            pnl=pnl, ticket=ticket,
                        )
                        self.risk_mgr.record_trade_result(pnl)

            # ── TP proximity alert ────────────────────────────────────────────
            if tp:
                pips_to_tp = abs(current - tp) / point / 10
                if pips_to_tp <= 3:
                    self.alerts.potential_exit(
                        symbol=symbol, direction=direction, ticket=ticket,
                        entry=entry, current=current, pnl=pnl, tp=tp,
                        reasons=[f"Price within {pips_to_tp:.1f} pips of TP"],
                    )

    # ── trailing stops ────────────────────────────────────────────────────────
    def _update_trailing_stops(self) -> None:
        if self.mode not in ("2", "3"):
            return
        positions = mt5.positions_get(magic=CONFIG.MAGIC_NUMBER)
        if not positions:
            return

        trail_points = int(getattr(CONFIG, "TRAILING_STOP_PIPS", 15) * 10)
        for pos in positions:
            try:
                self.executor.modify_trailing_stop(
                    ticket       = pos.ticket,
                    symbol       = pos.symbol,
                    trail_points = trail_points,
                )
            except Exception as e:
                logger.debug(f"Trailing stop error for {pos.ticket}: {e}")

    # ── end-of-day closure ────────────────────────────────────────────────────
    def _close_all_day_trades(self) -> None:
        if self.style != "scalper":
            return
        positions = mt5.positions_get(magic=CONFIG.MAGIC_NUMBER)
        if not positions:
            return
        logger.info(f"🌙 EOD: closing {len(positions)} position(s)…")
        for pos in positions:
            symbol    = pos.symbol
            direction = "BUY" if pos.type == 0 else "SELL"
            pnl       = pos.profit
            ticket    = pos.ticket

            tick     = mt5.symbol_info_tick(symbol)
            close_px = (
                (tick.bid if direction == "BUY" else tick.ask)
                if tick else pos.price_current
            )

            result = self.executor.close_position(ticket)
            if result:
                sym_info = mt5.symbol_info(symbol)
                point    = sym_info.point if sym_info else 0.00001
                pips     = (close_px - pos.price_open) / point / 10
                if direction == "SELL":
                    pips = -pips
                self.alerts.trade_closed(
                    symbol=symbol, direction=direction, ticket=ticket,
                    entry=pos.price_open, close=close_px, pnl=pnl,
                    pips=round(pips, 1), reason="EOD close",
                )
                self.dashboard.log_trade(
                    symbol=symbol, direction=direction,
                    pnl=pnl, ticket=ticket,
                )
                self.risk_mgr.record_trade_result(pnl)
                logger.info(f"  ✅ Closed {symbol} ticket={ticket} pnl=€{pnl:.2f}")
            else:
                logger.warning(f"  ❌ Could not close {symbol} ticket={ticket}")

    # ── daily summary ─────────────────────────────────────────────────────────
    def _daily_summary(self) -> None:
        stats = self.dashboard.get_today_stats()

        logger.info("=" * 50)
        logger.info("📊  DAILY SUMMARY")
        logger.info(f"   Signals   : {stats.get('signals',      0)}")
        logger.info(f"   Trades    : {stats.get('trades',       0)}")
        logger.info(f"   Winning   : {stats.get('winners',      0)}")
        logger.info(f"   Win rate  : {stats.get('win_rate',     0.0):.1f}%")
        logger.info(f"   Net P&L   : €{stats.get('net_pnl',    0.0):+.2f}")
        logger.info(f"   Best      : €{stats.get('best_trade',  0.0):+.2f}")
        logger.info(f"   Worst     : €{stats.get('worst_trade', 0.0):+.2f}")
        logger.info("=" * 50)

        self.alerts.daily_summary(
            trades       = stats.get("trades",        0),
            winners      = stats.get("winners",       0),
            losers       = stats.get("losers",        0),
            net_pnl      = stats.get("net_pnl",      0.0),
            win_rate     = stats.get("win_rate",     0.0),
            signals      = stats.get("signals",       0),
            best_trade   = stats.get("best_trade",   0.0),
            worst_trade  = stats.get("worst_trade",  0.0),
            gross_profit = stats.get("gross_profit", 0.0),
            gross_loss   = stats.get("gross_loss",   0.0),
        )
        try:
            self.dashboard.export_report()
        except Exception as e:
            logger.warning(f"Report export failed: {e}")

    # ── keyboard listener ─────────────────────────────────────────────────────
    def _start_keyboard_listener(self) -> None:
        def _listen():
            while True:
                try:
                    key = input().strip().upper()
                except EOFError:
                    break
                if   key == "Q": self.stop()
                elif key == "P":
                    self._paused = not self._paused
                    print(f"\n  {'⏸  Paused' if self._paused else '▶  Resumed'}\n")
                elif key == "S": self.alerts.sound.toggle()
                elif key == "T": self.alerts.telegram.toggle()
                elif key == "M": self._show_runtime_menu()

        threading.Thread(target=_listen, daemon=True).start()

    def _show_runtime_menu(self) -> None:
        print("\n" + "─" * 40)
        print("  🎛  RUNTIME MENU")
        print("─" * 40)
        print(f"  Mode      : {self.mode} — {self.MODE_NAMES[self.mode]}")
        print(f"  Style     : {self.style.title()}")
        print(f"  Watchlist : {', '.join(CONFIG.WATCHLIST)}")
        print(f"  Sound     : {'ON' if self.alerts.sound.enabled else 'OFF'}")
        print(f"  TG        : {'ON' if self.alerts.telegram.enabled else 'OFF'}")
        print("─" * 40)
        print("  [1/2/3] Change mode   [S] Toggle sound")
        print("  [T]     Toggle TG     [P] Pause/Resume")
        print("  [Q]     Quit")
        print("─" * 40 + "\n")

    # ── shutdown ──────────────────────────────────────────────────────────────
    def stop(self) -> None:
        logger.info("\n🛑 Shutting down GODBOT…")
        self._running = False
        self.alerts.system_offline("Manual shutdown")
        try:
            self.dashboard.export_report()
        except Exception:
            pass
        self.connector.disconnect()
        print("\n  👋 GODBOT stopped cleanly.\n")
        sys.exit(0)


# ────────────────────────────────────────────────────────────────────────────
#  Entry point
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    menu    = StartupMenu()
    profile = menu.run()
    system  = ForexSystem(profile)
    system.start()
