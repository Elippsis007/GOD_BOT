# =============================================================================
#  core/data_handler.py  –  GODBOT v3.0
# =============================================================================
#  Fixes / improvements applied in this revision:
#
#  A  [FIX] get_data() called mt5.copy_rates_from() directly instead of
#     going through MT5Connector.get_ohlcv(); this bypassed the singleton
#     connection guard, chunked fetch, reconnect logic, and timeframe
#     normaliser.  All data fetches now route through the connector.
#
#  B  [FIX] get_data() used datetime.now(pytz.utc) as the utc_from anchor
#     for copy_rates_from(), which returns bars STARTING at that timestamp
#     — i.e. zero or one bar (the current forming candle).  The correct
#     call for "most recent N bars" is copy_rates_from_pos(pos=0, count=N),
#     which is what MT5Connector.get_ohlcv() already does correctly.
#
#  C  [FIX] _cache_time stored datetime.now() (naïve local time) but
#     _is_cache_valid() also used datetime.now() (naïve local time) —
#     consistent but wrong when the host clock jumps (DST, NTP).  Both
#     now use time.monotonic() which is immune to wall-clock adjustments.
#
#  D  [FIX] get_tick_data() called mt5.copy_ticks_from() directly,
#     bypassing the connector.  Now delegates to
#     MT5Connector.get_latest_tick() for single-tick spread queries and
#     MT5Connector.get_ticks() for bulk tick frames, keeping all MT5
#     surface area inside MT5Connector.
#
#  E  [FIX] get_current_spread() called mt5.symbol_info_tick() and
#     mt5.symbol_info() directly and divided by point/10 — which is
#     wrong for JPY pairs (2-digit) and metals.  Now delegates to
#     MT5Connector.get_latest_tick() which already returns a correctly
#     computed spread_pips value.
#
#  F  [FIX] _build_dataframe() renamed tick_volume → Volume but did not
#     handle the real_volume fallback when tick_volume is absent.  It
#     also silently dropped the Spread column (present in MT5 rate arrays)
#     without using it.  The column-rename logic now mirrors the robust
#     resolution in MT5Connector._to_dataframe(), and the DataFrame is
#     already clean when it arrives from the connector so _build_dataframe
#     is now a lightweight column-normaliser rather than a raw builder.
#
#  G  [FIX] _add_session_info() used overlapping UTC hour windows that
#     caused session mis-tagging.  Assignments executed in order, so
#     "london" hours 7-16 were immediately overwritten by "newyork"
#     hours 13-22 for bars in the 13-16 overlap window, and "overlap"
#     was assigned last — but only to 12-16, missing the 12:00 bar.
#     Rewritten as a single pd.cut() call so windows are mutually
#     exclusive and correctly prioritised: overlap (12-16) > newyork
#     (16-22) > london (7-12) > asian (0-7) > off.
#
#  H  [FIX] get_multi_timeframe() used raw mt5.TIMEFRAME_D1 as the HTF
#     key value, which is a raw integer constant (16408).  After Fix A,
#     the call goes through get_data() → MT5Connector.get_ohlcv() which
#     accepts string or int, so "D1" is passed as the string form for
#     clarity and consistency.
#
#  I  [FIX] is_market_open() called mt5.symbol_info_tick() directly.
#     Now delegates to MT5Connector.get_latest_tick() and uses the
#     returned "time" key rather than re-parsing the raw tick timestamp.
#
#  J  [FIX] get_all_symbol_status() called mt5.symbol_info_tick()
#     directly for Bid/Ask display.  Now delegates to
#     MT5Connector.get_latest_tick() and uses the returned ask/bid keys.
#
#  K  [NEW] _normalise_columns() extracted as a separate helper so that
#     data arriving from the connector (lowercase OHLCV) is consistently
#     renamed to the Title-Case column names (Open/High/Low/Close/Volume)
#     expected by IndicatorEngine and SignalEngine.  Previously each path
#     (get_data, get_historical_range, get_multi_timeframe) had slightly
#     different rename logic.
#
#  L  [NEW] get_data() now accepts a string timeframe ("M5", "H1") in
#     addition to MT5 integer constants, delegating normalisation to
#     MT5Connector._normalise_tf() via get_ohlcv().
#
#  M  [FIX] get_historical_range() used pytz.utc.localize() on a
#     datetime that might already be tz-aware, which raises
#     ValueError in pytz.  Now uses datetime.replace(tzinfo=timezone.utc)
#     only when tzinfo is None.
#
#  N  [NEW] Stale-connection recovery: if get_ohlcv() returns None due
#     to a dropped connection, get_data() calls connector.reconnect()
#     once and retries before returning None to the caller.
# =============================================================================

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from config.settings import CONFIG
from core.mt5_connector import MT5Connector
from monitoring.logger import get_logger

logger = get_logger("DataHandler")

# Session boundary hours (UTC).  Used by _add_session_info.
# Windows are mutually exclusive; overlap takes highest priority.
_SESSION_BINS   = [0,  7, 12, 16, 22, 24]
_SESSION_LABELS = ["asian", "london", "overlap", "newyork", "off"]


class DataHandler:
    """
    Manages all data operations:
    - Multi-timeframe data fetching  (via MT5Connector — never raw mt5 calls)
    - Data cleaning and validation
    - Caching with monotonic-clock expiry
    - Tick data processing
    - Session tagging
    """

    def __init__(self, config=CONFIG) -> None:
        self.cfg = config
        self._connector = MT5Connector()   # singleton

        # [C] Monotonic timestamps avoid DST / NTP wall-clock jumps
        self._cache:       Dict[str, pd.DataFrame] = {}
        self._cache_mono:  Dict[str, float]        = {}
        self.CACHE_EXPIRY: float = 60.0            # seconds

    # =========================================================================
    # Primary data fetch
    # =========================================================================

    def get_data(
        self,
        symbol:    str,
        timeframe: Union[str, int],
        bars:      int = CONFIG.BARS_HISTORY,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Main entry point for OHLCV data.

        [A] All data fetches route through MT5Connector.get_ohlcv() so that
            the singleton connection guard, chunked fetch (Fix K in connector),
            reconnect logic, and timeframe normaliser are always applied.

        [B] No longer passes datetime.now() as utc_from — that caused
            copy_rates_from() to return only the current forming candle.
            MT5Connector.get_ohlcv(bars=N) uses copy_rates_from_pos(pos=0)
            which correctly returns the N most recent closed+forming bars.

        [L] Accepts string ("M5") or MT5 int constant — normalisation is
            delegated to the connector.

        [N] On the first None return, attempts one reconnect and retries.
        """
        cache_key = f"{symbol}_{timeframe}"

        if use_cache and self._is_cache_valid(cache_key):
            logger.debug("📦 Cache hit: %s", cache_key)
            return self._cache[cache_key]

        df = self._fetch_and_process(symbol, timeframe, bars)

        # [N] Stale-connection recovery: reconnect once and retry
        if df is None:
            logger.warning(
                "⚠️  get_data(%s, %s): first attempt returned None — "
                "attempting reconnect.", symbol, timeframe,
            )
            if self._connector.reconnect():
                df = self._fetch_and_process(symbol, timeframe, bars)

        if df is None:
            return None

        # [C] Store with monotonic timestamp
        self._cache[cache_key]      = df
        self._cache_mono[cache_key] = time.monotonic()

        logger.debug(
            "✅ %s | TF:%s | Bars:%d | Latest:%s",
            symbol, timeframe, len(df), df.index[-1],
        )
        return df

    def _fetch_and_process(
        self,
        symbol:    str,
        timeframe: Union[str, int],
        bars:      int,
    ) -> Optional[pd.DataFrame]:
        """Fetch → normalise columns → clean → session tag → validate."""
        try:
            df = self._connector.get_ohlcv(symbol, timeframe, bars=bars)
            if df is None or df.empty:
                logger.warning("⚠️  No data returned for %s %s.", symbol, timeframe)
                return None

            df = self._normalise_columns(df)    # [K] consistent Title-Case cols
            df = self._clean_data(df)
            df = self._add_session_info(df)

            if not self._validate_data(df, symbol):
                return None

            return df

        except Exception as exc:
            logger.error("get_data(%s, %s) failed: %s", symbol, timeframe, exc)
            return None

    # =========================================================================
    # Multi-timeframe fetch
    # =========================================================================

    def get_multi_timeframe(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetches primary, confirmation, and daily timeframes simultaneously.

        [H] HTF now passed as string "D1" instead of the raw mt5.TIMEFRAME_D1
            integer constant (16408), which is opaque and fragile.
        """
        result: Dict[str, pd.DataFrame] = {}
        tf_map = {
            "primary": self.cfg.PRIMARY_TF,
            "confirm": self.cfg.CONFIRM_TF,
            "htf":     "D1",               # [H]
        }
        for name, tf in tf_map.items():
            df = self.get_data(symbol, tf)
            if df is not None:
                result[name] = df
                logger.debug("📊 %s %s: %d bars", symbol, name, len(df))
            else:
                logger.warning("⚠️  %s %s: No data", symbol, name)
        return result

    # =========================================================================
    # Tick data
    # =========================================================================

    def get_tick_data(
        self,
        symbol: str,
        count:  int = 1_000,
    ) -> Optional[Dict]:
        """
        [D] Returns a dict with ask, bid, mid, spread_pips, spread_points,
        and time for the latest tick, sourced through MT5Connector so that
        the singleton connection guard is always active.

        For bulk tick frames (e.g. for VWAP calculation) use
        get_tick_frame() below.
        """
        tick = self._connector.get_latest_tick(symbol)
        if tick is None:
            logger.warning("⚠️  get_tick_data(%s): no tick returned.", symbol)
        return tick

    def get_tick_frame(
        self,
        symbol: str,
        count:  int = 1_000,
    ) -> Optional[pd.DataFrame]:
        """
        [D] Returns a DataFrame of recent ticks for spread analysis,
        VWAP construction, or micro-structure work.  Delegates entirely
        to MT5Connector.get_ticks().
        """
        df = self._connector.get_ticks(symbol, count=count)
        if df is None:
            logger.warning("⚠️  get_tick_frame(%s): no ticks returned.", symbol)
            return None

        # Add spread and mid columns if ask/bid are present
        if "ask" in df.columns and "bid" in df.columns:
            df["spread"] = df["ask"] - df["bid"]
            df["mid"]    = (df["ask"] + df["bid"]) / 2.0

        return df

    # =========================================================================
    # Spread
    # =========================================================================

    def get_current_spread(self, symbol: str) -> Optional[float]:
        """
        [E] Returns current spread in pips.

        Previously called mt5.symbol_info_tick() and mt5.symbol_info()
        directly and divided by point/10 — incorrect for JPY (2-digit)
        and metals.  MT5Connector.get_latest_tick() already handles the
        pip-size logic for all symbol types.
        """
        tick = self._connector.get_latest_tick(symbol)
        if tick is None:
            return None
        return round(tick["spread_pips"], 2)

    # =========================================================================
    # Historical range
    # =========================================================================

    def get_historical_range(
        self,
        symbol:    str,
        timeframe: Union[str, int],
        date_from: datetime,
        date_to:   datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Fetches OHLCV data between two specific dates — useful for
        backtesting and calendar-range analysis.

        [M] Uses datetime.replace(tzinfo=timezone.utc) instead of
            pytz.utc.localize() which raises ValueError on already-aware
            datetimes.
        """
        try:
            # [M] Safe tz-attachment
            if date_from.tzinfo is None:
                date_from = date_from.replace(tzinfo=timezone.utc)
            if date_to.tzinfo is None:
                date_to = date_to.replace(tzinfo=timezone.utc)

            bars_approx = self._estimate_bars(timeframe, date_from, date_to)
            df = self._connector.get_ohlcv(
                symbol, timeframe,
                bars=bars_approx,
                utc_from=date_from,
            )
            if df is None or df.empty:
                return None

            df = self._normalise_columns(df)   # [K]
            df = self._clean_data(df)

            # Trim to exact requested range
            df = df[(df.index >= pd.Timestamp(date_from)) &
                    (df.index <= pd.Timestamp(date_to))]
            return df if not df.empty else None

        except Exception as exc:
            logger.error(
                "get_historical_range(%s, %s) failed: %s", symbol, timeframe, exc
            )
            return None

    @staticmethod
    def _estimate_bars(
        timeframe: Union[str, int],
        date_from: datetime,
        date_to:   datetime,
    ) -> int:
        """
        Rough bar count estimate for a date range.  Used when requesting
        historical data so we ask for approximately the right number of bars.
        """
        seconds = (date_to - date_from).total_seconds()
        tf_seconds_map = {
            "M1": 60, "M2": 120, "M3": 180, "M4": 240, "M5": 300,
            "M10": 600, "M15": 900, "M30": 1_800,
            "H1": 3_600, "H4": 14_400, "D1": 86_400,
        }
        if isinstance(timeframe, str):
            bar_secs = tf_seconds_map.get(timeframe.upper(), 300)
        else:
            bar_secs = timeframe if timeframe > 86_400 else 300
        bars = max(int(seconds / bar_secs) + 10, 50)
        return min(bars, CONFIG.BARS_HISTORY)

    # =========================================================================
    # Market status
    # =========================================================================

    def is_market_open(self, symbol: str) -> bool:
        """
        [I] Checks if market is currently tradeable by examining the age
        of the most recent tick.  Delegates to MT5Connector.get_latest_tick()
        instead of calling mt5.symbol_info_tick() directly.
        """
        try:
            tick = self._connector.get_latest_tick(symbol)
            if tick is None:
                return False
            age = (datetime.now(timezone.utc) - tick["time"]).total_seconds()
            return age < 600
        except Exception:
            return False

    def get_all_symbol_status(self) -> pd.DataFrame:
        """
        [J] Returns market status for all configured symbols.
        Delegates to MT5Connector.get_latest_tick() instead of
        calling mt5.symbol_info_tick() directly.
        """
        rows: List[dict] = []
        for symbol in self.cfg.SYMBOLS:
            open_  = self.is_market_open(symbol)
            tick   = self._connector.get_latest_tick(symbol)
            spread = round(tick["spread_pips"], 2) if tick else None
            rows.append({
                "Symbol": symbol,
                "Status": "🟢 Open" if open_ else "🔴 Closed",
                "Bid":    tick["bid"] if tick else "N/A",
                "Ask":    tick["ask"] if tick else "N/A",
                "Spread": f"{spread} pips" if spread is not None else "N/A",
            })
        return pd.DataFrame(rows)

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [K] Rename lowercase connector output (open/high/low/close/volume)
        to the Title-Case column names (Open/High/Low/Close/Volume) expected
        by IndicatorEngine and SignalEngine.

        Also handles the broker Spread column and RealVolume column when
        present, and adds a Volume column of zeros if none is found.
        """
        rename_map = {
            "open":        "Open",
            "high":        "High",
            "low":         "Low",
            "close":       "Close",
            "volume":      "Volume",
            "spread":      "Spread",
            "real_volume": "RealVolume",
            "tick_volume": "Volume",      # fallback if volume not yet renamed
        }
        # Only rename columns that actually exist
        df = df.rename(columns={k: v for k, v in rename_map.items()
                                 if k in df.columns})

        # [F] Ensure Volume column exists
        if "Volume" not in df.columns:
            df["Volume"] = 0

        return df

    def _build_dataframe(self, rates) -> pd.DataFrame:
        """
        Legacy helper kept for callers that bypass get_data() (e.g.
        backtesting utilities that pass raw numpy arrays directly).
        For all live-trading paths use _normalise_columns() on the
        DataFrame already returned by MT5Connector.get_ohlcv().
        """
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        return self._normalise_columns(df)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes bad data and sorts by time.

        Validates OHLC integrity by detecting and DROPPING any bars where
        the broker has sent logically impossible prices, rather than
        silently overwriting them with max/min recalculations which would
        mask upstream data quality problems and feed corrupt candle data
        into the indicator engine.

        A bar is malformed when any of the following holds:
          High < max(Open, Close)  — the high cannot be below the body top
          Low  > min(Open, Close)  — the low cannot be above the body bottom
          High < Low               — impossible price relationship
        """
        if df is None or df.empty:
            return df

        price_cols = ["Open", "High", "Low", "Close"]

        # Remove duplicate timestamps — keep the most recent
        df = df[~df.index.duplicated(keep="last")]

        # Remove zero / negative prices — broker feed errors
        for col in price_cols:
            if col in df.columns:
                df = df[df[col] > 0]

        # Detect and drop malformed OHLC bars
        if all(c in df.columns for c in price_cols):
            invalid_high = df["High"] < df[["Open", "Close"]].max(axis=1)
            invalid_low  = df["Low"]  > df[["Open", "Close"]].min(axis=1)
            invalid_hl   = df["High"] < df["Low"]
            bad_bars     = invalid_high | invalid_low | invalid_hl

            if bad_bars.any():
                logger.warning(
                    "⚠️  Dropping %d malformed OHLC bar(s) — "
                    "broker feed integrity issue.", bad_bars.sum(),
                )
                df = df[~bad_bars]

        df.sort_index(inplace=True)
        return df

    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Ensures data meets minimum quality standards."""
        if df is None or df.empty:
            logger.warning("❌ %s: Empty DataFrame", symbol)
            return False

        if len(df) < 50:
            logger.warning("❌ %s: Not enough bars (%d)", symbol, len(df))
            return False

        price_cols = [c for c in ("Open", "High", "Low", "Close")
                      if c in df.columns]
        if not price_cols:
            logger.warning("❌ %s: No OHLC columns present.", symbol)
            return False

        null_pct = df[price_cols].isnull().mean().mean()
        if null_pct > 0.01:
            logger.warning(
                "❌ %s: Too many nulls (%.1f%%)", symbol, null_pct * 100
            )
            return False

        return True

    def _add_session_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [G] Tags each bar with its UTC-based trading session.

        The previous implementation used sequential .loc[] assignments
        with overlapping hour windows.  Because assignments execute in
        order, any bar in the 13-16 UTC window was first tagged "london",
        then immediately overwritten by "newyork", and the "overlap" label
        was never correctly applied to the 12:00 bar.

        Fix: use pd.cut() with mutually exclusive, correctly prioritised
        windows.  The bins cover 24 hours exhaustively.

        Session definitions (UTC):
          asian    00:00 – 07:00   Tokyo / Sydney
          london   07:00 – 12:00   London pre-overlap
          overlap  12:00 – 16:00   London / New York crossover (highest volatility)
          newyork  16:00 – 22:00   New York afternoon
          off      22:00 – 24:00   Low-liquidity close
        """
        if df.empty:
            return df

        hours = df.index.hour + df.index.minute / 60.0

        df["session"] = pd.cut(
            hours,
            bins=_SESSION_BINS,
            labels=_SESSION_LABELS,
            right=False,
            include_lowest=True,
        ).astype(str)

        return df

    def _is_cache_valid(self, key: str) -> bool:
        """
        [C] Uses time.monotonic() for expiry comparison so that wall-clock
        adjustments (DST changes, NTP corrections) do not invalidate or
        extend cache entries unexpectedly.

        The original used datetime.now() (naïve local time) for both
        storage and comparison — consistent but sensitive to clock jumps.
        """
        if key not in self._cache:
            return False
        age = time.monotonic() - self._cache_mono[key]
        return age < self.CACHE_EXPIRY

    def clear_cache(self) -> None:
        """Clears all cached DataFrames and their timestamps."""
        self._cache.clear()
        self._cache_mono.clear()
        logger.info("🧹 Data cache cleared")
