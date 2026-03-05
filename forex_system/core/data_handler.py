# OHLCV + tick data management
# core/data_handler.py
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import pytz
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from config.settings import CONFIG
from monitoring.logger import get_logger

logger = get_logger("DataHandler")


class DataHandler:
    """
    Manages all data operations:
    - Multi-timeframe data fetching
    - Data cleaning and validation
    - Caching to avoid redundant API calls
    - Tick data processing
    - Session filtering
    """

    def __init__(self, config=CONFIG):
        self.cfg              = config
        self._cache:      Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, datetime]     = {}
        self.CACHE_EXPIRY = 60  # seconds

    # ── Primary Data Fetch ────────────────────────────────
    def get_data(
        self,
        symbol:    str,
        timeframe: int,
        bars:      int = CONFIG.BARS_HISTORY,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Main entry point for OHLCV data.
        Returns clean, validated DataFrame.
        """
        cache_key = f"{symbol}_{timeframe}"

        if use_cache and self._is_cache_valid(cache_key):
            logger.debug(f"📦 Cache hit: {cache_key}")
            return self._cache[cache_key]

        try:
            utc_now = datetime.now(pytz.utc)
            rates   = mt5.copy_rates_from(symbol, timeframe, utc_now, bars)

            if rates is None or len(rates) == 0:
                logger.warning(f"⚠️ No data returned for {symbol}")
                return None

            df = self._build_dataframe(rates)
            df = self._clean_data(df)
            df = self._add_session_info(df)

            if not self._validate_data(df, symbol):
                return None

            self._cache[cache_key]      = df
            self._cache_time[cache_key] = datetime.now()

            logger.debug(
                f"✅ {symbol} | TF:{timeframe} | "
                f"Bars:{len(df)} | Latest:{df.index[-1]}"
            )
            return df

        except Exception as e:
            logger.error(f"get_data({symbol}) failed: {e}")
            return None

    # ── Multi Timeframe Data ──────────────────────────────
    def get_multi_timeframe(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetches primary, confirmation and daily timeframes
        for a symbol simultaneously.
        """
        result = {}
        tf_map = {
            "primary": self.cfg.PRIMARY_TF,
            "confirm": self.cfg.CONFIRM_TF,
            "htf":     mt5.TIMEFRAME_D1,
        }
        for name, tf in tf_map.items():
            df = self.get_data(symbol, tf)
            if df is not None:
                result[name] = df
                logger.debug(f"📊 {symbol} {name}: {len(df)} bars")
            else:
                logger.warning(f"⚠️ {symbol} {name}: No data")
        return result

    # ── Tick Data ─────────────────────────────────────────
    def get_tick_data(
        self,
        symbol: str,
        count:  int = 1000
    ) -> Optional[pd.DataFrame]:
        """Fetches granular tick data for spread analysis and precise entries."""
        try:
            utc_now = datetime.now(pytz.utc)
            ticks   = mt5.copy_ticks_from(
                symbol, utc_now, count, mt5.COPY_TICKS_ALL
            )
            if ticks is None or len(ticks) == 0:
                return None

            df = pd.DataFrame(ticks)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.set_index("time", inplace=True)
            df["spread"] = df["ask"] - df["bid"]
            df["mid"]    = (df["ask"] + df["bid"]) / 2
            return df

        except Exception as e:
            logger.error(f"get_tick_data({symbol}) failed: {e}")
            return None

    # ── Spread Analysis ───────────────────────────────────
    def get_current_spread(self, symbol: str) -> Optional[float]:
        """Returns current spread in pips."""
        try:
            tick     = mt5.symbol_info_tick(symbol)
            sym_info = mt5.symbol_info(symbol)
            if tick is None or sym_info is None:
                return None
            spread_pips = (tick.ask - tick.bid) / sym_info.point / 10
            return round(spread_pips, 2)
        except Exception as e:
            logger.error(f"get_spread({symbol}) failed: {e}")
            return None

    # ── Historical Range ──────────────────────────────────
    def get_historical_range(
        self,
        symbol:    str,
        timeframe: int,
        date_from: datetime,
        date_to:   datetime
    ) -> Optional[pd.DataFrame]:
        """Fetches data between two specific dates — useful for backtesting."""
        try:
            utc = pytz.utc
            if date_from.tzinfo is None:
                date_from = utc.localize(date_from)
            if date_to.tzinfo is None:
                date_to = utc.localize(date_to)

            rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
            if rates is None or len(rates) == 0:
                return None

            df = self._build_dataframe(rates)
            df = self._clean_data(df)
            return df

        except Exception as e:
            logger.error(f"get_historical_range({symbol}) failed: {e}")
            return None

    # ── Market Status ─────────────────────────────────────
    def is_market_open(self, symbol: str) -> bool:
        """Checks if market is currently tradeable."""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return False
            tick_time = datetime.fromtimestamp(tick.time, tz=pytz.utc)
            age = (datetime.now(pytz.utc) - tick_time).total_seconds()
            return age < 600
        except Exception:
            return False

    def get_all_symbol_status(self) -> pd.DataFrame:
        """Returns market status for all configured symbols."""
        rows = []
        for symbol in self.cfg.SYMBOLS:
            spread = self.get_current_spread(symbol)
            open_  = self.is_market_open(symbol)
            tick   = mt5.symbol_info_tick(symbol)
            rows.append({
                "Symbol": symbol,
                "Status": "🟢 Open" if open_ else "🔴 Closed",
                "Bid":    tick.bid  if tick else "N/A",
                "Ask":    tick.ask  if tick else "N/A",
                "Spread": f"{spread} pips" if spread else "N/A",
            })
        return pd.DataFrame(rows)

    # ── Internal Helpers ──────────────────────────────────
    def _build_dataframe(self, rates) -> pd.DataFrame:
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={
            "open":        "Open",
            "high":        "High",
            "low":         "Low",
            "close":       "Close",
            "tick_volume": "Volume",
            "spread":      "Spread",
            "real_volume": "RealVolume",
        }, inplace=True)
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes bad data and sorts by time.
        Validates OHLC integrity by logging and dropping any bars
        where the broker has sent logically impossible prices rather
        than silently overwriting them, which would feed corrupt
        candle data into the indicator engine.
        """
        # Remove duplicate timestamps — keep most recent
        df = df[~df.index.duplicated(keep="last")]

        # Remove zero / negative prices — these are broker feed errors
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if col in df.columns:
                df = df[df[col] > 0]

        # FIX: the original blindly recalculated High and Low from all
        # four OHLC columns using max/min, which silently overwrote whatever
        # the broker sent. This masked upstream data quality problems and
        # could produce candles with High < Open or Low > Close that
        # indicators would then compute against without any warning.
        #
        # The correct approach is to DETECT and DROP malformed bars rather
        # than silently alter them. A bar is malformed when:
        #   High < max(Open, Close)  — high can't be lower than body top
        #   Low  > min(Open, Close)  — low can't be higher than body bottom
        #   High < Low               — impossible price relationship
        #
        # Dropping these rows keeps the dataset honest. In practice fewer
        # than 0.1% of bars from a reliable broker will ever be affected.
        if all(c in df.columns for c in price_cols):
            invalid_high = df["High"] < df[["Open", "Close"]].max(axis=1)
            invalid_low  = df["Low"]  > df[["Open", "Close"]].min(axis=1)
            invalid_hl   = df["High"] < df["Low"]
            bad_bars     = invalid_high | invalid_low | invalid_hl

            if bad_bars.any():
                logger.warning(
                    f"⚠️ Dropping {bad_bars.sum()} malformed OHLC bar(s) — "
                    f"broker feed integrity issue"
                )
                df = df[~bad_bars]

        # Sort chronologically
        df.sort_index(inplace=True)
        return df

    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Ensures data meets minimum quality standards."""
        if df is None or df.empty:
            logger.warning(f"❌ {symbol}: Empty DataFrame")
            return False

        if len(df) < 50:
            logger.warning(f"❌ {symbol}: Not enough bars ({len(df)})")
            return False

        null_pct = df[["Open", "High", "Low", "Close"]].isnull().mean().mean()
        if null_pct > 0.01:
            logger.warning(f"❌ {symbol}: Too many nulls ({null_pct:.1%})")
            return False

        return True

    def _add_session_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tags each bar with its trading session."""
        hours = df.index.hour

        df["session"] = "off"
        df.loc[(hours >= 0)  & (hours < 8),  "session"] = "asian"
        df.loc[(hours >= 7)  & (hours < 16), "session"] = "london"
        df.loc[(hours >= 13) & (hours < 22), "session"] = "newyork"
        df.loc[(hours >= 12) & (hours < 16), "session"] = "overlap"
        return df

    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        # FIX: the original used .seconds which only returns the seconds
        # component of the timedelta (0-59). For a CACHE_EXPIRY of 60s
        # this happened to work but for any expiry > 59s it would silently
        # return stale data forever. total_seconds() is always correct.
        age = (datetime.now() - self._cache_time[key]).total_seconds()
        return age < self.CACHE_EXPIRY

    def clear_cache(self):
        self._cache.clear()
        self._cache_time.clear()
        logger.info("🧹 Data cache cleared")
