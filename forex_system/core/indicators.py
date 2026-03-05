# core/indicators.py
import os
import tempfile

# Portable Numba cache — works on any OS / user account
os.environ["NUMBA_CACHE_DIR"] = os.path.join(
    tempfile.gettempdir(), ".numba_cache"
)

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Optional
from config.settings import CONFIG
from monitoring.logger import get_logger

logger = get_logger("Indicators")

# Minimum bars required after dropna — below this the DataFrame is
# considered too short to be useful and compute_all() returns None.
MIN_BARS_AFTER_DROPNA = 50


def _col(df_or_result, prefix: str, fallback=None):
    """Find first column whose name starts with *prefix* (case-insensitive).

    Returns the column Series if found, a fallback Series of NaN otherwise.
    Logs a warning on miss so missing indicators are visible in the log.
    """
    cols = [c for c in df_or_result.columns if c.upper().startswith(prefix.upper())]
    if cols:
        return df_or_result[cols[0]]
    logger.warning(
        f"⚠️ Column prefix '{prefix}' not found. "
        f"Available: {list(df_or_result.columns)}"
    )
    if fallback is not None:
        return fallback
    return pd.Series([np.nan] * len(df_or_result), index=df_or_result.index)


class IndicatorEngine:
    """
    Computes a full suite of technical indicators and returns an enriched
    DataFrame.

    All pandas_ta column lookups use prefix matching so they work across
    library versions.  All OHLCV column access uses lowercase names
    (open, high, low, close, volume) consistent with MT5Connector and
    DataHandler output.
    """

    def __init__(self, config=CONFIG):
        self.cfg = config

    def compute_all(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Master method — adds all indicator columns and drops NaN rows."""
        if df is None or df.empty:
            logger.warning("Empty DataFrame passed to IndicatorEngine")
            return None

        # FIX: normalise column names to lowercase so this engine works
        # regardless of whether the caller used connector.get_ohlcv() (now
        # lowercase) or an older code path that used Title Case.
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Verify required base columns are present after normalisation
        required = {"open", "high", "low", "close"}
        missing  = required - set(df.columns)
        if missing:
            logger.error(f"IndicatorEngine: missing required columns {missing}")
            return None

        df = self._trend_indicators(df)
        df = self._momentum_indicators(df)
        df = self._volatility_indicators(df)
        df = self._volume_indicators(df)
        df = self._market_structure(df)

        df.dropna(inplace=True)

        # FIX: guard against an empty result after dropna — return None with
        # a warning so callers skip processing rather than operating on an
        # empty DataFrame silently.
        if len(df) < MIN_BARS_AFTER_DROPNA:
            logger.warning(
                f"IndicatorEngine: only {len(df)} rows remain after dropna "
                f"(minimum {MIN_BARS_AFTER_DROPNA}) — returning None"
            )
            return None

        return df

    # ── Trend ─────────────────────────────────────────────────────────────────
    def _trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # FIX: all column references changed from Title Case to lowercase

        # EMAs
        df["ema_fast"]  = ta.ema(df["close"], length=self.cfg.EMA_FAST)
        df["ema_slow"]  = ta.ema(df["close"], length=self.cfg.EMA_SLOW)
        df["ema_trend"] = ta.ema(df["close"], length=self.cfg.EMA_TREND)

        # MACD
        macd = ta.macd(
            df["close"],
            fast   = self.cfg.MACD_FAST,
            slow   = self.cfg.MACD_SLOW,
            signal = self.cfg.MACD_SIGNAL,
        )
        if macd is not None:
            df["macd"]        = _col(macd, "MACD_")
            df["macd_signal"] = _col(macd, "MACDs_")
            df["macd_hist"]   = _col(macd, "MACDh_")
        else:
            df["macd"] = df["macd_signal"] = df["macd_hist"] = np.nan

        # ADX
        adx = ta.adx(df["high"], df["low"], df["close"])
        if adx is not None:
            df["adx"]    = _col(adx, "ADX_")
            df["di_pos"] = _col(adx, "DMP_")
            df["di_neg"] = _col(adx, "DMN_")
        else:
            df["adx"] = df["di_pos"] = df["di_neg"] = np.nan

        # Ichimoku
        try:
            ich    = ta.ichimoku(df["high"], df["low"], df["close"])
            ich_df = ich[0] if isinstance(ich, tuple) else ich
            if ich_df is not None:
                df["tenkan"]   = _col(ich_df, "ITS_")
                df["kijun"]    = _col(ich_df, "IKS_")
                df["senkou_a"] = _col(ich_df, "ISA_")
                df["senkou_b"] = _col(ich_df, "ISB_")
            else:
                df["tenkan"] = df["kijun"] = \
                df["senkou_a"] = df["senkou_b"] = np.nan
        except Exception as e:
            logger.warning(f"Ichimoku error: {e}")
            df["tenkan"] = df["kijun"] = \
            df["senkou_a"] = df["senkou_b"] = np.nan

        return df

    # ── Momentum ──────────────────────────────────────────────────────────────
    def _momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # FIX: all column references changed from Title Case to lowercase

        # RSI
        df["rsi"] = ta.rsi(df["close"], length=self.cfg.RSI_PERIOD)

        # Stochastic
        stoch = ta.stoch(df["high"], df["low"], df["close"])
        if stoch is not None:
            df["stoch_k"] = _col(stoch, "STOCHk_")
            df["stoch_d"] = _col(stoch, "STOCHd_")
        else:
            df["stoch_k"] = df["stoch_d"] = np.nan

        # CCI
        df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)

        # Williams %R
        df["willr"] = ta.willr(df["high"], df["low"], df["close"])

        return df

    # ── Volatility ────────────────────────────────────────────────────────────
    def _volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # FIX: all column references changed from Title Case to lowercase

        # Bollinger Bands
        bb = ta.bbands(
            df["close"],
            length = self.cfg.BB_PERIOD,
            std    = self.cfg.BB_STD,
        )
        if bb is not None:
            df["bb_upper"] = _col(bb, "BBU_")
            df["bb_mid"]   = _col(bb, "BBM_")
            df["bb_lower"] = _col(bb, "BBL_")
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        else:
            df["bb_upper"] = df["bb_mid"] = \
            df["bb_lower"] = df["bb_width"] = np.nan

        # ATR
        df["atr"] = ta.atr(
            df["high"], df["low"], df["close"],
            length=self.cfg.ATR_PERIOD,
        )

        # Keltner Channel
        try:
            kc = ta.kc(df["high"], df["low"], df["close"])
            if kc is not None:
                df["kc_upper"] = _col(kc, "KCU")
                df["kc_lower"] = _col(kc, "KCL")
            else:
                df["kc_upper"] = df["kc_lower"] = np.nan
        except Exception as e:
            logger.warning(f"Keltner Channel error: {e}")
            df["kc_upper"] = df["kc_lower"] = np.nan

        return df

    # ── Volume ────────────────────────────────────────────────────────────────
    def _volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # FIX: all column references changed from Title Case to lowercase
        # volume column is now guaranteed lowercase after compute_all normalises

        vol = df.get("volume")   # returns None if column absent

        try:
            if vol is not None:
                df["obv"] = ta.obv(df["close"], vol)
            else:
                df["obv"] = np.nan
        except Exception:
            df["obv"] = np.nan

        try:
            if vol is not None:
                df["cmf"] = ta.cmf(df["high"], df["low"], df["close"], vol)
            else:
                df["cmf"] = np.nan
        except Exception:
            df["cmf"] = np.nan

        try:
            if vol is not None:
                df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], vol)
            else:
                df["vwap"] = np.nan
        except Exception:
            df["vwap"] = np.nan

        return df

    # ── Market Structure ──────────────────────────────────────────────────────
    def _market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        # FIX: all column references changed from Title Case to lowercase

        # FIX: replace ATR with a zero-safe denominator to prevent inf / NaN
        # propagation when ATR is zero during the warmup period.
        atr_safe = df["atr"].replace(0, np.nan)

        df["price_vs_ema_fast"] = (df["close"] - df["ema_fast"]) / atr_safe
        df["price_vs_ema_slow"] = (df["close"] - df["ema_slow"]) / atr_safe
        df["ema_spread"]        = (df["ema_fast"] - df["ema_slow"]) / atr_safe

        # Candle shape
        df["body"]       = df["close"] - df["open"]
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["is_bullish"] = (df["body"] > 0).astype(int)

        # Squeeze — BB inside KC = volatility compression
        df["squeeze"] = (
            (df["bb_lower"] > df["kc_lower"]) &
            (df["bb_upper"] < df["kc_upper"])
        ).astype(int)

        return df
