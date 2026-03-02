# core/indicators.py
import os
os.environ["NUMBA_CACHE_DIR"] = r"C:\Users\micha\.numba_cache"

import pandas as pd
import numpy as np
import pandas_ta as ta
from config.settings import CONFIG
from monitoring.logger import get_logger

logger = get_logger("Indicators")


def _col(df_or_result, prefix: str, fallback=None):
    """Find first column starting with a prefix — version-safe."""
    cols = [c for c in df_or_result.columns if c.upper().startswith(prefix.upper())]
    if cols:
        return df_or_result[cols[0]]
    logger.warning(f"⚠️ Column prefix '{prefix}' not found. Available: {list(df_or_result.columns)}")
    if fallback is not None:
        return fallback
    return pd.Series([np.nan] * len(df_or_result), index=df_or_result.index)


class IndicatorEngine:
    """
    Computes a full suite of technical indicators
    and returns an enriched DataFrame.
    All pandas_ta column lookups use prefix matching
    so they work across all library versions.
    """

    def __init__(self, config=CONFIG):
        self.cfg = config

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Master method — adds all indicator columns in-place."""
        if df is None or df.empty:
            logger.warning("Empty DataFrame passed to indicator engine")
            return df

        df = df.copy()
        df = self._trend_indicators(df)
        df = self._momentum_indicators(df)
        df = self._volatility_indicators(df)
        df = self._volume_indicators(df)
        df = self._market_structure(df)
        df.dropna(inplace=True)
        return df

    # ── Trend ─────────────────────────────────────────────
    def _trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # EMAs
        df["ema_fast"]  = ta.ema(df["Close"], length=self.cfg.EMA_FAST)
        df["ema_slow"]  = ta.ema(df["Close"], length=self.cfg.EMA_SLOW)
        df["ema_trend"] = ta.ema(df["Close"], length=self.cfg.EMA_TREND)

        # MACD
        macd = ta.macd(
            df["Close"],
            fast=self.cfg.MACD_FAST,
            slow=self.cfg.MACD_SLOW,
            signal=self.cfg.MACD_SIGNAL
        )
        if macd is not None:
            df["macd"]        = _col(macd, "MACD_")
            df["macd_signal"] = _col(macd, "MACDs_")
            df["macd_hist"]   = _col(macd, "MACDh_")
        else:
            df["macd"] = df["macd_signal"] = df["macd_hist"] = np.nan

        # ADX
        adx = ta.adx(df["High"], df["Low"], df["Close"])
        if adx is not None:
            df["adx"]    = _col(adx, "ADX_")
            df["di_pos"] = _col(adx, "DMP_")
            df["di_neg"] = _col(adx, "DMN_")
        else:
            df["adx"] = df["di_pos"] = df["di_neg"] = np.nan

        # Ichimoku
        try:
            ich = ta.ichimoku(df["High"], df["Low"], df["Close"])
            ich_df = ich[0] if isinstance(ich, tuple) else ich
            if ich_df is not None:
                df["tenkan"]   = _col(ich_df, "ITS_")
                df["kijun"]    = _col(ich_df, "IKS_")
                df["senkou_a"] = _col(ich_df, "ISA_")
                df["senkou_b"] = _col(ich_df, "ISB_")
            else:
                df["tenkan"] = df["kijun"] = df["senkou_a"] = df["senkou_b"] = np.nan
        except Exception as e:
            logger.warning(f"Ichimoku error: {e}")
            df["tenkan"] = df["kijun"] = df["senkou_a"] = df["senkou_b"] = np.nan

        return df

    # ── Momentum ──────────────────────────────────────────
    def _momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # RSI
        df["rsi"] = ta.rsi(df["Close"], length=self.cfg.RSI_PERIOD)

        # Stochastic
        stoch = ta.stoch(df["High"], df["Low"], df["Close"])
        if stoch is not None:
            df["stoch_k"] = _col(stoch, "STOCHk_")
            df["stoch_d"] = _col(stoch, "STOCHd_")
        else:
            df["stoch_k"] = df["stoch_d"] = np.nan

        # CCI
        df["cci"] = ta.cci(df["High"], df["Low"], df["Close"], length=20)

        # Williams %R
        df["willr"] = ta.willr(df["High"], df["Low"], df["Close"])

        return df

    # ── Volatility ────────────────────────────────────────
    def _volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Bollinger Bands — use prefix matching, NOT hardcoded name
        bb = ta.bbands(
            df["Close"],
            length=self.cfg.BB_PERIOD,
            std=self.cfg.BB_STD
        )
        if bb is not None:
            df["bb_upper"] = _col(bb, "BBU_")
            df["bb_mid"]   = _col(bb, "BBM_")
            df["bb_lower"] = _col(bb, "BBL_")
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        else:
            df["bb_upper"] = df["bb_mid"] = df["bb_lower"] = df["bb_width"] = np.nan

        # ATR
        df["atr"] = ta.atr(
            df["High"], df["Low"], df["Close"],
            length=self.cfg.ATR_PERIOD
        )

        # Keltner Channel
        try:
            kc = ta.kc(df["High"], df["Low"], df["Close"])
            if kc is not None:
                df["kc_upper"] = _col(kc, "KCU")
                df["kc_lower"] = _col(kc, "KCL")
            else:
                df["kc_upper"] = df["kc_lower"] = np.nan
        except Exception as e:
            logger.warning(f"Keltner Channel error: {e}")
            df["kc_upper"] = df["kc_lower"] = np.nan

        return df

    # ── Volume ────────────────────────────────────────────
    def _volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df["obv"]  = ta.obv(df["Close"], df["Volume"])
        except Exception:
            df["obv"] = np.nan
        try:
            df["cmf"]  = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"])
        except Exception:
            df["cmf"] = np.nan
        try:
            df["vwap"] = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
        except Exception:
            df["vwap"] = np.nan
        return df

    # ── Market Structure ──────────────────────────────────
    def _market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        # Price relative to EMAs (normalised by ATR)
        df["price_vs_ema_fast"] = (df["Close"] - df["ema_fast"]) / df["atr"]
        df["price_vs_ema_slow"] = (df["Close"] - df["ema_slow"]) / df["atr"]
        df["ema_spread"]        = (df["ema_fast"] - df["ema_slow"]) / df["atr"]

        # Candle shape
        df["body"]       = df["Close"] - df["Open"]
        df["upper_wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
        df["lower_wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
        df["is_bullish"] = (df["body"] > 0).astype(int)

        # Squeeze (BB inside KC = compression)
        df["squeeze"] = (
            (df["bb_lower"] > df["kc_lower"]) &
            (df["bb_upper"] < df["kc_upper"])
        ).astype(int)

        return df