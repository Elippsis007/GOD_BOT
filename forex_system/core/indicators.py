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

    FIX — Ichimoku lookahead bias eliminated:
      pandas_ta.ichimoku() shifts Senkou Span A/B 26 bars forward by
      default (standard charting convention).  This means bar[t] contains
      cloud values computed from bars t+1..t+26 — pure future leakage.
      Fixed by passing lookahead=False so cloud values are aligned to the
      bar at which they were computed, not projected forward.

    FIX — Profile-aware indicator periods:
      _resolve_periods() reads the active scalper profile at compute_all()
      time so switching M1 ↔ M5 at the startup menu automatically uses
      the correct EMA / RSI / ATR / MACD / BB / ADX periods without
      needing to reinstantiate IndicatorEngine.
    """

    def __init__(self, config=CONFIG):
        self.cfg = config

    # ── Period resolver — reads active profile at compute time ───────────────
    def _resolve_periods(self) -> dict:
        """
        Return the correct indicator periods for the currently active
        trading profile.

        For scalper mode reads CONFIG.get_scalper_profile() which returns
        either the M1 or M5 block depending on CONFIG.SCALPER_TF_SELECTED.
        Falls back to CONFIG flat attributes (day-trader / defaults) if
        the scalper profile is unavailable.

        Returns a flat dict with keys:
            ema_fast, ema_slow, ema_trend,
            rsi_period, macd_fast, macd_slow, macd_signal,
            bb_period, bb_std, atr_period, adx_period
        """
        try:
            p = self.cfg.get_scalper_profile()
            return {
                "ema_fast":    p.get("ema_fast",    self.cfg.EMA_FAST),
                "ema_slow":    p.get("ema_slow",    self.cfg.EMA_SLOW),
                "ema_trend":   p.get("ema_trend",   self.cfg.EMA_TREND),
                "rsi_period":  p.get("rsi_period",  self.cfg.RSI_PERIOD),
                "macd_fast":   p.get("macd_fast",   self.cfg.MACD_FAST),
                "macd_slow":   p.get("macd_slow",   self.cfg.MACD_SLOW),
                "macd_signal": p.get("macd_signal", self.cfg.MACD_SIGNAL),
                "bb_period":   p.get("bb_period",   self.cfg.BB_PERIOD),
                "bb_std":      p.get("bb_std",      self.cfg.BB_STD),
                "atr_period":  p.get("atr_period",  self.cfg.ATR_PERIOD),
                "adx_period":  p.get("adx_period",  self.cfg.ADX_PERIOD
                               if hasattr(self.cfg, "ADX_PERIOD") else 10),
            }
        except Exception:
            # Fallback — use flat CONFIG attributes (day-trader or defaults)
            return {
                "ema_fast":    self.cfg.EMA_FAST,
                "ema_slow":    self.cfg.EMA_SLOW,
                "ema_trend":   self.cfg.EMA_TREND,
                "rsi_period":  self.cfg.RSI_PERIOD,
                "macd_fast":   self.cfg.MACD_FAST,
                "macd_slow":   self.cfg.MACD_SLOW,
                "macd_signal": self.cfg.MACD_SIGNAL,
                "bb_period":   self.cfg.BB_PERIOD,
                "bb_std":      self.cfg.BB_STD,
                "atr_period":  self.cfg.ATR_PERIOD,
                "adx_period":  getattr(self.cfg, "ADX_PERIOD", 10),
            }

    def compute_all(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Master method — adds all indicator columns and drops NaN rows."""
        if df is None or df.empty:
            logger.warning("Empty DataFrame passed to IndicatorEngine")
            return None

        df = df.copy()
        df.columns = df.columns.str.lower()

        required = {"open", "high", "low", "close"}
        missing  = required - set(df.columns)
        if missing:
            logger.error(f"IndicatorEngine: missing required columns {missing}")
            return None

        # Resolve indicator periods from active profile at compute time
        periods = self._resolve_periods()

        df = self._trend_indicators(df, periods)
        df = self._momentum_indicators(df, periods)
        df = self._volatility_indicators(df, periods)
        df = self._volume_indicators(df)
        df = self._market_structure(df)

        df.dropna(inplace=True)

        if len(df) < MIN_BARS_AFTER_DROPNA:
            logger.warning(
                f"IndicatorEngine: only {len(df)} rows remain after dropna "
                f"(minimum {MIN_BARS_AFTER_DROPNA}) — returning None"
            )
            return None

        return df

    # ── Trend ─────────────────────────────────────────────────────────────────
    def _trend_indicators(self, df: pd.DataFrame, p: dict) -> pd.DataFrame:
        # EMAs — periods from active profile
        df["ema_fast"]  = ta.ema(df["close"], length=p["ema_fast"])
        df["ema_slow"]  = ta.ema(df["close"], length=p["ema_slow"])
        df["ema_trend"] = ta.ema(df["close"], length=p["ema_trend"])

        # MACD — periods from active profile
        macd = ta.macd(
            df["close"],
            fast   = p["macd_fast"],
            slow   = p["macd_slow"],
            signal = p["macd_signal"],
        )
        if macd is not None:
            df["macd"]        = _col(macd, "MACD_")
            df["macd_signal"] = _col(macd, "MACDs_")
            df["macd_hist"]   = _col(macd, "MACDh_")
        else:
            df["macd"] = df["macd_signal"] = df["macd_hist"] = np.nan

        # ADX — period from active profile
        adx = ta.adx(df["high"], df["low"], df["close"], length=p["adx_period"])
        if adx is not None:
            df["adx"]    = _col(adx, "ADX_")
            df["di_pos"] = _col(adx, "DMP_")
            df["di_neg"] = _col(adx, "DMN_")
        else:
            df["adx"] = df["di_pos"] = df["di_neg"] = np.nan

        # ── Ichimoku — FIX: lookahead=False eliminates Senkou Span leakage ──
        #
        # DEFAULT behaviour (lookahead=True / not set):
        #   pandas_ta shifts Senkou Span A and B 26 bars FORWARD so the
        #   cloud appears projected ahead on a chart.  At bar[t] this means
        #   senkou_a[t] = value computed from bars up to t+26 — future data.
        #   This leaks into both the ML training features (in_cloud,
        #   above_cloud) and the signal engine cloud checks.
        #
        # FIX (lookahead=False):
        #   Cloud values are aligned to the bar at which they were computed.
        #   senkou_a[t] = average of tenkan[t] and kijun[t] at that bar.
        #   No future bars are referenced — fully causal.
        #
        # Tenkan-sen and Kijun-sen are NOT affected by lookahead — they are
        # always computed from bars up to and including t.
        try:
            ich    = ta.ichimoku(
                df["high"], df["low"], df["close"],
                lookahead=False,   # ← THE FIX
            )
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
    def _momentum_indicators(self, df: pd.DataFrame, p: dict) -> pd.DataFrame:
        # RSI — period from active profile
        df["rsi"] = ta.rsi(df["close"], length=p["rsi_period"])

        # Stochastic — fixed periods (standard 14,3,3 — not profile-dependent)
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
    def _volatility_indicators(self, df: pd.DataFrame, p: dict) -> pd.DataFrame:
        # Bollinger Bands — periods from active profile
        bb = ta.bbands(
            df["close"],
            length = p["bb_period"],
            std    = p["bb_std"],
        )
        if bb is not None:
            df["bb_upper"] = _col(bb, "BBU_")
            df["bb_mid"]   = _col(bb, "BBM_")
            df["bb_lower"] = _col(bb, "BBL_")
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        else:
            df["bb_upper"] = df["bb_mid"] = \
            df["bb_lower"] = df["bb_width"] = np.nan

        # ATR — period from active profile
        df["atr"] = ta.atr(
            df["high"], df["low"], df["close"],
            length=p["atr_period"],
        )

        # Keltner Channel — fixed periods (used only for squeeze detection)
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
        vol = df.get("volume")

        try:
            df["obv"] = ta.obv(df["close"], vol) if vol is not None else np.nan
        except Exception:
            df["obv"] = np.nan

        try:
            df["cmf"] = ta.cmf(
                df["high"], df["low"], df["close"], vol
            ) if vol is not None else np.nan
        except Exception:
            df["cmf"] = np.nan

        try:
            df["vwap"] = ta.vwap(
                df["high"], df["low"], df["close"], vol
            ) if vol is not None else np.nan
        except Exception:
            df["vwap"] = np.nan

        return df

    # ── Market Structure ──────────────────────────────────────────────────────
    def _market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
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
