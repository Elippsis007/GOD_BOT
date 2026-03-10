# =============================================================================
# GODBOT v3.0 – core/indicators.py
# =============================================================================
#  Fixes / improvements applied in this revision:
#
#  A  dropna() uses subset=_REQUIRED_COLS only. _SCORED_COLS rows kept
#     even when NaN (stoch, cmf, vwap, squeeze). Stochastic and CMF/VWAP
#     moved from required to scored so warmup NaNs don't drop rows.
#
#  B  _col() accepts silent=False; optional lookups pass silent=True.
#
#  C  Volume validity uses majority test (>50% positive bars) not .any().
#     All volume-derived NaN columns set as pd.Series not np.nan scalar.
#
#  D  Squeeze only computed when both BB and KC are valid.
#
#  E  All exception handlers use _nan_series(df) not np.nan scalar.
#
#  F  compute_all() logs active TF + key periods + before/after row counts.
#
#  G  _resolve_periods() validates each key — None/zero/negative replaced
#     with CONFIG flat-attribute fallback.
#
#  H  replace([inf, -inf], np.nan) at end of _market_structure().
#
#  I  Stochastic K period from active scalper profile (M5→5, M1→3).
#     STOCH_BULL/BEAR_ZONE are scoring thresholds, NOT period parameters.
#
#  J  MIN_BARS_AFTER_DROPNA=60 for scan/train; MIN_BARS_FOR_PREDICT=30
#     for prediction. for_prediction=False parameter on compute_all().
#
#  K  [FIX] DataHandler returns Title-Case columns (Open/High/Low/Close).
#     compute_all() lowercases them with df.columns.str.lower() before
#     ANY processing so every indicator receives "close" not "Close".
#     Previously this was done but AFTER the missing-column guard, meaning
#     the guard would fail on "Open" not matching required input {"open"}.
#     Guard now runs AFTER the lowercase conversion.
#
#  L  [FIX] _resolve_periods() read p.get("stoch_k", 5) but the profile
#     dict key is "STOCH_K" (uppercase from get_scalper_profile()).
#     Now tries both cases so the correct period is always found.
#
#  M  [FIX] Ichimoku column prefix matching was brittle — pandas_ta ≥0.3.14
#     changed column names from "ITS_9" to "ISA_9_26" etc. across versions.
#     _col() already handles prefix matching so the prefixes themselves
#     must be checked. Corrected to the canonical pandas_ta prefixes and
#     added a version-agnostic fallback scan for any column containing
#     "tenkan", "kijun", "senkou" (case-insensitive).
#
#  N  [FIX] compute_all() lowercased all columns but _market_structure()
#     accesses df["open"] / df["close"] etc. directly. If the DataFrame
#     ever had mixed-case columns on entry, some accesses would silently
#     return NaN. The lowercase step is now a strict pre-condition enforced
#     at the top of compute_all() before any sub-method is called.
#
#  O  [NEW] HTF (higher-timeframe) confirmation columns added:
#     htf_ema_fast, htf_ema_slow, htf_ema_bull, htf_ema_bear.
#     compute_all() accepts an optional htf_df parameter. When provided,
#     the last HTF EMA values are forward-filled onto the primary df index
#     so signal_engine.evaluate() can gate trades on M15/H1 trend direction
#     without fetching a second DataFrame inside the signal engine itself.
#     This is the "single most important missing piece" from the audit.
#
#  P  [NEW] Candle-pattern recognition added: engulfing (bull/bear),
#     pin bar (bull/bear), inside bar, doji. These are boolean integer
#     columns (0/1) ready for use as ML features and signal engine gates.
#
#  Q  [NEW] CMF threshold documented in-code: the 0.05 threshold used by
#     signal_engine is noted here with the recommended 0.02 override for
#     tick-volume-based CMF (retail MT5 limitation noted in audit).
#
#  R  [FIX] VWAP from ta.vwap() resets daily but only works correctly
#     when the DataFrame index carries timezone-aware UTC timestamps.
#     Added explicit UTC-awareness guard before calling ta.vwap().
# =============================================================================

from __future__ import annotations

import os
import tempfile

# Portable Numba cache — works on any OS / user account
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), ".numba_cache"),
)

from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from config.settings import CONFIG
from monitoring.logger import get_logger

logger = get_logger("Indicators")

# ── Bar-count thresholds ──────────────────────────────────────────────────────
# [J] Separate minimums for training/scan vs prediction paths.
MIN_BARS_AFTER_DROPNA: int = 60   # training / live scan
MIN_BARS_FOR_PREDICT:  int = 30   # ML prediction path (~100 bars passed in)

# ── Column tier definitions ───────────────────────────────────────────────────
# [A] REQUIRED: non-NaN rows are dropped after compute_all().
_REQUIRED_COLS: list = [
    "open", "high", "low", "close",
    # Trend
    "ema_fast", "ema_slow", "ema_trend",
    # Momentum
    "macd", "macd_signal",
    "rsi",
    # Volatility
    "bb_upper", "bb_mid", "bb_lower",
    "atr",
    # Direction
    "adx", "di_pos", "di_neg",
]

# [A] SCORED: used by signal engine when present; rows NOT dropped on NaN.
_SCORED_COLS: list = [
    "stoch_k", "stoch_d",   # Filter 6
    "cmf",                   # Filter 7 (threshold 0.02 recommended — see [Q])
    "vwap",                  # Filter 10
    "squeeze",               # Filter 8
    # HTF confirmation [O]
    "htf_ema_fast", "htf_ema_slow",
    "htf_ema_bull", "htf_ema_bear",
    # Candle patterns [P]
    "pat_bull_engulf", "pat_bear_engulf",
    "pat_bull_pin", "pat_bear_pin",
    "pat_inside_bar", "pat_doji",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _nan_series(df: pd.DataFrame) -> pd.Series:
    """Return a NaN Series aligned to df's index. Use instead of np.nan."""
    return pd.Series(np.nan, index=df.index, dtype=np.float64)


def _col(
    df_or_result: pd.DataFrame,
    prefix:       str,
    silent:       bool = False,
    contains:     Optional[str] = None,
) -> pd.Series:
    """
    [B] Find the first column whose name starts with *prefix* (case-insensitive).
    If not found and *contains* is given, fall back to the first column
    whose name contains *contains* (case-insensitive).

    Returns the column Series if found, a NaN Series aligned to the
    DataFrame's index otherwise.

    silent=True downgrades the missing-column warning to DEBUG.
    """
    upper_prefix = prefix.upper()
    cols = [c for c in df_or_result.columns if c.upper().startswith(upper_prefix)]

    # [M] Version-agnostic fallback for Ichimoku and other renaming indicators
    if not cols and contains:
        lower_contains = contains.lower()
        cols = [c for c in df_or_result.columns if lower_contains in c.lower()]

    if cols:
        return df_or_result[cols[0]]

    msg = (
        f"Column prefix '{prefix}' not found. "
        f"Available: {list(df_or_result.columns)[:10]}"
    )
    if silent:
        logger.debug("[Indicators] %s", msg)
    else:
        logger.warning("⚠️  [Indicators] %s", msg)

    return pd.Series(np.nan, index=df_or_result.index, dtype=np.float64)


# =============================================================================
# IndicatorEngine
# =============================================================================

class IndicatorEngine:
    """
    Computes a full suite of technical indicators and returns an enriched
    DataFrame ready for signal_engine.evaluate() and ml_model._build_dataset().

    Design decisions
    ────────────────
    • All columns lowercased on entry to compute_all() before any guard
      or sub-method runs. [K][N]

    • dropna() uses subset=_REQUIRED_COLS only — optional indicators
      (Ichimoku, OBV, VWAP, Stochastic, CCI, Williams %R) can be NaN
      on early bars without removing valid rows. [A]

    • Period resolution reads CONFIG.get_scalper_profile() at compute_all()
      time — switching M1 ↔ M5 automatically uses the correct periods
      without reinstantiating IndicatorEngine. [G]

    • Stochastic K period from active profile (M5: k=5, M1: k=3). [I][L]

    • Ichimoku uses lookahead=False to eliminate Senkou Span leakage.
      Column lookup uses both prefix and contains-fallback for cross-version
      compatibility. [M]

    • HTF EMAs forward-filled onto primary index via optional htf_df
      parameter — enables signal_engine M15 trend gate without a second
      DataFrame fetch inside the signal engine. [O]

    • Candle pattern columns (engulfing, pin bar, inside bar, doji)
      computed as 0/1 integers, ready for ML features and signal gates. [P]

    • VWAP guarded for UTC-aware index before calling ta.vwap(). [R]

    • All exception handlers use _nan_series(df) not np.nan scalar. [E]
    """

    def __init__(self, config=CONFIG) -> None:
        self.cfg = config

    # ── Period resolver ───────────────────────────────────────────────────────

    def _resolve_periods(self) -> dict:
        """
        Return validated indicator periods for the currently active profile.

        Resolution order:
          1. CONFIG.get_scalper_profile()  — active M1 or M5 scalper profile
          2. CONFIG flat attributes         — day-trader / global defaults

        [G] Each key validated: must be a positive number. None/zero/negative
            falls back to the CONFIG attribute for that key.

        [I] stoch_k included: M5 profile → 5, M1 profile → 3.
            stoch_d and stoch_smooth_k fixed at 3 (standard smoothing).

        [L] Profile dict keys from get_scalper_profile() are uppercase
            (e.g. "STOCH_K"). _resolve_periods() tries both lower and upper
            case so the correct value is always found.
        """
        defaults = {
            "ema_fast":    self.cfg.EMA_FAST,
            "ema_slow":    self.cfg.EMA_SLOW,
            "ema_trend":   self.cfg.EMA_TREND,
            "rsi_period":  self.cfg.RSI_PERIOD,
            "macd_fast":   self.cfg.MACD_FAST,
            "macd_slow":   self.cfg.MACD_SLOW,
            "macd_signal": self.cfg.MACD_SIGNAL,
            "bb_period":   self.cfg.BB_PERIOD,
            "bb_std":      float(self.cfg.BB_STD),
            "atr_period":  self.cfg.ATR_PERIOD,
            "adx_period":  getattr(self.cfg, "ADX_PERIOD", 14),
            "stoch_k":     5,   # M5 default when profile unavailable
        }

        try:
            p = self.cfg.get_scalper_profile()
        except Exception as exc:
            logger.warning(
                "[Indicators] get_scalper_profile() failed (%s) — "
                "using CONFIG flat attributes.", exc,
            )
            return defaults

        resolved = {}
        for key, default_val in defaults.items():
            # [L] Try exact key, then uppercase version (profile returns uppercase)
            raw = p.get(key, p.get(key.upper(), default_val))

            try:
                val = float(raw)
                if val <= 0:
                    raise ValueError(f"{key}={val} is not positive")
                resolved[key] = val if key == "bb_std" else int(val)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "[Indicators] Invalid period '%s'=%r (%s) — "
                    "using default %s.", key, raw, exc, default_val,
                )
                resolved[key] = default_val

        return resolved

    # ── Master compute method ─────────────────────────────────────────────────

    def compute_all(
        self,
        df:             pd.DataFrame,
        for_prediction: bool = False,
        htf_df:         Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Add all indicator columns and return a cleaned DataFrame.

        Parameters
        ──────────
        df             : OHLCV DataFrame from DataHandler.get_data().
        for_prediction : True → uses MIN_BARS_FOR_PREDICT=30 threshold. [J]
        htf_df         : Optional higher-timeframe DataFrame (e.g. M15 or H1)
                         whose EMAs are forward-filled onto the primary index
                         for HTF trend confirmation in signal_engine. [O]

        Returns
        ───────
        Enriched DataFrame with _REQUIRED_COLS guaranteed non-NaN,
        or None if data is insufficient / invalid.
        """
        if df is None or df.empty:
            logger.warning("[Indicators] Empty DataFrame passed to compute_all().")
            return None

        df = df.copy()

        # [K][N] Lowercase ALL columns FIRST — before any guard or sub-method.
        df.columns = df.columns.str.lower()

        # Missing-column guard runs AFTER lowercase conversion [K]
        required_input = {"open", "high", "low", "close"}
        missing_input  = required_input - set(df.columns)
        if missing_input:
            logger.error(
                "[Indicators] Missing required OHLC columns: %s", missing_input
            )
            return None

        rows_before = len(df)
        periods     = self._resolve_periods()

        # [F] Log active profile + key periods
        active_tf = getattr(self.cfg, "SCALPER_TF_SELECTED", "?")
        logger.debug(
            "[Indicators] compute_all | TF=%s | "
            "EMA %d/%d/%d | RSI %d | MACD %d/%d/%d | "
            "BB %d | ATR %d | ADX %d | StochK %d | bars_in=%d",
            active_tf,
            periods["ema_fast"], periods["ema_slow"], periods["ema_trend"],
            periods["rsi_period"],
            periods["macd_fast"], periods["macd_slow"], periods["macd_signal"],
            periods["bb_period"], periods["atr_period"], periods["adx_period"],
            periods["stoch_k"], rows_before,
        )

        df = self._trend_indicators(df, periods)
        df = self._momentum_indicators(df, periods)
        df = self._volatility_indicators(df, periods)
        df = self._volume_indicators(df)
        df = self._market_structure(df)
        df = self._candle_patterns(df)              # [P]

        # [O] Forward-fill HTF EMA confirmation onto primary index
        if htf_df is not None:
            df = self._add_htf_ema(df, htf_df, periods)
        else:
            # Fill with NaN so downstream code can check for presence safely
            for col in ("htf_ema_fast", "htf_ema_slow",
                        "htf_ema_bull", "htf_ema_bear"):
                df[col] = _nan_series(df)

        # [A] Drop NaN only on REQUIRED columns that were actually computed
        drop_subset = [c for c in _REQUIRED_COLS if c in df.columns]
        df.dropna(subset=drop_subset, inplace=True)

        rows_after   = len(df)
        rows_dropped = rows_before - rows_after

        # [F] Log row counts after dropna
        logger.debug(
            "[Indicators] dropna | before=%d after=%d dropped=%d",
            rows_before, rows_after, rows_dropped,
        )

        # [J] Select bar threshold based on caller context
        min_bars = MIN_BARS_FOR_PREDICT if for_prediction else MIN_BARS_AFTER_DROPNA

        if rows_after < min_bars:
            logger.warning(
                "[Indicators] %d rows after dropna < minimum %d for %s path — "
                "returning None.",
                rows_after, min_bars,
                "prediction" if for_prediction else "scan",
            )
            return None

        return df

    # ── Trend indicators ──────────────────────────────────────────────────────

    def _trend_indicators(self, df: pd.DataFrame, p: dict) -> pd.DataFrame:
        """
        EMAs, MACD, ADX, Ichimoku.

        [M] Ichimoku column lookup uses both prefix and contains-fallback
            for cross-version compatibility with pandas_ta ≥ 0.3.14.
        [E] All exception handlers use _nan_series(df).
        """
        # ── EMAs ──────────────────────────────────────────────────────────────
        try:
            df["ema_fast"]  = ta.ema(df["close"], length=p["ema_fast"])
            df["ema_slow"]  = ta.ema(df["close"], length=p["ema_slow"])
            df["ema_trend"] = ta.ema(df["close"], length=p["ema_trend"])
        except Exception as exc:
            logger.warning("[Indicators] EMA error: %s", exc)
            df["ema_fast"] = df["ema_slow"] = df["ema_trend"] = _nan_series(df)

        # ── MACD ──────────────────────────────────────────────────────────────
        try:
            macd = ta.macd(
                df["close"],
                fast   = p["macd_fast"],
                slow   = p["macd_slow"],
                signal = p["macd_signal"],
            )
            if macd is not None and not macd.empty:
                df["macd"]        = _col(macd, "MACD_")
                df["macd_signal"] = _col(macd, "MACDs_")
                df["macd_hist"]   = _col(macd, "MACDh_")
            else:
                raise ValueError("ta.macd() returned None or empty")
        except Exception as exc:
            logger.warning("[Indicators] MACD error: %s", exc)
            df["macd"] = df["macd_signal"] = df["macd_hist"] = _nan_series(df)

        # ── ADX ───────────────────────────────────────────────────────────────
        try:
            adx = ta.adx(
                df["high"], df["low"], df["close"],
                length=p["adx_period"],
            )
            if adx is not None and not adx.empty:
                df["adx"]    = _col(adx, "ADX_")
                df["di_pos"] = _col(adx, "DMP_")
                df["di_neg"] = _col(adx, "DMN_")
            else:
                raise ValueError("ta.adx() returned None or empty")
        except Exception as exc:
            logger.warning("[Indicators] ADX error: %s", exc)
            df["adx"] = df["di_pos"] = df["di_neg"] = _nan_series(df)

        # ── Ichimoku — lookahead=False eliminates Senkou Span leakage ─────────
        # Standard charting shifts Senkou Span A/B 26 bars forward so bar[t]
        # sees cloud values from bars t+1..t+26 (future data).
        # lookahead=False aligns cloud values to the bar at which they were
        # computed — fully causal, no forward leakage into ML features.
        #
        # [M] pandas_ta renamed columns between versions:
        #   ≤0.3.13  →  ITS_9, IKS_26, ISA_9_26, ISB_52_26
        #   ≥0.3.14  →  ISA_9_26_52_26, ISB_9_26_52_26, etc.
        # _col() prefix search + contains fallback handles both versions.
        try:
            ich = ta.ichimoku(
                df["high"], df["low"], df["close"],
                lookahead=False,
            )
            ich_df = ich[0] if isinstance(ich, tuple) else ich
            if ich_df is not None and not ich_df.empty:
                df["tenkan"]   = _col(ich_df, "ITS_",  silent=True,
                                      contains="tenkan")
                df["kijun"]    = _col(ich_df, "IKS_",  silent=True,
                                      contains="kijun")
                df["senkou_a"] = _col(ich_df, "ISA_",  silent=True,
                                      contains="senkou_a")
                df["senkou_b"] = _col(ich_df, "ISB_",  silent=True,
                                      contains="senkou_b")
            else:
                raise ValueError("ta.ichimoku() returned None or empty")
        except Exception as exc:
            logger.debug("[Indicators] Ichimoku (non-critical): %s", exc)
            df["tenkan"]   = _nan_series(df)
            df["kijun"]    = _nan_series(df)
            df["senkou_a"] = _nan_series(df)
            df["senkou_b"] = _nan_series(df)

        return df

    # ── Momentum indicators ───────────────────────────────────────────────────

    def _momentum_indicators(self, df: pd.DataFrame, p: dict) -> pd.DataFrame:
        """
        RSI, Stochastic, CCI, Williams %R.

        [I][L] Stochastic K period from active profile:
                 M5 scalper → stoch_k = 5  (25-minute lookback)
                 M1 scalper → stoch_k = 3  (3-minute lookback)
                 Fallback   → stoch_k = 5  (M5 default)
               D and smooth_k fixed at 3 (standard smoothing).

               STOCH_BULL_ZONE and STOCH_BEAR_ZONE are SCORING THRESHOLD
               values used by signal_engine.py — they are NOT period
               parameters and must NOT be passed to ta.stoch().

        [Q] CMF threshold: retail MT5 provides tick_volume not real volume.
            signal_engine.py should use a threshold of 0.02 (not 0.05)
            for tick-volume-based CMF to account for the higher noise level.
        """
        # ── RSI ───────────────────────────────────────────────────────────────
        try:
            df["rsi"] = ta.rsi(df["close"], length=p["rsi_period"])
        except Exception as exc:
            logger.warning("[Indicators] RSI error: %s", exc)
            df["rsi"] = _nan_series(df)

        # ── Stochastic ────────────────────────────────────────────────────────
        # [I][L] K period from resolved profile — D and smooth_k always 3.
        stoch_k_period = p.get("stoch_k", 5)
        try:
            stoch = ta.stoch(
                df["high"], df["low"], df["close"],
                k        = stoch_k_period,
                d        = 3,
                smooth_k = 3,
            )
            if stoch is not None and not stoch.empty:
                df["stoch_k"] = _col(stoch, "STOCHk_")
                df["stoch_d"] = _col(stoch, "STOCHd_")
            else:
                raise ValueError("ta.stoch() returned None or empty")
        except Exception as exc:
            logger.warning("[Indicators] Stochastic error: %s", exc)
            df["stoch_k"] = df["stoch_d"] = _nan_series(df)

        # ── CCI ───────────────────────────────────────────────────────────────
        try:
            df["cci"] = ta.cci(
                df["high"], df["low"], df["close"], length=20
            )
        except Exception as exc:
            logger.debug("[Indicators] CCI (non-critical): %s", exc)
            df["cci"] = _nan_series(df)

        # ── Williams %R ───────────────────────────────────────────────────────
        try:
            df["willr"] = ta.willr(
                df["high"], df["low"], df["close"]
            )
        except Exception as exc:
            logger.debug("[Indicators] Williams %%R (non-critical): %s", exc)
            df["willr"] = _nan_series(df)

        return df

    # ── Volatility indicators ─────────────────────────────────────────────────

    def _volatility_indicators(self, df: pd.DataFrame, p: dict) -> pd.DataFrame:
        """
        Bollinger Bands, ATR, Keltner Channel.
        """
        # ── Bollinger Bands ───────────────────────────────────────────────────
        try:
            bb = ta.bbands(
                df["close"],
                length = p["bb_period"],
                std    = p["bb_std"],
            )
            if bb is not None and not bb.empty:
                df["bb_upper"] = _col(bb, "BBU_")
                df["bb_mid"]   = _col(bb, "BBM_")
                df["bb_lower"] = _col(bb, "BBL_")
                df["bb_width"] = (
                    (df["bb_upper"] - df["bb_lower"]) /
                    df["bb_mid"].replace(0, np.nan)
                )
            else:
                raise ValueError("ta.bbands() returned None or empty")
        except Exception as exc:
            logger.warning("[Indicators] Bollinger Bands error: %s", exc)
            df["bb_upper"] = df["bb_mid"] = \
            df["bb_lower"] = df["bb_width"] = _nan_series(df)

        # ── ATR ───────────────────────────────────────────────────────────────
        try:
            df["atr"] = ta.atr(
                df["high"], df["low"], df["close"],
                length=p["atr_period"],
            )
        except Exception as exc:
            logger.warning("[Indicators] ATR error: %s", exc)
            df["atr"] = _nan_series(df)

        # ── Keltner Channel ───────────────────────────────────────────────────
        # Optional — only used for squeeze detection. [B] silent=True. [D]
        try:
            kc = ta.kc(df["high"], df["low"], df["close"])
            if kc is not None and not kc.empty:
                df["kc_upper"] = _col(kc, "KCU", silent=True)
                df["kc_lower"] = _col(kc, "KCL", silent=True)
            else:
                raise ValueError("ta.kc() returned None or empty")
        except Exception as exc:
            logger.debug("[Indicators] Keltner Channel (non-critical): %s", exc)
            df["kc_upper"] = df["kc_lower"] = _nan_series(df)

        return df

    # ── Volume indicators ─────────────────────────────────────────────────────

    def _volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        OBV, CMF, VWAP.

        [C] Volume validity uses majority test (>50% positive bars).
        [C][E] All derived NaN columns use pd.Series not scalar np.nan.
        [Q] CMF threshold recommendation: 0.02 (not 0.05) for tick_volume.
        [R] VWAP requires UTC-aware index; guarded before calling ta.vwap().
        """
        # [C] Majority test
        if "volume" in df.columns:
            positive_count = (df["volume"] > 0).sum()
            has_volume = (
                df["volume"].notna().any() and
                positive_count > len(df) * 0.5
            )
        else:
            has_volume = False

        if not has_volume:
            logger.debug(
                "[Indicators] No valid volume data (%s) — "
                "OBV/CMF/VWAP set to NaN.",
                "column absent" if "volume" not in df.columns
                else "insufficient positive bars",
            )

        vol = df["volume"] if has_volume else None

        # ── OBV ───────────────────────────────────────────────────────────────
        try:
            df["obv"] = (
                ta.obv(df["close"], vol) if has_volume else _nan_series(df)
            )
        except Exception as exc:
            logger.debug("[Indicators] OBV error: %s", exc)
            df["obv"] = _nan_series(df)

        # ── CMF ───────────────────────────────────────────────────────────────
        # [Q] Recommended signal_engine threshold: 0.02 (not 0.05) because
        #     MT5 provides tick_volume (tick count) rather than real traded
        #     volume on retail EURUSD feeds. Tick-volume-based CMF is noisier
        #     than real-volume CMF and a tighter threshold filters too much.
        try:
            df["cmf"] = (
                ta.cmf(df["high"], df["low"], df["close"], vol)
                if has_volume else _nan_series(df)
            )
        except Exception as exc:
            logger.debug("[Indicators] CMF error: %s", exc)
            df["cmf"] = _nan_series(df)

        # ── VWAP ──────────────────────────────────────────────────────────────
        # ta.vwap() anchors at the first timestamp of each calendar day.
        # [R] Requires a UTC-aware DatetimeIndex to compute the daily anchor.
        #     If the index is tz-naive, convert to UTC before computing.
        try:
            if has_volume:
                idx = df.index
                if idx.tzinfo is None:
                    df.index = idx.tz_localize("UTC")

                df["vwap"] = ta.vwap(
                    df["high"], df["low"], df["close"], vol
                )

                # Restore original tz-awareness if we changed it
                if idx.tzinfo is None:
                    df.index = df.index.tz_localize(None)
            else:
                df["vwap"] = _nan_series(df)
        except Exception as exc:
            logger.debug("[Indicators] VWAP error: %s", exc)
            df["vwap"] = _nan_series(df)

        return df

    # ── Market structure ──────────────────────────────────────────────────────

    def _market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derived features: EMA spread, candle shape, squeeze flag.

        [D] Squeeze only computed when both BB and KC data are valid.
        [H] replace([inf, -inf], np.nan) before return.
        """
        atr_safe = df["atr"].replace(0, np.nan)

        # EMA-derived normalised distances (ATR units)
        df["price_vs_ema_fast"] = (df["close"] - df["ema_fast"]) / atr_safe
        df["price_vs_ema_slow"] = (df["close"] - df["ema_slow"]) / atr_safe
        df["ema_spread"]        = (df["ema_fast"] - df["ema_slow"]) / atr_safe

        # Candle shape
        df["body"]       = df["close"] - df["open"]
        df["upper_wick"] = df["high"]  - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["is_bullish"] = (df["body"] > 0).astype(int)

        # [D] Squeeze — Bollinger Bands inside Keltner Channel
        kc_valid = (
            "kc_upper" in df.columns and "kc_lower" in df.columns and
            df["kc_upper"].notna().any() and df["kc_lower"].notna().any()
        )
        bb_valid = (
            "bb_upper" in df.columns and "bb_lower" in df.columns and
            df["bb_upper"].notna().any() and df["bb_lower"].notna().any()
        )

        if kc_valid and bb_valid:
            df["squeeze"] = (
                (df["bb_lower"] > df["kc_lower"]) &
                (df["bb_upper"] < df["kc_upper"])
            ).astype(int)
            logger.debug(
                "[Indicators] Squeeze computed — %d/%d bars in compression.",
                int(df["squeeze"].sum()), len(df),
            )
        else:
            logger.debug(
                "[Indicators] KC or BB invalid — squeeze=0 (no compression)."
            )
            df["squeeze"] = 0

        # [H] Remove inf values from ATR-divided features
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        return df

    # ── Candle pattern recognition ────────────────────────────────────────────

    def _candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [P] Detect common price-action patterns as 0/1 integer columns.

        Patterns computed
        ─────────────────
        pat_bull_engulf  : Bullish engulfing — current candle's body fully
                           engulfs the previous candle's body and is bullish.
        pat_bear_engulf  : Bearish engulfing — same logic, current is bearish.
        pat_bull_pin     : Bullish pin bar — lower wick ≥ 2× body,
                           upper wick < body. Signals potential reversal up.
        pat_bear_pin     : Bearish pin bar — upper wick ≥ 2× body,
                           lower wick < body. Signals potential reversal down.
        pat_inside_bar   : Inside bar — current high < previous high AND
                           current low > previous low. Consolidation signal.
        pat_doji         : Body < 10% of total candle range. Indecision.

        All values are 0 or 1 (integer). NaN-safe: if a prior-bar value is
        unavailable the pattern is set to 0 for that row.
        """
        try:
            o   = df["open"]
            h   = df["high"]
            l   = df["low"]
            c   = df["close"]
            po  = o.shift(1)    # previous open
            ph  = h.shift(1)    # previous high
            pl  = l.shift(1)    # previous low
            pc  = c.shift(1)    # previous close

            body_size      = (c - o).abs()
            prev_body_size = (pc - po).abs()
            candle_range   = (h - l).replace(0, np.nan)

            upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
            lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l

            # Bullish engulfing
            df["pat_bull_engulf"] = (
                (c > o) &           # current bullish
                (pc < po) &         # previous bearish
                (c >= po) &         # current close >= previous open
                (o <= pc)           # current open  <= previous close
            ).astype(int).fillna(0)

            # Bearish engulfing
            df["pat_bear_engulf"] = (
                (c < o) &           # current bearish
                (pc > po) &         # previous bullish
                (c <= po) &         # current close <= previous open
                (o >= pc)           # current open  >= previous close
            ).astype(int).fillna(0)

            # Bullish pin bar
            df["pat_bull_pin"] = (
                (lower_wick >= 2 * body_size.replace(0, np.nan)) &
                (upper_wick <  body_size)
            ).astype(int).fillna(0)

            # Bearish pin bar
            df["pat_bear_pin"] = (
                (upper_wick >= 2 * body_size.replace(0, np.nan)) &
                (lower_wick <  body_size)
            ).astype(int).fillna(0)

            # Inside bar
            df["pat_inside_bar"] = (
                (h < ph) & (l > pl)
            ).astype(int).fillna(0)

            # Doji — body < 10% of candle range
            df["pat_doji"] = (
                body_size < 0.1 * candle_range
            ).astype(int).fillna(0)

        except Exception as exc:
            logger.warning("[Indicators] Candle pattern error: %s", exc)
            for col in ("pat_bull_engulf", "pat_bear_engulf",
                        "pat_bull_pin",   "pat_bear_pin",
                        "pat_inside_bar", "pat_doji"):
                df[col] = 0

        return df

    # ── HTF EMA confirmation ──────────────────────────────────────────────────

    def _add_htf_ema(
        self,
        df:      pd.DataFrame,
        htf_df:  pd.DataFrame,
        periods: dict,
    ) -> pd.DataFrame:
        """
        [O] Forward-fill HTF EMA values onto the primary timeframe index.

        Workflow
        ────────
        1. Lowercase htf_df columns.
        2. Compute EMA fast and slow on the HTF DataFrame using the same
           periods as the primary timeframe (profile-driven).
        3. Reindex to the primary df's DatetimeIndex using forward-fill
           (ffill) — every M5 bar gets the most recent closed M15/H1 EMA.
        4. Add htf_ema_bull and htf_ema_bear as 0/1 integer columns
           so signal_engine can gate trades with a simple boolean check:
             if ind["htf_ema_bull"].iloc[-1] == 1: # only take BUY signals
        5. No rows are dropped by this step — if HTF data is unavailable
           for any row the column remains NaN and the signal engine treats
           it as a neutral (no-confirmation) gate.

        This replaces the pattern where signal_engine would have to fetch
        a second DataFrame internally, keeping the signal engine stateless
        with respect to data access.
        """
        try:
            htf = htf_df.copy()
            htf.columns = htf.columns.str.lower()

            if "close" not in htf.columns:
                raise ValueError("htf_df missing 'close' column")

            htf["htf_ema_fast"] = ta.ema(htf["close"], length=periods["ema_fast"])
            htf["htf_ema_slow"] = ta.ema(htf["close"], length=periods["ema_slow"])

            # Keep only the EMA columns for the merge
            htf_emas = htf[["htf_ema_fast", "htf_ema_slow"]].copy()

            # Both indexes must be tz-aware for reindex to work correctly
            if htf_emas.index.tzinfo is None:
                htf_emas.index = htf_emas.index.tz_localize("UTC")
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC")

            # Reindex + forward-fill: each primary bar gets the last HTF EMA
            htf_emas = htf_emas.reindex(
                htf_emas.index.union(df.index)
            ).ffill().reindex(df.index)

            df["htf_ema_fast"] = htf_emas["htf_ema_fast"].values
            df["htf_ema_slow"] = htf_emas["htf_ema_slow"].values

            # Boolean confirmation columns
            df["htf_ema_bull"] = (
                df["htf_ema_fast"] > df["htf_ema_slow"]
            ).astype(int)
            df["htf_ema_bear"] = (
                df["htf_ema_fast"] < df["htf_ema_slow"]
            ).astype(int)

            bull_pct = df["htf_ema_bull"].mean() * 100
            logger.debug(
                "[Indicators] HTF EMAs attached — bull=%.1f%% bear=%.1f%%",
                bull_pct, 100 - bull_pct,
            )

        except Exception as exc:
            logger.warning("[Indicators] HTF EMA attachment failed: %s", exc)
            df["htf_ema_fast"] = _nan_series(df)
            df["htf_ema_slow"] = _nan_series(df)
            df["htf_ema_bull"] = _nan_series(df)
            df["htf_ema_bear"] = _nan_series(df)

        return df
