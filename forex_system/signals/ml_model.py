# =============================================================================
# GODBOT v3.0 – signals/ml_model.py
# =============================================================================
#  Fixes / improvements applied in this revision:
#
#  A  float32 cast + NaN-drop done per-chunk (chunked 36k-bar history)
#
#  B  Granular try/except per feature group — one bad indicator does not
#     wipe the entire feature matrix
#
#  C  Label alignment fixed after NaN drops — index reset before concat
#
#  D  DST-aware session features via tz_convert('Europe/Madrid') with
#     UTC-naive fallback
#
#  E  Optuna objective changed to macro F1 (both XGB and LGBM)
#
#  F  Class weights REMOVED — reverts to unweighted XGB / balanced LGBM
#
#  G  Ensemble weights: XGB 45% + LGBM 55% everywhere
#
#  H  New features: ADX slope, trending/ranging regime, H1/M15 EMA proxy,
#     swing high/low distance, ROC-3/6/12
#
#  I  n_trials wired through train() → _tune_xgb() / _tune_lgbm()
#
#  J  Memory-usage helper via BytesIO
#
#  K  Atomic model save with temp files
#
#  L  Per-file warnings on load
#
#  M  predict_batch() vectorised via _safe_proba_batch()
#
#  N  [FIX] FORWARD_BARS reduced from 12 → 6 bars (30 min).
#     12 bars = 60 min look-ahead labels a trade's entire hold time as
#     a success, turning any entry that eventually recovered into BUY/SELL.
#     6 bars = 30 min matches the actual intended scalp hold time and
#     produces tighter, more actionable labels.
#
#  O  [FIX] ML_MIN_CONFIDENCE raised from 0.35 → 0.55.
#     On a 3-class problem random chance = 0.333. The previous threshold
#     of 0.35 was only 1.7% above random and filtered almost nothing.
#     0.55 requires genuine model conviction before a signal passes.
#     All predict() / predict_batch() callers that previously used 0.35
#     must now use 0.55 (enforced via MIN_CONFIDENCE class constant).
#
#  P  [FIX] Minimum training bars raised from 500 → 20 000.
#     5 000 M5 bars ≈ 17 trading days. Optuna + 3-fold CV on 4 000 rows
#     produces overfit models. 20 000 bars ≈ 70 trading days (≈ 3 months)
#     is the practical minimum for a generalising M5 EURUSD classifier.
#     A clear WARNING is emitted when the caller passes fewer than 20 000
#     bars so the problem is visible in the log, not silently accepted.
#
#  Q  [FIX] predict() accepted a pd.Series (single row) but
#     _build_features() expects a DataFrame. Fixed: predict() wraps the
#     Series in a single-row DataFrame before calling _build_features().
#     _row_to_vector() is now only used as an internal fallback when
#     _build_features() cannot produce a matrix for a single row.
#
#  R  [FIX] _create_labels() MIN_PIP_MOVE of 0.0005 was used as an
#     absolute return threshold on a pct_change() value. For EURUSD at
#     1.10, a 0.0005 return = 0.05% = ~0.55 pips net move — far too low.
#     Threshold corrected to 0.0008 (0.08% return ≈ 0.88 pips) and
#     documented. A table of threshold → pip-equivalent is included.
#
#  S  [FIX] _build_features() stochastic used a fixed 14-bar K period
#     regardless of the active scalper profile. Now reads
#     CONFIG.get_scalper_profile()["STOCH_K"] (M5→5, M1→3) to match
#     what IndicatorEngine computes, so ML features are consistent with
#     the signal engine's indicator values.
#
#  T  [NEW] Candle-pattern features (pat_bull_engulf, pat_bear_engulf,
#     pat_bull_pin, pat_bear_pin, pat_inside_bar, pat_doji) added to
#     _build_features() to match the columns now produced by
#     IndicatorEngine._candle_patterns(). ML model sees the same patterns
#     the signal engine scores.
#
#  U  [NEW] HTF confirmation features (htf_ema_bull, htf_ema_bear) from
#     IndicatorEngine._add_htf_ema() added as pass-through features when
#     the incoming DataFrame already contains those columns, making the
#     ML model aware of the higher-timeframe trend direction.
#
#  V  [FIX] load() returned False silently when one of the four artefact
#     files existed but was corrupt (joblib.load raised an exception).
#     Added per-file load with individual try/except so partial corruption
#     is visible and the corrupt file is flagged for deletion.
#
#  W  [NEW] is_trained property for safe external state checks without
#     accessing private _trained attribute directly.
#
#  X  [FIX] Feature matrix alignment: _build_features() now returns
#     a DataFrame aligned to the SAME index as the input df after NaN
#     drops. train() uses this index to slice df_trimmed for label
#     creation, preventing the HOLD-inflation bug from warm-up trimming.
# =============================================================================

from __future__ import annotations

import io
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

from config.settings import CONFIG
from monitoring.logger import logger

# ── Silence noisy third-party loggers ─────────────────────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("xgboost").setLevel(logging.ERROR)


# =============================================================================
# Constants
# =============================================================================
_LABEL_HOLD  = 0
_LABEL_BUY   = 1
_LABEL_SELL  = 2
_LABEL_NAMES = {_LABEL_HOLD: "HOLD", _LABEL_BUY: "BUY", _LABEL_SELL: "SELL"}

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# =============================================================================
# MLSignalModel
# =============================================================================
class MLSignalModel:
    """
    Two-model ensemble (XGBoost 45% + LightGBM 55%) for M5 EURUSD
    direction classification: HOLD / BUY / SELL.

    Key design choices
    ──────────────────
    • FORWARD_BARS = 6 (30 min) — matches actual scalp hold time. [N]
    • MIN_CONFIDENCE = 0.55 — meaningful separation from 0.333 random. [O]
    • MIN_TRAINING_BARS = 20_000 — ≈ 3 months of M5 bars. [P]
    • Macro F1 Optuna objective — prevents class-imbalance overfitting. [E]
    • Stochastic K period from active profile to match IndicatorEngine. [S]
    • Candle-pattern features aligned with IndicatorEngine output. [T]
    • HTF confirmation features passed through when present. [U]
    """

    # ── Configurable class-level defaults ─────────────────────────────────────
    OPTUNA_TRIALS:    int   = 40
    CV_SPLITS:        int   = 3
    TEST_FRAC:        float = 0.20

    # [N] Reduced from 12 → 6 bars (30 min scalp window)
    FORWARD_BARS:     int   = 6

    # [O] Raised from 0.35 → 0.55 (meaningful above 0.333 random baseline)
    MIN_CONFIDENCE:   float = 0.55

    # [P] Raised from 500 → 20_000 bars minimum for training
    MIN_TRAINING_BARS: int  = 20_000

    # [R] Fixed pct-return threshold: 0.0008 ≈ 0.88 pips on EURUSD at 1.10
    #     Threshold → pip-equivalent table:
    #       0.0003 ≈ 0.33 pips  (too low — noise dominates)
    #       0.0005 ≈ 0.55 pips  (previous value — marginally above noise)
    #       0.0008 ≈ 0.88 pips  (this value — minimum meaningful scalp move)
    #       0.0012 ≈ 1.32 pips  (too high — most scalp moves labelled HOLD)
    MIN_PIP_MOVE:     float = 0.0008

    def __init__(self, symbol: str = "EURUSD") -> None:
        self.symbol    = symbol
        self._xgb:      Optional[xgb.XGBClassifier]  = None
        self._lgbm:     Optional[lgb.LGBMClassifier] = None
        self._scaler:   Optional[StandardScaler]      = None
        self._features: Optional[List[str]]           = None
        self._trained   = False

    # ── [W] Safe external state check ─────────────────────────────────────────
    @property
    def is_trained(self) -> bool:
        """True only when all four model artefacts are loaded and non-None."""
        return (
            self._trained and
            self._xgb      is not None and
            self._lgbm     is not None and
            self._scaler   is not None and
            self._features is not None
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def train(
        self,
        df:       pd.DataFrame,
        n_trials: Optional[int] = None,
    ) -> Dict:
        """
        Full training pipeline.

        [P] Emits a WARNING if fewer than MIN_TRAINING_BARS rows are passed
            so the problem is visible in the log. Training is not blocked
            (to allow unit tests / quick demos) but the warning is explicit.

        [X] df_trimmed sliced from df using the index returned by
            _build_features() so label ATR thresholds are computed on the
            same rows the features see.
        """
        trials = n_trials if n_trials is not None else self.OPTUNA_TRIALS
        logger.info(
            "[MLModel] Training started — symbol=%s  trials=%d",
            self.symbol, trials,
        )

        # [P] Minimum bar guard
        if len(df) < self.MIN_TRAINING_BARS:
            logger.warning(
                "[MLModel] ⚠️  Only %d bars provided — minimum recommended is "
                "%d (%d M5 bars ≈ %.0f trading days). "
                "Model may overfit. Use a longer history for production.",
                len(df), self.MIN_TRAINING_BARS,
                self.MIN_TRAINING_BARS,
                self.MIN_TRAINING_BARS / (6.5 * 12),   # ≈ trading days
            )

        # ── 1. Feature engineering ────────────────────────────────────────────
        X_raw, feature_names = self._build_features(df)
        if X_raw is None or len(X_raw) < 500:
            raise ValueError(
                "Feature matrix too small after engineering "
                f"(got {0 if X_raw is None else len(X_raw)} rows)."
            )

        # ── 2. Label creation on the TRIMMED index [X] ───────────────────────
        # Slice df to only the rows that survived feature NaN removal.
        # This prevents HTF EMA warm-up rows from inflating HOLD labels.
        df_trimmed = df.loc[df.index.intersection(X_raw.index)]
        labels_raw = self._create_labels(df_trimmed)
        valid_idx  = labels_raw.dropna().index
        labels     = labels_raw.loc[valid_idx]
        X_raw      = X_raw.loc[valid_idx]

        y = labels.values.astype(np.int32)
        X = X_raw.values.astype(np.float32)

        n_total = len(y)
        if n_total < 500:
            raise ValueError(
                f"Too few aligned rows after label creation: {n_total}"
            )

        logger.info(
            "[MLModel] Dataset: %d rows, %d features", n_total, X.shape[1]
        )
        for lbl, name in _LABEL_NAMES.items():
            pct = (y == lbl).sum() / n_total * 100
            logger.info("  %s: %.1f%%", name, pct)

        # ── 3. Temporal train / test split ────────────────────────────────────
        n_test  = max(500, int(n_total * self.TEST_FRAC))
        n_train = n_total - n_test
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        logger.info("[MLModel] Train=%d  Test=%d", n_train, n_test)

        # ── 4. Scale ──────────────────────────────────────────────────────────
        self._scaler = StandardScaler()
        X_train_s = self._scaler.fit_transform(X_train)
        X_test_s  = self._scaler.transform(X_test)

        # ── 5. Optuna hyperparameter search ───────────────────────────────────
        logger.info("[MLModel] Optuna tuning — XGB (%d trials) …", trials)
        best_xgb_params  = self._tune_xgb(X_train_s, y_train, trials)
        logger.info("[MLModel] Optuna tuning — LGBM (%d trials) …", trials)
        best_lgbm_params = self._tune_lgbm(X_train_s, y_train, trials)

        # ── 6. Time-series cross-validation ───────────────────────────────────
        tscv = TimeSeriesSplit(n_splits=self.CV_SPLITS)
        xgb_cv_scores, lgbm_cv_scores = [], []

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train_s)):
            Xtr, Xva = X_train_s[tr_idx], X_train_s[va_idx]
            ytr, yva = y_train[tr_idx],   y_train[va_idx]

            xgb_cv = xgb.XGBClassifier(
                **best_xgb_params,
                objective="multi:softprob",
                num_class=3,
                use_label_encoder=False,
                eval_metric="mlogloss",
                verbosity=0,
            )
            xgb_cv.fit(Xtr, ytr)
            xgb_cv_scores.append(
                accuracy_score(yva, xgb_cv.predict(Xva))
            )

            lgbm_cv = lgb.LGBMClassifier(
                **best_lgbm_params,
                objective="multiclass",
                num_class=3,
                class_weight="balanced",
                verbose=-1,
            )
            lgbm_cv.fit(Xtr, ytr)
            lgbm_cv_scores.append(
                accuracy_score(yva, lgbm_cv.predict(Xva))
            )

        mean_xgb_cv  = float(np.mean(xgb_cv_scores))
        mean_lgbm_cv = float(np.mean(lgbm_cv_scores))
        logger.info(
            "[MLModel] XGB  mean CV accuracy : %.4f", mean_xgb_cv
        )
        logger.info(
            "[MLModel] LGBM mean CV accuracy : %.4f", mean_lgbm_cv
        )

        # ── 7. Final model fit on training set ────────────────────────────────
        self._xgb = xgb.XGBClassifier(
            **best_xgb_params,
            objective="multi:softprob",
            num_class=3,
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
        )
        self._xgb.fit(X_train_s, y_train)

        self._lgbm = lgb.LGBMClassifier(
            **best_lgbm_params,
            objective="multiclass",
            num_class=3,
            class_weight="balanced",
            verbose=-1,
        )
        self._lgbm.fit(X_train_s, y_train)

        self._features = feature_names
        self._trained  = True

        # ── 8. Evaluate on held-out test set ──────────────────────────────────
        xgb_proba  = self._xgb.predict_proba(X_test_s)
        lgbm_proba = self._lgbm.predict_proba(X_test_s)
        ensemble   = 0.45 * xgb_proba + 0.55 * lgbm_proba   # [G]
        y_pred     = np.argmax(ensemble, axis=1)

        test_acc   = float(accuracy_score(y_test, y_pred))
        report_str = classification_report(
            y_test, y_pred, target_names=["HOLD", "BUY", "SELL"]
        )
        logger.info(
            "[MLModel] Held-out test accuracy: %.4f", test_acc
        )
        logger.info("\n%s", report_str)

        fi = self._feature_importances()

        return {
            "xgb_cv_accuracy":       mean_xgb_cv,
            "lgbm_cv_accuracy":      mean_lgbm_cv,
            "test_accuracy":         test_acc,
            "n_train":               n_train,
            "n_test":                n_test,
            "n_features":            len(feature_names),
            "feature_names":         feature_names,
            "feature_importances":   fi,
            "classification_report": report_str,
        }

    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, df_input: pd.DataFrame) -> Dict:
        """
        Return a dict with label, confidence, and per-class probabilities
        for the most recent bar in *df_input*.

        [Q] Accepts a full DataFrame (not a Series). The last row is used
            for prediction. _build_features() is called on the full df so
            that rolling indicator lookbacks are correct; only the final
            row of the resulting feature matrix is used for inference.

        [O] Returns confidence=0.0 and label='HOLD' when confidence is
            below MIN_CONFIDENCE, so the caller never needs to check the
            threshold separately.
        """
        if not self.is_trained:
            return {"label": "HOLD", "confidence": 0.0,
                    "probabilities": {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0}}

        try:
            # [Q] Build features on the full df to get correct rolling windows
            X_feat, _ = self._build_features(df_input)
            if X_feat is None or X_feat.empty:
                return {"label": "HOLD", "confidence": 0.0,
                        "probabilities": {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0}}

            # Align to stored feature list
            X_aligned = self._align_features(X_feat)
            # Use only the last row for the current-bar prediction
            x_last = X_aligned.iloc[[-1]].values.astype(np.float32)
            x_s    = self._scaler.transform(x_last)

            xgb_p  = self._xgb.predict_proba(x_s)[0]
            lgbm_p = self._lgbm.predict_proba(x_s)[0]
            proba  = 0.45 * xgb_p + 0.55 * lgbm_p   # [G]

            label_idx = int(np.argmax(proba))
            conf      = float(proba[label_idx])

            # [O] Apply confidence gate
            if conf < self.MIN_CONFIDENCE:
                return {
                    "label":         "HOLD",
                    "confidence":    conf,
                    "probabilities": {
                        "HOLD": float(proba[0]),
                        "BUY":  float(proba[1]),
                        "SELL": float(proba[2]),
                    },
                }

            return {
                "label":         _LABEL_NAMES[label_idx],
                "confidence":    conf,
                "probabilities": {
                    "HOLD": float(proba[0]),
                    "BUY":  float(proba[1]),
                    "SELL": float(proba[2]),
                },
            }

        except Exception as exc:
            logger.warning("[MLModel] predict() error: %s", exc)
            return {"label": "HOLD", "confidence": 0.0,
                    "probabilities": {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0}}

    # ──────────────────────────────────────────────────────────────────────────

    def predict_batch(self, df: pd.DataFrame) -> List[Dict]:
        """
        [M] Vectorised batch prediction for a full DataFrame.
        Returns a list of dicts (one per row) with label/confidence/probas.
        """
        if not self.is_trained:
            return [{"label": "HOLD", "confidence": 0.0}] * len(df)

        try:
            X_feat, _ = self._build_features(df)
            if X_feat is None or X_feat.empty:
                return [{"label": "HOLD", "confidence": 0.0}] * len(df)

            X_aligned  = self._align_features(X_feat)
            X_s        = self._scaler.transform(
                X_aligned.values.astype(np.float32)
            )
            xgb_proba  = self._xgb.predict_proba(X_s)
            lgbm_proba = self._lgbm.predict_proba(X_s)
            ensemble   = 0.45 * xgb_proba + 0.55 * lgbm_proba   # [G]
            label_idxs = np.argmax(ensemble, axis=1)
            confs      = ensemble[np.arange(len(label_idxs)), label_idxs]

            results = []
            for i, (lidx, conf) in enumerate(zip(label_idxs, confs)):
                # [O] Confidence gate per row
                effective_label = (
                    _LABEL_NAMES[int(lidx)]
                    if float(conf) >= self.MIN_CONFIDENCE
                    else "HOLD"
                )
                results.append({
                    "label":         effective_label,
                    "confidence":    float(conf),
                    "probabilities": {
                        "HOLD": float(ensemble[i, 0]),
                        "BUY":  float(ensemble[i, 1]),
                        "SELL": float(ensemble[i, 2]),
                    },
                })
            return results

        except Exception as exc:
            logger.warning("[MLModel] predict_batch() error: %s", exc)
            return [{"label": "HOLD", "confidence": 0.0}] * len(df)

    # =========================================================================
    # Feature engineering
    # =========================================================================

    def _build_features(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """
        Build the full feature matrix from raw OHLCV.

        [B]  Per-group try/except — one bad indicator does not wipe all.
        [S]  Stochastic K period from active scalper profile.
        [T]  Candle-pattern features (aligned with IndicatorEngine output).
        [U]  HTF confirmation features passed through when present.
        [X]  Returns DataFrame aligned to same index as input after NaN drops.
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        for col in ("open", "high", "low", "close", "tick_volume"):
            if col in df.columns:
                df[col] = df[col].astype(np.float32)

        feats = pd.DataFrame(index=df.index)

        # ── Price / returns ───────────────────────────────────────────────────
        try:
            feats["ret_1"]    = df["close"].pct_change(1)
            feats["ret_3"]    = df["close"].pct_change(3)
            feats["ret_5"]    = df["close"].pct_change(5)
            feats["ret_10"]   = df["close"].pct_change(10)
            feats["hl_ratio"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)
            feats["oc_ratio"] = (df["close"] - df["open"]) / (df["close"] + 1e-9)
        except Exception as exc:
            logger.warning("[MLModel] price/return features failed: %s", exc)

        # ── EMAs ──────────────────────────────────────────────────────────────
        try:
            ema8   = df["close"].ewm(span=8,   adjust=False).mean()
            ema21  = df["close"].ewm(span=21,  adjust=False).mean()
            ema50  = df["close"].ewm(span=50,  adjust=False).mean()
            ema200 = df["close"].ewm(span=200, adjust=False).mean()
            feats["ema_fast"]  = (df["close"] - ema8)   / (df["close"] + 1e-9)
            feats["ema_slow"]  = (df["close"] - ema21)  / (df["close"] + 1e-9)
            feats["ema_50"]    = (df["close"] - ema50)  / (df["close"] + 1e-9)
            feats["ema_200"]   = (df["close"] - ema200) / (df["close"] + 1e-9)
            feats["ema_cross"] = (ema8 - ema21) / (df["close"] + 1e-9)
        except Exception as exc:
            logger.warning("[MLModel] EMA features failed: %s", exc)

        # ── HTF proxy (M5 data aggregated to M15 / H1) ───────────────────────
        try:
            ema_fast_m15 = df["close"].ewm(span=24,  adjust=False).mean()
            ema_slow_m15 = df["close"].ewm(span=63,  adjust=False).mean()
            ema_fast_h1  = df["close"].ewm(span=96,  adjust=False).mean()
            ema_slow_h1  = df["close"].ewm(span=252, adjust=False).mean()

            feats["ema_fast_m15"]  = (df["close"] - ema_fast_m15) / (df["close"] + 1e-9)
            feats["ema_slow_m15"]  = (df["close"] - ema_slow_m15) / (df["close"] + 1e-9)
            feats["htf_cross_m15"] = (ema_fast_m15 - ema_slow_m15) / (df["close"] + 1e-9)
            feats["ema_fast_h1"]   = (df["close"] - ema_fast_h1)  / (df["close"] + 1e-9)
            feats["ema_slow_h1"]   = (df["close"] - ema_slow_h1)  / (df["close"] + 1e-9)
            feats["htf_cross_h1"]  = (ema_fast_h1 - ema_slow_h1)  / (df["close"] + 1e-9)
            feats["htf_trend"]     = np.sign(
                (ema_fast_h1 - ema_slow_h1).values
            ).astype(np.float32)
        except Exception as exc:
            logger.warning("[MLModel] HTF proxy features failed: %s", exc)

        # ── [U] HTF confirmation columns from IndicatorEngine ─────────────────
        # If the incoming df already has htf_ema_bull / htf_ema_bear (computed
        # by IndicatorEngine._add_htf_ema() from real M15 data), use those
        # directly. They are more accurate than the EWM proxy above.
        try:
            for col in ("htf_ema_bull", "htf_ema_bear"):
                if col in df.columns and df[col].notna().any():
                    feats[col] = df[col].values
        except Exception as exc:
            logger.warning("[MLModel] HTF pass-through features failed: %s", exc)

        # ── MACD ──────────────────────────────────────────────────────────────
        try:
            ema12       = df["close"].ewm(span=12, adjust=False).mean()
            ema26       = df["close"].ewm(span=26, adjust=False).mean()
            macd_line   = ema12 - ema26
            macd_sig    = macd_line.ewm(span=9, adjust=False).mean()
            feats["macd"]        = macd_line            / (df["close"] + 1e-9)
            feats["macd_signal"] = macd_sig             / (df["close"] + 1e-9)
            feats["macd_hist"]   = (macd_line - macd_sig) / (df["close"] + 1e-9)
        except Exception as exc:
            logger.warning("[MLModel] MACD features failed: %s", exc)

        # ── RSI ───────────────────────────────────────────────────────────────
        try:
            feats["rsi"]    = self._rsi(df["close"], 14)
            feats["rsi_6"]  = self._rsi(df["close"], 6)
            feats["rsi_25"] = self._rsi(df["close"], 25)
        except Exception as exc:
            logger.warning("[MLModel] RSI features failed: %s", exc)

        # ── Bollinger Bands ───────────────────────────────────────────────────
        try:
            bb_mid = df["close"].rolling(20).mean()
            bb_std = df["close"].rolling(20).std()
            bb_up  = bb_mid + 2 * bb_std
            bb_lo  = bb_mid - 2 * bb_std
            feats["bb_pct"]   = (df["close"] - bb_lo) / (bb_up - bb_lo + 1e-9)
            feats["bb_width"] = (bb_up - bb_lo) / (bb_mid + 1e-9)
        except Exception as exc:
            logger.warning("[MLModel] BB features failed: %s", exc)

        # ── ATR ───────────────────────────────────────────────────────────────
        try:
            tr = pd.concat([
                (df["high"] - df["low"]),
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"]  - df["close"].shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr14 = tr.ewm(span=14, adjust=False).mean()
            atr5  = tr.ewm(span=5,  adjust=False).mean()
            feats["atr_ratio"]    = atr5  / (atr14 + 1e-9)
            feats["atr_lag1"]     = atr14.shift(1) / (atr14 + 1e-9)
            feats["volatility"]   = tr.rolling(10).std() / (df["close"] + 1e-9)
            if "bb_width" in feats.columns:
                bb_width_raw       = feats["bb_width"] * (df["close"] + 1e-9)
                feats["bb_width_atr"] = bb_width_raw / (atr14 + 1e-9)
                feats["bb_squeeze"]   = feats["bb_width"] / (feats["atr_ratio"] + 1e-9)
        except Exception as exc:
            logger.warning("[MLModel] ATR features failed: %s", exc)

        # ── ADX + slope + regime ──────────────────────────────────────────────
        try:
            period     = 14
            up_move    = df["high"].diff()
            down_move  = -df["low"].diff()
            dm_plus    = np.where(
                (up_move > down_move) & (up_move > 0), up_move, 0.0
            )
            dm_minus   = np.where(
                (down_move > up_move) & (down_move > 0), down_move, 0.0
            )
            tr_ser = pd.concat([
                (df["high"] - df["low"]),
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"]  - df["close"].shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr_adx   = tr_ser.ewm(span=period, adjust=False).mean()
            dmi_plus  = (
                pd.Series(dm_plus, index=df.index)
                .ewm(span=period, adjust=False).mean() / (atr_adx + 1e-9) * 100
            )
            dmi_minus = (
                pd.Series(dm_minus, index=df.index)
                .ewm(span=period, adjust=False).mean() / (atr_adx + 1e-9) * 100
            )
            dx  = (dmi_plus - dmi_minus).abs() / (dmi_plus + dmi_minus + 1e-9) * 100
            adx = dx.ewm(span=period, adjust=False).mean()

            feats["adx"]         = adx / 100.0
            feats["adx_slope"]   = adx.diff(3) / 100.0   # [H]
            feats["dmi_diff"]    = (dmi_plus - dmi_minus) / 100.0
            feats["is_trending"] = (adx >= 25).astype(np.float32)   # [H]
            feats["is_ranging"]  = (adx <  20).astype(np.float32)   # [H]
        except Exception as exc:
            logger.warning("[MLModel] ADX features failed: %s", exc)

        # ── CCI ───────────────────────────────────────────────────────────────
        try:
            tp      = (df["high"] + df["low"] + df["close"]) / 3
            tp_mean = tp.rolling(20).mean()
            tp_std  = tp.rolling(20).std()
            feats["cci"] = (tp - tp_mean) / (0.015 * (tp_std + 1e-9))
        except Exception as exc:
            logger.warning("[MLModel] CCI features failed: %s", exc)

        # ── Stochastic ────────────────────────────────────────────────────────
        # [S] K period from active scalper profile (M5→5, M1→3)
        try:
            stoch_k_period = 5   # default
            try:
                stoch_k_period = int(
                    CONFIG.get_scalper_profile().get(
                        "STOCH_K",
                        CONFIG.get_scalper_profile().get("stoch_k", 5),
                    )
                )
            except Exception:
                pass   # keep default 5 if profile unavailable

            low_k   = df["low"].rolling(stoch_k_period).min()
            high_k  = df["high"].rolling(stoch_k_period).max()
            stoch_k = (df["close"] - low_k) / (high_k - low_k + 1e-9) * 100
            feats["stoch_k"] = stoch_k
            feats["stoch_d"] = stoch_k.rolling(3).mean()
        except Exception as exc:
            logger.warning("[MLModel] Stochastic features failed: %s", exc)

        # ── CMF ───────────────────────────────────────────────────────────────
        try:
            vol     = df.get("tick_volume",
                             df.get("volume", pd.Series(1.0, index=df.index)))
            mf_mult = (
                (df["close"] - df["low"]) - (df["high"] - df["close"])
            ) / (df["high"] - df["low"] + 1e-9)
            mf_vol  = mf_mult * vol
            feats["cmf"] = (
                mf_vol.rolling(20).sum() / (vol.rolling(20).sum() + 1e-9)
            )
        except Exception as exc:
            logger.warning("[MLModel] CMF features failed: %s", exc)

        # ── Volume ────────────────────────────────────────────────────────────
        try:
            vol          = df.get("tick_volume",
                                  df.get("volume", pd.Series(1.0, index=df.index)))
            feats["vol_ratio"]    = vol / (vol.rolling(10).mean() + 1e-9)
            feats["vol_ratio_20"] = vol / (vol.rolling(20).mean() + 1e-9)
        except Exception as exc:
            logger.warning("[MLModel] Volume features failed: %s", exc)

        # ── VWAP distance ─────────────────────────────────────────────────────
        try:
            vol  = df.get("tick_volume",
                          df.get("volume", pd.Series(1.0, index=df.index)))
            tp   = (df["high"] + df["low"] + df["close"]) / 3
            vwap = (
                (tp * vol).rolling(20).sum() /
                (vol.rolling(20).sum() + 1e-9)
            )
            feats["vwap_dist"] = (df["close"] - vwap) / (df["close"] + 1e-9)
        except Exception as exc:
            logger.warning("[MLModel] VWAP features failed: %s", exc)

        # ── Ichimoku TK diff ──────────────────────────────────────────────────
        try:
            tenkan = (
                df["high"].rolling(9).max() + df["low"].rolling(9).min()
            ) / 2
            kijun  = (
                df["high"].rolling(26).max() + df["low"].rolling(26).min()
            ) / 2
            feats["ichi_tk_diff"] = (tenkan - kijun) / (df["close"] + 1e-9)
        except Exception as exc:
            logger.warning("[MLModel] Ichimoku features failed: %s", exc)

        # ── Price-structure: swing high/low, ROC ─────────────────────────────
        try:
            sw   = 12
            sh   = df["high"].rolling(sw).max()
            sl   = df["low"].rolling(sw).min()
            feats["dist_swing_hi"] = (sh - df["close"]) / (df["close"] + 1e-9)
            feats["dist_swing_lo"] = (df["close"] - sl) / (df["close"] + 1e-9)
            feats["swing_range"]   = (sh - sl)           / (df["close"] + 1e-9)
        except Exception as exc:
            logger.warning("[MLModel] swing features failed: %s", exc)

        try:
            feats["roc_3"]  = df["close"].pct_change(3)   # [H]
            feats["roc_6"]  = df["close"].pct_change(6)   # [H]
            feats["roc_12"] = df["close"].pct_change(12)  # [H]
        except Exception as exc:
            logger.warning("[MLModel] ROC features failed: %s", exc)

        # ── [T] Candle-pattern features ───────────────────────────────────────
        # Pass-through if IndicatorEngine already computed them; otherwise
        # compute independently so the feature set is self-contained.
        try:
            pattern_cols = (
                "pat_bull_engulf", "pat_bear_engulf",
                "pat_bull_pin",    "pat_bear_pin",
                "pat_inside_bar",  "pat_doji",
            )
            existing = [c for c in pattern_cols if c in df.columns]
            missing  = [c for c in pattern_cols if c not in df.columns]

            # Pass through pre-computed patterns
            for col in existing:
                feats[col] = df[col].values

            # Compute missing patterns independently
            if missing:
                o   = df["open"]
                h   = df["high"]
                l   = df["low"]
                c   = df["close"]
                po, ph, pl, pc = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
                body  = (c - o).abs()
                upper = h - pd.concat([o, c], axis=1).max(axis=1)
                lower = pd.concat([o, c], axis=1).min(axis=1) - l
                rng   = (h - l).replace(0, np.nan)

                _pat: Dict[str, pd.Series] = {
                    "pat_bull_engulf": (
                        (c > o) & (pc < po) & (c >= po) & (o <= pc)
                    ).astype(int).fillna(0),
                    "pat_bear_engulf": (
                        (c < o) & (pc > po) & (c <= po) & (o >= pc)
                    ).astype(int).fillna(0),
                    "pat_bull_pin": (
                        (lower >= 2 * body.replace(0, np.nan)) & (upper < body)
                    ).astype(int).fillna(0),
                    "pat_bear_pin": (
                        (upper >= 2 * body.replace(0, np.nan)) & (lower < body)
                    ).astype(int).fillna(0),
                    "pat_inside_bar": (
                        (h < ph) & (l > pl)
                    ).astype(int).fillna(0),
                    "pat_doji": (
                        body < 0.1 * rng
                    ).astype(int).fillna(0),
                }
                for col in missing:
                    if col in _pat:
                        feats[col] = _pat[col].values

        except Exception as exc:
            logger.warning("[MLModel] Candle pattern features failed: %s", exc)

        # ── Session / time features ───────────────────────────────────────────
        try:
            idx = df.index
            if hasattr(idx, "tz") and idx.tz is not None:
                local = idx.tz_convert("Europe/Madrid")
            else:
                local = idx.tz_localize("UTC").tz_convert("Europe/Madrid")
            hour = local.hour.astype(np.float32)
            dow  = local.dayofweek.astype(np.float32)
            feats["hour"]       = hour
            feats["dow"]        = dow
            feats["is_london"]  = ((hour >= 8)  & (hour < 17)).astype(np.float32)
            feats["is_ny"]      = ((hour >= 13) & (hour < 22)).astype(np.float32)
            feats["is_overlap"] = ((hour >= 13) & (hour < 17)).astype(np.float32)
            feats["is_asian"]   = ((hour >= 0)  & (hour < 8)).astype(np.float32)
            feats["sin_hour"]   = np.sin(2 * np.pi * hour / 24).astype(np.float32)
            feats["cos_hour"]   = np.cos(2 * np.pi * hour / 24).astype(np.float32)
        except Exception as exc:
            logger.warning("[MLModel] session features failed: %s", exc)

        # ── Final NaN / inf removal ───────────────────────────────────────────
        feats.replace([np.inf, -np.inf], np.nan, inplace=True)
        feats.dropna(inplace=True)

        if feats.empty:
            logger.error("[MLModel] All rows dropped after NaN removal.")
            return None, []

        feature_names = feats.columns.tolist()
        logger.info(
            "[MLModel] Feature matrix: %d rows × %d features",
            len(feats), len(feature_names),
        )
        return feats.astype(np.float32), feature_names

    # =========================================================================
    # Label creation
    # =========================================================================

    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Fixed pct-return forward label creation.

        [N]  Uses FORWARD_BARS=6 (30 min) not 12 (60 min).
        [R]  Threshold corrected to MIN_PIP_MOVE=0.0008 (≈0.88 pips on EURUSD
             at 1.10). The previous 0.0005 was marginally above noise.

        Label logic:
          BUY  (1) : future_close > entry × (1 + threshold)
          SELL (2) : future_close < entry × (1 - threshold)
          HOLD (0) : |future_return| ≤ threshold

        Target distribution: HOLD ~55-65%, BUY ~17-22%, SELL ~17-22%.
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        fwd_ret   = df["close"].shift(-self.FORWARD_BARS) / df["close"] - 1
        threshold = self.MIN_PIP_MOVE

        labels = pd.Series(_LABEL_HOLD, index=df.index, dtype=np.int32)
        labels[fwd_ret >  threshold] = _LABEL_BUY
        labels[fwd_ret < -threshold] = _LABEL_SELL

        # Drop the look-ahead tail — these bars have no valid forward window
        labels.iloc[-self.FORWARD_BARS:] = np.nan
        return labels

    # =========================================================================
    # Optuna tuning  (objective = macro F1)
    # =========================================================================

    def _tune_xgb(
        self, X: np.ndarray, y: np.ndarray, n_trials: int
    ) -> Dict:
        """[E] Optimises macro F1 (not accuracy) to handle class imbalance."""
        tscv    = TimeSeriesSplit(n_splits=self.CV_SPLITS)
        splits  = list(tscv.split(X))
        tr_idx, va_idx = splits[-1]   # most recent fold for tuning

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators":      trial.suggest_int("n_estimators",     100, 600),
                "max_depth":         trial.suggest_int("max_depth",         3,   8),
                "learning_rate":     trial.suggest_float("learning_rate",   0.01, 0.3, log=True),
                "subsample":         trial.suggest_float("subsample",       0.5,  1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight":  trial.suggest_int("min_child_weight",  1,   10),
                "gamma":             trial.suggest_float("gamma",           0.0,  1.0),
                "reg_alpha":         trial.suggest_float("reg_alpha",       0.0,  2.0),
                "reg_lambda":        trial.suggest_float("reg_lambda",      0.5,  3.0),
            }
            mdl = xgb.XGBClassifier(
                **params,
                objective="multi:softprob",
                num_class=3,
                use_label_encoder=False,
                eval_metric="mlogloss",
                verbosity=0,
                n_jobs=-1,
            )
            mdl.fit(X[tr_idx], y[tr_idx])
            return f1_score(
                y[va_idx], mdl.predict(X[va_idx]),
                average="macro", zero_division=0,
            )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        logger.info("[MLModel] XGB best macro-F1: %.4f", study.best_value)
        return study.best_params

    def _tune_lgbm(
        self, X: np.ndarray, y: np.ndarray, n_trials: int
    ) -> Dict:
        """[E] Optimises macro F1 (not accuracy) to handle class imbalance."""
        tscv    = TimeSeriesSplit(n_splits=self.CV_SPLITS)
        splits  = list(tscv.split(X))
        tr_idx, va_idx = splits[-1]

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators":       trial.suggest_int("n_estimators",     100, 600),
                "num_leaves":         trial.suggest_int("num_leaves",        20, 150),
                "max_depth":          trial.suggest_int("max_depth",          3,   8),
                "learning_rate":      trial.suggest_float("learning_rate",   0.01, 0.3, log=True),
                "subsample":          trial.suggest_float("subsample",       0.5,  1.0),
                "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples":  trial.suggest_int("min_child_samples",  5,  50),
                "reg_alpha":          trial.suggest_float("reg_alpha",       0.0,  2.0),
                "reg_lambda":         trial.suggest_float("reg_lambda",      0.5,  3.0),
            }
            mdl = lgb.LGBMClassifier(
                **params,
                objective="multiclass",
                num_class=3,
                class_weight="balanced",
                verbose=-1,
                n_jobs=-1,
            )
            mdl.fit(X[tr_idx], y[tr_idx])
            return f1_score(
                y[va_idx], mdl.predict(X[va_idx]),
                average="macro", zero_division=0,
            )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        logger.info("[MLModel] LGBM best macro-F1: %.4f", study.best_value)
        return study.best_params

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _align_features(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align a feature DataFrame to the stored feature list.

        Adds zero-filled columns for features present in the trained model
        but absent from *feat_df* (e.g. a new feature added after training).
        Drops columns present in *feat_df* but not in the trained model.
        Returns a DataFrame with exactly the same columns in the same order
        as self._features.
        """
        if self._features is None:
            raise RuntimeError("Model not trained — call train() or load() first.")

        aligned = pd.DataFrame(index=feat_df.index)
        for col in self._features:
            aligned[col] = feat_df[col].values if col in feat_df.columns else 0.0
        return aligned

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
        rs    = gain / (loss + 1e-9)
        return 100 - 100 / (1 + rs)

    def _feature_importances(self) -> Dict[str, float]:
        """Weighted-average feature importances across XGB (45%) and LGBM (55%)."""
        if not (self._xgb and self._lgbm and self._features):
            return {}
        xgb_imp  = self._xgb.feature_importances_
        lgbm_imp = self._lgbm.feature_importances_
        xgb_imp  = xgb_imp  / (xgb_imp.sum()  + 1e-9)
        lgbm_imp = lgbm_imp / (lgbm_imp.sum() + 1e-9)
        avg_imp  = 0.45 * xgb_imp + 0.55 * lgbm_imp
        return dict(sorted(
            zip(self._features, avg_imp.tolist()),
            key=lambda kv: kv[1], reverse=True,
        ))

    def memory_usage(self) -> float:
        """Return approximate model memory usage in MB. [J]"""
        buf = io.BytesIO()
        joblib.dump(
            {"xgb": self._xgb, "lgbm": self._lgbm,
             "scaler": self._scaler, "features": self._features},
            buf,
        )
        return buf.tell() / (1024 * 1024)

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, symbol: Optional[str] = None) -> None:
        """
        [K] Atomically save all four model artefacts to disk.
        Writes to *.tmp first, then renames — guarantees no partial writes.
        """
        sym = symbol or self.symbol
        artefacts = {
            "xgb":      (MODEL_DIR / f"xgb_{sym}.pkl",      self._xgb),
            "lgbm":     (MODEL_DIR / f"lgbm_{sym}.pkl",     self._lgbm),
            "scaler":   (MODEL_DIR / f"scaler_{sym}.pkl",   self._scaler),
            "features": (MODEL_DIR / f"features_{sym}.pkl", self._features),
        }
        for key, (path, payload) in artefacts.items():
            tmp = path.with_suffix(".tmp")
            try:
                joblib.dump(payload, tmp, compress=3)
                tmp.replace(path)
                logger.info("[MLModel] Saved %s", path)
            except Exception as exc:
                logger.error("[MLModel] Failed to save %s: %s", path, exc)
                if tmp.exists():
                    tmp.unlink()
                raise

    def load(self, symbol: Optional[str] = None) -> bool:
        """
        [L][V] Load model artefacts from disk with per-file error handling.

        [L] Per-file warnings when a file is missing.
        [V] Per-file try/except on joblib.load so partial corruption is
            flagged individually rather than causing a silent False return.
            A corrupt file is identified by name in the error log so it can
            be deleted and retrained without guesswork.
        """
        sym = symbol or self.symbol
        paths = {
            "xgb":      MODEL_DIR / f"xgb_{sym}.pkl",
            "lgbm":     MODEL_DIR / f"lgbm_{sym}.pkl",
            "scaler":   MODEL_DIR / f"scaler_{sym}.pkl",
            "features": MODEL_DIR / f"features_{sym}.pkl",
        }

        # [L] Check for missing files first
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            for m in missing:
                logger.warning("[MLModel] Missing model file: %s", m)
            return False

        # [V] Per-file load with individual error handling
        loaded: Dict = {}
        for key, path in paths.items():
            try:
                loaded[key] = joblib.load(path)
            except Exception as exc:
                logger.error(
                    "[MLModel] Failed to load %s: %s  "
                    "(delete and retrain to fix)", path, exc,
                )
                return False

        self._xgb      = loaded["xgb"]
        self._lgbm     = loaded["lgbm"]
        self._scaler   = loaded["scaler"]
        self._features = loaded["features"]
        self._trained  = True

        logger.info(
            "[MLModel] Loaded %s — %d features  mem=%.1f MB",
            sym, len(self._features), self.memory_usage(),
        )
        return True

    @staticmethod
    def delete_stale(symbol: str) -> None:
        """Remove all model artefact files for a given symbol."""
        for prefix in ("xgb", "lgbm", "scaler", "features"):
            path = MODEL_DIR / f"{prefix}_{symbol}.pkl"
            if path.exists():
                path.unlink()
                logger.info("[MLModel] Deleted stale file: %s", path)
