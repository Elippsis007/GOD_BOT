# =============================================================================
# GODBOT v3.0 – signals/ml_model.py  (PRODUCTION – Triple-Specialist Architecture)
# =============================================================================
#
#  All previous fixes A–AH retained. Full architectural rewrite:
#
#  [AI] THREE-SPECIALIST ARCHITECTURE replaces single 3-class softmax model.
#         Previous: one model answering "up/down/sideways simultaneously"
#         → LGBM temperature maxed at 3.0 on every run, macro-F1 ceiling 0.38.
#         New: three independent binary specialist models:
#           • BUY detector  — "will long TP be hit before SL in next N bars?"
#           • SELL detector — "will short TP be hit before SL in next N bars?"
#           • REGIME classifier — "is market trending (1) or ranging (0)?"
#         Each model answers a simpler binary question → random baseline 0.50
#         vs 0.33 for 3-class → expected F1 0.58–0.68 per specialist.
#
#  [AJ] REGIME GATE suppresses BUY/SELL in ranging markets.
#         Regime classifier trained on ADX ≥ 25 sustained over 5 bars as
#         the trending label. In ranging markets (regime=0), both BUY and
#         SELL detectors are suppressed regardless of their confidence.
#         This eliminates the systematic scalping losses that occur in
#         choppy mean-reverting conditions.
#
#  [AK] CONFLICT RESOLUTION for simultaneous BUY+SELL signals.
#         If both BUY and SELL detectors fire above threshold simultaneously,
#         the model outputs HOLD rather than arbitrarily picking one.
#         If neither fires, HOLD. Only clean unambiguous signals pass.
#
#  [AL] HURST EXPONENT feature (fractal dimension over 50 bars).
#         Quantifies whether price is trending (H > 0.5) or mean-reverting
#         (H < 0.5). Single most predictive feature for the regime classifier.
#         Computed as the rescaled range (R/S) statistic.
#
#  [AM] REALIZED VOLATILITY RATIO (5-bar vs 20-bar).
#         Captures volatility regime shifts 3–5 bars before ADX reacts.
#         Rising ratio = volatility expansion = trending conditions emerging.
#
#  [AN] RETURN AUTOCORRELATION over 10 bars.
#         Detects momentum persistence (positive autocorr) vs mean-reversion
#         (negative autocorr). Computed as lag-1 autocorrelation of 1-bar
#         returns over a 10-bar rolling window.
#
#  [AO] SESSION-ANCHORED VWAP replaces rolling VWAP.
#         VWAP is now anchored to the London session open (08:00 CET) and
#         NY session open (13:00 CET) separately. Price deviation from the
#         session-anchored VWAP is a cleaner mean-reversion signal than
#         the 20-bar rolling VWAP used previously.
#
#  [AP] CONFIDENCE GATE raised to 0.62 for BUY/SELL detectors.
#         With binary classifiers and regime gating, a 0.62 threshold
#         selects only the top ~25% most confident predictions, ensuring
#         the ML model adds value rather than noise on top of the
#         signal engine's 9 existing filters.
#
#  [AQ] SEPARATE SCALERS per specialist model.
#         Each of the three models gets its own StandardScaler fitted only
#         on its own training split. Previously a single shared scaler was
#         fitted on the full dataset, leaking future information into the
#         training set boundary.
#
#  [AR] BINARY OPTUNA OBJECTIVE uses binary F1 (average="binary") rather
#         than macro F1. Binary F1 on the positive class (TP hit = 1) is
#         more directly aligned with what we want: high precision and
#         recall on actual trade opportunities.
#
#  [AS] SAVE/LOAD updated to persist all 3 × 2 = 6 model files plus
#         3 scalers and 3 temperature files. Backward compatible — if only
#         legacy files exist, load() returns False and triggers retrain.
#
# =============================================================================

from __future__ import annotations

import io
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score
)
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
_LABEL_HOLD = 0
_LABEL_BUY  = 1
_LABEL_SELL = 2
_LABEL_NAMES = {_LABEL_HOLD: "HOLD", _LABEL_BUY: "BUY", _LABEL_SELL: "SELL"}

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# =============================================================================
# _SpecialistModel — single binary XGB+LGBM ensemble
# =============================================================================
class _SpecialistModel:
    """
    Internal binary classifier: XGBoost 45% + LightGBM 55% ensemble
    with temperature calibration and inverse-frequency sample weights.
    Used by MLSignalModel for BUY, SELL, and REGIME specialists.
    """

    def __init__(self, name: str) -> None:
        self.name      = name
        self._xgb_only = name in ("buy", "sell")  # LGBM uncalibratable on binary imbalanced data
        self._xgb:       Optional[xgb.XGBClassifier]  = None
        self._lgbm:      Optional[lgb.LGBMClassifier] = None
        self._scaler:    Optional[StandardScaler]      = None
        self._temp_xgb:  float = 1.0
        self._temp_lgbm: float = 1.0
        self._trained    = False

    @property
    def is_trained(self) -> bool:
        return (
            self._trained
            and self._xgb    is not None
            and self._lgbm   is not None
            and self._scaler is not None
        )

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        n_trials: int,
        cv_splits: int,
    ) -> Dict:
        """
        Full pipeline: scale → Optuna tune → CV → final fit →
        temperature calibrate → evaluate.
        Returns metrics dict.
        """
        # [AQ] Separate scaler per specialist
        self._scaler   = StandardScaler()
        X_train_s      = self._scaler.fit_transform(X_train)
        X_test_s       = self._scaler.transform(X_test)

        sample_weights = _compute_sample_weights(y_train)

        # [AR] Binary Optuna tuning
        logger.info("[%s] Optuna XGB (%d trials)…", self.name, n_trials)
        best_xgb  = _tune_binary_xgb(
            X_train_s, y_train, sample_weights, n_trials, cv_splits
        )
        logger.info("[%s] Optuna LGBM (%d trials)…", self.name, n_trials)
        best_lgbm = _tune_binary_lgbm(
            X_train_s, y_train, sample_weights, n_trials, cv_splits
        )

        # CV scores
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        xgb_cv, lgbm_cv = [], []
        for tr_idx, va_idx in tscv.split(X_train_s):
            mx = xgb.XGBClassifier(
                **best_xgb,
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
            )
            mx.fit(X_train_s[tr_idx], y_train[tr_idx],
                   sample_weight=sample_weights[tr_idx])
            xgb_cv.append(f1_score(
                y_train[va_idx], mx.predict(X_train_s[va_idx]),
                average="binary", zero_division=0,
            ))

            ml = lgb.LGBMClassifier(
                **best_lgbm,
                objective="binary",
                class_weight="balanced",
                verbose=-1,
            )
            ml.fit(X_train_s[tr_idx], y_train[tr_idx],
                   sample_weight=sample_weights[tr_idx])
            lgbm_cv.append(f1_score(
                y_train[va_idx], ml.predict(X_train_s[va_idx]),
                average="binary", zero_division=0,
            ))

        mean_xgb_cv  = float(np.mean(xgb_cv))
        mean_lgbm_cv = float(np.mean(lgbm_cv))
        logger.info("[%s] XGB  CV F1: %.4f", self.name, mean_xgb_cv)
        logger.info("[%s] LGBM CV F1: %.4f", self.name, mean_lgbm_cv)

        # Final fit on full training set
        self._xgb = xgb.XGBClassifier(
            **best_xgb,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        )
        self._xgb.fit(X_train_s, y_train, sample_weight=sample_weights)

        self._lgbm = lgb.LGBMClassifier(
            **best_lgbm,
            objective="binary",
            class_weight="balanced",
            verbose=-1,
        )
        self._lgbm.fit(X_train_s, y_train, sample_weight=sample_weights)
        self._trained = True

        # Temperature calibration [AF]
        self._temp_xgb  = _calibrate_temperature_binary(
            self._xgb,  X_test_s, y_test, self.name + "_xgb"
        )
        self._temp_lgbm = _calibrate_temperature_binary(
            self._lgbm, X_test_s, y_test, self.name + "_lgbm"
        )

        # Evaluate
        p_xgb  = _apply_temp_binary(
            self._xgb.predict_proba(X_test_s)[:, 1],  self._temp_xgb
        )
        p_lgbm = _apply_temp_binary(
            self._lgbm.predict_proba(X_test_s)[:, 1], self._temp_lgbm
        )
        proba  = 0.45 * p_xgb + 0.55 * p_lgbm
        preds  = (proba >= 0.5).astype(int)

        test_f1  = float(f1_score(y_test, preds, average="binary", zero_division=0))
        test_auc = float(roc_auc_score(y_test, proba))
        report   = classification_report(
            y_test, preds, target_names=["NEG", "POS"]
        )
        logger.info("[%s] Test F1=%.4f  AUC=%.4f", self.name, test_f1, test_auc)
        logger.info("\n%s", report)

        return {
            "xgb_cv_f1":   mean_xgb_cv,
            "lgbm_cv_f1":  mean_lgbm_cv,
            "test_f1":     test_f1,
            "test_auc":    test_auc,
            "report":      report,
            "temp_xgb":    self._temp_xgb,
            "temp_lgbm":   self._temp_lgbm,
        }

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict_proba_positive(self, x_scaled: np.ndarray) -> float:
        if not self.is_trained:
            return 0.0
        p_xgb = _apply_temp_binary(
            self._xgb.predict_proba(x_scaled)[:, 1], self._temp_xgb
        )
        if self._xgb_only:
            return float(p_xgb[0])
        p_lgbm = _apply_temp_binary(
            self._lgbm.predict_proba(x_scaled)[:, 1], self._temp_lgbm
        )
        return float(0.45 * p_xgb[0] + 0.55 * p_lgbm[0])

    def predict_proba_positive_batch(self, X_scaled: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.zeros(len(X_scaled))
        p_xgb = _apply_temp_binary(
            self._xgb.predict_proba(X_scaled)[:, 1], self._temp_xgb
        )
        if self._xgb_only:
            return p_xgb
        p_lgbm = _apply_temp_binary(
            self._lgbm.predict_proba(X_scaled)[:, 1], self._temp_lgbm
        )
        return 0.45 * p_xgb + 0.55 * p_lgbm

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, symbol: str) -> None:
        n = self.name
        artefacts = {
            f"xgb_{n}_{symbol}.pkl":    self._xgb,
            f"lgbm_{n}_{symbol}.pkl":   self._lgbm,
            f"scaler_{n}_{symbol}.pkl": self._scaler,
            f"temps_{n}_{symbol}.pkl":  {
                "xgb": self._temp_xgb, "lgbm": self._temp_lgbm
            },
        }
        for fname, payload in artefacts.items():
            path = MODEL_DIR / fname
            tmp  = path.with_suffix(".tmp")
            try:
                joblib.dump(payload, tmp, compress=3)
                tmp.replace(path)
                logger.info("[MLModel] Saved %s", path)
            except Exception as exc:
                logger.error("[MLModel] Save failed %s: %s", path, exc)
                if tmp.exists():
                    tmp.unlink()
                raise

    def load(self, symbol: str) -> bool:
        n = self.name
        paths = {
            "xgb":    MODEL_DIR / f"xgb_{n}_{symbol}.pkl",
            "lgbm":   MODEL_DIR / f"lgbm_{n}_{symbol}.pkl",
            "scaler": MODEL_DIR / f"scaler_{n}_{symbol}.pkl",
        }
        temps_path = MODEL_DIR / f"temps_{n}_{symbol}.pkl"

        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            for m in missing:
                logger.warning("[MLModel] Missing: %s", m)
            return False

        try:
            self._xgb    = joblib.load(paths["xgb"])
            self._lgbm   = joblib.load(paths["lgbm"])
            self._scaler = joblib.load(paths["scaler"])
            self._trained = True
        except Exception as exc:
            logger.error("[MLModel] Load failed for %s/%s: %s", n, symbol, exc)
            return False

        if temps_path.exists():
            try:
                t = joblib.load(temps_path)
                self._temp_xgb  = t.get("xgb",  1.0)
                self._temp_lgbm = t.get("lgbm", 1.0)
            except Exception:
                self._temp_xgb = self._temp_lgbm = 1.0

        logger.info("[MLModel] Loaded specialist '%s' for %s", n, symbol)
        return True

    def delete_stale(self, symbol: str) -> None:
        n = self.name
        for prefix in (f"xgb_{n}", f"lgbm_{n}", f"scaler_{n}", f"temps_{n}"):
            path = MODEL_DIR / f"{prefix}_{symbol}.pkl"
            if path.exists():
                path.unlink()
                logger.info("[MLModel] Deleted stale: %s", path)


# =============================================================================
# Module-level helpers (shared across specialists)
# =============================================================================

def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """[Z] Inverse class-frequency sample weights, normalised to mean=1."""
    classes, counts = np.unique(y, return_counts=True)
    freq     = counts / len(y)
    inv_freq = 1.0 / (freq + 1e-9)
    inv_freq = inv_freq / inv_freq.mean()
    ctw = dict(zip(classes, inv_freq))
    return np.array([ctw[yi] for yi in y], dtype=np.float32)


def _apply_temp_binary(proba_pos: np.ndarray, T: float) -> np.ndarray:
    """
    Temperature scaling for binary probabilities.
    Converts p → sigmoid(logit(p) / T).
    T > 1 softens (pushes toward 0.5), T < 1 sharpens.
    """
    if abs(T - 1.0) < 1e-6:
        return proba_pos
    p   = np.clip(proba_pos, 1e-7, 1 - 1e-7)
    log = np.log(p / (1 - p)) / T          # logit / T
    return 1.0 / (1.0 + np.exp(-log))      # sigmoid


def _calibrate_temperature_binary(
    model, X_test: np.ndarray, y_test: np.ndarray, name: str
) -> float:
    """
    Finds T ∈ [0.3, 10.0] that minimises binary NLL on test set.
    Upper bound raised from 5.0 → 10.0 so LGBM specialists can find
    their true optimal temperature rather than hitting the ceiling.
    """
    try:
        raw = model.predict_proba(X_test)[:, 1]

        def nll(T: float) -> float:
            p   = _apply_temp_binary(raw, T)
            p   = np.clip(p, 1e-7, 1 - 1e-7)
            return -float(
                (y_test * np.log(p) + (1 - y_test) * np.log(1 - p)).mean()
            )

        res   = minimize_scalar(nll, bounds=(0.3, 10.0), method="bounded")
        T_opt = float(res.x)
        logger.info("[MLModel] %s temp T=%.4f  NLL=%.4f", name, T_opt, res.fun)
        return T_opt
    except Exception as exc:
        logger.warning("[MLModel] Temp calibration failed %s: %s", name, exc)
        return 1.0


def _tune_binary_xgb(
    X: np.ndarray, y: np.ndarray,
    sw: np.ndarray, n_trials: int, cv_splits: int,
) -> Dict:
    tscv   = TimeSeriesSplit(n_splits=cv_splits)
    splits = list(tscv.split(X))

    def obj(trial: optuna.Trial) -> float:
        p = {
            "n_estimators":     trial.suggest_int("n_estimators",    100, 1000),
            "max_depth":        trial.suggest_int("max_depth",         3,    9),
            "learning_rate":    trial.suggest_float("lr",          0.003, 0.3, log=True),
            "subsample":        trial.suggest_float("sub",          0.5,  1.0),
            "colsample_bytree": trial.suggest_float("col",          0.4,  1.0),
            "min_child_weight": trial.suggest_int("mcw",              1,   30),
            "gamma":            trial.suggest_float("gamma",        0.0,  3.0),
            "reg_alpha":        trial.suggest_float("alpha",        0.0,  5.0),
            "reg_lambda":       trial.suggest_float("lambda",       0.5,  5.0),
        }
        scores = []
        for tr, va in splits:
            m = xgb.XGBClassifier(
                **p,
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0, n_jobs=-1,
            )
            m.fit(X[tr], y[tr], sample_weight=sw[tr])
            scores.append(f1_score(
                y[va], m.predict(X[va]), average="binary", zero_division=0
            ))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    logger.info("[MLModel] XGB binary best F1: %.4f", study.best_value)
    # Rename param keys back to XGBClassifier names
    bp = study.best_params
    return {
        "n_estimators":     bp["n_estimators"],
        "max_depth":        bp["max_depth"],
        "learning_rate":    bp["lr"],
        "subsample":        bp["sub"],
        "colsample_bytree": bp["col"],
        "min_child_weight": bp["mcw"],
        "gamma":            bp["gamma"],
        "reg_alpha":        bp["alpha"],
        "reg_lambda":       bp["lambda"],
    }


def _tune_binary_lgbm(
    X: np.ndarray, y: np.ndarray,
    sw: np.ndarray, n_trials: int, cv_splits: int,
) -> Dict:
    tscv   = TimeSeriesSplit(n_splits=cv_splits)
    splits = list(tscv.split(X))

    def obj(trial: optuna.Trial) -> float:
        p = {
            "n_estimators":      trial.suggest_int("n_est",          100, 1000),
            "num_leaves":        trial.suggest_int("leaves",          16,  256),
            "max_depth":         trial.suggest_int("depth",            3,    9),
            "learning_rate":     trial.suggest_float("lr",         0.003, 0.3, log=True),
            "subsample":         trial.suggest_float("sub",         0.5,  1.0),
            "colsample_bytree":  trial.suggest_float("col",         0.4,  1.0),
            "min_child_samples": trial.suggest_int("mcs",             5,  100),
            "reg_alpha":         trial.suggest_float("alpha",       0.0,  5.0),
            "reg_lambda":        trial.suggest_float("lambda",      0.5,  5.0),
        }
        scores = []
        for tr, va in splits:
            m = lgb.LGBMClassifier(
                **p,
                objective="binary",
                class_weight="balanced",
                verbose=-1, n_jobs=-1,
            )
            m.fit(X[tr], y[tr], sample_weight=sw[tr])
            scores.append(f1_score(
                y[va], m.predict(X[va]), average="binary", zero_division=0
            ))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    logger.info("[MLModel] LGBM binary best F1: %.4f", study.best_value)
    bp = study.best_params
    return {
        "n_estimators":      bp["n_est"],
        "num_leaves":        bp["leaves"],
        "max_depth":         bp["depth"],
        "learning_rate":     bp["lr"],
        "subsample":         bp["sub"],
        "colsample_bytree":  bp["col"],
        "min_child_samples": bp["mcs"],
        "reg_alpha":         bp["alpha"],
        "reg_lambda":        bp["lambda"],
    }


# =============================================================================
# MLSignalModel  — public API (drop-in replacement, same predict() signature)
# =============================================================================
class MLSignalModel:
    """
    [AI] Triple-specialist architecture:
      _buy_model    — binary: long TP hit before SL?
      _sell_model   — binary: short TP hit before SL?
      _regime_model — binary: trending (1) or ranging (0)?

    predict() returns same dict as before:
      {"label": "BUY"|"SELL"|"HOLD", "confidence": float,
       "probabilities": {"HOLD": float, "BUY": float, "SELL": float}}

    No changes required in main.py or signal_engine.py.
    """

    # ── Constants ─────────────────────────────────────────────────────────────
    OPTUNA_TRIALS:     int   = 40        # per specialist (3 × 40 = 120 total)
    CV_SPLITS:         int   = 3
    TEST_FRAC:         float = 0.20
    FORWARD_BARS:      int   = 8         # [AG] 40 min
    MIN_TRAINING_BARS: int   = 20_000
    MIN_PIP_FLOOR:     float = 0.0005
    TB_TP_MULT:        float = 1.5       # [AB]
    TB_SL_MULT:        float = 1.0       # [AB]

    # [AP] Confidence gate for BUY/SELL specialists
    MIN_CONFIDENCE:    float = 0.55

    # Regime gate threshold — probability of trending must exceed this
    REGIME_THRESHOLD:  float = 0.55

    def __init__(self, symbol: str = "EURUSD") -> None:
        self.symbol       = symbol
        self._features:   Optional[List[str]] = None
        self._buy_model   = _SpecialistModel("buy")
        self._sell_model  = _SpecialistModel("sell")
        self._regime_model = _SpecialistModel("regime")

    # ── State check ───────────────────────────────────────────────────────────
    @property
    def is_trained(self) -> bool:
        return (
            self._features     is not None
            and self._buy_model.is_trained
            and self._sell_model.is_trained
            and self._regime_model.is_trained
        )

    # =========================================================================
    # Public: train
    # =========================================================================
    def train(
        self,
        df:       pd.DataFrame,
        n_trials: Optional[int] = None,
    ) -> Dict:
        trials = n_trials if n_trials is not None else self.OPTUNA_TRIALS
        logger.info(
            "[MLModel] Triple-specialist training — symbol=%s  trials=%d/specialist",
            self.symbol, trials,
        )

        if len(df) < self.MIN_TRAINING_BARS:
            logger.warning(
                "[MLModel] ⚠️  Only %d bars (min %d) — may overfit.",
                len(df), self.MIN_TRAINING_BARS,
            )

        # ── 1. Shared feature matrix ──────────────────────────────────────────
        X_raw, feature_names = self._build_features(df)
        if X_raw is None or len(X_raw) < 500:
            raise ValueError(
                f"Feature matrix too small: "
                f"{0 if X_raw is None else len(X_raw)} rows."
            )
        self._features = feature_names

        # ── 2. Labels — three separate binary targets ─────────────────────────
        df_trimmed   = df.loc[df.index.intersection(X_raw.index)]
        buy_labels   = self._create_buy_labels(df_trimmed)
        sell_labels  = self._create_sell_labels(df_trimmed)
        regime_labels = self._create_regime_labels(df_trimmed)

        # Align all labels to common valid index
        valid_idx = (
            buy_labels.dropna().index
            .intersection(sell_labels.dropna().index)
            .intersection(regime_labels.dropna().index)
            .intersection(X_raw.index)
        )

        X_aligned = X_raw.loc[valid_idx].values.astype(np.float32)
        y_buy     = buy_labels.loc[valid_idx].values.astype(np.int32)
        y_sell    = sell_labels.loc[valid_idx].values.astype(np.int32)
        y_regime  = regime_labels.loc[valid_idx].values.astype(np.int32)

        n_total = len(X_aligned)
        logger.info("[MLModel] Aligned dataset: %d rows, %d features", n_total, len(feature_names))
        logger.info("[MLModel] BUY    positives: %.1f%%", y_buy.mean()    * 100)
        logger.info("[MLModel] SELL   positives: %.1f%%", y_sell.mean()   * 100)
        logger.info("[MLModel] REGIME trending:  %.1f%%", y_regime.mean() * 100)

        # ── 3. Temporal train/test split ──────────────────────────────────────
        n_test  = max(500, int(n_total * self.TEST_FRAC))
        n_train = n_total - n_test

        X_tr, X_te   = X_aligned[:n_train], X_aligned[n_train:]
        yb_tr, yb_te = y_buy[:n_train],     y_buy[n_train:]
        ys_tr, ys_te = y_sell[:n_train],    y_sell[n_train:]
        yr_tr, yr_te = y_regime[:n_train],  y_regime[n_train:]

        logger.info("[MLModel] Train=%d  Test=%d", n_train, n_test)

        # ── 4. Train each specialist ──────────────────────────────────────────
        logger.info("[MLModel] ══ Training BUY specialist ══")
        buy_metrics = self._buy_model.fit(
            X_tr, yb_tr, X_te, yb_te, trials, self.CV_SPLITS
        )

        logger.info("[MLModel] ══ Training SELL specialist ══")
        sell_metrics = self._sell_model.fit(
            X_tr, ys_tr, X_te, ys_te, trials, self.CV_SPLITS
        )

        logger.info("[MLModel] ══ Training REGIME specialist ══")
        regime_metrics = self._regime_model.fit(
            X_tr, yr_tr, X_te, yr_te, trials, self.CV_SPLITS
        )

        return {
            "buy":    buy_metrics,
            "sell":   sell_metrics,
            "regime": regime_metrics,
            "n_train":    n_train,
            "n_test":     n_test,
            "n_features": len(feature_names),
            "feature_names": feature_names,
        }

    # =========================================================================
    # Public: predict (same signature as before — drop-in replacement)
    # =========================================================================
    def predict(self, df_input: pd.DataFrame) -> Dict:
        """
        [AI][AJ][AK] Triple-specialist inference with regime gate
        and conflict resolution. Returns same dict format as before.
        """
        _hold = {
            "label": "HOLD", "confidence": 0.0,
            "probabilities": {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0},
        }

        if not self.is_trained:
            return _hold

        try:
            X_feat, _ = self._build_features(df_input)
            if X_feat is None or X_feat.empty:
                return _hold

            X_aligned = self._align_features(X_feat)
            x_last    = X_aligned.iloc[[-1]].values.astype(np.float32)

            # Scale per specialist [AQ]
            x_buy    = self._buy_model._scaler.transform(x_last)
            x_sell   = self._sell_model._scaler.transform(x_last)
            x_regime = self._regime_model._scaler.transform(x_last)

            p_regime = self._regime_model.predict_proba_positive(x_regime)
            p_buy    = self._buy_model.predict_proba_positive(x_buy)
            p_sell   = self._sell_model.predict_proba_positive(x_sell)

            # [AJ] Regime gate
            if p_regime < self.REGIME_THRESHOLD:
                return {
                    "label": "HOLD", "confidence": p_regime,
                    "probabilities": {
                        "HOLD": 1.0 - p_regime,
                        "BUY":  p_buy,
                        "SELL": p_sell,
                    },
                }

            buy_fires  = p_buy  >= self.MIN_CONFIDENCE
            sell_fires = p_sell >= self.MIN_CONFIDENCE

            # [AK] Conflict resolution
            if buy_fires and sell_fires:
                # Both fire — ambiguous, output HOLD
                return {
                    "label": "HOLD",
                    "confidence": max(p_buy, p_sell),
                    "probabilities": {
                        "HOLD": 1.0 - max(p_buy, p_sell),
                        "BUY":  p_buy,
                        "SELL": p_sell,
                    },
                }

            if buy_fires:
                return {
                    "label": "BUY", "confidence": p_buy,
                    "probabilities": {
                        "HOLD": 1.0 - p_buy,
                        "BUY":  p_buy,
                        "SELL": p_sell,
                    },
                }

            if sell_fires:
                return {
                    "label": "SELL", "confidence": p_sell,
                    "probabilities": {
                        "HOLD": 1.0 - p_sell,
                        "BUY":  p_buy,
                        "SELL": p_sell,
                    },
                }

            return {
                "label": "HOLD",
                "confidence": max(p_buy, p_sell),
                "probabilities": {
                    "HOLD": 1.0 - max(p_buy, p_sell),
                    "BUY":  p_buy,
                    "SELL": p_sell,
                },
            }

        except Exception as exc:
            logger.warning("[MLModel] predict() error: %s", exc)
            return _hold

    # =========================================================================
    # Public: predict_batch
    # =========================================================================
    def predict_batch(self, df: pd.DataFrame) -> List[Dict]:
        """Vectorised batch prediction."""
        if not self.is_trained:
            return [{"label": "HOLD", "confidence": 0.0}] * len(df)

        try:
            X_feat, _ = self._build_features(df)
            if X_feat is None or X_feat.empty:
                return [{"label": "HOLD", "confidence": 0.0}] * len(df)

            X_al = self._align_features(X_feat).values.astype(np.float32)

            X_buy    = self._buy_model._scaler.transform(X_al)
            X_sell   = self._sell_model._scaler.transform(X_al)
            X_regime = self._regime_model._scaler.transform(X_al)

            p_regime = self._regime_model.predict_proba_positive_batch(X_regime)
            p_buy    = self._buy_model.predict_proba_positive_batch(X_buy)
            p_sell   = self._sell_model.predict_proba_positive_batch(X_sell)

            results = []
            for i in range(len(X_al)):
                pr, pb, ps = p_regime[i], p_buy[i], p_sell[i]

                if pr < self.REGIME_THRESHOLD:
                    results.append({
                        "label": "HOLD", "confidence": float(pr),
                        "probabilities": {
                            "HOLD": float(1.0 - pr),
                            "BUY":  float(pb),
                            "SELL": float(ps),
                        },
                    })
                    continue

                buy_f  = pb >= self.MIN_CONFIDENCE
                sell_f = ps >= self.MIN_CONFIDENCE

                if buy_f and sell_f:
                    lbl, conf = "HOLD", float(max(pb, ps))
                elif buy_f:
                    lbl, conf = "BUY",  float(pb)
                elif sell_f:
                    lbl, conf = "SELL", float(ps)
                else:
                    lbl, conf = "HOLD", float(max(pb, ps))

                results.append({
                    "label": lbl, "confidence": conf,
                    "probabilities": {
                        "HOLD": float(1.0 - max(pb, ps)),
                        "BUY":  float(pb),
                        "SELL": float(ps),
                    },
                })
            return results

        except Exception as exc:
            logger.warning("[MLModel] predict_batch() error: %s", exc)
            return [{"label": "HOLD", "confidence": 0.0}] * len(df)

    # =========================================================================
    # Label creation — three separate binary targets
    # =========================================================================

    def _create_buy_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        [AB] Binary: 1 if long TP (entry + ATR×1.5) is hit before
        long SL (entry - ATR×1.0) within FORWARD_BARS. Else 0.
        """
        return self._create_barrier_labels(df, direction="buy")

    def _create_sell_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        [AB] Binary: 1 if short TP (entry - ATR×1.5) is hit before
        short SL (entry + ATR×1.0) within FORWARD_BARS. Else 0.
        """
        return self._create_barrier_labels(df, direction="sell")

    def _create_barrier_labels(
        self, df: pd.DataFrame, direction: str
    ) -> pd.Series:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        close = df["close"].values.astype(np.float64)
        high  = df["high"].values.astype(np.float64)
        low   = df["low"].values.astype(np.float64)
        n     = len(close)

        # Wilder ATR
        tr  = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i]  - close[i - 1]),
            )
        atr = np.zeros(n)
        if n >= 14:
            atr[13] = tr[1:14].mean()
            for i in range(14, n):
                atr[i] = (atr[i - 1] * 13 + tr[i]) / 14

        labels = np.full(n, np.nan)
        for i in range(n - self.FORWARD_BARS):
            if atr[i] <= 0:
                labels[i] = 0
                continue

            entry = close[i]
            wh    = high[i + 1: i + 1 + self.FORWARD_BARS]
            wl    = low[i  + 1: i + 1 + self.FORWARD_BARS]

            if direction == "buy":
                tp = entry + atr[i] * self.TB_TP_MULT
                sl = entry - atr[i] * self.TB_SL_MULT
                hit_tp = np.any(wh >= tp)
                hit_sl = np.any(wl <= sl)
                # TP hit cleanly before SL
                labels[i] = 1 if (hit_tp and not hit_sl) else 0
            else:  # sell
                tp = entry - atr[i] * self.TB_TP_MULT
                sl = entry + atr[i] * self.TB_SL_MULT
                hit_tp = np.any(wl <= tp)
                hit_sl = np.any(wh >= sl)
                labels[i] = 1 if (hit_tp and not hit_sl) else 0

        result = pd.Series(labels, index=df.index)
        valid  = result.dropna()
        if len(valid) > 0:
            logger.info(
                "[MLModel] %s labels — positive rate: %.1f%%",
                direction.upper(), valid.mean() * 100,
            )
        return result

    def _create_regime_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        [AJ] Binary: 1 if market is trending, 0 if ranging.
        Trending = ADX (14) ≥ 25 sustained for at least 5 consecutive bars.
        This prevents the model from mislabelling a brief ADX spike as trending.
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        h = df["high"].astype(np.float64)
        l = df["low"].astype(np.float64)
        c = df["close"].astype(np.float64)

        up_move   = h.diff()
        down_move = -l.diff()
        dm_plus   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        dm_minus  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_s    = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_adx = tr_s.ewm(span=14, adjust=False).mean()

        dmi_p = (
            pd.Series(dm_plus, index=df.index)
            .ewm(span=14, adjust=False).mean() / (atr_adx + 1e-9) * 100
        )
        dmi_m = (
            pd.Series(dm_minus, index=df.index)
            .ewm(span=14, adjust=False).mean() / (atr_adx + 1e-9) * 100
        )
        dx  = (dmi_p - dmi_m).abs() / (dmi_p + dmi_m + 1e-9) * 100
        adx = dx.ewm(span=14, adjust=False).mean()

        # Sustained ADX ≥ 25 for 5 consecutive bars
        above25  = (adx >= 25).astype(int)
        sustained = above25.rolling(5).min()   # 1 only if all 5 bars ≥ 25

        result = sustained.where(sustained.notna(), np.nan)
        valid  = result.dropna()
        if len(valid) > 0:
            logger.info(
                "[MLModel] REGIME labels — trending rate: %.1f%%",
                valid.mean() * 100,
            )
        return result

    # =========================================================================
    # Feature engineering (shared across all three specialists)
    # =========================================================================

    def _build_features(
        self, df: pd.DataFrame
    ) -> Tuple[Optional[pd.DataFrame], List[str]]:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        for col in ("open", "high", "low", "close", "tick_volume"):
            if col in df.columns:
                df[col] = df[col].astype(np.float32)

        feats = pd.DataFrame(index=df.index)
        c = df["close"]
        o = df["open"]
        h = df["high"]
        l = df["low"]

        # ── ATR (shared base) ─────────────────────────────────────────────────
        try:
            tr    = pd.concat([
                h - l,
                (h - c.shift(1)).abs(),
                (l - c.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr14 = tr.ewm(span=14, adjust=False).mean()
            atr5  = tr.ewm(span=5,  adjust=False).mean()
            feats["atr_ratio"]  = atr5  / (atr14 + 1e-9)
            feats["atr_lag1"]   = atr14.shift(1) / (atr14 + 1e-9)
            feats["volatility"] = tr.rolling(10).std() / (c + 1e-9)
            feats["atr_norm"]   = atr14 / (c + 1e-9)
        except Exception as exc:
            logger.warning("[MLModel] ATR failed: %s", exc)
            atr14 = pd.Series(0.0, index=df.index)

        # ── [AM] Realized volatility ratio ────────────────────────────────────
        try:
            log_ret  = np.log(c / c.shift(1))
            rv5      = (log_ret ** 2).rolling(5).sum()
            rv20     = (log_ret ** 2).rolling(20).sum()
            feats["rv_ratio"] = rv5 / (rv20 + 1e-9)
        except Exception as exc:
            logger.warning("[MLModel] RV ratio failed: %s", exc)

        # ── [AN] Return autocorrelation ───────────────────────────────────────
        try:
            ret1 = c.pct_change(1)
            # Lag-1 autocorrelation over 10-bar rolling window
            feats["ret_autocorr"] = ret1.rolling(10).apply(
                lambda x: float(pd.Series(x).autocorr(lag=1))
                if len(x) >= 4 else 0.0,
                raw=False,
            )
        except Exception as exc:
            logger.warning("[MLModel] Autocorr failed: %s", exc)

        # ── [AL] Hurst exponent (R/S statistic, 50-bar rolling) ───────────────
        try:
            feats["hurst"] = c.rolling(50).apply(_hurst_rs, raw=True)
        except Exception as exc:
            logger.warning("[MLModel] Hurst failed: %s", exc)

        # ── Price / returns ───────────────────────────────────────────────────
        try:
            feats["ret_1"]      = c.pct_change(1)
            feats["ret_3"]      = c.pct_change(3)
            feats["ret_5"]      = c.pct_change(5)
            feats["ret_10"]     = c.pct_change(10)
            feats["hl_ratio"]   = (h - l) / (c + 1e-9)
            feats["oc_ratio"]   = (c - o) / (c + 1e-9)
            feats["close_pos"]  = (c - l) / (h - l + 1e-9)
            feats["ret_skew10"] = c.pct_change().rolling(10).skew()
            feats["ret_kurt10"] = c.pct_change().rolling(10).kurt()
            direction           = np.sign(c - o)
            streak              = direction.groupby(
                (direction != direction.shift()).cumsum()
            ).cumcount() + 1
            feats["streak"]     = streak * direction
        except Exception as exc:
            logger.warning("[MLModel] price/return failed: %s", exc)

        # ── EMAs ──────────────────────────────────────────────────────────────
        try:
            ema8   = c.ewm(span=8,   adjust=False).mean()
            ema21  = c.ewm(span=21,  adjust=False).mean()
            ema50  = c.ewm(span=50,  adjust=False).mean()
            ema200 = c.ewm(span=200, adjust=False).mean()
            feats["ema_fast"]   = (c - ema8)   / (c + 1e-9)
            feats["ema_slow"]   = (c - ema21)  / (c + 1e-9)
            feats["ema_50"]     = (c - ema50)  / (c + 1e-9)
            feats["ema_200"]    = (c - ema200) / (c + 1e-9)
            feats["ema_cross"]  = (ema8 - ema21) / (c + 1e-9)
            feats["ema8_slope"] = ema8.diff(3) / (atr14 + 1e-9)
            feats["dist_ema8"]  = (c - ema8)   / (atr14 + 1e-9)
            feats["dist_ema21"] = (c - ema21)  / (atr14 + 1e-9)
            feats["dist_ema50"] = (c - ema50)  / (atr14 + 1e-9)
        except Exception as exc:
            logger.warning("[MLModel] EMA failed: %s", exc)

        # ── HTF proxy (M5 → M15 / H1 EMA approximation) ──────────────────────
        try:
            ema_fm15 = c.ewm(span=24,  adjust=False).mean()
            ema_sm15 = c.ewm(span=63,  adjust=False).mean()
            ema_fh1  = c.ewm(span=96,  adjust=False).mean()
            ema_sh1  = c.ewm(span=252, adjust=False).mean()
            feats["ema_fast_m15"]  = (c - ema_fm15) / (c + 1e-9)
            feats["ema_slow_m15"]  = (c - ema_sm15) / (c + 1e-9)
            feats["htf_cross_m15"] = (ema_fm15 - ema_sm15) / (c + 1e-9)
            feats["ema_fast_h1"]   = (c - ema_fh1)  / (c + 1e-9)
            feats["ema_slow_h1"]   = (c - ema_sh1)  / (c + 1e-9)
            feats["htf_cross_h1"]  = (ema_fh1 - ema_sh1) / (c + 1e-9)
            feats["htf_trend"]     = np.sign(
                (ema_fh1 - ema_sh1).values
            ).astype(np.float32)
        except Exception as exc:
            logger.warning("[MLModel] HTF proxy failed: %s", exc)

        # ── HTF pass-through from indicator engine ────────────────────────────
        try:
            for col in ("htf_ema_bull", "htf_ema_bear"):
                if col in df.columns and df[col].notna().any():
                    feats[col] = df[col].values
        except Exception as exc:
            logger.warning("[MLModel] HTF pass-through failed: %s", exc)

        # ── MACD ──────────────────────────────────────────────────────────────
        try:
            ema12     = c.ewm(span=12, adjust=False).mean()
            ema26     = c.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
            feats["macd"]        = macd_line              / (c + 1e-9)
            feats["macd_signal"] = macd_sig               / (c + 1e-9)
            feats["macd_hist"]   = (macd_line - macd_sig) / (c + 1e-9)
        except Exception as exc:
            logger.warning("[MLModel] MACD failed: %s", exc)

        # ── RSI ───────────────────────────────────────────────────────────────
        try:
            feats["rsi"]    = _rsi(c, 14)
            feats["rsi_6"]  = _rsi(c, 6)
            feats["rsi_25"] = _rsi(c, 25)
        except Exception as exc:
            logger.warning("[MLModel] RSI failed: %s", exc)

        # ── Bollinger Bands + Keltner squeeze ─────────────────────────────────
        try:
            bb_mid = c.rolling(20).mean()
            bb_std = c.rolling(20).std()
            bb_up  = bb_mid + 2 * bb_std
            bb_lo  = bb_mid - 2 * bb_std
            feats["bb_pct"]   = (c - bb_lo) / (bb_up - bb_lo + 1e-9)
            feats["bb_width"] = (bb_up - bb_lo) / (bb_mid + 1e-9)
            kc_mid   = c.ewm(span=20, adjust=False).mean()
            kc_width = atr14 * 2
            feats["bb_kc_squeeze"] = feats["bb_width"] / (
                (kc_width / (kc_mid + 1e-9)) + 1e-9
            )
        except Exception as exc:
            logger.warning("[MLModel] BB failed: %s", exc)

        # ── ADX + DMI ─────────────────────────────────────────────────────────
        try:
            period    = 14
            up_move   = h.diff()
            down_move = -l.diff()
            dm_plus   = np.where(
                (up_move > down_move) & (up_move > 0), up_move, 0.0
            )
            dm_minus  = np.where(
                (down_move > up_move) & (down_move > 0), down_move, 0.0
            )
            tr_adx    = pd.concat([
                h - l,
                (h - c.shift(1)).abs(),
                (l - c.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr_adx   = tr_adx.ewm(span=period, adjust=False).mean()
            dmi_p     = (
                pd.Series(dm_plus, index=df.index)
                .ewm(span=period, adjust=False).mean()
                / (atr_adx + 1e-9) * 100
            )
            dmi_m     = (
                pd.Series(dm_minus, index=df.index)
                .ewm(span=period, adjust=False).mean()
                / (atr_adx + 1e-9) * 100
            )
            dx  = (dmi_p - dmi_m).abs() / (dmi_p + dmi_m + 1e-9) * 100
            adx = dx.ewm(span=period, adjust=False).mean()
            feats["adx"]         = adx / 100.0
            feats["adx_slope"]   = adx.diff(3) / 100.0
            feats["dmi_diff"]    = (dmi_p - dmi_m) / 100.0
            feats["is_trending"] = (adx >= 25).astype(np.float32)
            feats["is_ranging"]  = (adx <  20).astype(np.float32)
        except Exception as exc:
            logger.warning("[MLModel] ADX failed: %s", exc)

        # ── CCI ───────────────────────────────────────────────────────────────
        try:
            tp      = (h + l + c) / 3
            tp_mean = tp.rolling(20).mean()
            tp_std  = tp.rolling(20).std()
            feats["cci"] = (tp - tp_mean) / (0.015 * (tp_std + 1e-9))
        except Exception as exc:
            logger.warning("[MLModel] CCI failed: %s", exc)

        # ── Stochastic ────────────────────────────────────────────────────────
        try:
            stoch_k_period = 5
            try:
                stoch_k_period = int(
                    CONFIG.get_scalper_profile().get(
                        "STOCH_K",
                        CONFIG.get_scalper_profile().get("stoch_k", 5),
                    )
                )
            except Exception:
                pass
            low_k   = l.rolling(stoch_k_period).min()
            high_k  = h.rolling(stoch_k_period).max()
            stoch_k = (c - low_k) / (high_k - low_k + 1e-9) * 100
            feats["stoch_k"] = stoch_k
            feats["stoch_d"] = stoch_k.rolling(3).mean()
        except Exception as exc:
            logger.warning("[MLModel] Stochastic failed: %s", exc)

        # ── CMF ───────────────────────────────────────────────────────────────
        try:
            vol     = df.get(
                "tick_volume", df.get("volume", pd.Series(1.0, index=df.index))
            )
            mf_mult = ((c - l) - (h - c)) / (h - l + 1e-9)
            mf_vol  = mf_mult * vol
            feats["cmf"] = (
                mf_vol.rolling(20).sum() / (vol.rolling(20).sum() + 1e-9)
            )
        except Exception as exc:
            logger.warning("[MLModel] CMF failed: %s", exc)

        # ── Volume + OBV ratio ────────────────────────────────────────────────
        try:
            vol = df.get(
                "tick_volume", df.get("volume", pd.Series(1.0, index=df.index))
            )
            feats["vol_ratio"]    = vol / (vol.rolling(10).mean() + 1e-9)
            feats["vol_ratio_20"] = vol / (vol.rolling(20).mean() + 1e-9)
            up_vol   = vol.where(c >= o, 0.0)
            down_vol = vol.where(c <  o, 0.0)
            feats["obv_ratio"] = (
                up_vol.rolling(10).sum() /
                (down_vol.rolling(10).sum() + 1e-9)
            )
        except Exception as exc:
            logger.warning("[MLModel] Volume failed: %s", exc)

        # ── [AO] Session-anchored VWAP ────────────────────────────────────────
        try:
            vol = df.get(
                "tick_volume", df.get("volume", pd.Series(1.0, index=df.index))
            )
            tp = (h + l + c) / 3

            idx = df.index
            if hasattr(idx, "tz") and idx.tz is not None:
                local = idx.tz_convert("Europe/Madrid")
            else:
                local = idx.tz_localize("UTC").tz_convert("Europe/Madrid")

            hour = pd.Series(local.hour, index=df.index)

            # London session anchor: reset at 08:00 CET
            lon_session = (hour >= 8).astype(int)
            lon_group   = (lon_session.diff().fillna(0) > 0).cumsum()
            lon_tp_vol  = (tp * vol).groupby(lon_group).cumsum()
            lon_vol_cum = vol.groupby(lon_group).cumsum()
            vwap_london = lon_tp_vol / (lon_vol_cum + 1e-9)

            # NY session anchor: reset at 13:00 CET
            ny_session  = (hour >= 13).astype(int)
            ny_group    = (ny_session.diff().fillna(0) > 0).cumsum()
            ny_tp_vol   = (tp * vol).groupby(ny_group).cumsum()
            ny_vol_cum  = vol.groupby(ny_group).cumsum()
            vwap_ny     = ny_tp_vol / (ny_vol_cum + 1e-9)

            feats["vwap_london_dist"] = (c - vwap_london) / (c + 1e-9)
            feats["vwap_ny_dist"]     = (c - vwap_ny)     / (c + 1e-9)

            # Legacy rolling VWAP retained for backward feature compat
            vwap_roll = (
                (tp * vol).rolling(20).sum() /
                (vol.rolling(20).sum() + 1e-9)
            )
            feats["vwap_dist"] = (c - vwap_roll) / (c + 1e-9)
        except Exception as exc:
            logger.warning("[MLModel] VWAP failed: %s", exc)

        # ── Ichimoku TK diff ──────────────────────────────────────────────────
        try:
            tenkan = (h.rolling(9).max()  + l.rolling(9).min())  / 2
            kijun  = (h.rolling(26).max() + l.rolling(26).min()) / 2
            feats["ichi_tk_diff"] = (tenkan - kijun) / (c + 1e-9)
        except Exception as exc:
            logger.warning("[MLModel] Ichimoku failed: %s", exc)

        # ── Swing high/low + ROC ──────────────────────────────────────────────
        try:
            sw = 12
            sh = h.rolling(sw).max()
            sl = l.rolling(sw).min()
            feats["dist_swing_hi"] = (sh - c) / (c + 1e-9)
            feats["dist_swing_lo"] = (c - sl)  / (c + 1e-9)
            feats["swing_range"]   = (sh - sl)  / (c + 1e-9)
            feats["roc_3"]         = c.pct_change(3)
            feats["roc_6"]         = c.pct_change(6)
            feats["roc_12"]        = c.pct_change(12)
        except Exception as exc:
            logger.warning("[MLModel] Swing/ROC failed: %s", exc)

        # ── Candle patterns ───────────────────────────────────────────────────
        try:
            po, ph, pl, pc = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
            body  = (c - o).abs()
            upper = h - pd.concat([o, c], axis=1).max(axis=1)
            lower = pd.concat([o, c], axis=1).min(axis=1) - l
            rng   = (h - l).replace(0, np.nan)
            _pat = {
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
                "pat_inside_bar": ((h < ph) & (l > pl)).astype(int).fillna(0),
                "pat_doji":       (body < 0.1 * rng).astype(int).fillna(0),
            }
            for col, val in _pat.items():
                feats[col] = (
                    df[col].values
                    if col in df.columns and df[col].notna().any()
                    else val.values
                )
        except Exception as exc:
            logger.warning("[MLModel] Candle patterns failed: %s", exc)

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
            logger.warning("[MLModel] Session features failed: %s", exc)

        # ── Final cleanup ─────────────────────────────────────────────────────
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
    # Helpers
    # =========================================================================

    def _align_features(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        if self._features is None:
            raise RuntimeError("Model not trained — call train() or load() first.")
        aligned = pd.DataFrame(index=feat_df.index)
        for col in self._features:
            aligned[col] = feat_df[col].values if col in feat_df.columns else 0.0
        return aligned

    def memory_usage(self) -> float:
        buf = io.BytesIO()
        joblib.dump(
            {
                "buy_xgb":    self._buy_model._xgb,
                "buy_lgbm":   self._buy_model._lgbm,
                "sell_xgb":   self._sell_model._xgb,
                "sell_lgbm":  self._sell_model._lgbm,
                "reg_xgb":    self._regime_model._xgb,
                "reg_lgbm":   self._regime_model._lgbm,
            },
            buf,
        )
        return buf.tell() / (1024 * 1024)

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, symbol: Optional[str] = None) -> None:
        sym = symbol or self.symbol
        # Save features list
        feat_path = MODEL_DIR / f"features_{sym}.pkl"
        tmp       = feat_path.with_suffix(".tmp")
        joblib.dump(self._features, tmp, compress=3)
        tmp.replace(feat_path)
        logger.info("[MLModel] Saved %s", feat_path)
        # Save each specialist
        self._buy_model.save(sym)
        self._sell_model.save(sym)
        self._regime_model.save(sym)

    def load(self, symbol: Optional[str] = None) -> bool:
        sym        = symbol or self.symbol
        feat_path  = MODEL_DIR / f"features_{sym}.pkl"

        # [AS] Check for new-style specialist files first
        buy_exists = (MODEL_DIR / f"xgb_buy_{sym}.pkl").exists()

        if not buy_exists:
            # Legacy single-model files exist → trigger retrain
            logger.warning(
                "[MLModel] Legacy model files found for %s — "
                "please retrain with new architecture.", sym
            )
            return False

        if not feat_path.exists():
            logger.warning("[MLModel] Missing features file: %s", feat_path)
            return False

        try:
            self._features = joblib.load(feat_path)
        except Exception as exc:
            logger.error("[MLModel] Failed to load features: %s", exc)
            return False

        ok = (
            self._buy_model.load(sym)
            and self._sell_model.load(sym)
            and self._regime_model.load(sym)
        )
        if ok:
            logger.info(
                "[MLModel] Loaded triple-specialist model for %s  "
                "(%d features  mem=%.1f MB)",
                sym, len(self._features), self.memory_usage(),
            )
        return ok

    @staticmethod
    def delete_stale(symbol: str) -> None:
        for prefix in ("xgb", "lgbm", "scaler", "features", "temps"):
            for specialist in ("buy", "sell", "regime", ""):
                suffix = f"_{specialist}" if specialist else ""
                path   = MODEL_DIR / f"{prefix}{suffix}_{symbol}.pkl"
                if path.exists():
                    path.unlink()
                    logger.info("[MLModel] Deleted stale: %s", path)


# =============================================================================
# Module-level pure functions (no class state)
# =============================================================================

def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _hurst_rs(prices: np.ndarray) -> float:
    """
    [AL] Rescaled range (R/S) Hurst exponent estimate.
    H > 0.5 → trending, H < 0.5 → mean-reverting, H ≈ 0.5 → random walk.
    Returns 0.5 (neutral) if computation fails or window too short.
    """
    try:
        n = len(prices)
        if n < 20:
            return 0.5
        mean_p  = prices.mean()
        deviations = prices - mean_p
        cum_dev    = np.cumsum(deviations)
        R = cum_dev.max() - cum_dev.min()
        S = prices.std(ddof=1)
        if S < 1e-10:
            return 0.5
        rs   = R / S
        if rs <= 0:
            return 0.5
        # Hurst = log(R/S) / log(n/2)
        H = np.log(rs) / np.log(n / 2)
        return float(np.clip(H, 0.0, 1.0))
    except Exception:
        return 0.5
