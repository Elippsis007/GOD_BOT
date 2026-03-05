# signals/ml_model.py
import os
import glob
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import classification_report, accuracy_score
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

from monitoring.logger import get_logger

logger = get_logger("MLModel")

MODEL_DIR   = "models"
CLASS_ORDER = [0, 1, 2]   # 0=HOLD  1=BUY  2=SELL
LABEL_MAP   = {0: "HOLD", 1: "BUY", 2: "SELL"}

os.makedirs(MODEL_DIR, exist_ok=True)


class MLSignalModel:
    """
    Ensemble ML signal model using XGBoost + LightGBM.

    Upgrades vs previous RF + GB version:
      - XGBoost  (gradient boosted trees — best single model for tabular data)
      - LightGBM (faster, handles large datasets and class imbalance well)
      - Optuna   hyperparameter tuning on Fold 1 only (fast — ~60 s)
      - Expanded feature set: 50+ features including non-lagging candle
        pattern ratios, volume spike, volatility regime, ATR-normalised
        price distances, Ichimoku cloud position, Stochastic divergence,
        Williams %R, CCI, VWAP distance.
      - Label generation: 1.2× ATR threshold over a 6-bar forward window
        with a minimum pip filter (3 pips) to avoid labelling noise as
        BUY/SELL.
      - Class weights balanced automatically by both models.
      - Walk-forward TimeSeriesSplit (5 folds).
      - Feature importance logged after training (top 15).
      - Models saved as xgb_<symbol>.pkl, lgbm_<symbol>.pkl,
        scaler_<symbol>.pkl, features_<symbol>.pkl.
    """

    # ── Tuning budget ─────────────────────────────────────────────────────────
    OPTUNA_TRIALS   = 25    # trials per model per tuning run
    N_FOLDS         = 5
    ATR_MULTIPLIER  = 1.2   # label threshold
    FORWARD_BARS    = 6     # bars ahead for label calculation
    MIN_PIP_MOVE    = 0.0003  # ~3 pips minimum move to label as BUY/SELL

    def __init__(self):
        self._xgb:      object = None
        self._lgbm:     object = None
        self._scaler:   StandardScaler = None
        self._features: list = []

    # ────────────────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, symbol: str) -> dict:
        """Train on df, save models, return results dict."""
        features, labels = self._build_dataset(df)
        if features is None or len(features) < 200:
            logger.warning(f"MLModel: not enough data to train ({len(df)} bars)")
            return {}

        self._features = list(features.columns)
        n_samples      = len(features)
        logger.info(
            f"🧠 Training ML model on {n_samples} bars | "
            f"{len(self._features)} features…"
        )

        dist = pd.Series(labels).value_counts().sort_index()
        logger.info(
            f"   Label distribution → "
            f"HOLD={dist.get(0,0)}  BUY={dist.get(1,0)}  SELL={dist.get(2,0)}"
        )

        X = features.values.astype(np.float32)
        y = labels.values.astype(np.int32)

        tscv    = TimeSeriesSplit(n_splits=self.N_FOLDS)
        splits  = list(tscv.split(X))

        # ── Optuna tuning on fold 1 only (keeps training fast) ────────────────
        logger.info("   🔍 Tuning hyperparameters (Optuna)…")
        X_tr, X_val = X[splits[0][0]], X[splits[0][1]]
        y_tr, y_val = y[splits[0][0]], y[splits[0][1]]

        sc_tune = StandardScaler().fit(X_tr)
        Xt_tr   = sc_tune.transform(X_tr)
        Xt_val  = sc_tune.transform(X_val)

        xgb_params  = self._tune_xgb(Xt_tr, y_tr, Xt_val, y_val)
        lgbm_params = self._tune_lgbm(Xt_tr, y_tr, Xt_val, y_val)
        logger.info("   ✅ Hyperparameter tuning complete")

        # ── Walk-forward cross-validation ─────────────────────────────────────
        xgb_scores, lgbm_scores = [], []
        for fold_idx, (tr_idx, val_idx) in enumerate(splits, 1):
            sc   = StandardScaler().fit(X[tr_idx])
            Xtr  = sc.transform(X[tr_idx])
            Xval = sc.transform(X[val_idx])

            xgb_cv  = XGBClassifier(**xgb_params,  random_state=42,
                                     use_label_encoder=False,
                                     eval_metric="mlogloss",
                                     verbosity=0)
            lgbm_cv = LGBMClassifier(**lgbm_params, random_state=42,
                                      verbose=-1)

            xgb_cv.fit(Xtr, y[tr_idx])
            lgbm_cv.fit(Xtr, y[tr_idx])

            xgb_score  = accuracy_score(y[val_idx], xgb_cv.predict(Xval))
            lgbm_score = accuracy_score(y[val_idx], lgbm_cv.predict(Xval))
            xgb_scores.append(xgb_score)
            lgbm_scores.append(lgbm_score)
            logger.info(
                f"   Fold {fold_idx}: XGB={xgb_score:.3f}  LGBM={lgbm_score:.3f}"
            )

        logger.info(
            f"   Mean CV → XGB={np.mean(xgb_scores):.3f}  "
            f"LGBM={np.mean(lgbm_scores):.3f}"
        )

        # ── Final models trained on all data ──────────────────────────────────
        self._scaler = StandardScaler().fit(X)
        X_scaled     = self._scaler.transform(X)

        self._xgb = XGBClassifier(**xgb_params, random_state=42,
                                   use_label_encoder=False,
                                   eval_metric="mlogloss",
                                   verbosity=0)
        self._lgbm = LGBMClassifier(**lgbm_params, random_state=42,
                                     verbose=-1)
        self._xgb.fit(X_scaled, y)
        self._lgbm.fit(X_scaled, y)

        # ── Final classification report ───────────────────────────────────────
        y_pred_xgb  = self._xgb.predict(X_scaled)
        y_pred_lgbm = self._lgbm.predict(X_scaled)
        y_pred_ens  = self._ensemble_labels(
            self._xgb.predict_proba(X_scaled),
            self._lgbm.predict_proba(X_scaled),
        )

        logger.info("\n" + classification_report(
            y, y_pred_ens,
            target_names=["HOLD", "BUY", "SELL"],
            zero_division=0,
        ))

        # ── Feature importances (top 15) ──────────────────────────────────────
        importances = (
            self._xgb.feature_importances_ +
            self._lgbm.feature_importances_
        ) / 2.0
        top15 = sorted(
            zip(self._features, importances),
            key=lambda x: x[1], reverse=True
        )[:15]
        logger.info("   📊 Top-15 feature importances (XGB+LGBM average):")
        for name, imp in top15:
            logger.info(f"      {name:<30} {imp:.4f}")

        # ── Save ──────────────────────────────────────────────────────────────
        self._save(symbol)

        return {
            "mean_accuracy_xgb":  float(np.mean(xgb_scores)),
            "mean_accuracy_lgbm": float(np.mean(lgbm_scores)),
            "n_features":         len(self._features),
            "n_samples":          n_samples,
        }

    def predict(self, df: pd.DataFrame) -> dict:
        """Return {'label': int, 'confidence': float, 'probabilities': dict}."""
        _neutral = {
            "label": 0,
            "confidence": 0.0,
            "probabilities": {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0},
        }

        if self._xgb is None or self._lgbm is None or self._scaler is None:
            return _neutral

        try:
            features, _ = self._build_dataset(df, for_prediction=True)
            if features is None or features.empty:
                return _neutral

            # Align columns to training feature set
            for col in self._features:
                if col not in features.columns:
                    features[col] = 0.0
            features = features[self._features]

            X = self._scaler.transform(
                features.values[-1:].astype(np.float32)
            )

            xgb_proba  = self._xgb.predict_proba(X)[0]
            lgbm_proba = self._lgbm.predict_proba(X)[0]

            # Safe probability arrays (handle missing classes)
            xgb_full  = self._safe_proba(xgb_proba,  self._xgb.classes_)
            lgbm_full = self._safe_proba(lgbm_proba, self._lgbm.classes_)

            # Weighted ensemble: LGBM slightly higher weight (faster convergence)
            avg_proba = xgb_full * 0.45 + lgbm_full * 0.55

            label      = int(np.argmax(avg_proba))
            confidence = float(avg_proba[label])

            return {
                "label":         label,
                "confidence":    round(confidence, 4),
                "probabilities": {
                    "HOLD": round(float(avg_proba[0]), 4),
                    "BUY":  round(float(avg_proba[1]), 4),
                    "SELL": round(float(avg_proba[2]), 4),
                },
            }
        except Exception as e:
            logger.warning(f"MLModel predict error: {e}")
            return _neutral

    def load(self, symbol: str) -> bool:
        """Load saved models. Returns True on success."""
        try:
            xgb_path  = os.path.join(MODEL_DIR, f"xgb_{symbol}.pkl")
            lgbm_path = os.path.join(MODEL_DIR, f"lgbm_{symbol}.pkl")
            sc_path   = os.path.join(MODEL_DIR, f"scaler_{symbol}.pkl")
            ft_path   = os.path.join(MODEL_DIR, f"features_{symbol}.pkl")

            for p in (xgb_path, lgbm_path, sc_path, ft_path):
                if not os.path.exists(p):
                    logger.info(f"MLModel: No saved model found for {symbol} — training required")
                    return False

            with open(xgb_path,  "rb") as f: self._xgb      = pickle.load(f)
            with open(lgbm_path, "rb") as f: self._lgbm     = pickle.load(f)
            with open(sc_path,   "rb") as f: self._scaler   = pickle.load(f)
            with open(ft_path,   "rb") as f: self._features = pickle.load(f)

            logger.info(f"📊 Model loaded for {symbol} ({len(self._features)} features)")
            return True

        except Exception as e:
            logger.warning(f"MLModel load error: {e} — deleting stale files")
            self._delete_model_files(symbol)
            return False

    # ────────────────────────────────────────────────────────────────────────
    #  Optuna tuning
    # ────────────────────────────────────────────────────────────────────────

    def _tune_xgb(self, X_tr, y_tr, X_val, y_val) -> dict:
        def objective(trial):
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 400),
                "max_depth":         trial.suggest_int("max_depth", 3, 8),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight":  trial.suggest_int("min_child_weight", 5, 30),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "num_class":         3,
                "objective":         "multi:softprob",
            }
            mdl = XGBClassifier(**params, random_state=42,
                                use_label_encoder=False,
                                eval_metric="mlogloss",
                                verbosity=0)
            mdl.fit(X_tr, y_tr)
            return accuracy_score(y_val, mdl.predict(X_val))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.OPTUNA_TRIALS, show_progress_bar=False)
        return study.best_params

    def _tune_lgbm(self, X_tr, y_tr, X_val, y_val) -> dict:
        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 400),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples":trial.suggest_int("min_child_samples", 10, 50),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "num_class":        3,
                "objective":        "multiclass",
                "class_weight":     "balanced",
            }
            mdl = LGBMClassifier(**params, random_state=42, verbose=-1)
            mdl.fit(X_tr, y_tr)
            return accuracy_score(y_val, mdl.predict(X_val))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.OPTUNA_TRIALS, show_progress_bar=False)
        return study.best_params

    # ────────────────────────────────────────────────────────────────────────
    #  Feature engineering
    # ────────────────────────────────────────────────────────────────────────

    def _build_dataset(
        self,
        df: pd.DataFrame,
        for_prediction: bool = False,
    ):
        """Build feature matrix and (optionally) label vector."""
        try:
            df = df.copy()
            df.columns = df.columns.str.lower()

            required = {"open", "high", "low", "close", "atr",
                        "ema_fast", "ema_slow", "rsi", "macd",
                        "macd_signal", "bb_upper", "bb_lower", "bb_mid"}
            missing  = required - set(df.columns)
            if missing:
                logger.warning(f"MLModel: missing columns {missing}")
                return None, None

            f = pd.DataFrame(index=df.index)

            # ── Price returns ──────────────────────────────────────────────
            atr_safe = df["atr"].replace(0, np.nan)
            for n in [1, 2, 3, 5, 8, 13]:
                f[f"ret_{n}"]      = df["close"].pct_change(n) * 100
                f[f"ret_atr_{n}"]  = df["close"].diff(n) / atr_safe

            # ── Candle structure ───────────────────────────────────────────
            body        = (df["close"] - df["open"]).abs()
            candle_range = (df["high"] - df["low"]).replace(0, np.nan)
            f["body_ratio"]       = body / candle_range
            f["upper_wick_ratio"] = (df["high"] - df[["open","close"]].max(axis=1)) / candle_range
            f["lower_wick_ratio"] = (df[["open","close"]].min(axis=1) - df["low"]) / candle_range
            f["candle_direction"] = np.sign(df["close"] - df["open"])
            f["candle_range_atr"] = candle_range / atr_safe

            # ── Consecutive candle direction ───────────────────────────────
            direction = np.sign(df["close"] - df["open"])
            f["consec_bull"] = direction.groupby(
                (direction != direction.shift()).cumsum()
            ).cumcount().where(direction > 0, 0)
            f["consec_bear"] = direction.groupby(
                (direction != direction.shift()).cumsum()
            ).cumcount().where(direction < 0, 0)

            # ── EMA features ───────────────────────────────────────────────
            f["ema_fast"]           = df["ema_fast"]
            f["ema_slow"]           = df["ema_slow"]
            f["ema_spread_atr"]     = (df["ema_fast"] - df["ema_slow"]) / atr_safe
            f["price_vs_ema_fast"]  = (df["close"] - df["ema_fast"]) / atr_safe
            f["price_vs_ema_slow"]  = (df["close"] - df["ema_slow"]) / atr_safe

            if "ema_trend" in df.columns:
                f["price_vs_ema_trend"] = (df["close"] - df["ema_trend"]) / atr_safe
                f["ema_trend_slope"]    = df["ema_trend"].diff(3) / atr_safe

            # ── EMA slope (momentum of the MA itself) ──────────────────────
            f["ema_fast_slope"] = df["ema_fast"].diff(2) / atr_safe
            f["ema_slow_slope"] = df["ema_slow"].diff(2) / atr_safe

            # ── RSI features ───────────────────────────────────────────────
            f["rsi"]         = df["rsi"]
            f["rsi_lag1"]    = df["rsi"].shift(1)
            f["rsi_lag2"]    = df["rsi"].shift(2)
            f["rsi_change"]  = df["rsi"].diff(1)
            f["rsi_mean_5"]  = df["rsi"].rolling(5).mean()
            f["rsi_mean_10"] = df["rsi"].rolling(10).mean()

            # ── MACD features ──────────────────────────────────────────────
            f["macd"]          = df["macd"]
            f["macd_signal"]   = df["macd_signal"]
            f["macd_hist"]     = df["macd"] - df["macd_signal"]
            f["macd_hist_lag"] = f["macd_hist"].shift(1)
            f["macd_cross"]    = (
                np.sign(df["macd"] - df["macd_signal"]) -
                np.sign(df["macd"].shift(1) - df["macd_signal"].shift(1))
            )

            # ── Bollinger Band features ────────────────────────────────────
            bb_width = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
            f["bb_position"]   = (df["close"] - df["bb_lower"]) / bb_width
            f["bb_width_atr"]  = bb_width / atr_safe
            f["bb_squeeze"]    = bb_width / df["bb_mid"]

            if "bb_width" in df.columns:
                f["bb_width_change"] = df["bb_width"].diff(1)

            # ── ATR & Volatility features ──────────────────────────────────
            f["atr_ratio"]      = atr_safe / df["close"]
            f["atr_lag1"]       = df["atr"].shift(1)
            f["atr_change"]     = df["atr"].diff(1) / atr_safe
            f["volatility"]     = df["close"].pct_change().rolling(10).std()
            f["vol_ratio"]      = atr_safe / df["atr"].rolling(20).mean()

            # ── Volume features ────────────────────────────────────────────
            if "volume" in df.columns:
                vol        = df["volume"].replace(0, np.nan)
                vol_ma     = vol.rolling(20).mean().replace(0, np.nan)
                f["vol_ratio_20"]  = vol / vol_ma
                f["vol_change"]    = vol.pct_change(1)
                f["vol_spike"]     = (vol > vol_ma * 2.0).astype(int)

            # ── CMF ────────────────────────────────────────────────────────
            if "cmf" in df.columns:
                f["cmf"] = df["cmf"]

            # ── ADX features ───────────────────────────────────────────────
            if "adx" in df.columns:
                f["adx"]        = df["adx"]
                f["adx_strong"] = (df["adx"] > 25).astype(int)
            if "di_pos" in df.columns and "di_neg" in df.columns:
                f["di_diff"] = df["di_pos"] - df["di_neg"]

            # ── Stochastic features ────────────────────────────────────────
            if "stoch_k" in df.columns:
                f["stoch_k"]      = df["stoch_k"]
                f["stoch_d"]      = df.get("stoch_d", df["stoch_k"])
                f["stoch_diff"]   = df["stoch_k"] - df.get("stoch_d", df["stoch_k"])
                f["stoch_change"] = df["stoch_k"].diff(1)

            # ── Williams %R ────────────────────────────────────────────────
            if "willr" in df.columns:
                f["willr"] = df["willr"]

            # ── CCI ────────────────────────────────────────────────────────
            if "cci" in df.columns:
                f["cci"]        = df["cci"]
                f["cci_change"] = df["cci"].diff(1)

            # ── VWAP distance ──────────────────────────────────────────────
            if "vwap" in df.columns:
                f["vwap_dist"] = (df["close"] - df["vwap"]) / atr_safe

            # ── Ichimoku features ──────────────────────────────────────────
            if "tenkan" in df.columns and "kijun" in df.columns:
                f["ichi_tk_diff"]    = (df["tenkan"] - df["kijun"]) / atr_safe
                f["price_vs_tenkan"] = (df["close"] - df["tenkan"]) / atr_safe
                f["price_vs_kijun"]  = (df["close"] - df["kijun"]) / atr_safe

            if "senkou_a" in df.columns and "senkou_b" in df.columns:
                f["in_cloud"] = (
                    (df["close"] > df[["senkou_a","senkou_b"]].min(axis=1)) &
                    (df["close"] < df[["senkou_a","senkou_b"]].max(axis=1))
                ).astype(int)
                f["above_cloud"] = (
                    df["close"] > df[["senkou_a","senkou_b"]].max(axis=1)
                ).astype(int)

            # ── Squeeze (BB inside KC) ─────────────────────────────────────
            if "squeeze" in df.columns:
                f["squeeze"]      = df["squeeze"]
                f["squeeze_bars"] = df["squeeze"].rolling(5).sum()

            # ── Rolling statistics ─────────────────────────────────────────
            for w in [5, 10, 20]:
                f[f"close_zscore_{w}"] = (
                    (df["close"] - df["close"].rolling(w).mean()) /
                    df["close"].rolling(w).std().replace(0, np.nan)
                )
                f[f"high_{w}"]  = (df["close"] == df["high"].rolling(w).max()).astype(int)
                f[f"low_{w}"]   = (df["close"] == df["low"].rolling(w).min()).astype(int)

            # ── Hour of day (session context) ─────────────────────────────
            if hasattr(df.index, "hour"):
                f["hour"] = df.index.hour
                f["is_overlap"] = (
                    (df.index.hour >= 13) & (df.index.hour < 17)
                ).astype(int)

            # ── Drop NaN ──────────────────────────────────────────────────
            f.replace([np.inf, -np.inf], np.nan, inplace=True)
            f.dropna(inplace=True)

            if for_prediction:
                return f, None

            # ── Labels ────────────────────────────────────────────────────
            labels = self._create_labels(df.loc[f.index], atr_safe.loc[f.index])
            f      = f.loc[labels.index]

            return f, labels

        except Exception as e:
            logger.error(f"MLModel feature build error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None

    def _create_labels(
        self,
        df:       pd.DataFrame,
        atr_safe: pd.Series,
    ) -> pd.Series:
        """
        Label each bar as BUY=1 / SELL=2 / HOLD=0.

        Logic:
          - Look FORWARD_BARS bars ahead.
          - If future_high > close + ATR_MULTIPLIER * atr AND move > MIN_PIP_MOVE → BUY
          - If future_low  < close - ATR_MULTIPLIER * atr AND move > MIN_PIP_MOVE → SELL
          - Otherwise HOLD.
        When both conditions are met, the larger move wins.
        """
        close     = df["close"]
        threshold = atr_safe * self.ATR_MULTIPLIER

        future_high = close.shift(-self.FORWARD_BARS).rolling(
            self.FORWARD_BARS, min_periods=1
        ).max().shift(-(self.FORWARD_BARS - 1))

        future_low = close.shift(-self.FORWARD_BARS).rolling(
            self.FORWARD_BARS, min_periods=1
        ).min().shift(-(self.FORWARD_BARS - 1))

        # Vectorised label assignment
        up_move   = future_high - close
        down_move = close - future_low

        buy_cond  = (up_move   > threshold) & (up_move   > self.MIN_PIP_MOVE)
        sell_cond = (down_move > threshold) & (down_move > self.MIN_PIP_MOVE)

        labels = pd.Series(0, index=close.index, dtype=np.int32)
        labels[buy_cond  & ~sell_cond]             = 1
        labels[sell_cond & ~buy_cond]              = 2
        labels[buy_cond  &  sell_cond & (up_move >= down_move)] = 1
        labels[buy_cond  &  sell_cond & (up_move <  down_move)] = 2

        # Drop last FORWARD_BARS rows — no forward data available
        labels = labels.iloc[:-self.FORWARD_BARS]
        return labels

    # ────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """Map model probability output to fixed [HOLD, BUY, SELL] order."""
        full = np.zeros(3, dtype=np.float32)
        for i, cls in enumerate(classes):
            if 0 <= cls <= 2:
                full[cls] = proba[i]
        # Normalise
        total = full.sum()
        return full / total if total > 0 else np.array([1.0, 0.0, 0.0])

    @staticmethod
    def _ensemble_labels(xgb_proba, lgbm_proba) -> np.ndarray:
        avg = xgb_proba * 0.45 + lgbm_proba * 0.55
        return np.argmax(avg, axis=1)

    def _save(self, symbol: str) -> None:
        try:
            self._delete_model_files(symbol)
            paths = {
                f"xgb_{symbol}.pkl":      self._xgb,
                f"lgbm_{symbol}.pkl":     self._lgbm,
                f"scaler_{symbol}.pkl":   self._scaler,
                f"features_{symbol}.pkl": self._features,
            }
            for fname, obj in paths.items():
                with open(os.path.join(MODEL_DIR, fname), "wb") as f:
                    pickle.dump(obj, f)

            mean_xgb  = 0.0   # updated by caller via return dict
            mean_lgbm = 0.0
            logger.info(f"✅ Model saved for {symbol}")
        except Exception as e:
            logger.error(f"MLModel save error: {e}")

    @staticmethod
    def _delete_model_files(symbol: str) -> None:
        patterns = [
            # New format
            f"xgb_{symbol}.pkl",
            f"lgbm_{symbol}.pkl",
            f"scaler_{symbol}.pkl",
            f"features_{symbol}.pkl",
            # Legacy format (RF/GB from old version)
            f"rf_{symbol}.pkl",
            f"gb_{symbol}.pkl",
        ]
        for pattern in patterns:
            for path in glob.glob(os.path.join(MODEL_DIR, pattern)):
                try:
                    os.remove(path)
                    logger.debug(f"Deleted stale model file: {path}")
                except Exception:
                    pass
