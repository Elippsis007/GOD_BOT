# signals/ml_model.py
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from config.settings import CONFIG
from monitoring.logger import get_logger

logger = get_logger("MLModel")

# Fixed class order used everywhere — never inferred from data
CLASSES = [0, 1, 2]          # 0=HOLD  1=BUY  2=SELL


class MLSignalModel:
    """
    Ensemble ML model: Random Forest + Gradient Boosting
    with walk-forward cross-validation on time-series data.
    """
    MODEL_DIR = Path("models/")

    def __init__(self, config=CONFIG):
        self.cfg = config
        self.rf  = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_leaf=20, random_state=42,
            class_weight="balanced", n_jobs=-1,
        )
        self.gb  = GradientBoostingClassifier(
            n_estimators=150, max_depth=4,
            learning_rate=0.05, random_state=42,
        )
        self.scaler     = StandardScaler()
        self.is_trained = False
        self.MODEL_DIR.mkdir(exist_ok=True)

    # ── Column normalisation ──────────────────────────────────────────────────

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure 'volume' column exists and all names are lowercase."""
        df = df.copy()
        if "volume" not in df.columns:
            if "tick_volume" in df.columns:
                df = df.rename(columns={"tick_volume": "volume"})
            elif "real_volume" in df.columns:
                df = df.rename(columns={"real_volume": "volume"})
        df.columns = df.columns.str.lower()
        return df

    # ── Feature engineering ───────────────────────────────────────────────────

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._normalise_columns(df)

        # ── Base features from CONFIG ─────────────────────────────────────────
        # Only keep base columns that are actually present — avoids KeyError if
        # an optional indicator is missing from this bar's DataFrame.
        available_base = [c for c in self.cfg.ML_FEATURES if c in df.columns]
        if not available_base:
            logger.warning("No ML_FEATURES columns found in DataFrame — returning empty")
            return pd.DataFrame()

        X = df[available_base].copy()

        # ── Lagged features ───────────────────────────────────────────────────
        # FIX: removed 'ema_spread' — it is not produced by IndicatorEngine.
        # Replaced with 'ema_fast' and 'ema_slow' which are always present.
        lag_cols = ["rsi", "macd", "ema_fast", "ema_slow"]
        for col in lag_cols:
            if col in df.columns:
                for lag in range(1, 6):
                    X[f"{col}_lag{lag}"] = df[col].shift(lag)

        # ── Rolling statistics ────────────────────────────────────────────────
        if "rsi" in df.columns:
            X["rsi_mean_10"] = df["rsi"].rolling(10).mean()

        if "atr" in df.columns:
            atr_rolling = df["atr"].rolling(20).mean()
            X["atr_ratio"] = df["atr"] / atr_rolling.replace(0, np.nan)

        # FIX: 'bb_width' may not exist — compute it from bb_upper / bb_lower
        # if available, rather than assuming the column name.
        if "bb_upper" in df.columns and "bb_lower" in df.columns and "bb_mid" in df.columns:
            bb_width         = df["bb_upper"] - df["bb_lower"]
            bb_width_rolling = bb_width.rolling(20).mean()
            X["bb_squeeze"]  = bb_width / bb_width_rolling.replace(0, np.nan)
        elif "bb_width" in df.columns:
            # Accept the column if IndicatorEngine happens to produce it directly
            bb_width_rolling  = df["bb_width"].rolling(20).mean()
            X["bb_squeeze"]   = df["bb_width"] / bb_width_rolling.replace(0, np.nan)

        if "volume" in df.columns:
            X["vol_change"] = df["volume"].pct_change(5)

        return X.dropna()

    def _build_labels(self, df: pd.DataFrame, forward_bars: int = 3) -> pd.Series:
        """
        Label = 1 (BUY)  if future close > current + 0.5 * ATR
              = 2 (SELL) if future close < current - 0.5 * ATR
              = 0 (HOLD) otherwise
        """
        df = self._normalise_columns(df)

        future_close = df["close"].shift(-forward_bars)
        threshold    = df["atr"] * 0.5

        labels = pd.Series(0, index=df.index)
        labels[future_close > df["close"] + threshold] = 1
        labels[future_close < df["close"] - threshold] = 2
        return labels

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, symbol: str = "ALL") -> dict:
        logger.info(f"🧠 Training ML model on {len(df)} bars…")

        X = self._build_features(df)
        if X.empty:
            logger.warning("Feature build returned empty DataFrame — skipping training.")
            return {"cv_scores": [], "mean_accuracy": 0.0}

        y = self._build_labels(df).loc[X.index]

        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Remove last forward_bars rows — future labels are NaN / invalid there
        X = X.iloc[:-3]
        y = y.iloc[:-3]

        if len(X) < 100:
            logger.warning(
                f"⚠️  Only {len(X)} usable rows after feature build — skipping training."
            )
            return {"cv_scores": [], "mean_accuracy": 0.0}

        # ── Walk-forward cross-validation ─────────────────────────────────────
        # FIX: both RF and GB are now evaluated in every fold so their CV
        # performance is comparable and the final ensemble is fair.
        tscv      = TimeSeriesSplit(n_splits=5)
        rf_scores = []
        gb_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Each fold gets a fresh scaler fitted only on training slice
            fold_scaler  = StandardScaler()
            X_tr_sc      = fold_scaler.fit_transform(X_tr)
            X_val_sc     = fold_scaler.transform(X_val)

            fold_rf = RandomForestClassifier(
                n_estimators=200, max_depth=8,
                min_samples_leaf=20, random_state=42,
                class_weight="balanced", n_jobs=-1,
            )
            fold_gb = GradientBoostingClassifier(
                n_estimators=150, max_depth=4,
                learning_rate=0.05, random_state=42,
            )
            fold_rf.fit(X_tr_sc, y_tr)
            fold_gb.fit(X_tr_sc, y_tr)

            rf_scores.append(fold_rf.score(X_val_sc, y_val))
            gb_scores.append(fold_gb.score(X_val_sc, y_val))
            logger.info(
                f"  Fold {fold+1}: RF={rf_scores[-1]:.3f}  GB={gb_scores[-1]:.3f}"
            )

        # ── Final fit on full dataset ─────────────────────────────────────────
        X_scaled = self.scaler.fit_transform(X)
        self.rf.fit(X_scaled, y)
        self.gb.fit(X_scaled, y)
        self.is_trained = True

        # ── Persist to disk ───────────────────────────────────────────────────
        self.MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(self.rf,     self.MODEL_DIR / f"rf_{symbol}.pkl")
        joblib.dump(self.gb,     self.MODEL_DIR / f"gb_{symbol}.pkl")
        joblib.dump(self.scaler, self.MODEL_DIR / f"scaler_{symbol}.pkl")

        report = classification_report(
            y, self.rf.predict(X_scaled), zero_division=0
        )
        mean_rf = float(np.mean(rf_scores))
        mean_gb = float(np.mean(gb_scores))
        logger.info(f"\n{report}")
        logger.info(
            f"✅ Model saved — mean CV accuracy: RF={mean_rf:.3f}  GB={mean_gb:.3f}"
        )

        return {
            "cv_scores_rf": rf_scores,
            "cv_scores_gb": gb_scores,
            "mean_accuracy_rf": mean_rf,
            "mean_accuracy_gb": mean_gb,
        }

    # ── Prediction ────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_proba(model, X_scaled: np.ndarray, n_classes: int = 3) -> np.ndarray:
        """Return a fixed-length probability vector aligned to CLASSES [0,1,2].

        FIX: if a model was trained on fewer than 3 classes (e.g. only HOLD+BUY),
        its predict_proba() returns a shorter array. This method pads it to always
        produce a 3-element vector so RF + GB averaging never raises a ValueError.
        """
        raw    = model.predict_proba(X_scaled)[0]
        result = np.zeros(n_classes, dtype=float)
        for i, cls in enumerate(model.classes_):
            if cls < n_classes:
                result[cls] = raw[i]
        return result

    def predict(self, df: pd.DataFrame) -> dict:
        # FIX: untrained fallback now returns integer label (0) to match
        # ML_LABEL_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"} in main.py
        if not self.is_trained:
            return {
                "label":         0,
                "confidence":    0.0,
                "probabilities": {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0},
            }

        X = self._build_features(df)
        if X.empty:
            return {
                "label":         0,
                "confidence":    0.0,
                "probabilities": {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0},
            }

        X_last   = X.iloc[[-1]]
        X_scaled = self.scaler.transform(X_last)

        # FIX: use _safe_proba so mismatched class counts never raise ValueError
        rf_proba  = self._safe_proba(self.rf, X_scaled)
        gb_proba  = self._safe_proba(self.gb, X_scaled)
        avg_proba = (rf_proba + gb_proba) / 2.0

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

    # ── Persistence ───────────────────────────────────────────────────────────

    def load(self, symbol: str) -> bool:
        # FIX: broadened exception handling to catch corrupt/version-mismatched
        # model files (EOFError, ModuleNotFoundError, AttributeError, etc.)
        try:
            self.rf     = joblib.load(self.MODEL_DIR / f"rf_{symbol}.pkl")
            self.gb     = joblib.load(self.MODEL_DIR / f"gb_{symbol}.pkl")
            self.scaler = joblib.load(self.MODEL_DIR / f"scaler_{symbol}.pkl")
            self.is_trained = True
            logger.info(f"📦 Model loaded for {symbol}")
            return True
        except FileNotFoundError:
            logger.info(f"No saved model found for {symbol} — training required")
            return False
        except Exception as e:
            logger.warning(
                f"Model load failed for {symbol} ({type(e).__name__}: {e}) "
                f"— deleting stale files and retraining"
            )
            # Remove corrupt files so the next run doesn't hit the same error
            for prefix in ("rf", "gb", "scaler"):
                path = self.MODEL_DIR / f"{prefix}_{symbol}.pkl"
                if path.exists():
                    path.unlink()
            return False
