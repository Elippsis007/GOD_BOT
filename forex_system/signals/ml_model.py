# ML-based signal predictor
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


class MLSignalModel:
    """
    Ensemble ML model: Random Forest + Gradient Boosting
    with walk-forward validation on time-series data.
    """
    MODEL_DIR = Path("models/")

    def __init__(self, config=CONFIG):
        self.cfg    = config
        self.rf     = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_leaf=20, random_state=42,
            class_weight="balanced", n_jobs=-1,
        )
        self.gb     = GradientBoostingClassifier(
            n_estimators=150, max_depth=4,
            learning_rate=0.05, random_state=42,
        )
        self.scaler     = StandardScaler()
        self.is_trained = False
        self.MODEL_DIR.mkdir(exist_ok=True)

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Two-step column normalisation applied at the start of every
        feature/label builder so the rest of the code uses consistent names:

        1. tick_volume / real_volume  →  volume
        2. All column names           →  lowercase
           (guards against Close/Open capitalisation from any upstream source)
        """
        df = df.copy()
        if "volume" not in df.columns:
            if "tick_volume" in df.columns:
                df = df.rename(columns={"tick_volume": "volume"})
            elif "real_volume" in df.columns:
                df = df.rename(columns={"real_volume": "volume"})
        df.columns = df.columns.str.lower()
        return df

    # ── Feature Engineering ───────────────────────────────

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._normalise_columns(df)          # ← normalise once, here

        X = df[self.cfg.ML_FEATURES].copy()

        # Lagged features (1–5 bars)
        for col in ["rsi", "macd", "ema_spread"]:
            for lag in range(1, 6):
                X[f"{col}_lag{lag}"] = df[col].shift(lag)

        # Rolling statistics
        X["rsi_mean_10"] = df["rsi"].rolling(10).mean()
        X["atr_ratio"]   = df["atr"] / df["atr"].rolling(20).mean()
        X["bb_squeeze"]  = df["bb_width"] / df["bb_width"].rolling(20).mean()
        X["vol_change"]  = df["volume"].pct_change(5)   # lowercase 'volume'

        return X.dropna()

    def _build_labels(
        self, df: pd.DataFrame, forward_bars: int = 3
    ) -> pd.Series:
        """
        Label = 1 (BUY)  if future close > current + 0.5 * ATR
              = 2 (SELL) if future close < current - 0.5 * ATR
              = 0 (HOLD) otherwise
        """
        df = self._normalise_columns(df)          # ← same guard

        future_close = df["close"].shift(-forward_bars)   # lowercase 'close'
        threshold    = df["atr"] * 0.5

        labels = pd.Series(0, index=df.index)
        labels[future_close > df["close"] + threshold] = 1
        labels[future_close < df["close"] - threshold] = 2
        return labels

    # ── Training ──────────────────────────────────────────

    def train(self, df: pd.DataFrame, symbol: str = "ALL") -> dict:
        logger.info(f"🧠 Training ML model on {len(df)} bars...")

        X = self._build_features(df)
        y = self._build_labels(df).loc[X.index]

        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Remove last forward_bars rows (no future labels yet)
        X = X.iloc[:-3]
        y = y.iloc[:-3]

        if len(X) < 100:
            logger.warning(f"⚠️  Only {len(X)} usable rows after feature build — skipping training.")
            return {"cv_scores": [], "mean_accuracy": 0.0}

        # Walk-forward validation
        tscv   = TimeSeriesSplit(n_splits=5)
        scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            X_tr_sc  = self.scaler.fit_transform(X_tr)
            X_val_sc = self.scaler.transform(X_val)

            self.rf.fit(X_tr_sc, y_tr)
            score = self.rf.score(X_val_sc, y_val)
            scores.append(score)
            logger.info(f"  Fold {fold+1}: Accuracy = {score:.3f}")

        # Final fit on full dataset
        X_scaled = self.scaler.fit_transform(X)
        self.rf.fit(X_scaled, y)
        self.gb.fit(X_scaled, y)
        self.is_trained = True

        # Persist to disk
        self.MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(self.rf,     self.MODEL_DIR / f"rf_{symbol}.pkl")
        joblib.dump(self.gb,     self.MODEL_DIR / f"gb_{symbol}.pkl")
        joblib.dump(self.scaler, self.MODEL_DIR / f"scaler_{symbol}.pkl")

        report = classification_report(y, self.rf.predict(X_scaled))
        logger.info(f"\n{report}")
        logger.info(f"✅ Model saved — mean CV accuracy: {np.mean(scores):.3f}")

        return {"cv_scores": scores, "mean_accuracy": float(np.mean(scores))}

    # ── Prediction ────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> dict:
        if not self.is_trained:
            return {"label": "HOLD", "signal": 0, "confidence": 0.0}

        X = self._build_features(df)
        if len(X) == 0:
            return {"label": "HOLD", "signal": 0, "confidence": 0.0}

        X_last   = X.iloc[[-1]]
        X_scaled = self.scaler.transform(X_last)

        # Ensemble vote: average RF + GB probabilities
        rf_proba  = self.rf.predict_proba(X_scaled)[0]
        gb_proba  = self.gb.predict_proba(X_scaled)[0]
        avg_proba = (rf_proba + gb_proba) / 2

        signal     = int(np.argmax(avg_proba))
        confidence = float(avg_proba[signal])

        label_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        return {
            "signal":        signal,
            "label":         label_map[signal],
            "confidence":    round(confidence, 4),
            "probabilities": {
                "HOLD": round(float(avg_proba[0]), 4),
                "BUY":  round(float(avg_proba[1]), 4),
                "SELL": round(float(avg_proba[2]), 4),
            },
        }

    # ── Persistence ───────────────────────────────────────

    def load(self, symbol: str) -> bool:
        try:
            self.rf     = joblib.load(self.MODEL_DIR / f"rf_{symbol}.pkl")
            self.gb     = joblib.load(self.MODEL_DIR / f"gb_{symbol}.pkl")
            self.scaler = joblib.load(self.MODEL_DIR / f"scaler_{symbol}.pkl")
            self.is_trained = True
            logger.info(f"📦 Model loaded for {symbol}")
            return True
        except FileNotFoundError:
            logger.warning(f"No saved model for {symbol}, training required")
            return False
