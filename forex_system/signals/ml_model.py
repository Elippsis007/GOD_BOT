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
CLASSES = [0, 1, 2]   # 0=HOLD  1=BUY  2=SELL


class MLSignalModel:
    """
    Ensemble ML model: Random Forest + Gradient Boosting
    with walk-forward cross-validation on time-series data.

    Improvements over v1:
    - Richer feature set (price action, momentum, volatility, volume)
    - ATR-normalised returns instead of raw prices
    - Trend context features (price vs EMA50, EMA crossover state)
    - Candle pattern features (body size, upper/lower wick)
    - Stronger label generation (1.0x ATR threshold, 5-bar forward window)
    - Hyper-parameter tuning for better generalisation
    """

    MODEL_DIR = Path("models/")

    def __init__(self, config=CONFIG):
        self.cfg = config
        self.rf  = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=15,
            max_features="sqrt",
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            min_samples_leaf=15,
            random_state=42,
        )
        self.scaler     = StandardScaler()
        self.is_trained = False
        self.MODEL_DIR.mkdir(exist_ok=True)

    # ── Column normalisation ──────────────────────────────────────────────────

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
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

        required = {"close", "open", "high", "low", "atr"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            logger.warning(f"Missing required columns for ML features: {missing}")
            return pd.DataFrame()

        X = pd.DataFrame(index=df.index)

        # ── 1. ATR-normalised returns (scale-invariant momentum) ──────────────
        # These replace raw prices so the model generalises across
        # different price levels and volatility regimes.
        atr = df["atr"].replace(0, np.nan)
        X["ret_1"]  = (df["close"] - df["close"].shift(1))  / atr
        X["ret_3"]  = (df["close"] - df["close"].shift(3))  / atr
        X["ret_5"]  = (df["close"] - df["close"].shift(5))  / atr
        X["ret_10"] = (df["close"] - df["close"].shift(10)) / atr

        # ── 2. Candle pattern features ────────────────────────────────────────
        # Body size and wick ratios reveal rejection / continuation intent.
        candle_range = (df["high"] - df["low"]).replace(0, np.nan)
        body         = (df["close"] - df["open"]).abs()
        upper_wick   = df["high"] - df[["close", "open"]].max(axis=1)
        lower_wick   = df[["close", "open"]].min(axis=1) - df["low"]

        X["body_ratio"]        = body        / candle_range
        X["upper_wick_ratio"]  = upper_wick  / candle_range
        X["lower_wick_ratio"]  = lower_wick  / candle_range
        X["candle_direction"]  = np.sign(df["close"] - df["open"])

        # ── 3. RSI features ───────────────────────────────────────────────────
        if "rsi" in df.columns:
            X["rsi"]          = df["rsi"]
            X["rsi_lag1"]     = df["rsi"].shift(1)
            X["rsi_lag3"]     = df["rsi"].shift(3)
            X["rsi_change"]   = df["rsi"] - df["rsi"].shift(3)
            X["rsi_mean_10"]  = df["rsi"].rolling(10).mean()
            X["rsi_overbought"] = (df["rsi"] > 70).astype(int)
            X["rsi_oversold"]   = (df["rsi"] < 30).astype(int)

        # ── 4. MACD features ──────────────────────────────────────────────────
        if "macd" in df.columns and "macd_signal" in df.columns:
            X["macd"]          = df["macd"]
            X["macd_signal"]   = df["macd_signal"]
            X["macd_hist"]     = df["macd"] - df["macd_signal"]
            X["macd_hist_lag1"]= (df["macd"] - df["macd_signal"]).shift(1)
            # Crossover: +1 when macd crosses above signal, -1 when below
            hist_now  = df["macd"] - df["macd_signal"]
            hist_prev = hist_now.shift(1)
            X["macd_cross"] = np.where(
                (hist_now > 0) & (hist_prev <= 0),  1,
                np.where(
                    (hist_now < 0) & (hist_prev >= 0), -1, 0
                )
            )

        # ── 5. EMA trend context ──────────────────────────────────────────────
        if "ema_fast" in df.columns and "ema_slow" in df.columns:
            X["ema_fast"]    = df["ema_fast"]
            X["ema_slow"]    = df["ema_slow"]
            # Distance from price to each EMA, normalised by ATR
            X["price_vs_ema_fast"] = (df["close"] - df["ema_fast"]) / atr
            X["price_vs_ema_slow"] = (df["close"] - df["ema_slow"]) / atr
            # EMA spread normalised by ATR — captures trend strength
            X["ema_spread_atr"]    = (df["ema_fast"] - df["ema_slow"]) / atr
            # Bullish stack: fast > slow = 1, else -1
            X["ema_stack"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)

        if "ema_trend" in df.columns:
            X["price_vs_ema_trend"] = (df["close"] - df["ema_trend"]) / atr

        # ── 6. Bollinger Band features ────────────────────────────────────────
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            bb_mid   = (df["bb_upper"] + df["bb_lower"]) / 2
            bb_width = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
            X["bb_position"] = (df["close"] - bb_mid) / (bb_width / 2)
            X["bb_width_atr"] = bb_width / atr
            bb_width_ma       = bb_width.rolling(20).mean()
            X["bb_squeeze"]   = bb_width / bb_width_ma.replace(0, np.nan)

        # ── 7. ATR volatility features ────────────────────────────────────────
        X["atr_ratio"]   = df["atr"] / df["atr"].rolling(20).mean().replace(0, np.nan)
        X["atr_lag1"]    = df["atr"].shift(1) / atr
        X["volatility"]  = df["close"].pct_change().rolling(10).std()

        # ── 8. Volume features ────────────────────────────────────────────────
        if "volume" in df.columns:
            vol_ma           = df["volume"].rolling(20).mean().replace(0, np.nan)
            X["vol_ratio"]   = df["volume"] / vol_ma
            X["vol_change"]  = df["volume"].pct_change(5)
            X["vol_spike"]   = (df["volume"] > vol_ma * 2).astype(int)

        # ── 9. Rolling momentum features ─────────────────────────────────────
        X["momentum_5"]  = df["close"].pct_change(5)
        X["momentum_10"] = df["close"].pct_change(10)
        X["high_low_5"]  = (df["high"].rolling(5).max() - df["low"].rolling(5).min()) / atr

        # ── 10. Stochastic features ───────────────────────────────────────────
        if "stoch_k" in df.columns:
            X["stoch_k"]      = df["stoch_k"]
            X["stoch_overbought"] = (df["stoch_k"] > 80).astype(int)
            X["stoch_oversold"]   = (df["stoch_k"] < 20).astype(int)

        if "adx" in df.columns:
            X["adx"] = df["adx"]
            X["adx_strong"] = (df["adx"] > 25).astype(int)

        return X.dropna()

    def _build_labels(self, df: pd.DataFrame, forward_bars: int = 5) -> pd.Series:
        """
        Label = 1 (BUY)  if future close > current close + 1.0 * ATR
              = 2 (SELL) if future close < current close - 1.0 * ATR
              = 0 (HOLD) otherwise

        Uses 1.0x ATR threshold (up from 0.5x) to generate
        higher-quality, less-noisy labels. 5-bar forward window
        (up from 3) gives price more time to move decisively.
        """
        df = self._normalise_columns(df)

        future_close = df["close"].shift(-forward_bars)
        threshold    = df["atr"] * 1.0      # stronger signal required

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

        # Remove last 5 rows — future labels are invalid there
        X = X.iloc[:-5]
        y = y.iloc[:-5]

        if len(X) < 200:
            logger.warning(
                f"⚠️  Only {len(X)} usable rows after feature build — skipping training."
            )
            return {"cv_scores": [], "mean_accuracy": 0.0}

        # Log class distribution so we can spot imbalance early
        dist = y.value_counts().to_dict()
        logger.info(f"   Label distribution → HOLD={dist.get(0,0)}  BUY={dist.get(1,0)}  SELL={dist.get(2,0)}")

        # ── Walk-forward cross-validation ─────────────────────────────────────
        tscv      = TimeSeriesSplit(n_splits=5)
        rf_scores = []
        gb_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_scaler = StandardScaler()
            X_tr_sc     = fold_scaler.fit_transform(X_tr)
            X_val_sc    = fold_scaler.transform(X_val)

            fold_rf = RandomForestClassifier(
                n_estimators=300, max_depth=10,
                min_samples_leaf=15, max_features="sqrt",
                random_state=42, class_weight="balanced", n_jobs=-1,
            )
            fold_gb = GradientBoostingClassifier(
                n_estimators=200, max_depth=5,
                learning_rate=0.03, subsample=0.8,
                min_samples_leaf=15, random_state=42,
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

        # ── Feature importance (top 10) ───────────────────────────────────────
        feat_importance = pd.Series(
            self.rf.feature_importances_, index=X.columns
        ).sort_values(ascending=False).head(10)
        logger.info(f"📊 Top-10 features:\n{feat_importance.to_string()}")

        # ── Persist to disk ───────────────────────────────────────────────────
        self.MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(self.rf,     self.MODEL_DIR / f"rf_{symbol}.pkl")
        joblib.dump(self.gb,     self.MODEL_DIR / f"gb_{symbol}.pkl")
        joblib.dump(self.scaler, self.MODEL_DIR / f"scaler_{symbol}.pkl")

        report   = classification_report(y, self.rf.predict(X_scaled), zero_division=0)
        mean_rf  = float(np.mean(rf_scores))
        mean_gb  = float(np.mean(gb_scores))
        logger.info(f"\n{report}")
        logger.info(
            f"✅ Model saved — mean CV accuracy: RF={mean_rf:.3f}  GB={mean_gb:.3f}"
        )

        return {
            "cv_scores_rf":    rf_scores,
            "cv_scores_gb":    gb_scores,
            "mean_accuracy_rf": mean_rf,
            "mean_accuracy_gb": mean_gb,
        }

    # ── Prediction ────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_proba(model, X_scaled: np.ndarray, n_classes: int = 3) -> np.ndarray:
        """Return a fixed 3-element probability vector aligned to CLASSES [0,1,2]."""
        raw    = model.predict_proba(X_scaled)[0]
        result = np.zeros(n_classes, dtype=float)
        for i, cls in enumerate(model.classes_):
            if cls < n_classes:
                result[cls] = raw[i]
        return result

    def predict(self, df: pd.DataFrame) -> dict:
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

        rf_proba  = self._safe_proba(self.rf,  X_scaled)
        gb_proba  = self._safe_proba(self.gb,  X_scaled)
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
            for prefix in ("rf", "gb", "scaler"):
                path = self.MODEL_DIR / f"{prefix}_{symbol}.pkl"
                if path.exists():
                    path.unlink()
            return False
