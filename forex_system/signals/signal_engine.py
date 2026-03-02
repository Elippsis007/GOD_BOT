# Rule-based signal generator
# signals/signal_engine.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from config.settings import CONFIG
from monitoring.logger import get_logger

logger = get_logger("SignalEngine")

class SignalType(Enum):
    BUY    = "BUY"
    SELL   = "SELL"
    HOLD   = "HOLD"

@dataclass
class TradingSignal:
    symbol:     str
    signal:     SignalType
    strength:   float        # 0.0 – 1.0
    confidence: float        # 0.0 – 1.0
    entry:      float
    sl:         float
    tp:         float
    atr:        float
    reasons:    list
    timestamp:  str

class SignalEngine:
    """
    Multi-confluence signal engine.
    Each sub-filter scores +1 (bull) / -1 (bear) / 0 (neutral).
    A signal fires when confluence score >= threshold.
    """

    BULL_THRESHOLD = 4   # minimum bullish sub-signals
    BEAR_THRESHOLD = 4

    def __init__(self, config=CONFIG):
        self.cfg = config

    def evaluate(
        self, df: pd.DataFrame, symbol: str
    ) -> Optional[TradingSignal]:
        if df is None or len(df) < 2:
            return None

        last  = df.iloc[-1]
        prev  = df.iloc[-2]
        score = 0
        reasons = []

        # ── Filter 1: Trend Alignment (EMA Stack) ─────────
        if last["ema_fast"] > last["ema_slow"] > last["ema_trend"]:
            score += 1; reasons.append("✅ EMA bullish stack")
        elif last["ema_fast"] < last["ema_slow"] < last["ema_trend"]:
            score -= 1; reasons.append("❌ EMA bearish stack")

        # ── Filter 2: MACD Crossover ───────────────────────
        bull_cross = prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]
        bear_cross = prev["macd"] > prev["macd_signal"] and last["macd"] < last["macd_signal"]
        if bull_cross:
            score += 1; reasons.append("✅ MACD bullish crossover")
        elif bear_cross:
            score -= 1; reasons.append("❌ MACD bearish crossover")

        # ── Filter 3: RSI Zone ────────────────────────────
        if 40 < last["rsi"] < self.cfg.RSI_OVERBOUGHT:
            score += 1; reasons.append(f"✅ RSI healthy ({last['rsi']:.1f})")
        elif self.cfg.RSI_OVERSOLD < last["rsi"] < 60:
            score -= 1; reasons.append(f"❌ RSI bearish zone ({last['rsi']:.1f})")

        # ── Filter 4: Bollinger Band Position ─────────────
        if last["Close"] > last["bb_mid"] and last["Close"] < last["bb_upper"]:
            score += 1; reasons.append("✅ Price above BB midline")
        elif last["Close"] < last["bb_mid"] and last["Close"] > last["bb_lower"]:
            score -= 1; reasons.append("❌ Price below BB midline")

        # ── Filter 5: ADX Trend Strength ──────────────────
        if last["adx"] > 25:
            if last["di_pos"] > last["di_neg"]:
                score += 1; reasons.append(f"✅ Strong uptrend ADX={last['adx']:.1f}")
            else:
                score -= 1; reasons.append(f"❌ Strong downtrend ADX={last['adx']:.1f}")

        # ── Filter 6: Stochastic ──────────────────────────
        if last["stoch_k"] > last["stoch_d"] and last["stoch_k"] < 80:
            score += 1; reasons.append("✅ Stochastic bullish")
        elif last["stoch_k"] < last["stoch_d"] and last["stoch_k"] > 20:
            score -= 1; reasons.append("❌ Stochastic bearish")

        # ── Filter 7: Volume Confirmation ─────────────────
        if last["cmf"] > 0.1:
            score += 1; reasons.append("✅ Positive CMF (buying pressure)")
        elif last["cmf"] < -0.1:
            score -= 1; reasons.append("❌ Negative CMF (selling pressure)")

        # ── Filter 8: Squeeze Breakout ────────────────────
        if last["squeeze"] == 0 and prev["squeeze"] == 1:
            reasons.append("⚡ Volatility squeeze breakout detected!")
            score += (1 if last["is_bullish"] else -1)

        # ── Determine Signal ───────────────────────────────
        strength   = abs(score) / 8.0
        atr        = last["atr"]
        entry      = last["Close"]

        if score >= self.BULL_THRESHOLD:
            sl = entry - (atr * 1.5)
            tp = entry + (atr * 1.5 * self.cfg.RR_RATIO)
            signal_type = SignalType.BUY
            confidence  = min(score / 8.0, 1.0)
        elif score <= -self.BEAR_THRESHOLD:
            sl = entry + (atr * 1.5)
            tp = entry - (atr * 1.5 * self.cfg.RR_RATIO)
            signal_type = SignalType.SELL
            confidence  = min(abs(score) / 8.0, 1.0)
        else:
            return None  # No trade

        signal = TradingSignal(
            symbol=symbol,
            signal=signal_type,
            strength=round(strength, 3),
            confidence=round(confidence, 3),
            entry=round(entry, 5),
            sl=round(sl, 5),
            tp=round(tp, 5),
            atr=round(atr, 5),
            reasons=reasons,
            timestamp=str(df.index[-1])
        )
        logger.info(f"🎯 Signal [{signal.signal.value}] on {symbol} | Score={score} | Conf={confidence:.0%}")
        return signal
