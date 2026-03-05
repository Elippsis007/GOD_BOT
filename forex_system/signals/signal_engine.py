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
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


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
    8 filters total — max possible score is +8 or -8.
    """

    def __init__(self, config=CONFIG, trading_style: str = "scalper"):
        self.cfg           = config
        self.trading_style = trading_style
        self._update_thresholds()

    def _update_thresholds(self) -> None:
        """Sets BULL/BEAR thresholds from config based on trading style."""
        if self.trading_style == "scalper" and hasattr(self.cfg, "SCALPER_SIGNAL_SCORE"):
            self.BULL_THRESHOLD = self.cfg.SCALPER_SIGNAL_SCORE
            self.BEAR_THRESHOLD = self.cfg.SCALPER_SIGNAL_SCORE
        elif self.trading_style == "daytrader" and hasattr(self.cfg, "DAYTRADER_SIGNAL_SCORE"):
            self.BULL_THRESHOLD = self.cfg.DAYTRADER_SIGNAL_SCORE
            self.BEAR_THRESHOLD = self.cfg.DAYTRADER_SIGNAL_SCORE
        else:
            self.BULL_THRESHOLD = 4
            self.BEAR_THRESHOLD = 4

    def evaluate(
        self, df: pd.DataFrame, symbol: str
    ) -> Optional[TradingSignal]:
        if df is None or len(df) < 2:
            return None

        last    = df.iloc[-1]
        prev    = df.iloc[-2]
        score   = 0
        reasons = []

        # ── Filter 1: Trend Alignment (EMA Stack) ─────────
        if last["ema_fast"] > last["ema_slow"] > last["ema_trend"]:
            score += 1
            reasons.append("✅ EMA bullish stack")
        elif last["ema_fast"] < last["ema_slow"] < last["ema_trend"]:
            score -= 1
            reasons.append("❌ EMA bearish stack")

        # ── Filter 2: MACD Crossover ───────────────────────
        bull_cross = (
            prev["macd"] < prev["macd_signal"] and
            last["macd"] > last["macd_signal"]
        )
        bear_cross = (
            prev["macd"] > prev["macd_signal"] and
            last["macd"] < last["macd_signal"]
        )
        if bull_cross:
            score += 1
            reasons.append("✅ MACD bullish crossover")
        elif bear_cross:
            score -= 1
            reasons.append("❌ MACD bearish crossover")

        # ── Filter 3: RSI Zone ────────────────────────────
        # FIX: the original had two overlapping conditions that left a gap
        # and could hit the wrong branch:
        #
        #   Bullish : 40 < RSI < 70  (RSI_OVERBOUGHT)
        #   Bearish : 30 < RSI < 60  (RSI_OVERSOLD < RSI < 60)
        #
        # Problem 1 — GAP: RSI between 60 and 70 matched neither branch,
        # contributing nothing to the score even in a strong uptrend.
        #
        # Problem 2 — OVERLAP: RSI between 40 and 60 satisfied BOTH
        # conditions simultaneously. Because Python evaluates the first
        # `if` branch and skips `elif`, an RSI of 45 always scored bullish
        # and could never score bearish — masking genuine downtrends.
        #
        # FIX: split into three fully exclusive, fully covering zones:
        #   Bullish : RSI_OVERSOLD(30) < RSI <= 60   → healthy momentum up
        #   Neutral : 60 < RSI < RSI_OVERBOUGHT(70)  → no score (transition)
        #   Bearish : RSI_OVERBOUGHT(70) <= RSI       → overbought, fade
        #   AND mirror for the sell side:
        #   Bearish : RSI_OVERSOLD(30) <= RSI < 60   → healthy momentum down
        #   Bullish : RSI < RSI_OVERSOLD(30)          → oversold, bounce
        #
        # Simplified to the clearest mutually exclusive split:
        rsi = last["rsi"]
        if self.cfg.RSI_OVERSOLD < rsi < 60:
            # Rising from oversold into mid-range — bullish momentum
            score += 1
            reasons.append(f"✅ RSI bullish zone ({rsi:.1f})")
        elif 60 < rsi < self.cfg.RSI_OVERBOUGHT:
            # Between 60 and overbought — bearish lean, trend fading
            score -= 1
            reasons.append(f"❌ RSI bearish zone ({rsi:.1f})")
        elif rsi >= self.cfg.RSI_OVERBOUGHT:
            # Overbought — strong bearish signal
            score -= 1
            reasons.append(f"❌ RSI overbought ({rsi:.1f})")
        elif rsi <= self.cfg.RSI_OVERSOLD:
            # Oversold — strong bullish signal
            score += 1
            reasons.append(f"✅ RSI oversold ({rsi:.1f})")
        # RSI exactly at 60 scores neutral — intentional dead zone

        # ── Filter 4: Bollinger Band Position ─────────────
        if last["Close"] > last["bb_mid"] and last["Close"] < last["bb_upper"]:
            score += 1
            reasons.append("✅ Price above BB midline")
        elif last["Close"] < last["bb_mid"] and last["Close"] > last["bb_lower"]:
            score -= 1
            reasons.append("❌ Price below BB midline")

        # ── Filter 5: ADX Trend Strength ──────────────────
        if last["adx"] > 25:
            if last["di_pos"] > last["di_neg"]:
                score += 1
                reasons.append(f"✅ Strong uptrend ADX={last['adx']:.1f}")
            else:
                score -= 1
                reasons.append(f"❌ Strong downtrend ADX={last['adx']:.1f}")

        # ── Filter 6: Stochastic ──────────────────────────
        if last["stoch_k"] > last["stoch_d"] and last["stoch_k"] < 80:
            score += 1
            reasons.append("✅ Stochastic bullish")
        elif last["stoch_k"] < last["stoch_d"] and last["stoch_k"] > 20:
            score -= 1
            reasons.append("❌ Stochastic bearish")

        # ── Filter 7: Volume Confirmation ─────────────────
        if last["cmf"] > 0.1:
            score += 1
            reasons.append("✅ Positive CMF (buying pressure)")
        elif last["cmf"] < -0.1:
            score -= 1
            reasons.append("❌ Negative CMF (selling pressure)")

        # ── Filter 8: Squeeze Breakout ────────────────────
        if last["squeeze"] == 0 and prev["squeeze"] == 1:
            reasons.append("⚡ Volatility squeeze breakout detected!")
            score += (1 if last["is_bullish"] else -1)

        # ── Determine Signal ───────────────────────────────
        strength = abs(score) / 8.0
        atr      = last["atr"]
        entry    = last["Close"]

        if score >= self.BULL_THRESHOLD:
            sl          = entry - (atr * 1.5)
            tp          = entry + (atr * 1.5 * self.cfg.RR_RATIO)
            signal_type = SignalType.BUY
            confidence  = min(score / 8.0, 1.0)
        elif score <= -self.BEAR_THRESHOLD:
            sl          = entry + (atr * 1.5)
            tp          = entry - (atr * 1.5 * self.cfg.RR_RATIO)
            signal_type = SignalType.SELL
            confidence  = min(abs(score) / 8.0, 1.0)
        else:
            return None  # confluence too weak — no trade

        signal = TradingSignal(
            symbol     = symbol,
            signal     = signal_type,
            strength   = round(strength, 3),
            confidence = round(confidence, 3),
            entry      = round(entry, 5),
            sl         = round(sl,    5),
            tp         = round(tp,    5),
            atr        = round(atr,   5),
            reasons    = reasons,
            timestamp  = str(df.index[-1]),
        )
        logger.info(
            f"🎯 Signal [{signal.signal.value}] on {symbol} | "
            f"Score={score} | Conf={confidence:.0%}"
        )
        return signal
