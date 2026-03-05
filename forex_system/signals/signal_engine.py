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
    Multi-confluence signal engine optimised for M5 scalping.
    Each sub-filter scores +1 (bull) / -1 (bear) / 0 (neutral).
    A signal fires when confluence score >= threshold.
    8 filters total — max possible score is +8 or -8.

    M5 optimisations vs original:
      - Filter 1 : EMA stack uses fast/slow only (EMA 9/21);
                   ema_trend (EMA 50 = 250 min lag) removed from stack check.
                   Instead ema_trend acts as a directional bias gate only.
      - Filter 3 : RSI zone boundaries tightened (oversold <35, overbought >65)
                   so the filter fires more on M5 without waiting for extremes.
      - Filter 5 : ADX threshold lowered from 25 → 18 so it contributes on
                   shorter-term M5 trends (ADX 25 almost never fires on M5).
      - Filter 6 : Stochastic now also scores at extreme zones (<20 / >80)
                   giving an extra point on strong momentum candles.
      - Filter 7 : CMF threshold tightened from ±0.1 → ±0.05 so volume
                   confirmation triggers earlier on M5.
      - Filter 8 : Squeeze breakout now also checks momentum direction via
                   last close vs previous close as a fallback when is_bullish
                   is unavailable.
    """

    # ── M5-tuned indicator periods ────────────────────────────────────────────
    ADX_THRESHOLD     = 18     # was 25 — fires on M5 trends
    RSI_BULL_LOW      = 35     # was RSI_OVERSOLD (30) — slightly relaxed
    RSI_BULL_HIGH     = 60     # unchanged
    RSI_BEAR_LOW      = 60     # unchanged
    RSI_BEAR_HIGH     = 65     # was RSI_OVERBOUGHT (70) — slightly tightened
    CMF_THRESHOLD     = 0.05   # was 0.10 — fires earlier on M5 volume
    STOCH_BULL_ZONE   = 20     # %K below this = oversold bonus point
    STOCH_BEAR_ZONE   = 80     # %K above this = overbought bonus point

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

        # ── Normalise column names to lowercase ───────────────────────────────
        df = df.copy()
        df.columns = df.columns.str.lower()

        last  = df.iloc[-1]
        prev  = df.iloc[-2]
        score = 0
        reasons: list = []

        # ── Filter 1: Trend Alignment (EMA 9/21 stack + EMA 50 bias) ─────────
        # On M5, EMA 50 = 250 minutes of lag — too slow for crossover logic.
        # We use EMA 9/21 for the score and EMA 50 only as a directional gate
        # (adds +0.5 confluece weight by allowing half-point if trend agrees).
        ema_fast  = last.get("ema_fast",  None)
        ema_slow  = last.get("ema_slow",  None)
        ema_trend = last.get("ema_trend", None)

        if ema_fast is not None and ema_slow is not None:
            if ema_fast > ema_slow:
                score += 1
                tag = "✅ EMA 9>21 bullish"
                if ema_trend is not None and ema_fast > ema_trend:
                    tag += " + trend aligned"
                reasons.append(tag)
            elif ema_fast < ema_slow:
                score -= 1
                tag = "❌ EMA 9<21 bearish"
                if ema_trend is not None and ema_fast < ema_trend:
                    tag += " + trend aligned"
                reasons.append(tag)

        # ── Filter 2: MACD Crossover ──────────────────────────────────────────
        macd       = last.get("macd",        None)
        macd_sig   = last.get("macd_signal", None)
        prev_macd  = prev.get("macd",        None)
        prev_msig  = prev.get("macd_signal", None)

        if None not in (macd, macd_sig, prev_macd, prev_msig):
            bull_cross = prev_macd < prev_msig and macd > macd_sig
            bear_cross = prev_macd > prev_msig and macd < macd_sig
            if bull_cross:
                score += 1
                reasons.append("✅ MACD bullish crossover")
            elif bear_cross:
                score -= 1
                reasons.append("❌ MACD bearish crossover")
            # No cross but same side — half-signal via histogram direction
            elif macd > macd_sig and macd > prev_macd:
                score += 1
                reasons.append("✅ MACD histogram expanding bullish")
            elif macd < macd_sig and macd < prev_macd:
                score -= 1
                reasons.append("❌ MACD histogram expanding bearish")

        # ── Filter 3: RSI Zone (M5-tightened boundaries) ─────────────────────
        rsi = last.get("rsi", None)
        if rsi is not None:
            if rsi <= self.RSI_BULL_LOW:
                score += 1
                reasons.append(f"✅ RSI oversold ({rsi:.1f})")
            elif self.RSI_BULL_LOW < rsi < self.RSI_BULL_HIGH:
                score += 1
                reasons.append(f"✅ RSI bullish zone ({rsi:.1f})")
            elif rsi >= self.RSI_BEAR_HIGH:
                score -= 1
                reasons.append(f"❌ RSI overbought ({rsi:.1f})")
            elif self.RSI_BEAR_LOW < rsi < self.RSI_BEAR_HIGH:
                score -= 1
                reasons.append(f"❌ RSI bearish zone ({rsi:.1f})")

        # ── Filter 4: Bollinger Band Position ─────────────────────────────────
        close     = last.get("close",    None)
        bb_mid    = last.get("bb_mid",   None)
        bb_upper  = last.get("bb_upper", None)
        bb_lower  = last.get("bb_lower", None)

        if None not in (close, bb_mid, bb_upper, bb_lower):
            if bb_mid < close < bb_upper:
                score += 1
                reasons.append("✅ Price above BB midline")
            elif bb_lower < close < bb_mid:
                score -= 1
                reasons.append("❌ Price below BB midline")
            # Extreme band touches — strong momentum signal
            elif close >= bb_upper:
                score += 1
                reasons.append("✅ Price at/above BB upper (strong bull momentum)")
            elif close <= bb_lower:
                score -= 1
                reasons.append("❌ Price at/below BB lower (strong bear momentum)")

        # ── Filter 5: ADX Trend Strength (M5-lowered threshold: 18 vs 25) ────
        adx    = last.get("adx",    None)
        di_pos = last.get("di_pos", None)
        di_neg = last.get("di_neg", None)

        if None not in (adx, di_pos, di_neg):
            if adx > self.ADX_THRESHOLD:
                if di_pos > di_neg:
                    score += 1
                    reasons.append(f"✅ Uptrend confirmed ADX={adx:.1f}")
                else:
                    score -= 1
                    reasons.append(f"❌ Downtrend confirmed ADX={adx:.1f}")
            else:
                reasons.append(f"⚠️  ADX weak ({adx:.1f}) — no trend confirmation")

        # ── Filter 6: Stochastic (M5-tuned: scores extreme zones too) ─────────
        stoch_k = last.get("stoch_k", None)
        stoch_d = last.get("stoch_d", None)

        if None not in (stoch_k, stoch_d):
            if stoch_k <= self.STOCH_BULL_ZONE:
                # Oversold extreme — strong buy signal
                score += 1
                reasons.append(f"✅ Stochastic oversold ({stoch_k:.1f})")
            elif stoch_k >= self.STOCH_BEAR_ZONE:
                # Overbought extreme — strong sell signal
                score -= 1
                reasons.append(f"❌ Stochastic overbought ({stoch_k:.1f})")
            elif stoch_k > stoch_d and stoch_k < self.STOCH_BEAR_ZONE:
                score += 1
                reasons.append(f"✅ Stochastic bullish crossup ({stoch_k:.1f})")
            elif stoch_k < stoch_d and stoch_k > self.STOCH_BULL_ZONE:
                score -= 1
                reasons.append(f"❌ Stochastic bearish crossdown ({stoch_k:.1f})")

        # ── Filter 7: Volume Confirmation via CMF (M5 threshold: 0.05) ────────
        cmf = last.get("cmf", None)
        if cmf is not None:
            if cmf > self.CMF_THRESHOLD:
                score += 1
                reasons.append(f"✅ Positive CMF={cmf:.3f} (buying pressure)")
            elif cmf < -self.CMF_THRESHOLD:
                score -= 1
                reasons.append(f"❌ Negative CMF={cmf:.3f} (selling pressure)")

        # ── Filter 8: Volatility Squeeze Breakout ─────────────────────────────
        squeeze    = last.get("squeeze",    None)
        prev_sq    = prev.get("squeeze",    None)
        is_bullish = last.get("is_bullish", None)

        if squeeze is not None and prev_sq is not None:
            if squeeze == 0 and prev_sq == 1:
                # Squeeze just released — determine direction
                if is_bullish is not None:
                    direction = 1 if is_bullish else -1
                else:
                    # Fallback: use close-vs-prev-close direction
                    prev_close = prev.get("close", None)
                    if prev_close is not None and close is not None:
                        direction = 1 if close > prev_close else -1
                    else:
                        direction = 0

                if direction != 0:
                    score += direction
                    label = "bullish" if direction == 1 else "bearish"
                    reasons.append(f"⚡ Squeeze breakout ({label})")

        # ── Determine Signal ───────────────────────────────────────────────────
        strength = abs(score) / 8.0
        atr      = last.get("atr",   0.0001)
        entry    = last.get("close", 0.0)

        logger.debug(
            f"📊 {symbol} score={score} "
            f"(threshold ±{self.BULL_THRESHOLD}) | "
            f"filters: {' | '.join(reasons)}"
        )

        if score >= self.BULL_THRESHOLD:
            sl          = round(entry - (atr * 1.5),              5)
            tp          = round(entry + (atr * 1.5 * self.cfg.RR_RATIO), 5)
            signal_type = SignalType.BUY
            confidence  = round(min(score / 8.0, 1.0), 3)

        elif score <= -self.BEAR_THRESHOLD:
            sl          = round(entry + (atr * 1.5),              5)
            tp          = round(entry - (atr * 1.5 * self.cfg.RR_RATIO), 5)
            signal_type = SignalType.SELL
            confidence  = round(min(abs(score) / 8.0, 1.0), 3)

        else:
            logger.debug(
                f"⛔ Gate 2 BLOCKED — {symbol} score={score} "
                f"below threshold ±{self.BULL_THRESHOLD}"
            )
            return None  # confluence too weak — no trade

        signal = TradingSignal(
            symbol     = symbol,
            signal     = signal_type,
            strength   = round(strength, 3),
            confidence = confidence,
            entry      = round(entry, 5),
            sl         = sl,
            tp         = tp,
            atr        = round(atr, 5),
            reasons    = reasons,
            timestamp  = str(df.index[-1]),
        )

        logger.info(
            f"🎯 Signal [{signal_type.value}] on {symbol} | "
            f"Score={score} | Conf={confidence:.0%} | "
            f"Entry={entry:.5f} SL={sl:.5f} TP={tp:.5f}"
        )
        return signal
