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
    9 filters total — max possible score is +9 or -9.

    Filter 9 (new): Price Structure — detects higher highs/higher lows (BUY)
    or lower highs/lower lows (SELL) using the last 6 candles.
    This replaces ADX as the primary trend confirmation on M5.
    """

    # ── M5-tuned indicator periods ────────────────────────────────────────────
    ADX_THRESHOLD     = 18
    RSI_BULL_LOW      = 35
    RSI_BULL_HIGH     = 60
    RSI_BEAR_LOW      = 60
    RSI_BEAR_HIGH     = 65
    CMF_THRESHOLD     = 0.05
    STOCH_BULL_ZONE   = 20
    STOCH_BEAR_ZONE   = 80
    STRUCTURE_LOOKBACK = 6   # candles to check for HH/HL or LH/LL

    def __init__(self, config=CONFIG, trading_style: str = "scalper"):
        self.cfg           = config
        self.trading_style = trading_style
        self._update_thresholds()

    def _update_thresholds(self) -> None:
        if self.trading_style == "scalper" and hasattr(self.cfg, "SCALPER_SIGNAL_SCORE"):
            self.BULL_THRESHOLD = self.cfg.SCALPER_SIGNAL_SCORE
            self.BEAR_THRESHOLD = self.cfg.SCALPER_SIGNAL_SCORE
        elif self.trading_style == "daytrader" and hasattr(self.cfg, "DAYTRADER_SIGNAL_SCORE"):
            self.BULL_THRESHOLD = self.cfg.DAYTRADER_SIGNAL_SCORE
            self.BEAR_THRESHOLD = self.cfg.DAYTRADER_SIGNAL_SCORE
        else:
            self.BULL_THRESHOLD = 4
            self.BEAR_THRESHOLD = 4

    def _price_structure(self, df: pd.DataFrame) -> int:
        """
        Filter 9 — Price Structure detector.
        Looks at the last STRUCTURE_LOOKBACK candles and checks for:
          BUY  (+1): higher highs AND higher lows  (uptrend structure)
          SELL (-1): lower highs  AND lower lows   (downtrend structure)
          NEUTRAL (0): mixed / choppy structure

        Uses candle highs and lows directly — no lag, reacts immediately
        to price action unlike EMA or ADX.
        """
        try:
            window = df.iloc[-(self.STRUCTURE_LOOKBACK + 1):-1]
            if len(window) < self.STRUCTURE_LOOKBACK:
                return 0

            highs = window["high"].values
            lows  = window["low"].values

            # Check last 3 swing highs and lows
            hh = all(highs[i] > highs[i - 1] for i in range(1, len(highs)))
            hl = all(lows[i]  > lows[i - 1]  for i in range(1, len(lows)))
            lh = all(highs[i] < highs[i - 1] for i in range(1, len(highs)))
            ll = all(lows[i]  < lows[i - 1]  for i in range(1, len(lows)))

            if hh and hl:
                return 1   # bullish structure
            elif lh and ll:
                return -1  # bearish structure
            else:
                return 0   # no clear structure
        except Exception:
            return 0

    def evaluate(
        self, df: pd.DataFrame, symbol: str
    ) -> Optional[TradingSignal]:
        if df is None or len(df) < 2:
            return None

        df = df.copy()
        df.columns = df.columns.str.lower()

        last  = df.iloc[-1]
        prev  = df.iloc[-2]
        score = 0
        reasons: list = []

        # ── Filter 1: Trend Alignment (EMA 9/21) ─────────────────────────────
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
        macd      = last.get("macd",        None)
        macd_sig  = last.get("macd_signal", None)
        prev_macd = prev.get("macd",        None)
        prev_msig = prev.get("macd_signal", None)

        if None not in (macd, macd_sig, prev_macd, prev_msig):
            bull_cross = prev_macd < prev_msig and macd > macd_sig
            bear_cross = prev_macd > prev_msig and macd < macd_sig
            if bull_cross:
                score += 1
                reasons.append("✅ MACD bullish crossover")
            elif bear_cross:
                score -= 1
                reasons.append("❌ MACD bearish crossover")
            elif macd > macd_sig and macd > prev_macd:
                score += 1
                reasons.append("✅ MACD histogram expanding bullish")
            elif macd < macd_sig and macd < prev_macd:
                score -= 1
                reasons.append("❌ MACD histogram expanding bearish")

        # ── Filter 3: RSI Zone ────────────────────────────────────────────────
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
        close    = last.get("close",    None)
        bb_mid   = last.get("bb_mid",   None)
        bb_upper = last.get("bb_upper", None)
        bb_lower = last.get("bb_lower", None)

        if None not in (close, bb_mid, bb_upper, bb_lower):
            if bb_mid < close < bb_upper:
                score += 1
                reasons.append("✅ Price above BB midline")
            elif bb_lower < close < bb_mid:
                score -= 1
                reasons.append("❌ Price below BB midline")
            elif close >= bb_upper:
                score += 1
                reasons.append("✅ Price at/above BB upper (strong bull momentum)")
            elif close <= bb_lower:
                score -= 1
                reasons.append("❌ Price at/below BB lower (strong bear momentum)")

        # ── Filter 5: ADX Trend Strength ──────────────────────────────────────
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

        # ── Filter 6: Stochastic ──────────────────────────────────────────────
        stoch_k = last.get("stoch_k", None)
        stoch_d = last.get("stoch_d", None)

        if None not in (stoch_k, stoch_d):
            if stoch_k <= self.STOCH_BULL_ZONE:
                score += 1
                reasons.append(f"✅ Stochastic oversold ({stoch_k:.1f})")
            elif stoch_k >= self.STOCH_BEAR_ZONE:
                score -= 1
                reasons.append(f"❌ Stochastic overbought ({stoch_k:.1f})")
            elif stoch_k > stoch_d and stoch_k < self.STOCH_BEAR_ZONE:
                score += 1
                reasons.append(f"✅ Stochastic bullish crossup ({stoch_k:.1f})")
            elif stoch_k < stoch_d and stoch_k > self.STOCH_BULL_ZONE:
                score -= 1
                reasons.append(f"❌ Stochastic bearish crossdown ({stoch_k:.1f})")

        # ── Filter 7: CMF Volume ──────────────────────────────────────────────
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
                if is_bullish is not None:
                    direction = 1 if is_bullish else -1
                else:
                    prev_close = prev.get("close", None)
                    if prev_close is not None and close is not None:
                        direction = 1 if close > prev_close else -1
                    else:
                        direction = 0
                if direction != 0:
                    score += direction
                    label = "bullish" if direction == 1 else "bearish"
                    reasons.append(f"⚡ Squeeze breakout ({label})")

        # ── Filter 9: Price Structure (HH/HL or LH/LL) ───────────────────────
        structure = self._price_structure(df)
        if structure == 1:
            score += 1
            reasons.append("✅ Bullish structure (HH + HL confirmed)")
        elif structure == -1:
            score -= 1
            reasons.append("❌ Bearish structure (LH + LL confirmed)")
        else:
            reasons.append("⚠️  No clear price structure")

        # ── Determine Signal ───────────────────────────────────────────────────
        strength = abs(score) / 9.0
        atr      = last.get("atr",   0.0001)
        entry    = last.get("close", 0.0)

        logger.debug(
            f"📊 {symbol} score={score} "
            f"(threshold ±{self.BULL_THRESHOLD}) | "
            f"filters: {' | '.join(reasons)}"
        )

        if score >= self.BULL_THRESHOLD:
            sl          = round(entry - (atr * 1.5),                    5)
            tp          = round(entry + (atr * 1.5 * self.cfg.RR_RATIO), 5)
            signal_type = SignalType.BUY
            confidence  = round(min(score / 9.0, 1.0), 3)

        elif score <= -self.BEAR_THRESHOLD:
            sl          = round(entry + (atr * 1.5),                    5)
            tp          = round(entry - (atr * 1.5 * self.cfg.RR_RATIO), 5)
            signal_type = SignalType.SELL
            confidence  = round(min(abs(score) / 9.0, 1.0), 3)

        else:
            logger.debug(
                f"⛔ Gate 2 BLOCKED — {symbol} score={score} "
                f"below threshold ±{self.BULL_THRESHOLD}"
            )
            return None

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